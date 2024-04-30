from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI
import os
from langchain.agents import load_tools
from langchain import hub
from langchain.chains import LLMMathChain
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.utilities import PythonREPL , GoogleSerperAPIWrapper , WikipediaAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
from crewai import Agent, Task, Process, Crew
from langchain_community.agent_toolkits import GmailToolkit
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools
from tools.browser_tools import BrowserTools


from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-pro")
llm1 = OpenAI()
prompt = hub.pull("hwchase17/react")


search_tool = Tool(
    name="Scrape google searches",
    func=GoogleSerperAPIWrapper().run,
    description="useful for when you need to ask the agent to search the internet",
)

code_executor =  Tool(
        name = "python_repl",
        func=PythonREPL().run,
        description = "useful when you need to use python to answer a question. You should input python code"
)

llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
math_tool = Tool(
        func=llm_math_chain.run,
        name="Calculator",
        description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions.",
)

def email(query):
    toolkit = GmailToolkit()
    agent_1 = create_react_agent(llm, tools, prompt)
    agent_executor_1 = AgentExecutor(agent=agent_1, tools=toolkit.get_tools(), verbose=True,handle_parsing_errors=True)
    agent_executor_1.invoke({'input' : query})

email_tool = Tool(
    func = email,
    name = "Email",
    description = "Useful when you have any operation regarding email you have to pass the complete query as argument"
)

def general(query:str):
    return llm.invoke(query)

general_tool = Tool(
    func = general,
    name = "General",
    description = "Useful when you have to answer any general question"
)


def Research(topic : str):
    explorer = Agent(
    role="Senior Researcher",
    goal=f"Find and explore on {topic} for 2024",
    backstory=f"""You are and Expert strategist that knows how to spot emerging trends related to {topic}.
    You're great at finding and exploring on {topic}. You turned scraped data into detailed reports with useful insights.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm = llm
    )


    writer = Agent(
    role="Senior Technical Writer",
    goal=f"Write engaging and interesting blog post about {topic}, layman vocabulary",
    backstory=f"""You are an Expert Writer on technical innovation, especially in the field of {topic}. You know how to write in
    engaging, interesting but simple, straightforward and concise. You know how to present complicated technical terms to general audience in a
    fun way by using layman words.ONLY use scraped data from the internet for the blog.""",
    verbose=True,
    allow_delegation=True,
    llm = llm
    )


    critic = Agent(
    role="Expert Writing Critic",
    goal="Provide feedback and criticize blog post drafts. Make sure that the tone and writing style is compelling, simple and concise",
    backstory= f"""You are an Expert at providing feedback to the technical writers. You can tell when a blog text isn't concise,
    simple or engaging enough. You know how to provide helpful feedback that can improve any text. You know how to make sure that text
    stays technical and insightful by using layman terms.
    """,
    verbose=True,
    allow_delegation=True,
    llm = llm
    )


    task_report = Task(
    description=f"""Use and summarize scraped data from the internet to make a detailed report on the latest news and trends regarding {topic}. Use ONLY
    scraped data to generate the report. Your final answer MUST be a full analysis report, text only, ignore any code or anything that
    isn't text. The report has to have bullet points and with 5-10 exciting facts about {topic}. 
    """,
    agent=explorer,
    )


    task_blog = Task(
    description=f"""Write a blog article with text only and with a short but impactful headline and at least 10 paragraphs. Blog should summarize
    the report on latest ai tools found on localLLama subreddit. Style and tone should be compelling and concise, fun, technical but also use
    layman words for the general public. Name specific new, interesting facts about {topic}. Don't
    write "**Paragraph [number of the paragraph]:**", instead start the new paragraph in a new line. 
    For your Outputs use the following markdown format:
    ```
    ## [Title of post](link to project)
    - Interesting facts
    - Own thoughts on how it connects to the overall theme of the newsletter
    ## [Title of second post](link to project)
    - Interesting facts
    - Own thoughts on how it connects to the overall theme of the newsletter
    ```
    """,
    agent=writer,
    )


    task_critique = Task(
    description="""The Output MUST have the following markdown format:
    ```
    ## [Title of post](link to project)
    - Interesting facts
    - Own thoughts on how it connects to the overall theme of the newsletter
    ## [Title of second post](link to project)
    - Interesting facts
    - Own thoughts on how it connects to the overall theme of the newsletter
    ```
    Make sure that it does and if it doesn't, rewrite it accordingly.
    """,
    agent=critic,
    )


    crew = Crew(
    agents=[explorer, writer, critic],
    tasks=[task_report, task_blog, task_critique],
    verbose=2,
    process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
    )


    result = crew.kickoff()

    print("######################")
    return result

Researcher = Tool(
    func=Research,
    name="Researcher",
    description="Useful only when the user mentions to research. Only input the topic of research you want."
)

def startup(idea :str):
    marketer = Agent(
    role="Market Research Analyst",
    goal="Find out how big is the demand for my products and suggest how to reach the widest possible customer base",
    backstory="""You are an expert at understanding the market demand, target audience, and competition. This is crucial for
		validating whether an idea fulfills a market need and has the potential to attract a wide audience. You are good at coming up
		with ideas on how to appeal to widest possible audience.
		""",
    verbose=True,  # enable more detailed or extensive output
    allow_delegation=True,  # enable collaboration between agent
      llm=llm, # to load gemini,
    tools = [search_tool]
    )

    technologist = Agent(
    role="Technology Expert",
    goal="Make assessment on how technologically feasable the company is and what type of technologies the company needs to adopt in order to succeed",
    backstory="""You are a visionary in the realm of technology, with a deep understanding of both current and emerging technological trends. Your
		expertise lies not just in knowing the technology but in foreseeing how it can be leveraged to solve real-world problems and drive business innovation.
		You have a knack for identifying which technological solutions best fit different business models and needs, ensuring that companies stay ahead of
		the curve. Your insights are crucial in aligning technology with business strategies, ensuring that the technological adoption not only enhances
		operational efficiency but also provides a competitive edge in the market.""",
    verbose=True,  # enable more detailed or extensive output
    allow_delegation=True,  # enable collaboration between agent
      llm=llm ,# to load gemin,i
    tools = [search_tool]
    )

    business_consultant = Agent(
    role="Business Development Consultant",
    goal="Evaluate and advise on the business model, scalability, and potential revenue streams to ensure long-term sustainability and profitability",
    backstory="""You are a seasoned professional with expertise in shaping business strategies. Your insight is essential for turning innovative ideas
		into viable business models. You have a keen understanding of various industries and are adept at identifying and developing potential revenue streams.
		Your experience in scalability ensures that a business can grow without compromising its values or operational efficiency. Your advice is not just
		about immediate gains but about building a resilient and adaptable business that can thrive in a changing market.""",
    verbose=True,  # enable more detailed or extensive output
    allow_delegation=True,  # enable collaboration between agent
      llm=llm ,# to load gemini
    tools = [search_tool]
    )

    task1 = Task(
    description=f"""Analyze what the market demand for {idea}.
		Write a detailed report with description of what the ideal customer might look like, and how to reach the widest possible audience. The report has to
		be concise with at least 10 bullet points and it has to address the most important areas when it comes to marketing this type of business.
    """,
    agent=marketer)
    task2 = Task(
    description=f"""Analyze how to produce a high quality {idea}. Write a detailed report
		with description of which technologies the business needs to use in order to make {idea} possible. The report has to be concise with
		at least 10  bullet points and it has to address the most important areas when it comes to manufacturing this type of business.
    """,
    agent=technologist,
    )
    
    task3 = Task(
    description=f"""Analyze and summarize marketing and technological report and write a detailed business plan with
		description of how to make a sustainable and profitable {idea} business.
		The business plan has to be concise with
		at least 10  bullet points, 5 goals and it has to contain a time schedule for which goal should be achieved and when.
    """,
    agent=business_consultant)

    crew = Crew(
    agents=[marketer, technologist, business_consultant],
    tasks=[task1, task2, task3],
    verbose=2,
    process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
    )
    result = crew.kickoff()
    print("######################")
    return result


startup_researcher = Tool(
    name="Startup Researcher",
    func=startup,
    description="Useful when user specifically asks to do indepth analysis or research about a new startup. This tool is only for startup questions and nothing else. Only input idea of the startup",
)


def LinkedIn(topic : str):
    coach = Agent(
    role='Senior Career Coach',
    goal= f"Discover and examine on {topic} for 2024",
    backstory= f"You're an expert in spotting new trends and essential skills regarding {topic}.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm = llm
    )

    influencer = Agent(
    role='LinkedIn Influencer Writer',
    goal="Write catchy, emoji-filled LinkedIn posts within 200 words",
    backstory= f"You're a specialised writer on LinkedIn, focusing on {topic}.",
    verbose=True,
    allow_delegation=True,
    llm=llm
    )

    critic = Agent(
    role='Expert Writing Critic',
    goal="Give constructive feedback on post drafts",
    backstory="You're skilled in offering straightforward, effective advice to tech writers. Ensure posts are concise, under 200 words, with emojis and hashtags.",
    verbose=True,
    allow_delegation=True,
    llm=llm
    )

    task_search = Task(
    description= f"Compile a report listing skills and requirement regarding {topic}, presented in bullet points",
    agent=coach
        )
    task_post = Task(
        description=f"Create a LinkedIn post with a brief headline and a maximum of 200 words, focusing {topic}",
        agent=influencer
        )
    task_critique = Task(
        description=f"Refine the post for brevity, ensuring an engaging headline (no more than 30 characters) about {topic} and keeping within a 200-word limit",
        agent=critic
        )
    crew = Crew(
    agents=[coach, influencer, critic],
    tasks=[task_search, task_post, task_critique],
    verbose=2,
    process=Process.sequential
    )
    result = crew.kickoff()
    print("#############")
    return (result)


linkedin_post_generator = Tool(
    name = "linkedin_post",
    func = LinkedIn,
    description = "Useful for when you need to write a linkedin post. You should input the topic of the post"
)

def stock_analysis(company : str):
    finance_analyst = Agent(
        role='The Best Financial Analyst',
      goal="""Impress all customers with your financial data 
      and market trends analysis""",
      backstory="""The most seasoned financial analyst with 
      lots of expertise in stock market analysis and investment
      strategies that is working for a super important customer.""",
      verbose=True,
      tools=[
          CalculatorTools.calculate,
          search_tool
      ],
      llm = llm
    )

    research_analyst = Agent(
        role='Staff Research Analyst',
        goal="""Being the best at gather, interpret data and amaze
        your customer with it""",
        backstory="""Known as the BEST research analyst, you're
        skilled in sifting through news, company announcements, 
        and market sentiments. Now you're working on a super 
        important customer""",
        verbose=True,
        tools=[
            CalculatorTools.calculate,
            search_tool
        ],
        llm = llm
    )

    investment_advisor = Agent(
        role='Private Investment Advisor',
      goal="""Impress your customers with full analyses over stocks
      and completer investment recommendations""",
      backstory="""You're the most experienced investment advisor
      and you combine various analytical insights to formulate
      strategic investment advice. You are now working for
      a super important customer you need to impress.""",
      verbose=True,
      tools=[
        SearchTools.search_internet,
        SearchTools.search_news,
        CalculatorTools.calculate,
        search_tool
      ],
      llm = llm
    )

    task1 = Task(
        description=f"""
        Collect and summarize recent news articles, press
        releases, and market analyses related to the stock and
        its industry.
        Pay special attention to any significant events, market
        sentiments, and analysts' opinions. Also include upcoming 
        events like earnings and others.
  
        Your final answer MUST be a report that includes a
        comprehensive summary of the latest news, any notable
        shifts in market sentiment, and potential impacts on 
        the stock.
        Also make sure to return the stock ticker.
        
        {company}
  
        Make sure to use the most recent data as possible.
  
        Selected company by the customer: {company}
      """,
      agent=research_analyst
    )

    task2 = Task(description=f"""
        Conduct a thorough analysis of the {company}'s financial
        health and market performance. 
        This includes examining key financial metrics such as
        P/E ratio, EPS growth, revenue trends, and 
        debt-to-equity ratio. 
        Also, analyze the stock's performance in comparison 
        to its industry peers and overall market trends.

        Your final report MUST expand on the summary provided
        but now including a clear assessment of the stock's
        financial standing, its strengths and weaknesses, 
        and how it fares against its competitors in the current
        market scenario.{company}

        Make sure to use the most recent data possible.
      """,
      agent=finance_analyst
    )

    task3 = Task(description=f"""
        Analyze the latest 10-Q and 10-K filings for
        the {company}'s stock in question. 
        Focus on key sections like Management's Discussion and
        Analysis, financial statements, insider trading activity, 
        and any disclosed risks.
        Extract relevant data and insights that could influence
        the stock's future performance.

        Your final answer must be an expanded report that now
        also highlights significant findings from these filings,
        including any red flags or positive indicators for
        your customer.
        {company}        
      """,
      agent=finance_analyst
    )

    task4 = Task(description=f"""
        Review and synthesize the analyses provided by the
        Financial Analyst and the Research Analyst.
        Combine these insights to form a comprehensive
        investment recommendation. 
        
        You MUST Consider all aspects, including financial
        health, market sentiment, and qualitative data.

        Make sure to include a section that shows insider 
        trading activity, and upcoming events like earnings.

        Your final answer MUST be a recommendation for your
        customer. It should be a full super detailed report, providing a 
        clear investment stance and strategy with supporting evidence.
        Make it pretty and well formatted for your customer.
        {company}
      """,
      agent=investment_advisor
    )
    crew = Crew(
    agents=[
        research_analyst,
        finance_analyst,
        investment_advisor
    ],
    tasks=[task1,task2,task3,task4],
    verbose=2,
    process=Process.sequential
    )
    result = crew.kickoff()
    print("#############")
    return (result)

stock_analyser = Tool(
    name="Stock Analyser",
    func=stock_analysis,
    description="Use when you are specifically asked to do stock analysis about a company. Pass just the name of the company"
)

tools = [stock_analyser , startup_researcher , Researcher , general_tool , email_tool , search_tool , code_executor , math_tool , linkedin_post_generator]
agent = create_react_agent(llm1, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

# answer = agent_executor.invoke({"input" : "Who is pm of india"})

# print(answer['output'])

def general(query : str):
    return agent_executor.invoke({"input" : query})['output']