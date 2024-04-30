import os
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv
load_dotenv()

llm1 = OpenAI()
model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.2,convert_system_message_to_human=True)
def solve_excel(file,query):
    file_extension = file.name.split('.')[-1].lower()
    # file_extension = file.split('.')[-1].lower()
    if (file_extension == "xlsx"):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    agent = create_pandas_dataframe_agent(
        model,
        df,
        verbose=True
    )
    response = agent.invoke(query)['output']
    return response
# print(solve_excel("test.csv","What columns are there"))
# print(solve_excel("test.csv","What columns are there"))
# agent = create_csv_agent(
#     model,
#     "test.csv",
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# )

# df = pd.read_csv("test.csv")
# agent = create_pandas_dataframe_agent(
#         model,
#         df,
#         verbose=True
#     )
# response = agent.invoke("What columns are there")['output']
# print (response)