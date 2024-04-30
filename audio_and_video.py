import whisper
import moviepy.editor as mp
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import LLMChain
from langchain.agents import Tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from dotenv import load_dotenv
load_dotenv()

transcriber = whisper.load_model("base")

model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.2,convert_system_message_to_human=True)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)


def solve_audio_video(file,query):
   file = file.name
   # file_name = file
   file_extension = file.split('.')[-1].lower()
   if file_extension == 'mp4':
      video = mp.VideoFileClip(file)
      audio = video.audio
      audio.write_audiofile("audio.mp3")
      result = transcriber.transcribe("audio.mp3")
   else:
      result = transcriber.transcribe(file)
   context = result["text"]
   def ask_doc(question :str):
      texts = text_splitter.split_text(context)
      vector_index = Chroma.from_texts(texts, embeddings).as_retriever()
      qa_chain = RetrievalQA.from_chain_type(
         model,
         retriever=vector_index,
         return_source_documents=True
      )
      result = qa_chain.invoke({"query": question})
      return result["result"]

   def summarise(input = ""):
      refine_prompt_template = """
                  Write a concise summary of the following text delimited by triple backquotes.
                  Return your response in bullet points which covers the key points of the text.
                  ```{text}```
                  BULLET POINT SUMMARY:
                  """
      refine_prompt = PromptTemplate(
         template=refine_prompt_template, input_variables=["text"]
      )
      chain = LLMChain(
      prompt = refine_prompt,
      llm = model
      )
      return chain.invoke(context)

   summarise_tool = Tool(
      name = 'Summariser',
      func = summarise,
      description="Useful when it is required to summarise"
   )

   question_tool = Tool(
      name = 'Questionaire',
      func = ask_doc,
      description="Useful when you need to answer a question"
   )

   tools = [summarise_tool,question_tool]

   prompt = hub.pull("hwchase17/react")
   agent = create_react_agent(model, tools, prompt)

   agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

   result = agent_executor.invoke({"input": query})
   return result['output']


# solve_audio_video("temp.mp3","Summarise")

   # file_name = "video.mp4"


   # Load the video file
   # video = mp.VideoFileClip("video.mp4")

   # audio = video.audio
   # # Write the audio to a file
   # result = model.transcribe("audio.mp3")
   # print(result["text"])
