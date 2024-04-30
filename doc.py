import docx
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint
import google.generativeai as genai
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
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()
file_name = "./temp.docx"


model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.2,convert_system_message_to_human=True)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)


def extract_from_doc(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    context = '\n'.join(fullText)
    return context

def extract_from_pdf(pdf_file):
  pdf_loader = PyPDFLoader(pdf_file)
  pages = pdf_loader.load_and_split()
  context = "\n".join(str(p.page_content) for p in pages)
  return context

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")




def solve_doc(file,query):
   file_name = file.name
   file_extension = file_extension = file_name.split('.')[-1].lower()
   if file_extension == 'docx':
      context = extract_from_doc(file_name)
   else:
      context = extract_from_pdf(file_name)

   def ask_doc(question :str):
      texts = text_splitter.split_text(context)
      vector_index = Chroma.from_texts(texts, embeddings).as_retriever()
      qa_chain = RetrievalQA.from_chain_type(
         model,
         retriever=vector_index,
         return_source_documents=True
      )
      result = qa_chain({"query": question})
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
      return chain.run(context)

   summarise_tool = Tool(
      name = 'Summariser',
      func = summarise,
      description="Useful when questions asked regarding the document"
   )

   # question_tool = Tool(
   #    name = 'Questionaire',
   #    func = ask_doc,
   #    description="Useful when you are asked a question except summarisation so pass just the question as input"
   # )

   tools = [summarise_tool]

   prompt = hub.pull("hwchase17/react")
   agent = create_react_agent(model, tools, prompt)

   agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

   return agent_executor.invoke({"input": query})['output']

