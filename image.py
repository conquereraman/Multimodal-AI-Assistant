import os
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import PIL.Image as pil
from dotenv import load_dotenv
load_dotenv()

llm = OpenAI()
question = "What is in the image"

model = genai.GenerativeModel('gemini-pro-vision')

template = """ You are given a description of what is in the image and then you will be given what the user wants based on your knowledge provide the answer
description :
{description}
question:
{question}
"""


def solve_image(file,question):
    img = pil.open(file)
    description = model.generate_content(img).text
    prompt_template = PromptTemplate(template = template,input_variables = ['description' , 'question'])
    prompt_template.format(
        description = description,
        question = question
    )
    llm_chain = LLMChain(
        prompt = prompt_template,
        llm = llm,
        verbose = 1
    )
    return (llm_chain.invoke({"description" : description , "question" : question})['text'])

# llm1 = OpenAI()


