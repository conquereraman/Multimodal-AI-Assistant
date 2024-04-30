import streamlit as st
from general import general
from excel import solve_excel
from audio_and_video import solve_audio_video
from doc import solve_doc
from image import solve_image
from langchain import hub
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import os

# llm = ChatGoogleGenerativeAI(model="gemini-pro",streaming=True)
llm = OpenAI()


def generate_response(uploaded_file, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        filename = uploaded_file.name

        # Save the file to the current directory
        with open(os.path.join(filename), "wb") as f:
            f.write(uploaded_file.getvalue())
        extension = uploaded_file.name.split('.')[-1].lower()
        if(extension == 'xlsx' or extension == 'csv'):
            return solve_excel(uploaded_file, query_text)
            # return "Its an excel"
        elif(extension == 'mp4' or extension == 'mp3' or extension == 'wav'):
            return solve_audio_video(uploaded_file,query_text)
        elif(extension == 'docx' or extension == 'pdf'):
            return solve_doc(uploaded_file,query_text)
            # return "Its a document"
        elif(extension == 'png' or extension == 'jpg' or extension == 'jpeg'):
            return solve_image(uploaded_file,query_text)
        else:
            return "Error"
    else:
        return general(query_text)


# Page title
st.set_page_config(page_title='MultiModal AI Assistant')
st.title('MultiModal AI Assistant')

# File upload
uploaded_file = st.file_uploader('Upload an article')
if uploaded_file is not None:
    st.write(uploaded_file.name)
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.')

# Form input and query
if(st.button("Submit")):
    if(len(query_text)):
        prompt = """
        You are an expert Prompt Writer for Large Language Models.

        Your goal is to improve the prompt given below for {task} :
        --------------------

        Prompt: {lazy_prompt}

        --------------------

        Here are several tips on writing great prompts:

        -------

        Correct the grammar and make the prompt precise and single liner 

        ---------

        For example
        Bad: "Bro image what"
        Good: "Describe the content of the image."

        Bad: "Tell me about a cat"
        Good: "Write about the characteristics and behavior of a cat."

        Bad: "Make a poem"
        Good: "Create a poem on any topic that inspires you."

        Bad: "Write about the beach"
        Good: "Write a piece evoking the atmosphere and sensations of a day at the beach."

        Bad: "Story about love"
        Good: "Craft a narrative exploring the theme of love in any setting or context."

        Bad: "Discuss trees"
        Good: "Share your thoughts on the importance of trees in the environment."

        Bad: "Talk about history"
        Good: "Reflect on the significance of historical events and their impact on society."

        -----

        Now, improve the prompt.

        IMPROVED PROMPT:"""

        obj = PromptTemplate(
            template = prompt,
            input_variables = ['lazy_prompt','task']
        )
        
        llm_chain = LLMChain(
            prompt = obj,
            llm = llm,
            verbose = 1
        )
        query_text = llm_chain.invoke({'lazy_prompt': query_text,'task' : "passing to assistant to do the task"})['text']

        st.write(query_text)
        response = generate_response(uploaded_file, query_text)
        if(len(response)):
            st.write(response)
