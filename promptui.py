# from langchain_google_genai import ChatGoogleGenerativeAI
from groq import Groq
from dotenv import load_dotenv
import groq
import streamlit as st
import os
from langchain_core.prompts import PromptTemplate

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
st.header('Research Tool')
paper_title = st.selectbox("Select Research Paper", ["Attention Is All You Need", "BERT", "GPT 3"])
style_input = st.selectbox("Select Explanation Style", ["Simple Explanation", "Technical", "Mathematical","Code Oriented"])
lenght_input = st.selectbox("Select Explanation Length", ["Short", "Medium", "Long"])

template = PromptTemplate(
    template="Summarize the research paper titled '{paper_title}' in a {style_input} style and {lenght_input} length.",
    input_variables=["paper_title", "style_input", "lenght_input"]
)

prompt=template.format(
    paper_title=paper_title,
    style_input=style_input,
    lenght_input=lenght_input
)
# user_input = st.text_input("Enter your prompt:")
if st.button('summarize'):
    result = client.chat.completions.create(
        model='groq/compound',
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    st.write(result.choices[0].message.content)
    
    