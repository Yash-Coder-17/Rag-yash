from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain_core.prompts import PromptTemplate
import pyttsx3
import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import numpy as np

load_dotenv()

sd.default.device= 2

Loader= PyPDFLoader('yash.pdf')
docs = Loader.load()
splitter= RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)
result=splitter.split_documents(docs)
vector_store= Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory='Chroma-db',
    collection_name='yash'
)
vector_store.add_documents(result)
retriever= vector_store.as_retriever(search_type="similarity",search_kwargs={"k":1})
model= ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.9, max_tokens=1024)


print("Start speaking")
y=sd.rec(frames=44100*5, samplerate=44100, channels=2)
sd.wait()
print("stop speaking")
y=np.int16(y*32767)
write("audio.wav",44100,y)
r=sr.Recognizer()
with sr.AudioFile("audio.wav")as source:
    audio=r.record(source)

try:
    query=r.recognize_google(audio)
    print("You said:", query)
except:
    print("Sorry, I could not understand the audio.")



# query=input("Enter your query:")
text= retriever.invoke(query)
context= "\n".join([doc.page_content for doc in text])
template= PromptTemplate(
    template='''
    You are an intelligent assistant helping answer questions from a document.

    Use ONLY the information provided in the context below.

    If the answer is not found in the context, respond with:
    "I don't know."

    Context:
    ---------
    {context}
    ---------

    Question:
    {query}

    Answer clearly and concisely.
''',
    input_variables=["context","query"]
)

prompt= template.format(context=context, query=query)
response= model.invoke(prompt)
print(response.content)
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)
engine.say(response.content)
engine.runAndWait()