from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import streamlit as st
# from langchain_community.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

Loader = PyPDFLoader('rag/yash.pdf')
docs = Loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=10
) 
result = splitter.split_documents([docs[0]])                        #split_document isliye use kiyae kyuki string kao document mai convert karne kae liyae
vectorstore = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    ),
    persist_directory='Chromadb',
    collection_name='yash'
)
vectorstore.add_documents(result)
retriever = vectorstore.as_retriever(Searchtype = "Similarity", search_kwargs={"k": 1})
model = ChatGoogleGenerativeAI(model = "models/gemini-2.5-flash", temperature=0.9, max_tokens=1024)
st.title("Yash RAG System")
st.subheader("Hello! I am Yash Singh")
col1, col2 = st.columns(2)
with col1:
    st.text("I enjoy learning about new technologies and sharing my knowledge with others. I am passionate about AI, machine learning, and data science. I am always eager to explore new ideas and collaborate on exciting projects. Let's connect and learn together!")
with col2:
    st.image("rag/yash17.jpeg", width = 150)
st.subheader("What do you want to know about me?")
query = st.text_input("Enter your question:")
if st.button('Get Answer'):

    text = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in text])                       #document kao strng mai convert karne kae liye

    prompt_template = """
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
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "query"]
    )

    final_prompt = prompt.format(context=context, query=query)

    response = model.invoke(final_prompt)
    st.write(response.content)
