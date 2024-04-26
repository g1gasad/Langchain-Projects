import streamlit as st 
from PyPDF2 import PdfFileReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain.vectorstores import faiss
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfFileReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text 

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = faiss.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to
    provide all the details, if the answer is not in the provided context just say, 
    'answer is not available in the context.', don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain=load_qa_chain(model, chain_type='stuff', prompt=promt)
    return chain
    