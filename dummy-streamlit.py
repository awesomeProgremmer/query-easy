import streamlit as st
from main import load_doc
from main import split_documents
from main import create_embeddings
from main import build_vector_store
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os


load_dotenv()

st.title("dummy rag")
st.write("ask any question about the pdf document")

@st.cache_resource

