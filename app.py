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

st.title("📄 PDF Chat Assistant")
st.write("Ask any question about your PDF documents!")

@st.cache_resource
def build_qa_chain():
    with st.spinner("Loading PDF and building knowledge base..."):
        documents = load_doc()
        chunks = split_documents(documents)
        embeddings = create_embeddings()
        vectorstore = build_vector_store(chunks, embeddings)
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )
    return qa

qa = build_qa_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if query := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa.invoke(query)
            answer = result["result"]
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})