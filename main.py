from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def load_doc(path: str = 'data'):
    loader = DirectoryLoader(path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f'documents loaded of length: {len(documents)}')
    return documents


def split_documents(documents, chunk_size: int = 500, overlap: int = 50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_documents(documents)


def create_embeddings(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    print(f'Loading Embedding model {model_name}')
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print('embedding model is ready')
    return embeddings


def build_vector_store(chunks, embeddings):
    print("\nbuilding vectorstore ....")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("vector store ready")
    return vectorstore


def main():
    documents = load_doc()
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vectorstore = build_vector_store(chunks, embeddings)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    query = "What is this document about? Give a one sentence answer."
    result = qa.invoke(query)

    print("\n❓ Question:", query)
    print("💬 Answer:", result['result'])


if __name__ == "__main__":
    main()