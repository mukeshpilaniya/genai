import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os

# Streamlit Page Config
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄 Chat with your PDF using LangChain & Ollama")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Save the file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF using PyPDFLoader
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Initialize Ollama Embeddings
        embedding_model = OllamaEmbeddings(model="deepseek-r1:1.5b")  # You can use "gemma", "llama3" etc.

        # Create FAISS vector database
        vector_db = FAISS.from_documents(chunks, embedding_model)

        # Initialize LLM
        llm = Ollama(model="deepseek-r1:1.5b")  # Load Ollama LLM

        # Create RetrievalQA Chain
        retriever = vector_db.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("PDF processed! You can now ask questions.")

        # Chat Input
        query = st.text_input("Ask a question about the document:")

        if query:
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({"query": query})
                st.write("**Answer:**", response["result"])
