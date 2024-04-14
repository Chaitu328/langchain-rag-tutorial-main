import bs4
import tempfile
import os
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
import getpass

# Take environment variables from .env.
load_dotenv()

st.title("Testing Promps from Document")

# Create form
with st.form("user_inputs"):
    # File uploaded
    uploaded_file = st.file_uploader("Upload a file")

    # Input field
    input_field = st.text_input("Enter your Prompt")

    # Add button
    button = st.form_submit_button("Answer")

def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, "uploaded_file.txt")
        with open(temp_filepath, "wb") as f:
            f.write(file.read())  # Use read() instead of getvalue() to handle bytes
        # Use TextLoader instead of PyPDFLoader
        loader = TextLoader(temp_filepath, encoding='UTF-8')  # Assuming UTF-8 encoding
        docs.extend(loader.load())
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(docs)
    # store
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    # RAG prompt
    prompt = hub.pull("rlm/rag-prompt")
    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    result = qa_chain({"query": input_field})
    return result["result"], vectorstore  # Return result and vectorstore

# Check if file is uploaded
if not uploaded_file:
    st.info("Please upload to continue.")
    st.stop()

# Configure retriever and vectorstore
retriever, vectorstore = configure_retriever([uploaded_file])  

# Perform action on button click
if button and retriever is not None:
    st.write(retriever)
    # Cleanup vectorstore
    vectorstore.delete_collection()
