import tempfile
import os
from langchain.document_loaders import TextLoader, PDFPlumberLoader, UnstructuredXMLLoader, Docx2txtLoader
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_chroma import Chroma
from langchain.agents import AgentType
from langchain.agents import load_tools 
from langchain.agents import initialize_agent
from langchain_openai import ChatOpenAI
from langchain.llms import OpenAI
from langchain import hub
import streamlit as st
from dotenv import load_dotenv

# Take environment variables from .env.
load_dotenv()

st.title("Testing Prompts from Document")

# Create form
with st.form("user_inputs"):
    # File uploaded
    uploaded_file = st.file_uploader("Upload a file")

    # Input field
    input_field = st.text_input("Enter your Prompt")

    # Add button
    button = st.form_submit_button("Answer")

def load_documents(uploaded_files):
    docs = []
    for file in uploaded_files:
        file_extension = file.name.split('.')[-1].lower()
        file_content = file.getvalue()  # Get the file content
        if file_extension == 'pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_content)
                temp_file.seek(0)  # Move to the beginning of the file
                temp_filepath = temp_file.name
            loader = PDFPlumberLoader(temp_filepath)
            docs.extend(loader.load())
        elif file_extension == 'xml':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as temp_file:
                temp_file.write(file_content)
                temp_file.seek(0)
                temp_filepath = temp_file.name
            loader = UnstructuredXMLLoader(temp_filepath)
            docs.extend(loader.load())
        elif file_extension == 'xlsx':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
                temp_file.write(file_content)
                temp_file.seek(0)
                temp_filepath = temp_file.name
            loader = UnstructuredExcelLoader(temp_filepath)
            docs.extend(loader.load())
        elif file_extension == 'docx':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                temp_file.write(file_content)
                temp_file.seek(0)
                temp_filepath = temp_file.name
            loader = Docx2txtLoader(temp_filepath)
            docs.extend(loader.load())
        else:
            # Default loader for text files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                temp_file.write(file_content)
                temp_file.seek(0)
                temp_filepath = temp_file.name
            loader = TextLoader(temp_filepath, encoding='UTF-8')
            docs.extend(loader.load())
    return docs

def configure_retriever(uploaded_files, input_field):
    # Load documents
    docs = load_documents(uploaded_files)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(docs)

    # Create vectorstore
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

    # Configure RetrievalQA chain
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

    # Execute the query
    result = qa_chain({"query": input_field})

    # Return result and vectorstore
    return result["result"], vectorstore

# Check if file is uploaded
if not uploaded_file:
    st.info("Please upload a file to continue.")
    st.stop()

# serapi
def get_output(input_field):
    load_dotenv()

    serpapi_api_key2 = os.environ["serpapi_api_key"]
    llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

    tool = load_tools(["serpapi"], serpapi_api_key=serpapi_api_key2, llm=llm)
    agent = initialize_agent(tool, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    result = agent.invoke(input_field)

    return result["output"]

# Perform action on button click
if button:
    result, vectorstore = configure_retriever([uploaded_file], input_field)
    if any(substring in result.lower() for substring in ["i don't know", "context is not known", "values are not known", "no, ", "did not meet in the provided context."]):
        # Use the Zero-shot react model to get the output
    # If the result indicates that the context is unknown, use the Zero-shot react model
       result = get_output(input_field)

st.write(result)
# Cleanup vectorstore
vectorstore.delete_collection()


