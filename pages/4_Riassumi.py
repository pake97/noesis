import os, tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import hmac
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import mysql.connector
import base64
st.logo('logo.png', icon_image='logo.png')
st.set_page_config(page_title="Noesis")
# Streamlit app
st.header('Riassumi un documento PDF')
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["google_key"]

def check_password():
    """Returns `True` if the user had the correct password."""

    def login():    
        user = {"user":""}
        # Encode the password in Base64
        encoded_password = base64.b64encode(password.encode()).decode()
        
        payload = {
            "username": username,
            "password": encoded_password
        }

        # Send the POST request
        response = requests.post(st.secrets["login_api"], json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            print("Request was successful")
            user = response.json()
            print("Response Data:", user)
        elif response.status_code == 404:
            print("User not found")
        else:
            print(f"Failed with status code: {response.status_code}")
            print("Response Data:", response.text)
        
    
        
        
        st.session_state["user"] = user["user"]
        
    # Return True if the password is validated.
    if st.session_state.get("user", None):
        return True

# Show input for password.
    username = st.text_input(
        "Username", type="default", key="username"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    # Show input for password.
    password = st.text_input(
        "Password", type="password", key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
        
        
    st.button("Login", on_click=login)
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is 

# Get OpenAI API key and source document input
source_doc = st.file_uploader("Upload Source Document", type="pdf")

# If the 'Summarize' button is clicked
number = st.number_input("Numero di parole", min_value=1, max_value=1000, value=200, step=1, format="%d")
refine = st.text_area("Rifinisci il prompt", value="")
if st.button("Summarize"):
    # Validate inputs
    
    with st.spinner('Attendi...'):
        
        # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(source_doc.read())
        loader = PyPDFLoader(tmp_file.name)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(pages)
        
        
        os.remove(tmp_file.name)

        # Initialize the OpenAI module, load and run the summarize chain
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
        chain = load_summarize_chain(llm, chain_type="stuff")
        
        summary = chain.run(input_documents=texts, question="Scrivi un riassunto in italiano di {words} parole {refine}.".format(words=number, refine=refine))
        summary = llm.stream("Traduci in italiano: {summary}".format(summary=summary))
        st.write_stream(summary) 
