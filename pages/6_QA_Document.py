import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import requests
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAI
import os, tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from io import BytesIO

import base64
from streamlit_searchbox import st_searchbox
import boto3
os.environ["GOOGLE_API_KEY"] = st.secrets["google_key"]
session = boto3.Session( aws_access_key_id=st.secrets["aws_access_key_id"], aws_secret_access_key=st.secrets["aws_secret_access_key"])
s3 = session.resource('s3')
my_bucket = s3.Bucket('salesian2024')
s3_client = boto3.client('s3', aws_access_key_id=st.secrets["aws_access_key_id"], aws_secret_access_key=st.secrets["aws_secret_access_key"])
st.set_page_config(page_title='QA Documento')


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





    
def generate_response(uploaded_file, query_text, uploaded):
    # Load document if file is uploaded
    if uploaded_file is not None:
        loader = ""
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
            loader = PyPDFLoader(tmp_file.name)
            
        else:
            loader = PyPDFLoader(uploaded_file)
        pages = loader.load_and_split()
        if uploaded:
            os.remove(tmp_file.name)
        
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.create_documents([p.page_content for p in pages])
        context = ""
        text = "\n\n".join(str(p.page_content) for p in pages)
        context+=text
        # Select embeddings
        #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Create a vectorstore from documents
        #db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        #retriever = db.as_retriever()
        # Create QA chain
        chat_model = GoogleGenerativeAI(model="gemini-pro")
        #qa = RetrievalQA.from_chain_type(llm=GoogleGenerativeAI(model="gemini-pro"), chain_type='stuff', retriever=retriever)
        return chat_model.invoke([HumanMessage(content="Dato questo documento, rispondi alla domanda data. \nDocumento"+context+"\n\n"+query_text+"\n Domanda : "+query_text)])
        
        #return qa.run(query_text)

@st.cache_data(ttl=3600)
def getDocs():
    
    
    response = requests.get(st.secrets["documents_api"]+"?role="+st.session_state.user["role"])
    
    # Check if the request was successful
    if response.status_code == 200:
        print("Request was successful")
        data = response.json()
        print("Response Data:", data)
        return data["documents"]
    else:
        print(f"Failed with status code: {response.status_code}")
        print("Response Data:", response.text)
        return []




if not check_password():
    st.stop()  # Do not continue if check_password is 

if "qadata" not in st.session_state:
    data = getDocs()
    st.session_state["qadata"] = data


st.logo('logo.png', icon_image='logo.png')



# Page title

st.title('QA con documento caricato')

# File upload
uploaded_file = st.file_uploader('Carica un documento', type='pdf')
st.write('Oppure seleziona un documento già caricato:')

download_file = st.selectbox(
    "",
    [x["url"] for x in st.session_state["qadata"]])
# Query text
query_text = st.text_input('La tua domanda:', placeholder = '...')

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Invia', disabled=not(uploaded_file or download_file and query_text))
    if submitted:
        #try:
        with st.spinner('Calcolando...'):
            if(uploaded_file is None):
                if("visite-ispettoriali" in download_file or "verbali" in download_file or "visite-straordinarie" in download_file):
                    print(download_file)
                    uploaded_file = s3_client.download_file('preprocessing-noesis', download_file, '/tmp/temp_file.pdf')
                    response = generate_response('/tmp/temp_file.pdf', query_text, False)
                    result.append(response)    
                else:
                    uploaded_file = s3_client.download_file('salesian2024', download_file, '/tmp/temp_file.pdf')
                    response = generate_response('/tmp/temp_file.pdf', query_text, False)
                    result.append(response)
            else:
                response = generate_response(uploaded_file, query_text, True)
                result.append(response)
        #except Exception as e:
        #    st.exception(f"An error occurred: {e}")
if len(result):
    
    st.info(response)
    