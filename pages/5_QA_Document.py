import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAI
import os, tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from io import BytesIO
from streamlit_searchbox import st_searchbox
import boto3
os.environ["GOOGLE_API_KEY"] = st.secrets["google_key"]
session = boto3.Session( aws_access_key_id=st.secrets["aws_access_key_id"], aws_secret_access_key=st.secrets["aws_secret_access_key"])
s3 = session.resource('s3')
my_bucket = s3.Bucket('salesian2024')
s3_client = boto3.client('s3', aws_access_key_id=st.secrets["aws_access_key_id"], aws_secret_access_key=st.secrets["aws_secret_access_key"])
s3_docs = [doc.key for doc in my_bucket.objects.all()]


    
    
def generate_response(uploaded_file, query_text, uploaded):
    # Load document if file is uploaded
    if uploaded_file is not None:
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
            loader = PyPDFLoader(tmp_file.name)
            os.remove(tmp_file.name)
        else:
            loader = PyPDFLoader(uploaded_file)
        pages = loader.load_and_split()
        
        
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.create_documents([p.page_content for p in pages])
        # Select embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=GoogleGenerativeAI(model="gemini-pro"), chain_type='stuff', retriever=retriever)
        return qa.run(query_text)


st.logo('logo.png', icon_image='logo.png')



# Page title
st.set_page_config(page_title='QA Documento')
st.title('QA con documento caricato')

# File upload
uploaded_file = st.file_uploader('Carica un documento', type='pdf')
st.write('Oppure seleziona un documento già caricato:')

download_file = st.selectbox(
    "",
    s3_docs)
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
    