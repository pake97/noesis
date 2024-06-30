import streamlit as st
import os.path
import pandas as pd
import pathlib
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyPDFLoader
import hmac
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
import os
import boto3
from botocore.exceptions import NoCredentialsError

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pymilvus import MilvusClient
CLUSTER_ENDPOINT="https://in03-fbca0cedac2e113.api.gcp-us-west1.zillizcloud.com" # Set your cluster endpoint
TOKEN="b23e248d8a82ba9c8718cca8e0d27c67b9de17122a90c05e52090e540abad8b5bf084766511cf9ba9ba33a09f2b055014a776b9d" # Set your token
st.set_page_config(page_title="Noesis")

st.logo('logo.png', icon_image='logo.png')


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is 

class Connector:
    def __init__(self):
        self.client = MilvusClient(
            uri=CLUSTER_ENDPOINT,
            token=TOKEN
        )

    def get_client(self):
        return self.client

    def create_collection(self, collection_name, dimension):
        return self.client.create_collection({
            "collection_name": collection_name,
            "dimension": dimension,
            "index_file_size": 1024,
            "metric_type": "L2"
        })
    
    def insert_data(self, collection_name, data):
        return self.client.insert(collection_name=collection_name, data=data)
    
    def search(self, collection_name, vector, top_k):
        return self.client.search(collection_name=collection_name, data=vector, limit=top_k, search_params={"metric_type": "COSINE"})
    
    
from langchain_google_genai import GoogleGenerativeAIEmbeddings
os.environ["GOOGLE_API_KEY"] = "AIzaSyDzDsl5G4lD2c07IzcAbpb61Pp11QZIobg"
class Embedder:
    def __init__(self):
        self.embeddings =GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    
    def get_embeddings(self, documents):
        embeds = self.embeddings.embed_documents(documents)
        return embeds
    
    
class document_loader():
    def __init__(self):
        self.s3_bucket_name = "salesian2024"
        self.s3_client = boto3.client('s3', aws_access_key_id="AKIA6ODU5YGVV7PSITMV", aws_secret_access_key="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")
        self.embedder = Embedder()
        self.connector = Connector()
        
        
    def load_pdf(self, pdf_path):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        context = "\n\n".join(str(p.page_content) for p in pages)
        metadata = [p.metadata for p in pages]
        source = loader.source
        texts = text_splitter.split_text(context)
        return texts,metadata, source
        
    def load_document_to_s3(self, file):
        try:
            s3_key = os.path.basename(file.name)
            self.s3_client.upload_fileobj(file, self.s3_bucket_name, s3_key)
            return s3_key
        except FileNotFoundError:
            st.warning("The file was not found", icon="🚫")
        except NoCredentialsError:
            st.warning("Credentials not available", icon="🚫")
        except Exception as e:
            st.warning(e, icon="🚫")
            
    def upload_documents(self, documents):
        for document in documents:
            self.load_document_to_s3(document)
            docs,metadata, source = self.load_pdf(document)
            embeddings = self.embedder.get_embeddings(documents)
            data = []
            print(len(embeddings))
            for emb in embeddings:
                data.append({"vector":emb,"url":document})
            print(data)
            res = self.connector.insert_data("documents", data)
            


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is 

st.write("""
# File Upload
""")
uploaded_file = st.file_uploader("Scegli un file", type=[ "pdf"])
if uploaded_file is not None:
    if(uploaded_file.name.endswith('.pdf')):
        st.session_state["preview"] = ''
        st.session_state["preview"] = uploaded_file.name
    else:
        st.session_state["upload_state"] = "Only PDF files are supported!"
        st.text_area("Upload State", "", key="upload_state")
        
def upload():
    if uploaded_file is None:
        st.session_state["upload_state"] = "Upload a file first!"
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        temp_file = "./"+uploaded_file.name
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
        loader = PyPDFLoader(temp_file)
        pages = loader.load_and_split()
        context = "\n\n".join(str(p.page_content) for p in pages)
        metadata = [p.metadata for p in pages]
        source = loader.source
        texts = text_splitter.split_text(context)
        loader = document_loader()
        loaded = loader.load_document_to_s3(uploaded_file)
        embeddings = loader.embedder.get_embeddings(texts)
        data = []
        print(len(embeddings))
        for emb in embeddings:
            data.append({"vector":emb,"url":loaded})
        print(data)
        res = loader.connector.insert_data("documents", data)
        if(loaded):
            st.warning("Saved " + loaded + " successfully!", icon="✅")
            st.session_state["upload_state"] = "Saved " + loaded + " successfully!"
st.button("Upload file to Sandbox", on_click=upload)

