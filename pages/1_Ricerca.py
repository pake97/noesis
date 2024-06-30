import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import hmac
import os
import boto3
from pymilvus import MilvusClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
CLUSTER_ENDPOINT="https://in03-fbca0cedac2e113.api.gcp-us-west1.zillizcloud.com" # Set your cluster endpoint
TOKEN="b23e248d8a82ba9c8718cca8e0d27c67b9de17122a90c05e52090e540abad8b5bf084766511cf9ba9ba33a09f2b055014a776b9d" # Set your token
os.environ["GOOGLE_API_KEY"] = "AIzaSyDzDsl5G4lD2c07IzcAbpb61Pp11QZIobg"
class Embedder:
    def __init__(self):
        self.embeddings =GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    
    def get_embeddings(self, documents):
        embeds = self.embeddings.embed_documents(documents)
        return embeds
    


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
        return self.client.search(collection_name=collection_name, data=vector, limit=top_k, search_params={"metric_type": "COSINE"}, output_fields=['url'])
def download_file(url):
    st.session_state["clicked"] = True
    s3_client = boto3.client('s3', aws_access_key_id="AKIA6ODU5YGVV7PSITMV", aws_secret_access_key="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")
    downloaded = s3_client.download_file('salesian2024', url[0], '/tmp/temp_file.pdf')
    
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


connector = Connector()
st.logo('logo.png', icon_image='logo.png')

st.markdown("# Cerca documenti relativi alla tua ricerca")
query = st.text_area("Inserisci la tua ricerca")
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Cerca', disabled=not(query))
    if submitted:
        #try:
        with st.spinner('Calcolando...'):
            embs = Embedder()

            embedding = embs.get_embeddings(query)


            res =connector.search("documents", embedding, top_k=5)
            for re in res[0]:
                result.append(re['entity']['url'].split("/")[-1])
            
        #except Exception as e:
        #    st.exception(f"An error occurred: {e}")
if(result):
    

    
    for count,res in enumerate(result):
        st.write(res)       
        st.markdown("<a href='https://salesian2024.s3.eu-north-1.amazonaws.com/{file}'>Download</button></a>".format(file=res), unsafe_allow_html = True)
        
