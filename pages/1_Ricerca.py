import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import hmac
import os
import requests
import boto3
from pymilvus import MilvusClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import base64
os.environ["GOOGLE_API_KEY"] = st.secrets["google_key"]
class Embedder:
    def __init__(self):
        self.embeddings =GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    
    def get_embeddings(self, documents):
        embeds = self.embeddings.embed_documents(documents)
        return embeds
    


class Connector:
    def __init__(self):
        self.client = MilvusClient(
            uri=st.secrets["ZILLIZ_CLUSTER_ENDPOINT"],
            token=st.secrets["ZILLIZ_TOKEN"]
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
        doc_filter = "role in ["
        if st.session_state["user"]["role"] == "ispettore":
            doc_filter += "'ispettore', 'residente', 'consigliere', 'direttore']"
        elif st.session_state["user"]["role"] == "residente":
            doc_filter += "'residente', 'consigliere', 'direttore']"
        elif st.session_state["user"]["role"] == "consigliere":
            doc_filter += "'consigliere', 'direttore']"
        else:
            doc_filter += "'direttore']"
        print("FILTRO",doc_filter)   
        return self.client.search(collection_name=collection_name, data=vector[:10], limit=5, output_fields=['url'], filter=doc_filter)
def download_file(url):
    st.session_state["clicked"] = True
    s3_client = boto3.client('s3', aws_access_key_id=st.secrets['aws_access_key_id'], aws_secret_access_key=st.secrets['aws_secret_access_key'])
    downloaded = s3_client.download_file('salesian2024', url[0], '/tmp/temp_file.pdf')
    
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
        
