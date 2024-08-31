import streamlit as st
import hmac
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain.document_loaders import PyPDFLoader
import boto3
import requests
import mysql.connector
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from urllib.error import URLError
import uuid
from pymilvus import MilvusClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd
import base64

import random
from streamlit_searchbox import st_searchbox
session = boto3.Session( aws_access_key_id=st.secrets["aws_access_key_id"], aws_secret_access_key=st.secrets["aws_secret_access_key"])
s3 = session.resource('s3')
my_bucket = s3.Bucket('salesian2024')
s3_client = boto3.client('s3', aws_access_key_id=st.secrets["aws_access_key_id"], aws_secret_access_key=st.secrets["aws_secret_access_key"])


st.logo('logo.png', icon_image='logo.png')

st.set_page_config(page_title="Noesis")

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

@st.cache_data(ttl=120)
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


def deleteDoc(doc):
    
    payload = {
    "id": doc["id"]  # Replace with the actual user ID you want to delete
    }

    # Send the DELETE request
    response = requests.delete(st.secrets['delete_document_api'], json=payload)
    client = MilvusClient(
            uri=st.secrets["ZILLIZ_CLUSTER_ENDPOINT"],
            token=st.secrets["ZILLIZ_TOKEN"]
        )
    
    res = client.delete(
    collection_name="documents",
    filter="primary_key in ["+str(doc["zilliz_id"])+"]"
    )
    
    
    data=getDocs()
    st.session_state["data"] = data


if not check_password():
    st.stop()  # Do not continue if check_password is 

if "data" not in st.session_state:
    data = getDocs()
    st.session_state["data"] = data
    

st.title("Documenti")
st.session_state["filter"] = ""
st.session_state["_filter"] = ""

title = st.text_input("Cerca")
col1,col2,col3 = st.columns(3)
with col1:
    st.subheader("Documenti")
with col2:
    st.subheader("Accessibilità")
with col3:
    st.subheader("Azioni")

c = st.container( border=True)

for doc in list(filter(lambda x:title in x["url"] or title.upper() in x["url"] or title.capitalize() in x["url"], st.session_state["data"])):#s3_docs:
    cd = c.container( border=True)
    col1,col2,col3 = cd.columns(3)
    with col1:
        
        st.write(doc["url"])
    with col2:
        
        st.write(doc["level"])
    with col3:
        
        with st.popover("Elimina"):
            name = st.write("Sicuro di voler eliminare il documento?")
            st.button("Elimina", key=doc, on_click=deleteDoc, args=(doc,))
        st.link_button("Scarica","https://salesian2024.s3.eu-north-1.amazonaws.com/"+doc["url"])
    
