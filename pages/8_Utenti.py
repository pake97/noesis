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
import base64
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from urllib.error import URLError
from pymilvus import MilvusClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd
import random
st.logo('logo.png', icon_image='logo.png')

st.set_page_config(page_title="Noesis")

def check_password():
    """Returns `True` if the user had the correct password."""

    def login():    
        user = ""
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
            st.session_state["user"] = user["user"]
            print("Response Data:", user)
        elif response.status_code == 404:
            print("User not found")
        else:
            print(f"Failed with status code: {response.status_code}")
            print("Response Data:", response.text)
        
    
        
        
        
        
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

def save_user(id, username, password, role):
    
    print("SAVING",id, st.session_state['new_username'], st.session_state['new_password'], st.session_state['role'])
    
    # Encode the password in Base64
    encoded_password = base64.b64encode(st.session_state['new_password'].encode()).decode()
    # Define the payload with username, password, and role
    payload = {
        "username": st.session_state['new_username'],
        "password": encoded_password,
        "role":  st.session_state['role'],
        "id": id
    }

    # Send the POST request
    response = requests.post(st.secrets["put_user_api"], json=payload)

    # Check if the request was successful
    if response.status_code == 201:
        print("User created successfully")
    else:
        print(f"Failed with status code: {response.status_code}")
        print("Response Data:", response.text)
        st.error("Errore nella creazione dell'utente")
    df = getUser()
    df["select"] = False
    st.session_state["users_data"] = df
    st.session_state["add"] = False
    


def getUser():
    response = requests.get(st.secrets["users_api"])
    
    # Check if the request was successful
    if response.status_code == 200:
        print("Request was successful")
        data = response.json()
        print("Response Data:", data)
        return pd.DataFrame(data["users"])
    else:
        print(f"Failed with status code: {response.status_code}")
        print("Response Data:", response.text)
        return pd.DataFrame([])


if "data" not in st.session_state:
    df = getUser()
    df["select"] = False
    st.session_state["users_data"] = df

if "editor_key" not in st.session_state:
    st.session_state["editor_key"] = random.randint(0, 100000)

if "last_selected_row" not in st.session_state:
    st.session_state["last_selected_row"] = None


def get_row_and_clear_selection():
    key = st.session_state["editor_key"]
    df = st.session_state["users_data"]
    selected_rows = st.session_state[key]["edited_rows"]
    selected_rows = [int(row) for row in selected_rows if selected_rows[row]["select"]]
    try:
        last_row = selected_rows[-1]
    except IndexError:
        return
    df["select"] = False
    st.session_state["users_data"] = df
    st.session_state["editor_key"] = random.randint(0, 100000)
    st.session_state["last_selected_row"] = df.iloc[last_row]


def abort():
    st.session_state["last_selected_row"] = None
    st.session_state["editor_key"] = random.randint(0, 100000)



if not check_password():
    st.stop()  # Do not continue if check_password is 

st.title("Utenti Noesis")

def delete_user(row):
    print("TO DELETE",row['id'])
    print(type(int(row['id'])))
    payload = {
    "id": int(row['id'])# Replace with the actual user ID you want to delete
    }

    # Send the DELETE request
    response = requests.delete(st.secrets['delete_user_api'], json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        print("User deleted successfully")
    elif response.status_code == 404:
        print("User not found")
    else:
        print(f"Failed with status code: {response.status_code}")
        print("Response Data:", response.text)
    df = getUser()
    df["select"] = False
    st.session_state["users_data"] = df
    st.session_state["last_selected_row"] = None
    get_row_and_clear_selection()
        
    


    

if(st.session_state.user["role"]=='ispettore'):
    st.data_editor(
    st.session_state["users_data"],
    key=st.session_state["editor_key"],
    on_change=get_row_and_clear_selection,
    )

    last_row = st.session_state["last_selected_row"]

    if last_row is not None:
        st.write("Utente selezioanto:")
        st.table(last_row)
        col1,col2 = st.columns(2)
        with col1 : 
            st.button("Elimina Utente", on_click=delete_user, args=(last_row,))
        with col2 : 
            st.button("Annulla", on_click=abort)
    else : 
        st.subheader("Aggiungi Utente")
        form = st.form("add_user")
        new_username=form.text_input("Nuovo Username",key="new_username")
        new_password=form.text_input("Nuova Password",key="new_password")
        role=form.selectbox("Ruolo",['ispettore','residente','consigliere','direttore'],key="role")
        form.form_submit_button(label="Salva", help=None, on_click=save_user, args=(st.session_state["users_data"].shape[0],new_username, new_password, role), kwargs=None, type="secondary", disabled=False, use_container_width=False)
            
        
        
    
        
        
else:
    st.error("Non sei autorizzato ad accedere a questa pagina")