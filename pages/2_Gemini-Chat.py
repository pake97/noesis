import streamlit as st
import hmac
import os
import requests
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
import tiktoken
import logging

import mysql.connector
import base64



os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["google_key"]
def check_password():
    """Returns `True` if the user had the correct password."""

    def login():    
        
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
st.logo('logo.png', icon_image='logo.png')

st.set_page_config(page_title="Noesis")
st.title("Gemini AI Chatbot")


chat_model = ChatGoogleGenerativeAI(model="gemini-pro")
messages = [
    
]
if "messagesgemini" not in st.session_state:
    st.session_state.messagesgemini = []
if "aimessagesgemini" not in st.session_state:
    st.session_state.aimessagesgemini = []

# Display chat messages from history on app rerun
for message in st.session_state.messagesgemini:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Messaggio"):
    st.session_state.messagesgemini.append({"role": "user", "content": prompt})
    with st.chat_message("user"):

        if(len(prompt)>2500):
            chunks = [] 
            # divide the prompt in chunks of 2500 characters
            for i in range(0, len(prompt), 2500):
                chunks.append(prompt[i:i + 2500])
            for chunk in chunks:   
                st.session_state.aimessagesgemini.append(HumanMessage(content=chunk))
        else : 
            st.session_state.aimessagesgemini.append(HumanMessage(content=prompt))
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        stream =chat_model.stream(st.session_state.aimessagesgemini)
        logging.info(stream)
        response = st.write_stream(stream)
        
        st.session_state.aimessagesgemini.append(AIMessage(content=response))
    st.session_state.messagesgemini.append({"role": "assistant", "content": response})



