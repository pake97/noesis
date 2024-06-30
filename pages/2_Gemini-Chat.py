import streamlit as st
import hmac
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
import tiktoken
import logging





os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["google_key"]
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



