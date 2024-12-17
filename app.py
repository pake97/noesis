

import hmac
import streamlit as st
import requests
import base64








st.logo('logo.png', icon_image='logo.png')

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
    st.stop()  # Do not continue if check_password is not True.

# Main Streamlit app starts here
st.image('logo.png')
st.write("# BENVENUTO")
st.page_link("pages/2_Gemini-Chat.py", label="Gemini Chat", icon="💬")
st.page_link("pages/2_Salesiani-Chat.py", label="Salesiani Chat", icon="💬")
st.page_link("pages/1_Ricerca.py", label="Ricerca", icon="🔍")
st.page_link("pages/5_Riassumi.py", label="Riassumi un documento", icon="📄")
st.page_link("pages/7_Upload.py", label="Carica File", icon="📤")