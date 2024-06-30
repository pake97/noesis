

import hmac
import streamlit as st










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
    st.stop()  # Do not continue if check_password is not True.

# Main Streamlit app starts here
st.image('logo.png')
st.write("# BENVENUTO")
st.page_link("pages/2_Gemini-Chat.py", label="Gemini Chat", icon="💬")
st.page_link("pages/2_Salesiani-Chat.py", label="Salesiani Chat", icon="💬")
st.page_link("pages/1_Ricerca.py", label="Ricerca", icon="🔍")
st.page_link("pages/4_Riassumi.py", label="Riassumi un documento", icon="📄")
st.page_link("pages/5_QA_Document.py", label="QA con documento", icon="📚")
st.page_link("pages/6_Upload.py", label="Carica File", icon="📤")