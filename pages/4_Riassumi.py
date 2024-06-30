import os, tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
st.logo('logo.png', icon_image='logo.png')
st.set_page_config(page_title="Noesis")
# Streamlit app
st.header('Riassumi un documento PDF')
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

# Get OpenAI API key and source document input
source_doc = st.file_uploader("Upload Source Document", type="pdf")

# If the 'Summarize' button is clicked
number = st.number_input("Numero di parole", min_value=1, max_value=1000, value=200, step=1, format="%d")
refine = st.text_area("Rifinisci il prompt", value="")
if st.button("Summarize"):
    # Validate inputs
    
    with st.spinner('Attendi...'):
        
        # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(source_doc.read())
        loader = PyPDFLoader(tmp_file.name)
        pages = loader.load_and_split()
        st.write(type(pages))
        st.write(type(pages[0]))
        text=""
        
        for p in pages:
            text+=p.page_content
        
        os.remove(tmp_file.name)

        # Create embeddings for the pages and insert into Chroma database
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        

        # Initialize the OpenAI module, load and run the summarize chain
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
        chain = load_summarize_chain(llm, chain_type="stuff")
        
        summary = chain.run(input_documents=text, question="Scrivi un riassunto in italiano di {words} parole {refine}.".format(words=number, refine=refine))
        summary = llm.stream("Traduci in italiano: {summary}".format(summary=summary))
        st.write_stream(summary) 
