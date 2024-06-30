import streamlit as st
import hmac
import os
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain.document_loaders import PyPDFLoader
import boto3
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
        return self.client.search(collection_name=collection_name, data=vector, limit=top_k, search_params={"metric_type": "COSINE"}, output_fields=['url'])
def download_file(url):
    st.session_state["clicked"] = True
    s3_client = boto3.client('s3', aws_access_key_id=st.secrets['aws_access_key_id'], aws_secret_access_key=st.secrets['aws_secret_access_key'])
    downloaded = s3_client.download_file('salesian2024', url[0], '/tmp/temp_file.pdf')
    



review_template_str = """Il tuo compito è quello di rispondere alle domande fornite 
riguardo al mondo dei Salesiani della INE (Italia Nord Est). Usa il seguente contesto
per rispondere. Sii il più dettagliato possibile, ma non inventare informazioni, 
se non conosci la risposta, scrivi che non lo sai.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)
messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

review_chain = review_prompt_template | chat_model





os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
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
st.title("Gemini Ai Chatbot con i documenti Salesiani")

chat_model =ChatOpenAI(
    #model="gpt-4o",
    model ="gpt-3.5-turbo-0125",
    temperature=0,
    max_tokens=None,
    timeout=None)

messages = [
    
]
st.session_state.messages=[]

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
connector = Connector()
if prompt := st.chat_input("Invia messagio al Chatbot Salesiani:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        with st.spinner('Calcolando...'):
            embs = Embedder()

            embedding = embs.get_embeddings(prompt)

            context=""
            res =connector.search("documents", embedding, top_k=2)
            for re in res[0]:
                loader = PyPDFLoader('https://salesian2024.s3.eu-north-1.amazonaws.com/'+re['entity']['url'].split("/")[-1])
                pages = loader.load_and_split()
                text = "\n\n".join(str(p.page_content) for p in pages)
                context+=text
                # Load the PDF document from the URL
                #loader.load_from_url('https://salesian2024.s3.eu-north-1.amazonaws.com/'+re['entity']['url'].split("/")[-1])
                # Extract text from the loaded PDF
            messages.append(SystemMessage(content=context))
            messages.append(HumanMessage(content=prompt))
        

    with st.chat_message("assistant"):
        
        stream =chat_model.stream(messages, stream=True)
        response = st.write_stream(stream)
        
        messages.append(AIMessage(content=stream))
    st.session_state.messages.append({"role": "assistant", "content": response})



