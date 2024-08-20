import streamlit as st
import hmac
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
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
import pandas as pd

os.environ["GOOGLE_API_KEY"] = st.secrets["google_key"]


acronimi={"INE":"(Ispettoria) Italia Nord Est",
"INE":"(Ispettoria) Italia Nord Est",
"ILE":"(Ispettoria) Italia Lombardo Emiliana",
"ILE":"(Ispettoria) Italia Lombardo Emiliana",
"ISI":"(Ispettoria) Italia Sicula",
"ISI":"(Ispettoria) Italia Sicula",
"IME":"(Ispettoria) Italia Meridionale",
"IME":"(Ispettoria) Italia Meridionale",
"ICP":"(Ispettoria) Italia Circoscrizione Piemonte e Valle d'Aosta",
"ICP":"(Ispettoria) Italia Circoscrizione Piemonte e Valle d'Aosta",
"ICC":"(Ispettoria) Italia Centrale",
"ICC":"(Ispettoria) Italia Centrale",
"IVE":"Italia Veneto Est",
"IVE":"Italia Veneto Est",
"IVO":"Italia Veneto Ovest",
"IVO":"Italia Veneto Ovest",
"PEPS":"Progetto Educativo Pastorale Salesiano",
"PEPS":"Progetto Educativo Pastorale Salesiano",
"CEP":"Comunità Educativa Pastorale o Comunità Educativo Pastorale",
"CEP":"Comunità Educativa Pastorale o Comunità Educativo Pastorale",
"CEI":"Conferenza Episcopale Italiana",
"CEI":"Conferenza Episcopale Italiana",
"CIC":"Codex Iuris Canonici (= Codice di Diritto Canonico)",
"CIC":"Codex Iuris Canonici (= Codice di Diritto Canonico)",
"RM":"Rettor Maggiore",
"RM":"Rettor Maggiore",
"ACG":"Atti del Consiglio Generale",
"ACG":"Atti del Consiglio Generale",
"CI":"Capitolo Ispettoriale ",
"CI":"Capitolo Ispettoriale ",
"CGA":"Coordinatore/trice della Gestione Economica ed Amministrativa",
"CGA":"Coordinatore/trice della Gestione Economica ed Amministrativa",
"SDB":"Salesiani di don Bosco ",
"SDB":"Salesiani di don Bosco ",
"FMA":"Figlie di Maria Ausiliatrice ",
"FMA":"Figlie di Maria Ausiliatrice ",
"VDB":"Volontarie di don Bosco ",
"VDB":"Volontarie di don Bosco ",
"PGS":"Polisportive Giovanili Salesiane ",
"PGS":"Polisportive Giovanili Salesiane ",
"MGS":"Movimento Giovanile Salesiano",
"MGS":"Movimento Giovanile Salesiano",
"TGS":"Turismo Giovanile Salesiano",
"TGS":"Turismo Giovanile Salesiano",
"CFP":"Centro di Formazione Professionale",
"CFP":"Centro di Formazione Professionale",
"SFP":"Scuola della Formazione Professionale ",
"SFP":"Scuola della Formazione Professionale ",
"POI":"Piano Operativo Ispettoriale",
"POI":"Piano Operativo Ispettoriale",
"CNOS":"Centro Nazionale Opere Salesiane",
"CNOS":"Centro Nazionale Opere Salesiane",
"CNOS":"FAP - Centro Nazionale Opere Salesiane Formazione ed Aggiornamento Professionale",
"CNOS":"FAP - Centro Nazionale Opere Salesiane Formazione ed Aggiornamento Professionale",
"CNOS":"Scuola - Centro Nazionale Opere Salesiane settore Scuola",
"CNOS":"Scuola - Centro Nazionale Opere Salesiane settore Scuola",
"SCS":"Servizio Civile e Sociale",
"SCS":"Servizio Civile e Sociale",
"AGESC":"Associazione Genitori Scuole Cattoliche",
"AGESC":"Associazione Genitori Scuole Cattoliche",
"FP":"Formazione Professionale",
"FP":"Formazione Professionale",
"AV":"Animazione Vocazionale",
"AV":"Animazione Vocazionale",
"PIF":"Piano di Formazione Ispettoriale",
"PIF":"Piano di Formazione Ispettoriale",
"SCU":"Servizio Civile Universale",
"SCU":"Servizio Civile Universale",
"UPS":"Università Pontificia Salesiana",
"UPS":"Università Pontificia Salesiana",
"IUS":"Istituto Universitario Salesiano",
"IUS":"Istituto Universitario Salesiano",
"IUSVE":"Istituto Universitario Salesiano Venezia/Verona",
"IUSVE":"Istituto Universitario Salesiano Venezia/Verona",
"MSNA":"Minori Stranieri Non Accompagnati",
"MSNA":"Minori Stranieri Non Accompagnati",
"ADS":"Amici di San Domenico Savio",
"ADS":"Amici di San Domenico Savio",
"APG":"Animatori di Pastorale Giovanile",
"APG":"Animatori di Pastorale Giovanile",
"ISSM":"Istituto Salesiano San Marco",
"ISSM":"Istituto Salesiano San Marco",
"SDBM":"Salesiani Don Bosco Mestre ",
"SDBM":"Salesiani Don Bosco Mestre ",
"ISSZ":"Istituto Salesiano San Zeno",
"ISSZ":"Istituto Salesiano San Zeno",
"DAB":"Comunità Educativa per Minori e Centro Diurno di Albarè di Costermano",
"DAB":"Comunità Educativa per Minori e Centro Diurno di Albarè di Costermano",
"MOG":"Modello Organizzativo e Gestionale",
"MOG":"Modello Organizzativo e Gestionale",
"ODV":"Organismo di Vigilanza",
"ODV":"Organismo di Vigilanza",
"DPO":"Data Protection Officer ",
"DPO":"Data Protection Officer ",
"Cost":"Costituzioni della Congregazione Salesiana",
"Cost":"Costituzioni della Congregazione Salesiana",
"Reg":"Regolamenti della Congregazione Salesiana",
"Reg":"Regolamenti della Congregazione Salesiana"
}



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


chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
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
def resetChat():
    st.session_state.messagessalesiani = []
    st.session_state.aimessagessalesiani = []
st.set_page_config(page_title="Noesis")
st.title("Gemini Ai Chatbot con i documenti Salesiani")
st.button("Reset", on_click=resetChat)

chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
messages = [
    
]


if "messagessalesiani" not in st.session_state:
    st.session_state.messagessalesiani = []
if "aimessagessalesiani" not in st.session_state:
    st.session_state.aimessagessalesiani = []

for message in st.session_state.messagessalesiani:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
st.session_state.source_salesiani = ""
connector = Connector()
if prompt := st.chat_input("Invia messagio al Chatbot Salesiani:"):
    for k in list(acronimi.keys()):
        if k in prompt:
            prompt = prompt.replace(k,acronimi[k])
    st.session_state.messagessalesiani.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        with st.spinner('Calcolando...'):
            st.session_state.source_salesiani = ""
            if(len(st.session_state.aimessagessalesiani)==0):
                response = chat_model.invoke("Data la seguende domanda, capisci se ti vengono chieste informazioni riguardo alle sedi Salesiane della INE (Italia Nord Est), oppure alle scuole, strutture o opere. Anche domande riguardanti i direttori. Rispondi solo SI o NO. domanda :" + prompt+"Risposta:")
                if("NO" in response):
                    embs = Embedder()

                    embedding = embs.get_embeddings(prompt)

                    context=""
                    res =connector.search("documents", embedding, top_k=2)
                    for re in res[0]:
                        loader = PyPDFLoader('https://salesian2024.s3.eu-north-1.amazonaws.com/'+re['entity']['url'].split("/")[-1])
                        st.session_state.source_salesiani += re['entity']['url'].split("/")[-1]+","
                        pages = loader.load_and_split()
                        text = "\n\n".join(str(p.page_content) for p in pages)
                        context+=text
                        # Load the PDF document from the URL
                        #loader.load_from_url('https://salesian2024.s3.eu-north-1.amazonaws.com/'+re['entity']['url'].split("/")[-1])
                        # Extract text from the loaded PDF
                        
                    st.session_state.aimessagessalesiani.append(SystemMessage(content=context))
                        
                else : 
                    context = pd.read_csv("INE.csv",sep="|").to_string()
                    st.session_state.aimessagessalesiani.append(SystemMessage(content=context))
                    st.session_state.source_salesiani = "Elenco delle sedi Salesiane della INE"
            st.session_state.aimessagessalesiani.append(HumanMessage(content=prompt))


    with st.chat_message("assistant"):
        
        stream =chat_model.stream(st.session_state.aimessagessalesiani)
        response = st.write_stream(stream)
        st.session_state.aimessagessalesiani.append(AIMessage(content=response))
    st.session_state.messagessalesiani.append({"role": "assistant", "content": response})
    st.text("Fonte: "+st.session_state.source_salesiani)



