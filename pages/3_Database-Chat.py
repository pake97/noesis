import streamlit as st
import hmac
import pandas as pd
import os
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
import tiktoken
import logging
import requests
import re
import hmac
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from io import StringIO
import matplotlib.pyplot as plt
import json
import base64

SQL_FAIL_MESSAGE = "SQL_ERROR"

IS_PLOT = """
    Basandoti sulla domanda seguente, capisci se l'utente ha chiesto un grafico a linea, un histogramma o nessuno. Rispondi solo una di queste opzioni : Histogram, Line Plot, None.
    Domanda: {question}
    Risposta:"""
HIST_DATA = """
    Basandoti sui seguenti risultati, scrivili in formato csv per passarli alla libreria pandas in un Dataframe usando ';' come separatore se ci sono più colonne altrimenti vai a capo, in modo che io possa disegnare un histogramma per rispondere alla domanda seguente. Ritorna un dizionario con i seguenti campi : csv, nome_colonne. Includi i nomi delle colonne nel csv. Il csv deve essere pronto per essere letto dalla libreria json di python. 
    Domanda: {question}
    Risultati: {results}
    Risposta:
"""
LINE_DATA = """
    Basandoti sui seguenti risultati, scrivili in formato csv per passarli alla libreria pandas in un Dataframe usando ';' come separatore se ci sono più colonne altrimenti vai a capo, in modo che io possa disegnare un line plot per rispondere alla domanda seguente. Ritorna un dizionario con i seguenti campi : csv, nome_colonne . Includi i nomi delle colonne nel csv. Il csv deve essere pronto per essere letto dalla libreria json di python.
    Domanda: {question}
    Risultati: {results}
    Risposta:
"""

def init():
    model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest')
    #model = ChatOpenAI(model='gpt-3.5-turbo')
   
    
    return model, None

def text2sql(question):
    model, db = init()
    # Using Closure desgin pattern to pass the db to the model
    def get_schema(_):
    
        return """

CREATE TABLE CasaSalesiana (
    ID INT PRIMARY KEY AUTO_INCREMENT,
    Nome VARCHAR(255) NOT NULL,
    Descrizione TEXT,
    Città VARCHAR(255),
    Provincia VARCHAR(255),
    Regione VARCHAR(255),
    Luogo ENUM('Triveneto', 'Moldavia', 'Romania')
);


CREATE TABLE Salesiano (
    ID INT PRIMARY KEY AUTO_INCREMENT,
    Nome VARCHAR(255) NOT NULL,
    Cognome VARCHAR(255) NOT NULL,
    DataDiNascita DATE,
    CittaNascita VARCHAR(255),
    RegioneNascita VARCHAR(255),
    Ruolo ENUM('Direttore', 'Insegnante', 'Amministrativo', 'Ausiliario', 'Economo', 'Delegato', 'Tirocinante', 'Diacono', 'Sacerdote', 'Cooperatore', 'Laico', 'Volontario', 'Altro'),
    CasaSalesianaID INT,
    DataInizioMandato DATE,
    DataFineMandato DATE,
    Laureato BOOLEAN,
    Abilitato BOOLEAN,
    DataProfessione DATE,
    FOREIGN KEY (CasaSalesianaID) REFERENCES CasaSalesiana(ID)
);
CREATE TABLE OperaSalesiana (
    ID INT PRIMARY KEY AUTO_INCREMENT,
    Nome VARCHAR(255) NOT NULL,
    Descrizione TEXT,
    Città VARCHAR(255),
    Provincia VARCHAR(255),
    Regione VARCHAR(255),
    Luogo ENUM('Triveneto', 'Moldavia', 'Romania'),
    OperaSalesianaID INT,
    FOREIGN KEY (OperaSalesianaID) REFERENCES OperaSalesiana(ID),
    CasaSalesianaID INT,
    FOREIGN KEY (CasaSalesianaID) REFERENCES CasaSalesiana(ID)
);


CREATE TABLE ScuolaSalesiana (
    ID INT PRIMARY KEY AUTO_INCREMENT,
    Nome VARCHAR(255) NOT NULL,
    Indirizzo VARCHAR(255),
    Città VARCHAR(255),
    Provincia VARCHAR(255),
    Regione VARCHAR(255),
    Tipo ENUM('Primaria', 'Secondaria Primo Grado', 'Secondaria Secondo Grado', 'Università'),
    Luogo ENUM('Triveneto', 'Moldavia', 'Romania'),
    CasaSalesianaID INT,
    FOREIGN KEY (CasaSalesianaID) REFERENCES CasaSalesiana(ID),
    OperaSalesianaID INT,
    FOREIGN KEY (OperaSalesianaID) REFERENCES OperaSalesiana(ID)
);

CREATE TABLE AttivitaSalesiana (
    ID INT PRIMARY KEY AUTO_INCREMENT,
    Nome VARCHAR(255) NOT NULL,
    Descrizione TEXT,
    Città VARCHAR(255),
    Provincia VARCHAR(255),
    Regione VARCHAR(255),
    Luogo ENUM('Triveneto', 'Moldavia', 'Romania'),
    OperaSalesianaID INT,
    FOREIGN KEY (OperaSalesianaID) REFERENCES OperaSalesiana(ID),
    CasaSalesianaID INT,
    FOREIGN KEY (CasaSalesianaID) REFERENCES CasaSalesiana(ID)
);



CREATE TABLE Oratorio (
    ID INT PRIMARY KEY AUTO_INCREMENT,
    Nome VARCHAR(255) NOT NULL,
    Descrizione TEXT,
    Città VARCHAR(255),
    Provincia VARCHAR(255),
    Regione VARCHAR(255),
    Luogo ENUM('Triveneto', 'Moldavia', 'Romania'),
    OperaSalesianaID INT,
    FOREIGN KEY (OperaSalesianaID) REFERENCES OperaSalesiana(ID),
    CasaSalesianaID INT,
    FOREIGN KEY (CasaSalesianaID) REFERENCES CasaSalesiana(ID)
);


CREATE TABLE Parrocchia (
    ID INT PRIMARY KEY AUTO_INCREMENT,
    Nome VARCHAR(255) NOT NULL,
    Indirizzo VARCHAR(255),
    Città VARCHAR(255),
    Provincia VARCHAR(255),
    Regione ENUM('Triveneto', 'Moldavia', 'Romania'),
    CasaSalesianaID INT,
    FOREIGN KEY (CasaSalesianaID) REFERENCES CasaSalesiana(ID),
    OperaSalesianaID INT,
    FOREIGN KEY (OperaSalesianaID) REFERENCES OperaSalesiana(ID)
);

CREATE TABLE Auto (
    ID INT PRIMARY KEY AUTO_INCREMENT,
    Modello VARCHAR(255),
    AnnoAcquisto INT,
    AnnoCreazione INT,
    Targa VARCHAR(50),
    OperaSalesianaID INT,
    FOREIGN KEY (OperaSalesianaID) REFERENCES OperaSalesiana(ID),
    CasaSalesianaID INT,
    FOREIGN KEY (CasaSalesianaID) REFERENCES CasaSalesiana(ID),
    ParrocchiaID INT,
    FOREIGN KEY (ParrocchiaID) REFERENCES Parrocchia(ID),
    AttivitaSalesianaID INT,
    FOREIGN KEY (AttivitaSalesianaID) REFERENCES AttivitaSalesiana(ID),
    OratorioID INT,
    FOREIGN KEY (OratorioID) REFERENCES Oratorio(ID)
);


CREATE TABLE Studente (
    ID INT PRIMARY KEY AUTO_INCREMENT,
    Nome VARCHAR(255) NOT NULL,
    Cognome VARCHAR(255) NOT NULL,
    DataDiNascita DATE,
    CodiceFiscale VARCHAR(255),
    livello ENUM('Primaria', 'Secondaria Primo Grado', 'Secondaria Secondo Grado', 'Università'),
    anno INT,
    ScuolaSalesianaID INT,
    CasaSalesianaID INT,
    AnnoIscrizione INT,
    Retta DECIMAL(10, 2),
    FOREIGN KEY (ScuolaSalesianaID) REFERENCES ScuolaSalesiana(ID),
    FOREIGN KEY (CasaSalesianaID) REFERENCES CasaSalesiana(ID)
);

CREATE TABLE Dipendente (
    ID INT PRIMARY KEY AUTO_INCREMENT,
    Nome VARCHAR(255) NOT NULL,
    Cognome VARCHAR(255) NOT NULL,
    Sesso VARCHAR(10),
    CodiceFiscale VARCHAR(255),
    DataDiNascita DATE,
    CittaNascita VARCHAR(255),
    RegioneNascita VARCHAR(255),
    DataAssunzione DATE,
    DataCessazione DATE,
    Ruolo ENUM('Direttore', 'Insegnante', 'Amministrativo', 'Ausiliario', 'Economo', 'Delegato', 'Tirocinante', 'Diacono', 'Sacerdote', 'Cooperatore', 'Laico', 'Volontario', 'Altro'),
    OperaSalesianaID INT,
    FOREIGN KEY (OperaSalesianaID) REFERENCES OperaSalesiana(ID),
    CasaSalesianaID INT,
    FOREIGN KEY (CasaSalesianaID) REFERENCES CasaSalesiana(ID),
    ScuolaSalesianaID INT,
    FOREIGN KEY (ScuolaSalesianaID) REFERENCES ScuolaSalesiana(ID),
    ParrocchiaID INT,
    FOREIGN KEY (ParrocchiaID) REFERENCES Parrocchia(ID),
    AttivitaSalesianaID INT,
    FOREIGN KEY (AttivitaSalesianaID) REFERENCES AttivitaSalesiana(ID),
    OratorioID INT,
    FOREIGN KEY (OratorioID) REFERENCES Oratorio(ID)
);

CREATE TABLE Bilancio (
    ID INT PRIMARY KEY AUTO_INCREMENT,
    Anno INT,
    Descrizione VARCHAR(255) NOT NULL,
    TipoVoce ENUM('Entrata', 'Spesa', 'Attivo', 'Passivo') NOT NULL,
    Importo DECIMAL(15, 2) NOT NULL,
    Categoria VARCHAR(255) NOT NULL,
    OperaSalesianaID INT,
    FOREIGN KEY (OperaSalesianaID) REFERENCES OperaSalesiana(ID),
    CasaSalesianaID INT,
    FOREIGN KEY (CasaSalesianaID) REFERENCES CasaSalesiana(ID),
    ParrocchiaID INT,
    FOREIGN KEY (ParrocchiaID) REFERENCES Parrocchia(ID),
    AttivitaSalesianaID INT,
    FOREIGN KEY (AttivitaSalesianaID) REFERENCES AttivitaSalesiana(ID),
    OratorioID INT,
    FOREIGN KEY (OratorioID) REFERENCES Oratorio(ID)
);

CREATE TABLE Consumo (
    ID INT PRIMARY KEY AUTO_INCREMENT,
    CasaSalesianaID INT,
    Anno INT,
    costo DECIMAL(10, 2),
    TipoConsumo ENUM('luce', 'acqua', 'gas'),
    Valore DECIMAL(10, 2),
    FOREIGN KEY (CasaSalesianaID) REFERENCES CasaSalesiana(ID)
);

"""
    template = """Basandoti sullo schema del database seguente, scrivi una query MySQL che risponda alla domanda dell'utente. Ricorda che 'Triveneto', 'Moldavia' e 'Romania' sono i valori possibili per il campo 'Luogo' nelle tabelle 'CasaSalesiana', 'OperaSalesiana', 'ScuolaSalesiana', 'AttivitaSalesiana', 'Oratorio' e 'Parrocchia'. Mentre La Ispettoria INE (Italia Nord Est) comprende tutti i luoghi.
    {schema}
    Domanda: {question}
    SQL Query:"""
    prompt = ChatPromptTemplate.from_template(template)
    sql_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | model.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    return sql_response.invoke({"question": question})  

def execute_sql(query):
    query = query.replace("```sql", "").replace("```", "")
    print('EXECUTE SQL')
    print(query)
    print('----------------')
    #db = SQLDatabase.from_uri(st.secrets["mysql"], sample_rows_in_table_info=0)
    update_action_list = ['UPDATE','ADD','DELETE','DROP','MODIFY','INSERT']
    try:
        if any(item in query for item in update_action_list)==False:# no update actions
            #result = db.run(query)
            payload = {
                "query": query # Replace with your actual SQL query
            }
            response = requests.post(st.secrets["run_query_api"],json=payload)
    
            # Check if the request was successful
            if response.status_code == 200:
                print("Request was successful")
                data = response.json()
                print("Response Data:", data)
                return data["result"]
            else:
                print(f"Failed with status code: {response.status_code}")
                print("Response Data:", response.text)
                return "No results found."
            
                
        else: return 'Finished' #update actions return no result but "Finished"
    except Exception as e:
        error_message = str(e)
        print(SQL_FAIL_MESSAGE+"Motivo:",error_message+" Fine")
        return SQL_FAIL_MESSAGE

def sqlresult2text(question,sql_query,sql_result):
    # Using Closure desgin pattern to pass the db to the model
    model, db = init()
    def get_schema_info(_):
        return get_schema()
    ## To natural language
    
    template = """
    Basandoti sullo schema database seguente, la domanda, la query SQL e la risposta SQL, scrivi una descrizione in linguaggio naturale della risposta SQL. Se il risultato della query è vuoto, spiega che nessun risultato è stato trovato, invitando a inserire i dati nel database o a specificare meglio la domanda.
    {schema}
    Domanda: {question}
    SQL Query: {query}
    Risutlato SQL: {response}"""


    prompt_response = ChatPromptTemplate.from_template(template)


    text_response = (
        RunnablePassthrough.assign(schema=get_schema_info)
        | prompt_response
        | model
    )

    # execute the model 
    return   text_response, {"question": question,"query":sql_query,"response":sql_result}
st.logo('logo.png', icon_image='logo.png')

os.environ["GOOGLE_API_KEY"] = st.secrets["google_key"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
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




st.set_page_config(page_title="Noesis")
st.title("Chatta con il Database")


messages = [
    
]


if "messagesdb" not in st.session_state:
    st.session_state.messagesdb = []
if "aimessagesdb" not in st.session_state:
    st.session_state.aimessagesdb = []

for message in st.session_state.messagesdb:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Chiedi al Database:"):
    st.session_state.messagesdb.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        question = st.session_state.messagesdb[-1]["content"]
        print("Domanda",question)
        sql=text2sql(question)     
        print("SQL",sql)
        result = execute_sql(sql)
        
        st.write(result)
        #model, params = sqlresult2text(prompt,sql,result)
        #stream = model.stream(params)
        #response = st.write_stream(stream)
        
        model, db = init()
        prompt_response = ChatPromptTemplate.from_template(IS_PLOT)
        text_response = (
            prompt_response
            | model
        )
        is_plot= text_response.invoke({"question": question})
        if(is_plot.content=="Histogram"):
            prompt_response = ChatPromptTemplate.from_template(HIST_DATA)
            text_response = (
            prompt_response
            | model
            )
            
            hist_csv = text_response.invoke({"question": question, "results": result})
            print(question)
            print(result)
            print(hist_csv.content)
            df = pd.read_csv(StringIO(json.loads(hist_csv.content)['csv']), sep=';')
            columns=json.loads(hist_csv.content)['nome_colonne'].split(';')
            st.dataframe(df)
            
            st.bar_chart(df, x=columns[0], y=columns[1])
        elif(is_plot.content=="Line Plot"):
            prompt_response = ChatPromptTemplate.from_template(LINE_DATA)
            text_response = (
            prompt_response
            | model
            )
            line_csv = text_response.invoke({"question": question, "results": result})
            df = pd.read_csv(StringIO(json.loads(line_csv.content)['csv']), sep=';')
            columns=json.loads(line_csv.content)['nome_colonne'].split(';')
            st.dataframe(df)
            st.line_chart(df)
        else:
            pass
        
        st.session_state.aimessagesdb.append(AIMessage(content=result))
    st.session_state.messagesdb.append({"role": "assistant", "content": result})



