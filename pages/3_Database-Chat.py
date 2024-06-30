import streamlit as st
import hmac
import pandas as pd
import os
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
import tiktoken
import logging
import re
from langchain.utilities import SQLDatabase
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from io import StringIO
import matplotlib.pyplot as plt
import json


SQL_FAIL_MESSAGE = "SQL_ERROR"

IS_PLOT = """
    Basandoti sulla domanda seguente, capisci se l'utente ha chiesto un grafico a linea, un histogramma o nessuno. Rispondi solo una di queste opzioni : Histogram, Line Plot, None.
    Domanda: {question}
    Risposta:"""
HIST_DATA = """
    Basandoti sui seguenti risultati, scrivili in formato csv per passarli alla libreria pandas in un Dataframe usando ';' come separatore se ci sono più colonne altrimenti vai a capo, in modo che io possa disegnare un histogramma per rispondere alla domanda seguente. Ritorna un dizionario con i seguenti campi : csv, nome_colonne. Includi i nomi delle colonne nel csv.
    Domanda: {question}
    Risultati: {results}
    Risposta:
"""
LINE_DATA = """
    Basandoti sui seguenti risultati, scrivili in formato csv per passarli alla libreria pandas in un Dataframe usando ';' come separatore se ci sono più colonne altrimenti vai a capo, in modo che io possa disegnare un line plot per rispondere alla domanda seguente. Ritorna un dizionario con i seguenti campi : csv, nome_colonne . Includi i nomi delle colonne nel csv.
    Domanda: {question}
    Risultati: {results}
    Risposta:
"""

def init():
    #model = ChatGoogleGenerativeAI(model='gemini-pro')
    #model = ChatVertexAI(model='chat-bison@002')
    model=ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2)
    # Database
    #db = SQLDatabase.from_uri(f"sqlite:///./{db_name}.db", sample_rows_in_table_info=0)
    db = SQLDatabase.from_uri(f""+st.secrets["mysql"], sample_rows_in_table_info=0)
    return model,db

def text2sql(question):
    model,db = init()
    # Using Closure desgin pattern to pass the db to the model
    def get_schema(_):
        return db.get_table_info()
    template = """Basandoti sullo schema del database seguente, scrivi una query MySQL che risponda alla domanda dell'utente:
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
    print('EXECUTE SQL')
    print(query)
    print('----------------')
    db = SQLDatabase.from_uri(f""+st.secrets["mysql"], sample_rows_in_table_info=0)
    update_action_list = ['UPDATE','ADD','DELETE','DROP','MODIFY','INSERT']
    try:
        if any(item in query for item in update_action_list)==False:# no update actions
            result = db.run(query)
            if result:
                return result
            else:
                return "No results found."
        else: return 'Finished' #update actions return no result but "Finished"
    except Exception as e:
        error_message = str(e)
        print(SQL_FAIL_MESSAGE+"Motivo:",error_message+" Fine")
        return SQL_FAIL_MESSAGE

def sqlresult2text(question,sql_query,sql_result):
    # Using Closure desgin pattern to pass the db to the model
    model,db = init()
    def get_schema(_):
        return db.get_table_info()
    ## To natural language
    
    template = """
    Basandoti sullo schema database seguente, la domanda, la query SQL e la risposta SQL, scrivi una descrizione in linguaggio naturale della risposta SQL. Se il risultato della query è vuoto, spiega che nessun risultato è stato trovato, invitando a inserire i dati nel database o a specificare meglio la domanda.
    {schema}
    Domanda: {question}
    SQL Query: {query}
    Risutlato SQL: {response}"""


    prompt_response = ChatPromptTemplate.from_template(template)


    text_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt_response
        | model
    )

    # execute the model 
    return   text_response, {"question": question,"query":sql_query,"response":sql_result}
st.logo('logo.png', icon_image='logo.png')
st.session_state.messages=[]

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




st.set_page_config(page_title="Noesis")
st.title("Chatta con il Database")


messages = [
    
]


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Chiedi al Database:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        question = st.session_state.messages[-1]["content"]
        sql=text2sql(question)     
        st.write(sql)   
        result = execute_sql(sql)
        st.write(result)
        model, params = sqlresult2text(prompt,sql,result)
        stream = model.stream(params)
        response = st.write_stream(stream)
        
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
            columns=json.loads(hist_csv.content)['nome_colonne']
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
            columns=json.loads(line_csv.content)['nome_colonne']
            st.dataframe(df)
            st.line_chart(df)
        else:
            pass
        
        messages.append(AIMessage(content=stream))
    st.session_state.messages.append({"role": "assistant", "content": response})



