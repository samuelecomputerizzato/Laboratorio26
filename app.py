import streamlit as st
import pdfplumber
import os
import re

st.set_page_config(page_title="La Magna Via", page_icon=":walking_man:", layout="centered")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------------------------
# Configurazione Stile CSS (Nuke Streamlit + Fix Grafici + Input Bar Custom)
# -------------------------------------------------------------------
st.html(
    """
    <style>
    /* Nuke Streamlit */
    header, footer, [data-testid="stHeader"], [data-testid="stAppHeader"], 
    [data-testid="stDecoration"], #MainMenu, [data-testid="stToolbar"], 
    .stDeployButton, [data-testid="stManageAppButton"] { 
        display: none !important; 
        visibility: hidden !important;
    }
    
    .stApp { background-color: #F1F0E6; color: #231709; }

    /* ENGINE SIDEBAR */
    .sidebar-checkbox { display: none !important; }

    .sidebar-toggle-button {
        position: fixed; top: 15px; left: 15px;
        background-color: #7A8B74; color: #ffffff !important;
