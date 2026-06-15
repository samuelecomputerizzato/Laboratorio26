import streamlit as st
import pdfplumber
import os
import re  # Importato per la gestione della formattazione del testo

st.set_page_config(page_title="La Magna Via", page_icon=":walking_man:", layout="centered")

# -------------------------------------------------------------------
# Funzione di utilità per formattare i messaggi mantenendo il Markdown attivo
# -------------------------------------------------------------------
def formatta_messaggio(testo: str, colore: str, font_size: str = "15px") -> str:
    """
    Converte la sintassi dei grassetti e degli a capo in HTML sicuro,
    evitando che gli asterischi rimangano visibili all'interno dei div personalizzati.
    """
    # Converte **testo** in <strong>testo</strong>
    testo_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', testo)
    # Converte i ritorni a capo in tag <br> per non perdere la struttura del testo
    testo_html = testo_html.replace('\n', '<br>')
    return f'<div style="color: {colore}; font-size: {font_size}; line-height: 1.5;">{testo_html}</div>'


# Langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------------------------
# Configurazione Stile CSS (Nuova Sidebar a Scomparsa + Nuke Streamlit)
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    
    /* 1. NUKE TOTALE DI STREAMLIT (AZZERAMENTO DI OGNI LOGO, MENU E TASTO CLOUD) */
    header, footer, [data-testid="stHeader"], [data-testid="stAppHeader"], [data-testid="stDecoration"], 
    #MainMenu, [data-testid="stToolbar"], .stDeployButton, [data-testid="stManageAppButton"],
    div[class*="viewerBadge"], div[class*="StyledFooter"], a[href*="streamlit.io"],
    div[class*="StyledActionButton"], [data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important
