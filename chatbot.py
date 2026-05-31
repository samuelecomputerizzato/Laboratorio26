# Il mio chatbot online

import streamlit as st
import pdfplumber

st.header("Assistenza online")

with st.sidebar:
    st.title("Il mio documento")
    documento = st.file_uploader("Carica il tuo pdf:", type=["pdf"])