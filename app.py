import streamlit as st
import pdfplumber
import os

# Langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="La Magna Via", page_icon=":walking_man:", initial_sidebar_state="expanded")

# -------------------------------------------------------------------
# Configurazione Stile CSS (Allineato e Senza Errori)
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* 1. STILE GLOBALE APP E PAGINA CENTRALE */
    .stApp {
        background-color: #B5A585;
        background-attachment: fixed;
        color: #231709;
    }
    
    /* Configurazione scritta "Chiedi al chatbot" */
    .stTextInput label div p {
        color: #542E17 !important;
        font-size: 24px !important;
        font-weight: bold !important;
    }

    /* Rettangolo di input */
    .stTextInput input {
        background-color: #793921;
        color: #000000;
    }

    /* ELIMINA LA BARRA GRIGIA: Rende trasparenti i messaggi della chat */
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] > div,
    .stChatMessage {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding-left: 0px !important;
    }

    /* Cambia il colore di sfondo della barra laterale */
    [data-testid="stSidebar"] {
        background-color: #b25431; 
    }

    /* SISTEMAZIONE DELLA TENDINA (POPOVER BODY) */
    [data-testid="stPopoverBody"] {
        background-color: transparent !important; 
        border: none !important;              
        border-radius: 12px !important;       
        box-shadow: 0px 8px 24px rgba(0, 0, 0, 0.4) !important; 
        min-width: 320px !important;          
        max-width: 85vw !important;           
        padding: 18px !important;
    }

    /* Modifica il pulsante prima di cliccarlo */
    [data-testid="stPopover"] button {
        background-color: transparent !important; 
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    
    [data-testid="stPopover"] button:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* OTTIMIZZAZIONE SPECIFICA PER IL TELEFONO */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            width: 85vw !important; 
        }
        
        [data-testid="stPopoverBody"] {
            min-width: 280px !important;
            max-width: 80vw !important;
            position: fixed !important;  
            left: 50% !important;        
            transform: translateX(-50%) !important; 
            top: auto !important;        
        }
    } /* <-- PRIMA MANCAVA QUESTA CHIUSURA CHIAVE DEL MEDIA QUERY */
    /* ================================================================= */
    /* 4. RESET DELLA FASCIA IN BASSO E CURA DEL RETTANGOLO DI INPUT    */
    /* ================================================================= */
    
    /* Applica la trasparenza SOLO alla fascia esterna fissa sul fondo */
    [data-testid="stChatFloatingInputContainer"] {
        background-color: transparent !important;
        background: transparent !important;
        background-image: none !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* FORZA LA VISIBILITÀ E IL COLORE DEL RETTANGOLO DOVE DIGITI */
    .stChatInput,
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] textarea {
        background-color: #793921 !important; /* Marrone scuro per staccare dal beige */
        color: #ffffff !important;            /* Scritta bianca mentre digiti */
        border-radius: 10px !important;
        border: 1px solid #542E17 !important;
    }
    
    /* Gestione del testo d'aiuto dentro il rettangolo ("Chiedi alla Via...") */
    [data-testid="stChatInput"] textarea::placeholder {
        color: rgba(255, 255, 255, 0.6) !important;
    }

    /* Mantiene visibile l'icona della freccia per inviare il messaggio */
    [data-testid="stChatInput"] button {
        background-color: transparent !important;
        color: #ffffff !important;
    }
        
    </style>
    """,
    unsafe_allow_html=True
)

# Interfaccia centrale (LOGO e Titolo)
col1, col2, col3 = st.columns([1, 2, 1]) 
with col2:
    st.image("LOGO.png")

st.header("La Magna Via", text_alignment="center")

# -------------------------------------------------------------------
# Barra Laterale (Sidebar) - Allineamento Pulsanti Corretto
# -------------------------------------------------------------------
st.sidebar.image("LOGO.png", width=120)
st.sidebar.header("Ultreya, viandante!")
st.sidebar.write("---") 

# PRIMO PULSANTE: Il Codice del Viandante
with st.sidebar.popover("📜 Il Codice del Viandante", use_container_width=True):
    st.markdown("<p style='text-align: center; font-style: italic; margin-bottom: 15px;'>Il rispetto è il primo passo del pellegrino.</p>", unsafe_allow_html=True)
    st.markdown("""
    * 🍃 **Rispetta la natura:** non lasciare traccia, solo impronte. Porta sempre con te i tuoi rifiuti e i mozziconi. Il fuoco è un nemico: non accenderlo mai.
    * 🏡 **Rispetta il territorio:** sei ospite di terreni agricoli: chiudi i cancelli e non calpestare i raccolti. Chiedi sempre prima di cogliere frutti.
    * 🤫 **Rispetta il silenzio:** il cammino è meditazione. Rispetta la quiete nei borghi, nei monasteri e negli ospitali.
    * 🎒 **Sii essenziale:** viaggia leggero. Negli ostelli, sii ordinato e rispettoso: non è un hotel, ma una casa condivisa.
    * 🤝 **Sii solidale:** aiuta chi è in difficoltà. Un sorriso o un consiglio possono fare la differenza per un altro viandante.
    * 🙏 **Sii grato e umile:** ringrazia chi ti ospita. Accetta con curiosità i ritmi e la cultura che incontri.
    """)
    
# SECONDO PULSANTE: Le tue Credenziali (Spostato a sinistra, indipendente dal primo)
with st.sidebar.popover("📍 Le tue Credenziali", use_container_width=True):
    st.markdown("<p style='text-align: center; font-style: italic; margin-bottom: 15px;'>La tua Credenziale è la memoria del tuo spirito, custodiscila con cura.</p>", unsafe_allow_html=True)
    st.markdown("""
    * **Palermo:** Cattedrale (9:00-17:30) | Centro "Padre Nostro" (feriali 9:30-12:30; mar/gio 15:00-18:00).
    * **Monreale:** Duomo (8:30-12:45 / 14:30-17:00).
    * **Altofonte:** Ufficio Comunale, Parrocchie.
    * **Santa Cristina Gela:** Ufficio Comunale.
    * **Corleone:** Ufficio Comunale, Parrocchie.
    * **Prizzi:** Sportello Turistico (lun-ven 9:00-14:00) | Museo (sab 16:00-20:00; dom 9:00-13:00).
    * **Castronovo di Sicilia:** Ufficio Turistico, Parrocchia.
    * **Cammarata:** Comune, Ufficio Turistico.
    * **Sutera:** Ufficio Comunale, Parrocchia, Museo del Pellegrino.
    * **Grotte:** Centralino Comune (Piazza Umberto I), Parrocchia.
    * **Joppolo Giancaxio:** Ufficio Comunale, Parrocchie, Ristoratori.
    * **Agrigento:** Mudia (Via Duomo 96) per il Testimonium, Parrocchie.
    """)
    
