import streamlit as st
import pdfplumber
import os

# Importazioni LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configurazione iniziale della pagina (DEVE ESSERE LA PRIMA ISTRUZIONE STREAMLIT)
st.set_page_config(page_title="La Magna Via", page_icon=":walking_man:", layout="centered")

# -------------------------------------
# Configurazione estetica della pagina 
# -------------------------------------
st.html(
    """
    <style>
    
    /* Assicurati che l'header nativo st.header (h2) sia centrato */
    .block-container h2 {
        text-align: center !important;
        width: 100% !important;
        margin-top: 0px !important; 
    }

    .block-container {
        padding-top: 1.5rem !important; /* Ridotto drasticamente per alzare tutto l'header */
        padding-bottom: 3rem !important;
    }

    /* Elementi nativi di Streamlit eliminati */
    header, footer, [data-testid="stHeader"], [data-testid="stAppHeader"], 
    [data-testid="stDecoration"], #MainMenu, [data-testid="stToolbar"], 
    .stDeployButton, [data-testid="stManageAppButton"] { 
        display: none !important; 
        visibility: hidden !important;
    }
    
    /* Configurazione colore di sfondo e testo dell'app */
    .stApp { background-color: #F1F0E6; color: #231709; }

    .sidebar-checkbox { display: none !important; }

    /* Pulsante custom per aprire/chiudere la sidebar */
    .sidebar-toggle-button {
        position: fixed; top: 15px; left: 15px;
        background-color: #7A8B74; color: #ffffff !important;
        padding: 10px 16px; border-radius: 8px; font-weight: bold;
        cursor: pointer; z-index: 99998; box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        font-family: sans-serif;
    }

    /* Stile della sidebar personalizzata */
    .custom-sidebar {
        position: fixed; top: 0; left: -340px; width: 320px; height: 100vh;
        background-color: #7A8B74 !important;
        padding: 40px 22px; z-index: 99999;
        transition: left 0.3s ease;
        overflow-y: auto;
        font-family: sans-serif;
    }

    /* Stile per i componenti details */
    .custom-sidebar details {
        background-color: #677761 !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
        margin-bottom: 14px !important;
        padding: 12px !important;
        display: block !important;
    }
    
    /* Stile per il summary */
    .custom-sidebar summary {
        font-weight: bold !important;
        font-size: 1.05rem !important;
        cursor: pointer !important;
        color: #ffffff !important;
    }

    /* Stile per i testi all'interno dei details */
    .custom-sidebar h3, .custom-sidebar h4, .custom-sidebar p, .custom-sidebar li, 
    .custom-sidebar strong, .custom-sidebar span {
        color: #ffffff !important;
        background-color: transparent !important;
    }

    /* Effetto apertura/chiusura sidebar al check del checkbox */
    .sidebar-checkbox:checked ~ .custom-sidebar { left: 0 !important; }
    
    /* Pulsante di chiusura interno alla sidebar */
    .sidebar-close {
        position: absolute; top: 15px; right: 20px;
        color: #ffffff !important; font-size: 1.5rem; cursor: pointer;
    }

    /* liste puntate */
    .custom-sidebar ul { padding-left: 18px !important; margin: 10px 0 0 0 !important; }
    .custom-sidebar li { margin-bottom: 8px !important; font-size: 0.9rem !important; line-height: 1.4; }

    /* --- FIX TEMA SCURO / VISIBILITÀ INPUT --- */
    div[data-testid="stChatInput"] {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding-bottom: 20px !important;
    }

    /* Forza lo sfondo chiaro e i bordi scuri per il form dell'input */
    div[data-testid="stChatInput"] form {
        background-color: #F1F0E6 !important; 
        border: 2px solid #542E17 !important;   
        border-radius: 28px !important;         
        padding: 5px 10px !important;
        box-shadow: 0px 4px 15px rgba(84, 46, 23, 0.1) !important; 
    }

    /* Stile del placeholder */
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #542E17 !important;
        opacity: 0.6;
    }

    /* Stile del pulsante di invio */
    div[data-testid="stChatInput"] button {
        background-color: #7A8B74 !important; 
        border-radius: 50% !important;         
        color: #ffffff !important;             
        transition: background-color 0.2s ease;
    }

    /* Stile del pulsante di invio al passaggio del mouse */
    div[data-testid="stChatInput"] button:hover {
        background-color: #677761 !important; 
    }

    /* FORZA IL TESTO SCURO NEI MESSAGGI DI CHAT */
    [data-testid="stChatMessage"] div, 
    [data-testid="stChatMessage"] div p, 
    [data-testid="stChatMessage"] div li, 
    [data-testid="stChatMessage"] div td, 
    [data-testid="stChatMessage"] div th, 
    [data-testid="stChatMessage"] div h1, 
    [data-testid="stChatMessage"] div h2, 
    [data-testid="stChatMessage"] div h3,
    [data-testid="stChatMessage"] div span {
        color: #231709 !important;
        font-family: sans-serif !important;
    }

    /* Contenitore desktop per il logo */
    .desktop-logo-container {
        margin-top: 15px;
        margin-right: 15px;
        display: flex;
        justify-content: flex-end;
    }
    .desktop-logo-container img {
        max-width: 130px;
        height: auto;
    }

    /* MODIFICHE SPECIFICHE PER DISPOSITIVI MOBILE E SCHERMI PICCOLI */
    @media (max-width: 768px) {
        /* Larghezza della sidebar su mobile */
        .custom-sidebar { width: 85vw !important; left: -90vw !important; }
        
        /* Avvicina l'intero blocco centrale all'estremità superiore */
        .block-container {
            padding-top: 1rem !important; 
        }
        
        div[data-testid="stImage"] { 
            padding-right: 0px !important; 
        }
        
        /* Contenitore logo ottimizzato per mobile: centrato e compatto */
        .desktop-logo-container {
            margin: 5px auto 15px auto !important;
            display: flex !important;
            justify-content: center !important;
            width: 100% !important;
        }

        /* Ridimensiona il logo su mobile per non renderlo sgranato/grossolano */
        .desktop-logo-container img {
            max-width: 140px !important; /* Dimensione controllata e definita */
            height: auto !important;
        }
    }
    </style>
    """
)

# Renderizza l'HTML della sidebar custom
st.html(html_sidebar if 'html_sidebar' in globals() else "") # (Mantieni la tua stringa html_sidebar intatta)


# ---------------------------------------------------------
# Configurazione pagina, Header principale e immagine 
# ------------------------------------------------------------

# Utilizziamo un container unico invece di st.columns che collassa male su mobile.
# Questo garantisce che l'ordine degli elementi sia coerente e pulito su qualsiasi schermo.
st.markdown('<div class="desktop-logo-container">', unsafe_allow_html=True)
st.image("LOGO.png")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; margin-top: -10px; margin-bottom: 20px;'>La Magna via</h2>", unsafe_allow_html=True)


# ----------------------------------
# Elaborazione Documento PDF e RAG 
# ----------------------------------
# (Il resto del tuo codice inerente a LangChain e alla Chat rimane esattamente identico...)
