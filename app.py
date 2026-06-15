import streamlit as st
import pdfplumber
import os
import re

st.set_page_config(page_title="La Magna Via", page_icon=":walking_man:", layout="centered")

# -------------------------------------------------------------------
# Funzione di utilità per formattare i messaggi
# -------------------------------------------------------------------
def formatta_messaggio(testo: str, colore: str, font_size: str = "15px") -> str:
    testo_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', testo)
    testo_html = testo_html.replace('\n', '<br>')
    return f'<div style="color: {colore}; font-size: {font_size}; line-height: 1.5;">{testo_html}</div>'

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------------------------
# Configurazione Stile CSS (Blindato)
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Nuke Streamlit */
    header, footer, [data-testid="stHeader"], [data-testid="stAppHeader"], 
    [data-testid="stDecoration"], #MainMenu { display: none !important; }
    
    .stApp { background-color: #F1F0E6; color: #231709; }

    /* SIDEBAR BLINDATA */
    .sidebar-checkbox { display: none !important; }

    .sidebar-toggle-button {
        position: fixed; top: 15px; left: 15px;
        background-color: #7A8B74; color: #ffffff !important;
        padding: 10px 16px; border-radius: 8px; font-weight: bold;
        cursor: pointer; z-index: 99998; box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }

    .custom-sidebar {
        position: fixed; top: 0; left: -340px; width: 320px; height: 100vh;
        background-color: #7A8B74 !important; /* Verde solido */
        padding: 40px 22px; z-index: 99999;
        transition: left 0.3s ease;
        overflow-y: auto;
    }

    /* Forza il testo bianco e leggibile */
    .custom-sidebar, .custom-sidebar * {
        color: #ffffff !important;
        background-color: transparent !important; /* Rimuove evidenziazioni */
        font-family: sans-serif;
    }

    .sidebar-checkbox:checked ~ .custom-sidebar { left: 0 !important; }

    .custom-sidebar details {
        background-color: #677761 !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px; margin-bottom: 15px; padding: 10px;
    }
    
    .sidebar-close {
        position: absolute; top: 15px; right: 20px;
        color: #ffffff !important; font-size: 1.5rem; cursor: pointer;
    }

    @media (max-width: 480px) {
        .custom-sidebar { width: 85vw !important; left: -90vw !important; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# INIEZIONE STRUTTURA SIDEBAR
# -------------------------------------------------------------------
html_sidebar = """
<input type="checkbox" id="side-menu-switch" class="sidebar-checkbox">
<label for="side-menu-switch" class="sidebar-toggle-button">☰ Lo spazio del pellegrino</label>

<div class="custom-sidebar">
    <label for="side-menu-switch" class="sidebar-close">✕</label>
    
    <h3 style="text-align: center;">Menu del Viandante</h3>
    <hr style="border-top: 1px solid white;">
    
    <details>
        <summary>📜 Il Codice del Viandante</summary>
        <div style="margin-top: 10px; font-size: 0.9rem;">
            <ul>
                <li><strong>Rispetta la natura:</strong> non lasciare traccia.</li>
                <li><strong>Rispetta il territorio:</strong> sei ospite.</li>
                <li><strong>Rispetta il silenzio:</strong> cammino è meditazione.</li>
            </ul>
        </div>
    </details>

    <details>
        <summary>📍 Le tue Credenziali</summary>
        <div style="margin-top: 10px; font-size: 0.9rem;">
            <ul>
                <li><strong>Palermo:</strong> Cattedrale</li>
                <li><strong>Monreale:</strong> Duomo</li>
                <li><strong>Corleone:</strong> Parrocchie</li>
            </ul>
        </div>
    </details>
</div>
"""
st.markdown(html_sidebar, unsafe_allow_html=True)

# -------------------------------------------------------------------
# Interfaccia centrale
# -------------------------------------------------------------------
st.image("LOGO.png")
st.markdown("<h1 style='color: #542E17;'>La Magna Via</h1>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Elaborazione Documento PDF e RAG (Logica invariata)
# -------------------------------------------------------------------
cartella_corrente = os.path.dirname(__file__)
documento = os.path.join(cartella_corrente, "TAPPE AGGIORNATE.pdf")
catena = None

if os.path.exists(documento):
    @st.cache_data(show_spinner="Sto leggendo il PDF...")
    def estrai_testo_pdf(percorso_pdf):
        testo = ""
        with pdfplumber.open(percorso_pdf) as pdf:
            for pagina in pdf.pages:
                testo += (pagina.extract_text() or "") + "\n"
        return testo.strip()
    
    testo = estrai_testo_pdf(documento)
    
    @st.cache_resource(show_spinner=False)
    def setup_rag(testo_estratto):
        taglierina = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        frammenti = [f for f in taglierina.split_text(testo_estratto) if f.strip()]
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=st.secrets["OPENAI_API_KEY"])
        vettori = FAISS.from_texts(frammenti, embedding=embeddings)
        return vettori

    vettori = setup_rag(testo)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sei un assistente dedicato ai pellegrini della Magna Via. Usa il contesto: {context}"),
        ("human", "{question}")
    ])
    
    modello_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=st.secrets["OPENAI_API_KEY"])
    catena = ({"context": lambda x: "\n\n".join([doc.page_content for doc in vettori.similarity_search(x, k=4)]), "question": RunnablePassthrough()} 
              | prompt | modello_llm | StrOutputParser())

# -------------------------------------------------------------------
# Chat
# -------------------------------------------------------------------
if "cronologia" not in st.session_state: st.session_state.cronologia = []

for messaggio in st.session_state.cronologia:
    with st.chat_message(messaggio["role"], avatar="LOGO.png" if messaggio["role"] == "assistant" else "Utente.png"):
        st.markdown(formatta_messaggio(messaggio["content"], "#231709"))

if catena and (input_utente := st.chat_input("Chiedi alla Via...")):
    st.session_state.cronologia.append({"role": "user", "content": input_utente})
    with st.chat_message("user", avatar="Utente.png"):
        st.markdown(formatta_messaggio(input_utente, "#4A2E1B"))
    
    with st.chat_message("assistant", avatar="LOGO.png"):
        risposta = catena.invoke(input_utente)
        st.markdown(formatta_messaggio(risposta, "#3D2314"))
    
    st.session_state.cronologia.append({"role": "assistant", "content": risposta})
    st.rerun()
