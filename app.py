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
# Configurazione Stile CSS (Blindato + Ripristino Pulsanti)
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Nuke Streamlit */
    header, footer, [data-testid="stHeader"], [data-testid="stAppHeader"], 
    [data-testid="stDecoration"], #MainMenu { display: none !important; }
    
    .stApp { background-color: #F1F0E6; color: #231709; }

    /* ENGINE SIDEBAR */
    .sidebar-checkbox { display: none !important; }

    .sidebar-toggle-button {
        position: fixed; top: 15px; left: 15px;
        background-color: #7A8B74; color: #ffffff !important;
        padding: 10px 16px; border-radius: 8px; font-weight: bold;
        cursor: pointer; z-index: 99998; box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        font-family: sans-serif;
    }

    .custom-sidebar {
        position: fixed; top: 0; left: -340px; width: 320px; height: 100vh;
        background-color: #7A8B74 !important;
        padding: 40px 22px; z-index: 99999;
        transition: left 0.3s ease;
        overflow-y: auto;
        font-family: sans-serif;
    }

    /* Gestione dei blocchi a fisarmonica (I tuoi pulsanti) */
    .custom-sidebar details {
        background-color: #677761 !important; /* Sfondo solido per il pulsante */
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
        margin-bottom: 14px !important;
        padding: 12px !important;
        display: block !important;
    }
    
    .custom-sidebar summary {
        font-weight: bold !important;
        font-size: 1.05rem !important;
        cursor: pointer !important;
        color: #ffffff !important;
        list-style: list-item !important;
    }

    /* Protezione testo anti-evidenziazione e anti-tema scuro */
    .custom-sidebar h3,
    .custom-sidebar p,
    .custom-sidebar li,
    .custom-sidebar strong,
    .custom-sidebar span {
        color: #ffffff !important;
        background-color: transparent !important;
    }

    .sidebar-checkbox:checked ~ .custom-sidebar { left: 0 !important; }
    
    .sidebar-close {
        position: absolute; top: 15px; right: 20px;
        color: #ffffff !important; font-size: 1.5rem; cursor: pointer;
    }

    .custom-sidebar ul { padding-left: 18px !important; margin: 10px 0 0 0 !important; }
    .custom-sidebar li { margin-bottom: 8px !important; font-size: 0.9rem !important; line-height: 1.4; }

    @media (max-width: 480px) {
        .custom-sidebar { width: 85vw !important; left: -90vw !important; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# INIEZIONE STRUTTURA SIDEBAR CON CONTENUTO COMPLETO
# -------------------------------------------------------------------
html_sidebar = """
<input type="checkbox" id="side-menu-switch" class="sidebar-checkbox">
<label for="side-menu-switch" class="sidebar-toggle-button">☰ Lo spazio del pellegrino</label>

<div class="custom-sidebar">
    <label for="side-menu-switch" class="sidebar-close">✕</label>
    
    <h3 style="text-align: center; margin-bottom: 5px;">Menu del Viandante</h3>
    <p style="text-align: center; font-size: 0.85rem; font-style: italic; opacity: 0.9; margin-bottom: 20px;">Informazioni per il Cammino</p>
    <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.3); margin-bottom: 20px;">
    
    <!-- PULSANTE 1: IL CODICE DEL VIANDANTE -->
    <details>
        <summary>📜 Il Codice del Viandante</summary>
        <p style="font-style: italic; text-align: center; margin-top: 10px; font-size: 0.85rem; opacity: 0.9;">Il rispetto è il primo passo del pellegrino.</p>
        <ul>
            <li><strong>Rispetta la natura:</strong> non lasciare traccia, solo impronte. Porta sempre con te i tuoi rifiuti e i mozziconi. Il fuoco è un nemico: non accenderlo mai.</li>
            <li><strong>Rispetta il territorio:</strong> sei ospite di terreni agricoli: chiudi i cancelli e non calpestare i raccolti. Chiedi sempre prima di cogliere frutti.</li>
            <li><strong>Rispetta il silenzio:</strong> il cammino è meditazione. Rispetta la quiete nei borghi, nei monasteri e negli ospitali.</li>
            <li><strong>Sii essenziale:</strong> viaggia leggero. Negli ostelli, sii ordinato e respektoso: non è un hotel, ma una casa condivisa.</li>
            <li><strong>Sii solidale:</strong> aiuta chi è in difficoltà. Un sorriso o un consiglio possono fare la differenza per un altro viandante.</li>
            <li><strong>Sii grato e umile:</strong> ringrazia chi ti ospita. Accetta con curiosità i ritmi e la cultura che incontri.</li>
        </ul>
    </details>

    <!-- PULSANTE 2: LE CREDENZIALI -->
    <details>
        <summary>📍 Le tue Credenziali</summary>
        <p style="font-style: italic; text-align: center; margin-top: 10px; font-size: 0.85rem; opacity: 0.9;">La tua Credenziale è la memoria del tuo spirito, custodiscila con cura.</p>
        <ul>
            <li><strong>Palermo:</strong> Cattedrale (9:00-17:30) | Centro "Padre Nostro" (feriali 9:30-12:30; mar/gio 15:00-18:00).</li>
            <li><strong>Monreale:</strong> Duomo (8:30-12:45 / 14:30-17:00).</li>
            <li><strong>Altofonte:</strong> Ufficio Comunale, Parrocchie.</li>
            <li><strong>Santa Cristina Gela:</strong> Ufficio Comunale.</li>
            <li><strong>Corleone:</strong> Ufficio Comunale, Parrocchie.</li>
            <li><strong>Prizzi:</strong> Sportello Turistico (lun-ven 9:00-14:00) | Museo (sab 16:00-20:00; dom 9:00-13:00).</li>
            <li><strong>Castronovo di Sicilia:</strong> Ufficio Turistico, Parrocchia.</li>
            <li><strong>Cammarata:</strong> Comune, Ufficio Turistico.</li>
            <li><strong>Sutera:</strong> Ufficio Comunale, Parrocchia, Museo del Pellegrino.</li>
            <li><strong>Grotte:</strong> Centralino Comune (Piazza Umberto I), Parrocchia.</li>
            <li><strong>Joppolo Giancaxio:</strong> Ufficio Comunale, Parrocchie, Ristoratori.</li>
            <li><strong>Agrigento:</strong> Mudia (Via Duomo 96) per il Testimonium, Parrocchie.</li>
        </ul>
    </details>
</div>
"""
st.markdown(html_sidebar, unsafe_allow_html=True)

# -------------------------------------------------------------------
# Interfaccia centrale
# -------------------------------------------------------------------
st.image("LOGO.png")
st.markdown("<h1 style='text-align: center; color: #542E17;'>La Magna Via</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #542E17; font-weight: bold; font-size: 1.1rem; margin-bottom: 20px;'>Ultreya, viandante!</p>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Elaborazione Documento PDF e RAG
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
        ("system", '''Sei “La Magna via”, un assistente digitale dedicato ai pellegrini della Magna Via Francigena in Sicilia.
 
Il tuo ruolo è accompagner l’utente durante il cammino fornendo:
- informazioni pratiche (punti/fontanelle/fonti d'acqua, distanza, difficoltà delle tappe, punti di appproviggionamento)
- supporto culturale e narrativo sul territorio
- indicazioni su ospitalità, ristoro e luoghi di interesse
 
Regole di comportamento:
- Usa esclusivamente le informazioni presenti nel contesto fornito
- Non inventare informazioni mancanti
- Se l’informazione richiesta non è disponibile nel contesto, rispondi in modo accogliente e coerente con il ruolo di guida del cammino:
“Caro pellegrino, al momento non riesco a guidarti su questa informazione :cry:.”
- Se non sai la risposta non devi inserire :blush: dopo 'Caro pellegrino'
- Nel caso in cui l'utente ponga una domanda in una lingua diversa dall'italiano rispondi nella stessa lingua.
- Quando l'utente ti pone una domanda senza utilizzare la lingua italiana devi recuperare le informazioni al pdf e tradurle allineandoti alla lingua utilizzata dall'utente
- Nel caso in cui l'utente utilizzi un alfabeto diverso dalle lingue indoeuropee (cirillico, alfabeti asiatici ecc.) rispondi utilizzando lo stesso alfabeto
- Quando il pellegrino scriverà "Ultreya" tu dovrai rispondere "Et suseia!" con entusiasmo.
- i nomi delle tappe e informazioni importanti come km, presenza di cani, acqua, cibo ecc. devono essere visualizzati in grassetto nella cronologia della chat. 

Le risposte devono essere:
- chiare
- utili durante il cammino
- semplici da consultare anche in mobilità
- coerenti con l’esperienza del pellegrinaggio
- accoglienti e orientate all’accompagnamento del pellegrino.
- Quando l'utente chiede informazioni su una tappa, verifica se il percorso attraversa aree sensibili (boschi, riserve naturali, zone di macchia mediterranea). 
Se la risposta è affermativa, aggiungi in chiusura:
':herb: Cammina da custode (vai a capo)
La Magna Via è un dono prezioso, proteggiamola insieme dal rischio incendi. Per favore, evita di fumare nei boschi and porta sempre con te i mozziconi fino al prossimo borgo. Non lasciare traccia, solo impronte. Grazie!'

Contesto:\n{context}'''),
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
    
