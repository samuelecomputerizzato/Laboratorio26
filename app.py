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
# Configurazione Stile CSS (Layout con Sidebar Custom + Nuke Streamlit)
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
        height: 0px !important;
        width: 0px !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    /* 2. STILE GLOBALE APP E PAGINA CENTRALE */
    .stApp {
        background-color: #F1F0E6;
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

    /* Rende trasparenti i messaggi della chat per eliminare la barra grigia */
    [data-testid="stChatMessage"], [data-testid="stChatMessage"] > div, .stChatMessage {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding-left: 0px !important;
    }

    /* Forza i testi della chat a rimanere scuri */
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li, [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em, [data-testid="stChatMessage"] span, [data-testid="stChatMessage"] div {
        color: #231709 !important;
    }

    /* FORZA LA CENTRATURA PERFETTA E RIDUCE IL LOGO CENTRALE */
    .main [data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin-left: auto !important;
        margin-right: auto !important;
        width: 100% !important;
    }

    .main [data-testid="stImage"] img {
        display: block !important;
        margin: 0 auto !important;
        max-width: 180px !important;
        height: auto !important;
    }

    h1 {
        font-size: 2.2rem !important;
        margin-top: 15px !important;
        text-align: center !important;
        width: 100% !important;
        color: #542E17 !important;
    }

    /* =================================================================
       3. ENGINE DELLA NUOVA SIDEBAR CUSTOM (HTML/CSS PRIVATO)
       ================================================================= */
    
    /* Stile dei blocchi collassabili <details> */
    .custom-sidebar details {
        background-color: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 8px;
        margin-bottom: 12px;
        padding: 12px;
        transition: all 0.3s ease;
    }
    
    .custom-sidebar summary {
        font-weight: bold;
        font-size: 1.05rem;
        cursor: pointer;
        outline: none;
        color: #ffffff !important;
        user-select: none;
    }
    
    .custom-sidebar .details-content {
        margin-top: 10px;
        font-size: 0.92rem;
        line-height: 1.5;
        color: #f8f9fa !important;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 8px;
    }

    .custom-sidebar .details-content ul {
        padding-left: 18px;
        margin: 0;
    }

    .custom-sidebar .details-content li {
        margin-bottom: 8px;
        color: #f8f9fa !important;
    }

    .custom-sidebar .details-content strong {
        color: #ffffff !important;
    }

    /* CONFIGURAZIONE RESPONSIVA: COMPORTAMENTO PC VS TELEFONO */
    
    /* --- CONFIGURAZIONE MONITOR GRANDE (Vera Sidebar a Sinistra) --- */
    @media (min-width: 992px) {
        .custom-sidebar {
            position: fixed;
            top: 0;
            left: 0;
            width: 300px;
            height: 100vh;
            background-color: #7A8B74;
            padding: 30px 20px;
            box-shadow: 3px 0 15px rgba(0,0,0,0.08);
            overflow-y: auto;
            z-index: 99999;
        }
        
        /* Sposta la colonna della chat verso destra per non accavallarsi alla sidebar */
        .main .block-container {
            max-width: 680px !important;
            margin-left: 360px !important;
            margin-right: auto !important;
            padding-top: 40px !important;
        }
    }

    /* --- CONFIGURAZIONE MOBILE/TABLET (Diventa un Menu Superiore Elegante) --- */
    @media (max-width: 991px) {
        .custom-sidebar {
            background-color: #7A8B74;
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 25px;
            width: 100%;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        }
        .custom-sidebar h3 {
            font-size: 1.2rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# INIEZIONE HTML: La Nuova Sidebar Custom (Indipendente da Streamlit)
# -------------------------------------------------------------------
html_sidebar = """
<div class="custom-sidebar">
    <h3 style="color: #ffffff !important; text-align: center; margin-top: 0; margin-bottom: 10px; font-family: sans-serif;">Menu del Viandante</h3>
    <p style="color: #e2e8f0 !important; text-align: center; font-size: 0.85rem; margin-bottom: 20px; font-style: italic;">Informazioni Utili per la Via</p>
    <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.2); margin-bottom: 20px;">
    
    <details>
        <summary>📜 Il Codice del Viandante</summary>
        <div class="details-content">
            <p style="font-style: italic; text-align: center; margin-bottom: 10px;">Il rispetto è il primo passo del pellegrino.</p>
            <ul>
                <li><strong>Rispetta la natura:</strong> non lasciare traccia, solo impronte. Porta sempre con te i tuoi rifiuti e i mozziconi. Il fuoco è un nemico: non accenderlo mai.</li>
                <li><strong>Rispetta il territorio:</strong> sei ospite di terreni agricoli: chiudi i cancelli e non calpestare i raccolti.</li>
                <li><strong>Rispetta il silenzio:</strong> il cammino è meditazione. Rispetta la quiete nei borghi e negli ospitali.</li>
                <li><strong>Sii essenziale:</strong> viaggia leggero. Negli ostelli, sii ordinato e rispettoso.</li>
                <li><strong>Sii solidale:</strong> aiuta chi è in difficoltà. Un sorriso può fare la differenza.</li>
                <li><strong>Sii grato e umile:</strong> ringrazia chi ti ospita.</li>
            </ul>
        </div>
    </details>

    <details>
        <summary>📍 Le tue Credenziali</summary>
        <div class="details-content">
            <p style="font-style: italic; text-align: center; margin-bottom: 10px;">La tua Credenziale è la memoria del tuo spirito.</p>
            <ul>
                <li><strong>Palermo:</strong> Cattedrale (9:00-17:30) | Centro "Padre Nostro" (feriali 9:30-12:30; mar/gio 15:00-18:00).</li>
                <li><strong>Monreale:</strong> Duomo (8:30-12:45 / 14:30-17:00).</li>
                <li><strong>Altofonte:</strong> Ufficio Comunale, Parrocchie.</li>
                <li><strong>Santa Cristina Gela:</strong> Ufficio Comunale.</li>
                <li><strong>Corleone:</strong> Ufficio Comunale, Parrocchie.</li>
                <li><strong>Prizzi:** Sportello Turistico (lun-ven 9:00-14:00) | Museo (sab 16:00-20:00; dom 9:00-13:00).</li>
                <li><strong>Castronovo di Sicilia:</strong> Ufficio Turistico, Parrocchia.</li>
                <li><strong>Cammarata:</strong> Comune, Ufficio Turistico.</li>
                <li><strong>Sutera:</strong> Ufficio Comunale, Parrocchia, Museo del Pellegrino.</li>
                <li><strong>Grotte:</strong> Centralino Comune, Parrocchia.</li>
                <li><strong>Joppolo Giancaxio:</strong> Ufficio Comunale, Parrocchie.</li>
                <li><strong>Agrigento:</strong> Mudia (Via Duomo 96) per il Testimonium, Parrocchie.</li>
            </ul>
        </div>
    </details>
</div>
"""
# Inietta la barra laterale custom nell'applicazione
st.markdown(html_sidebar, unsafe_allow_html=True)

# -------------------------------------------------------------------
# Interfaccia centrale (LOGO e Titolo)
# -------------------------------------------------------------------
st.image("LOGO.png")
st.markdown("<h1>La Magna Via</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #542E17; font-weight: bold; font-size: 1.1rem; margin-bottom: 20px;'>Ultreya, viandante!</p>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Elaborazione Documento PDF e RAG
# -------------------------------------------------------------------
cartella_corrente = os.path.dirname(__file__)
documento = os.path.join(cartella_corrente, "TAPPE AGGIORNATE.pdf")

catena = None

if os.path.exists(documento):
    
    @st.cache_data(show_spinner="Sto leggendo il PDF...")
    def estrai_testo_pdf(percorso_pdf: str) -> str:
        testo = ""
        with pdfplumber.open(percorso_pdf) as pdf: 
            for pagina in pdf.pages:
                testo_pagina = pagina.extract_text() or ""
                testo = testo + testo_pagina + "\n"
        return testo.strip()
    
    testo = estrai_testo_pdf(documento)

    @st.cache_data(show_spinner=False)
    def crea_frammenti(testo_estratto: str):
        taglierina = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=1000,
            chunk_overlap=200
        )
        return [f for f in taglierina.split_text(testo_estratto) if f.strip()]

    frammenti = crea_frammenti(testo)

    if not frammenti:
        st.error("Il file PDF è stato trovato, ma non è stato possibile estrarre testo.")
    else:
        @st.cache_resource(show_spinner=False)
        def crea_vectorstore(lista_frammenti):
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=st.secrets["OPENAI_API_KEY"]
            )
            return FAISS.from_texts(lista_frammenti, embedding=embeddings) 
        
        vettori = crea_vectorstore(frammenti)
        
        def formatta_documento(documenti):
            return "\n\n".join([doc.page_content for doc in documenti])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             '''Sei “La Magna via”, un assistente digitale dedicato ai pellegrini della Magna Via Francigena in Sicilia.
 
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
- i nomi  delle tappe e informazioni importanti come km, presenza di cani, acqua, cibo ecc. devono essere visualizzati in grassetto nella cronologia della chat. 

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

        comparatore = vettori.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4}
        )
        
        modello_llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.3,
            max_tokens=1000,
            openai_api_key=st.secrets["OPENAI_API_KEY"]
        )
        
        catena = (
            {"context": comparatore | formatta_documento, "question": RunnablePassthrough()}
            | prompt
            | modello_llm
            | StrOutputParser()
        )
else:
    st.error(f"Non ho trovato il file PDF nel percorso: {documento}.")

# -------------------------------------------------------------------
# Gestione Cronologia e Messaggi Chat
# -------------------------------------------------------------------
if "cronologia" not in st.session_state:
    st.session_state.cronologia = []

# Mostra la cronologia a schermo
st.write("---")
for messaggio in st.session_state.cronologia:
    if messaggio["role"] == "user":
        icona = "Utente.png"  
        colore_testo = "#4A2E1B"  
    else:
        icona = "LOGO.png"  
        colore_testo = "#3D2314"  

    with st.chat_message(messaggio["role"], avatar=icona):
        testo_colorato = formatta_messaggio(messaggio["content"], colore_testo, "15px")
        st.markdown(testo_colorato, unsafe_allow_html=True)
        
# Blocco di Input Interattivo ancorato in basso
if catena is not None:
    if input_utente := st.chat_input("Chiedi alla Via..."):
        
        st.session_state.cronologia.append({"role": "user", "content": input_utente})
        with st.chat_message("user", avatar="Utente.png"):
            st.markdown(formatta_messaggio(input_utente, "#4A2E1B", "16px"), unsafe_allow_html=True)
        
        with st.chat_message("assistant", avatar="LOGO.png"):
            with st.spinner("Il chatbot sta rispondendo..."):
                risposta_bot = catena.invoke(input_utente)
                st.markdown(formatta_messaggio(risposta_bot, "#3D2314", "16px"), unsafe_allow_html=True)
        
        st.session_state.cronologia.append({"role": "assistant", "content": risposta_bot})
        st.rerun()
