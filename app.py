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
# Configurazione Stile CSS (Corretto senza st.divider all'interno)
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color:#B5A585;
        background-attachment: fixed;
        color: #231709;
        font-size: 36px
        font-weight: bold !important;

    }
    
    /* Configurazione scritta "Chiedi al chatbot" */
    .stTextInput label div p {
        color: #542E17 !important;
        font-size: 24px !important;
        font-weight: bold !important;
    }
    
    /* Rettangolo di input */
    .stTextInput input {
        background-color: #4F7942;
        color: #ffffff;
    }
    /* Cambia il colore di sfondo della barra laterale */
    [data-testid="stSidebar"] {
        background-color: #b25431; 
    }
    
    
    /* Configurazione dell'applicazione */
    .stApp {
        background-color: #B5A585;
        background-attachment: fixed;
        color: #524E17;
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
        color: #ffffff;
    }
        /* ELIMINA LA BARRA GRIGIA: Rende trasparenti i contenitori dei messaggi */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding-left: 0px !important; /* Opzionale: allinea meglio l'avatar a sinistra */
     }

   /* SISTEMAZIONE DELLA TENDINA (POPOVER BODY) - VERSIONE PULITA       */
    /* ================================================================= */
    
    [data-testid="stPopoverBody"] {
        background-color: #transparent !important; /* Mantiene lo sfondo interno per leggere il testo bianco */
        border: none !important;              /* LEVA IL QUADRATO/BORDO ATTORNO */
        border-radius: 12px !important;       /* Smussa gli angoli per renderlo meno spigoloso */
        box-shadow: 0px 8px 24px rgba(0, 0, 0, 0.4) !important; /* Un'ombra morbida al posto del bordo */
        min-width: 320px !important;          /* Ottimo per PC */
        max-width: 85vw !important;           /* Perfetto per non toccare i bordi del telefono */
        padding: 18px !important;
    }

    /* Modifica il pulsante "Il Codice del Viandante" prima di cliccarlo */
    [data-testid="stPopover"] button {
        background-color: transparent !important; /* Rende il bottone trasparente integrato nella sidebar */
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    
    [data-testid="stPopover"] button:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* ================================================================= */
    /* OTTIMIZZAZIONE SPECIFICA PER IL TELEFONO (RESPONSIVE CENTERING)   */
    /* ================================================================= */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            width: 85vw !important; /* Diamo un po' di respiro in più alla sidebar su mobile */
        }
        
        /* FORZA IL CENTRAMENTO DELLA TENDINA SU MOBILE */
        [data-testid="stPopoverBody"] {
            min-width: 280px !important;
            max-width: 80vw !important;
            position: fixed !important;  /* Lo sblocca dalla posizione nativa di Streamlit */
            left: 50% !important;        /* Lo sposta al centro esatto della sidebar */
            transform: translateX(-50%) !important; /* Bilancia il posizionamento per una centratura perfetta */
            top: auto !important;        /* Evita che scivoli troppo in alto o in basso */
        }
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

           
# Interfaccia grafica sidebar

# -------------------------------------------------------------------
# Barra Laterale (Sidebar) - Il Codice del Viandante
# -------------------------------------------------------------------
st.sidebar.image("LOGO.png", width=120)
st.sidebar.header("I tuoi passi")

st.sidebar.write("---") # Una linea di separazione visiva

# Creazione del pulsante con menù a scomparsa (popover)
with st.sidebar.popover("📜 Il Codice del Viandante", use_container_width=True):
    st.markdown("### 🗺️ Le Regole del Cammino")
    st.write("Caro pellegrino, per vivere al meglio la Magna Via Francigena, custodisci queste indicazioni nel tuo cuore:")
    
    # Serie di scritte all'interno del menù
    st.markdown("""
    **📜Il Codice del Viandante
* 📍 Guida per il tuo Cammino
Il rispetto è il primo passo del pellegrino.
* 🍃 Rispetta la Natura
Non lasciare traccia, solo impronte. Porta sempre con te i tuoi rifiuti e i mozziconi. Il fuoco è un nemico: non accenderlo mai.
* 🏡 Rispetta il Territorio
Sei ospite di terreni agricoli: chiudi i cancelli e non calpestare i raccolti. Chiedi sempre prima di cogliere frutti.
* 🤫 Rispetta il Silenzio
Il cammino è meditazione. Rispetta la quiete nei borghi, nei monasteri e negli ospitali.
* 🎒 Sii Essenziale
Viaggia leggero. Negli ostelli, sii ordinato e rispettoso: non è un hotel, ma una casa condivisa.
* 🤝 Sii Solidale
Aiuta chi è in difficoltà. Un sorriso o un consiglio possono fare la differenza per un altro viandante.
* 🙏 Sii Grato e Umile
Ringrazia chi ti ospita. Accetta con curiosità i ritmi e la cultura che incontri. La tua Credenziale è la memoria del tuo spirito: custodiscila con cura.
    
    """)
    
    st.caption("Buon cammino, custode della Via! ✨")
# -------------------------------------------------------------------
# Elaborazione Documento PDF e RAG
# -------------------------------------------------------------------
cartella_corrente = os.path.dirname(__file__)
documento = os.path.join(cartella_corrente, "TAPPE AGGIORNATE.pdf")

# Inizializziamo la variabile del retriever (catena) fuori dall'if
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
        st.error("Il file PDF è stato trovato, ma non è stato possibile estrarre testo. È un PDF scannerizzato?")
    else:
        @st.cache_resource(show_spinner=False)
        def crea_vectorstore(lista_frammenti):
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=st.secrets["OPENAI_API_KEY"]
            )
            return FAISS.from_texts(lista_frammenti, embedding=embeddings) 
        
        vettori = crea_vectorstore(frammenti)
        
        # --- CONFIGURAZIONE LANGCHAIN (Spostata qui per essere pronta subito) ---
        def formatta_documento(documenti):
            return "\n\n".join([doc.page_content for doc in documenti])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             '''Sei “La Magna via”, un assistente digitale dedicato ai pellegrini della Magna Via Francigena in Sicilia.
 
Il tuo ruolo è accompagnare l’utente durante il cammino fornendo:
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
 
Le risposte devono essere:
- chiare
- utili durante il cammino
- semplici da consultare anche in mobilità
- coerenti con l’esperienza del pellegrinaggio
- accoglienti e orientate all’accompagnamento del pellegrino.
- Quando l'utente chiede informazioni su una tappa, verifica se il percorso attraversa aree sensibili (boschi, riserve naturali, zone di macchia mediterranea). 
Se la risposta è affermativa, aggiungi in chiusura:
':herb: Cammina da custode (vai a capo)
La Magna Via è un dono prezioso, proteggiamola insieme dal rischio incendi. Per favore, evita di fumare nei boschi e porta sempre con te i mozziconi fino al prossimo borgo. Non lasciare traccia, solo impronte. Grazie!'
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

def invia():
    input_utente = st.session_state.domanda_utente
    
    if input_utente and catena is not None:  
        # 1. Salva la domanda dell'utente
        st.session_state.cronologia.append({"role": "user", "content": input_utente})
        
        # 2. Genera la risposta REALE usando LangChain
        with st.spinner("Il chatbot sta rispondendo..."):
            risposta_bot = catena.invoke(input_utente)
        
        # 3. Salva la risposta del bot
        st.session_state.cronologia.append({"role": "assistant", "content": risposta_bot})
        
        # 4. Resetta il campo di input
        st.session_state.domanda_utente = ""

# Mostra la cronologia a schermo
st.write("---")
for messaggio in st.session_state.cronologia:

    # Assegna l'avatar corretto in base al ruolo
    if messaggio["role"] == "user":
        icona = "Utente.png"  
        # Puoi anche differenziare le tonalità se vuoi (es. un marrone leggermente diverso per l'utente)
        colore_testo = "#4A2E1B"  # Dark Chocolate
    else:
        icona = "LOGO.png"  
        colore_testo = "#3D2314"  # Un Dark Chocolate ancora più intenso per il Bot

    with st.chat_message(messaggio["role"], avatar=icona):
        # Usiamo st.markdown con un tag span per applicare il colore dark chocolate
        testo_colorato = f'<span style="color: {colore_testo}; font-size: 15px;">{messaggio["content"]}</span>'
        st.markdown(testo_colorato, unsafe_allow_html=True)
        

# Blocco di Input Interattivo ancorato in basso
if catena is not None:
    # L'operatore := cattura l'input quando l'utente preme invio sulla tastiera o sul tasto freccia
    if input_utente := st.chat_input("Chiedi alla Via..."):
        
        # 1. Salva e mostra subito il messaggio dell'utente
        st.session_state.cronologia.append({"role": "user", "content": input_utente})
        with st.chat_message("user", avatar="Utente.png"):
            st.markdown(f'<div style="color: #4A2E1B; font-size: 16px;">{input_utente}</div>', unsafe_allow_html=True)
        
        # 2. Genera e mostra live la risposta del Modello RAG
        with st.chat_message("assistant", avatar="LOGO.png"):
            with st.spinner("Il chatbot sta rispondendo..."):
                risposta_bot = catena.invoke(input_utente)
                st.markdown(f'<div style="color: #3D2314; font-size: 16px; line-height: 1.5;">{risposta_bot}</div>', unsafe_allow_html=True)
        
        # 3. Salva la risposta del bot nella cronologia per mantenerla nei refresh futuri
        st.session_state.cronologia.append({"role": "assistant", "content": risposta_bot})
        
        # 4. Aggiorna la pagina per allineare tutto lo stato interno
        st.rerun()
