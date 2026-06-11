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

st.set_page_config(page_title="Verbum viae", page_icon=":walking_man:")
st.sidebar.image("ImmagineOrizzontale.webp",width=100)
st.sidebar.header("I tuoi passi")
# Personalizzazione colori
st.markdown(
    """
    <style>
    .stApp {
        background-color:#B5A585;
        background-attachment: fixed;
        color: #37261B;
        font-size: 36px;
    }
    st.divider()
    /* Configurazione scritta "Chiedi al chatbot" */
    .stTextInput label div p {
        color: #4A2A20 !important;
        font-size: 24px !important;
        font-weight: bold !important;
    }
    
    /* Rettangolo di input */
    .stTextInput input {
        background-color: #4F7942;
        color: #ffffff;
     st.divider()   
    }
    </style>
    """,
    unsafe_allow_html=True)

st.header("Verbum viae")
st.image("ImmagineOrizzontale.webp")




# Gestione dinamica del percorso dell'immagine


# FIX PERCORSO FILE: Trova il PDF nella stessa cartella di questo file app.py
cartella_corrente = os.path.dirname(__file__)
documento = os.path.join(cartella_corrente, "Tappe.pdf")

# Estrazione del contenuto e spezzettamento
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
        # Rimuoviamo stringhe vuote prima dello split
        return [f for f in taglierina.split_text(testo_estratto) if f.strip()]

    frammenti = crea_frammenti(testo)

    # CONTROLLO DI SICUREZZA: Evita il crash se il PDF è vuoto o fatto di sole immagini
    if not frammenti:
        st.error("Il file PDF è stato trovato, ma non è stato possibile estrarre testo. È un PDF scannerizzato (immagine)?")
    else:
        @st.cache_resource(show_spinner=False)
        def crea_vectorstore(lista_frammenti):
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=st.secrets["OPENAI_API_KEY"]
            )
            return FAISS.from_texts(lista_frammenti, embedding=embeddings) 
        
        vettori = crea_vectorstore(frammenti)

    


if "cronologia" not in st.session_state:
    st.session_state.cronologia = []

# -------------------------------------------------------------------
# Gestione prompt e input
# -------------------------------------------------------------------
def invia():
    # Recupera il testo inserito
    input_utente = st.session_state.domanda_utente
    
    if input_utente:  # Evita di inviare messaggi vuoti
        # Salva la domanda nell'ultimo stato inviato
        st.session_state.domanda_inviata = input_utente
        
        # Aggiungi il messaggio dell'utente alla cronologia
        st.session_state.cronologia.append({"role": "user", "content": input_utente})
        
        # --- QUI ANDRÀ LA LOGICA DEL TUO CHATBOT ---
        # Esempio di risposta fissa del bot:
        risposta_bot = f"Hai detto: '{input_utente}'. Sono il tuo assistente!"
        st.session_state.cronologia.append({"role": "assistant", "content": risposta_bot})
        # ------------------------------------------
        
        # Resetta il campo di input
        st.session_state.domanda_utente = ""

# 2. Mostra la cronologia della chat LONTANO dal fondo (prima del campo di input)
st.title("Il tuo Chatbot")

for messaggio in st.session_state.cronologia:
    with st.chat_message(messaggio["role"]):
        st.write(messaggio["content"])

# 3. Campo di input con il testo ALL'INTERNO (placeholder)
st.text_input(
    "Label nascosta", # Etichetta obbligatoria per accessibilità
    placeholder="Chiedi alla Via...", 
    key="domanda_utente", 
    on_change=invia,
    label_visibility="collapsed" # Nasconde l'etichetta sopra la barra
)


        def formatta_documento(documenti):
            return "\n\n".join([doc.page_content for doc in documenti])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             '''Sei “Sicily Pilgrim Assistant”, un assistente digitale dedicato ai pellegrini della Magna Via Francigena in Sicilia.
 
Il tuo ruolo è accompagnare l’utente durante il cammino fornendo:
- informazioni pratiche (acqua, distanza, difficoltà delle tappe)
- supporto culturale e narrativo sul territorio
- indicazioni su ospitalità, ristoro e luoghi di interesse
 
Regole di comportamento:
- Usa esclusivamente le informazioni presenti nel contesto fornito
- Non inventare informazioni mancanti
- Se l’informazione richiesta non è disponibile nel contesto, rispondi in modo accogliente e coerente con il ruolo di guida del cammino
 
Rispondi:
“Caro pellegrino, al momento non riesco a guidarti su questa informazione.”
 
Le risposte devono essere:
- chiare
- utili durante il cammino
- semplici da consultare anche in mobilità
- coerenti con l’esperienza del pellegrinaggio
- accoglienti e orientate all’accompagnamento del pellegrino'. 
Contesto:\n{context}'''),
            ("human", "{question}")
        ])

        comparatore = vettori.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4}
        )
        
        # FIX MODELLO: Cambiato in gpt-4o-mini (veloce ed economico)
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
        
        if domanda_utente:
            risposta = catena.invoke(domanda_utente)
            st.write(risposta)
else:
    st.error(f"Non ho trovato il file PDF nel percorso: {documento}. Assicurati che sia nella cartella 'RAG Samuele'.")
