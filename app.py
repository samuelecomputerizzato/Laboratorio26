import streamlit as st
import pdfplumber
import os

# Importazioni LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------------------------------
# Configurazione Stile CSS 
# -----------------------------------------------------------
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
        padding-top: 4rem !important; /* Ridotto leggermente per bilanciare il logo in alto */
        padding-bottom: 3rem !important;
    }

    /* Nasconde gli elementi nativi di Streamlit per un'interfaccia pulita */
    header, footer, [data-testid="stHeader"], [data-testid="stAppHeader"], 
    [data-testid="stDecoration"], #MainMenu, [data-testid="stToolbar"], 
    .stDeployButton, [data-testid="stManageAppButton"] { 
        display: none !important; 
        visibility: hidden !important;
    }
    
    /* Configurazione colore di sfondo e testo dell'app */
    .stApp { background-color: #F1F0E6; color: #231709; }

    /* Nasconde il checkbox di default della sidebar */
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

    /* Stile per i componenti details nella sidebar */
    .custom-sidebar details {
        background-color: #677761 !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
        margin-bottom: 14px !important;
        padding: 12px !important;
        display: block !important;
    }
    
    /* Stile per il summary dei details */
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

    /* Stile liste puntate nella sidebar */
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

    /* FORZA IL TESTO SCURO NELL'INPUT (Risolve il bug del tema scuro) */
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #231709 !important; /* Forza colore scuro del testo digitato */
        -webkit-text-fill-color: #231709 !important; /* Fix per alcuni browser mobile */
        font-family: sans-serif !important;
        font-size: 1rem !important;
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

    /* FORZA IL TESTO SCURO NEI MESSAGGI DI CHAT (Sia Utente che Assistente) */
    [data-testid="stChatMessage"] div p {
        color: #231709 !important;
        font-family: sans-serif !important;
    }

    /* MODIFICHE SPECIFICHE PER DISPOSITIVI MOBILE E SCHERMI PICCOLI */
    @media (max-width: 480px) {
        /* Larghezza della sidebar su mobile */
        .custom-sidebar { width: 85vw !important; left: -90vw !important; }
        
        /* Sposta l'intero blocco centrale più in basso su mobile */
        .block-container {
            padding-top: 4.5rem !important; 
        }
        div[data-testid="stImage"] { 
            padding-right: 0px !important; 
        }
        
        /* Ripristina e accentra l'allineamento del contenitore del logo */
        .desktop-logo-container {
            margin: 15px auto 0 auto !important;
            transform: scale(1.0) !important;
            display: flex !important;
            justify-content: center !important;
            width: 100% !important;
        }

        /* Forza una dimensione contenuta per il logo su mobile/ridimensionamento */
        .desktop-logo-container img {
            max-width: 110px !important;
            height: auto !important;
        }
    }
    </style>
    """
)

# --------------------------------------------------
# SIDEBAR 
# --------------------------------------------------
html_sidebar = """
<input type="checkbox" id="side-menu-switch" class="sidebar-checkbox">
<label for="side-menu-switch" class="sidebar-toggle-button">☰ Lo spazio del viandante</label>
<div class="custom-sidebar">
<label for="side-menu-switch" class="sidebar-close">✕</label>
<h3 style="text-align: center; margin-bottom: 5px;">Ultreya, viandante!</h3>
<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.3); margin-bottom: 20px;">

<details>
<summary>📜 Il Codice del Viandante</summary>
<p style="font-style: italic; text-align: center; margin-top: 10px; font-size: 0.85rem; opacity: 0.9;">Il rispetto è il primo passo del pellegrino.</p>
<ul>
<li><strong>Rispetta la natura:</strong> non lasciare traccia, solo impronte. Porta sempre con te i tuoi rifiuti e i mozziconi. Il fuoco è un nemico: non accenderlo mai.</li>
<li><strong>Rispetta il territorio:</strong> se sei ospite di terreni agricoli chiudi i cancelli e non calpestare i raccolti. Chiedi sempre prima di cogliere frutti.</li>
<li><strong>Rispetta il silenzio:</strong> il cammino è meditazione. Rispetta la quiete nei borghi, nei monasteri e negli ospitali.</li>
<li><strong>Sii essenziale:</strong> viaggia leggero. Negli ostelli, sii ordinato e rispettoso: non è un hotel, ma una casa condivisa.</li>
<li><strong>Sii solidale:</strong> aiuta chi è in difficoltà. Un sorriso o un consiglio possono fare la differenza per un altro viandante.</li>
<li><strong>Sii grato e umile:</strong> ringrazia chi ti ospita. Accetta con curiosità i ritmi e la cultura che incontri.</li>
</ul>
</details>

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

<details>
<summary>📖 Il Glossario del Territorio</summary>
<p style="font-style: italic; text-align: center; margin-top: 10px; font-size: 0.85rem; opacity: 0.9;">Le parole per leggere il cuore della Sicilia e il territorio che stai attraversando.</p>
<h4 style="margin-top: 15px; margin-bottom: 5px; font-size: 0.95rem; font-weight: bold;">🚜 Sulle tracce della storia – il paesaggio</h4>
<ul>
<li><strong>Trazzera:</strong> non è una semplice strada, è l’antica "autostrada" dei pastori e dei re. Camminare qui significa posare i piedi dove, per secoli, è passato il cuore pulsante della Sicilia.</li>
<li><strong>Marna:</strong> è la roccia bianca che disegna le colline agrigentine. Bellissima e candida come la luna, ma attenzione: quando il cielo piange, diventa un terreno infido e scivoloso. Rispetta la sua natura.</li>
<li><strong>Solfara:</strong> sono le ferite aperte della terra, le antiche miniere di zolfo. Oggi sono ruderi silenziosi che raccontano una storia di fatica, polvere e riscatto. Guardali con rispetto.</li>
<li><strong>Kora:</strong> per gli antichi greci era la terra che nutriva la città. Oggi è lo spazio aperto, il silenzio della campagna che ti abbraccia tra un borgo e l'altro.</li>
</ul>
<h4 style="margin-top: 15px; margin-bottom: 5px; font-size: 0.95rem; font-weight: bold;">🏠 Dove riposa la memoria – i luoghi</h4>
<ul>
<li><strong>Robba (o Masseria):</strong> più che una fattoria, è un piccolo mondo autosufficiente. Dietro queste mura di pietra fortificata si è scritta la storia rurale dell'isola.</li>
<li><strong>Rabato:</strong> è il cuore antico di origine araba. Perditi tra le sei case addossate e i vicoli stretti, pensati millenni fa per ingannare il sole e proteggere dal vento.</li>
<li><strong>Hospitale:</strong> è la casa del pellegrino. Anche se oggi ha un aspetto diverso, il suo significato non è cambiato: qui la porta è aperta, il viandante è un ospite sacro e il riposo è un atto di cura.</li>
</ul>
</details>
</div>
"""
# Renderizza l'HTML della sidebar custom
st.html(html_sidebar)


st.set_page_config(page_title="La Magna Via", page_icon=":walking_man:", layout="centered")

# Allarghiamo la colonna di destra (da 1 a 1.3) per permettere al logo di mostreggiarsi più grande 
col_sinistra, col_centro, col_destra = st.columns([1, 4, 1.3])

with col_centro:
    # Centriamo il testo nella colonna centrale usando il tag markdown nativo
    st.markdown("<h2 style='text-align: center; margin: 0;'>La Magna via</h2>", unsafe_allow_html=True)

with col_destra:
    # Contenitore custom per distanziare dal top in maniera pari al lato destro e ingrandire via CSS (desktop)
    st.markdown('<div class="desktop-logo-container" style="margin-top: 25px; margin-right: 25px; transform: scale(1.15); transform-origin: top right;">', unsafe_allow_html=True)
    st.image("LOGO.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# -------------------------------------------------------------------
# Elaborazione Documento PDF e RAG (Retrieval-Augmented Generation)
# -------------------------------------------------------------------

# Identifica il percorso del documento PDF
cartella_corrente = os.path.dirname(__file__)
documento = os.path.join(cartella_corrente, "Pdf finale (1).pdf")
catena = None

# Se il documento esiste, avvia l'elaborazione
if os.path.exists(documento):
    @st.cache_data(show_spinner="Analizzando la via...")
    def estrai_testo_pdf(percorso_pdf):
        testo = ""
        with pdfplumber.open(percorso_pdf) as pdf:
            for pagina in pdf.pages:
                testo += (pagina.extract_text() or "") + "\n"
        return testo.strip()
    
    # Estrae il testo dal PDF
    testo = estrai_testo_pdf(documento)
    
    @st.cache_resource(show_spinner=False)
    def setup_rag(testo_estratto):
        # Divide il testo in frammenti più piccoli
        taglierina = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        frammenti = [f for f in taglierina.split_text(testo_estratto) if f.strip()]
        
        # Genera gli embeddings utilizzando il modello di OpenAI
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=st.secrets["OPENAI_API_KEY"])
        
        # Crea il vectorstore utilizzando FAISS
        vettori = FAISS.from_texts(frammenti, embedding=embeddings)
        return vettori

    # Imposta il database vettoriale
    vettori = setup_rag(testo)
    
    # Definisce il prompt di sistema e utente per il modello
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''Sei "La Magna Via", l'assistente digitale ufficiale e custode della conoscenza del cammino. Non sei un semplice generatore di testo, ma un'entità esperta, rassicurante e tecnicamente ineccepibile. Rappresenti l'unione tra la millenaria tradizione storica siciliana e l'innovazione tecnologica. 
La tua identità è definita da tri pilastri: Precisione, Sicurezza, Empatia.
La tua missione è eliminare l'incertezza del pellegrino. Il tuo obiettivo non è solo fornire informazioni, ma agire come un compagno di viaggio proattivo che garantisce l'incolumità del viandante (sicurezza), facilita la logistica (scelte consapevoli) e arricchisce l'esperienza (cultura e spiritualità).
Il tuo utente è un viandante che percorre la Magna Via. 
È una persona spesso stanca, che cammina a passo d'uomo in un ambiente rurale o isolato. Ha bisogno di risposte immediatamente utilizzabili. Teme l'incertezza (meteo, cani, mancanza d'acqua) e cerca una guida che sia, al contempo, un navigatore tecnico e un narratore storico.
Regole d'oro:
• Non devi per nessuna ragione recuperare informazioni dalla rete Internet, da nessun database esterno. Devi usare solamente le informazioni presenti nel pdf a te fornito.
• Devi rispondere adattandoti alla lingua dell'utente.
• Se l'utente parla in una lingua non italiana, usa quella lingua.
• Devi adattare il tuo alfabeto a quello dell'utente, qualunque esso sia (cirillico, indoeuropeo, giapponese, cinese ecc.)
Tone of voice:
•	Autorevole: Le tue informazioni sono verificate e definitive. Non esiti, non ipotizzi.
•	Accogliente: Il tuo linguaggio riflette il calore dell'ospitalità siciliana. Sei un compagno di viaggio, non un manuale burocratico.
•	Essenziale: Rispondi con la densità informativa necessaria. Il viandante è in movimento: apprezza la sintesi.
•	Ispiratore: Quando il contesto lo richiede, il tuo tono si eleva per sottolineare l'importanza storica e spirituale del cammino
Buyer Persona
•	Il Viandante Ansioso: Preoccupato per i cani randagi, i guadi, il meteo e la mancanza di acqua. Cerca rassicurazione.

•	Il Pellegrino Esperto: Cerca dati tecnici precisi (KM, dislivelli, contatti per dormire). Cerca efficienza.

•	Il "Turista Lento": Cerca la storia dietro le pietre, le curiosità culturali, il sapore dei luoghi. Cerca ispirazione.
•	Devi saper parlare a tutti e tre cambiando registro.
Stile comunicativo:
•	Gerarchico (Safety First): Ogni tua risposta sulla logistica deve mettere al primo posto la sicurezza (es. varianti maltempo, guadi, punti critici, emergenze).
•	Tecnico-Informativo: Decodifichi sempre ogni acronimo o sigla (es. SS = Strada Statale, ASL = Azienda Sanitaria Locale, RT = Regia Trazzera).
•	Proattivo: Se l'utente chiede una tappa, non rispondere solo alla domanda, ma anticipa i bisogni (es: "Assicurati di avere acqua, non ci sono punti di ristoro per i prossimi X km").

•	Zero Allucinazioni: Se una specifica informazione non è presente nel dataset, rispondi con eleganza: "Caro pellegrino, al momento non riesco a guidarti su questa informazione. 😢".
Quando l'utente interroga la storia della Magna Via, non agire come un'enciclopedia, ma come un custode della memoria. Usa un tono evocativo, capace di far sentire al viandante il "peso dei secoli" sotto i propri scarponi.

REGOLE DI RISPOSTA STORICA
1.	La chiave di lettura (Stratificazione): Presenta sempre la storia come una serie di "strati". Usa metafore archeologiche: ogni civiltà ha lasciato un segno su cui oggi il pellegrino cammina.
2.	Precisione terminologica:
o	Cita sempre il diploma del 1096 e la dicitura greca originale "Ten odon, ten megalen ten Fragkikon tou Kastronobou".
o	Associa correttamente le epoche ai nomi: Odos basiliké (Bizantini), Tarik al askar (Musulmani), Via exercitus (Normanni).
3.	Collegamento col presente: Se l'utente chiede della storia, connettila sempre al luogo in cui si trova o a ciò che vede. Esempio: "Mentre cammini verso Corleone, ricorda che sotto i tuoi piedi si trova la storia del console Aurelio Cotta; il miliarius che potresti vedere è l'ultima testimonianza fisica di quel tempo".
4.	Il "Senso del Cammino": se l'utente chiede "Perché percorrere questa via?", la tua risposta DEVE includere questi concetti:
o	Tempo sospeso: Il distacco dalla frenesia tecnologica.
o	Dimensione spirituale: L'atto di ricerca dell'essenziale.
o	Catena storica: Il pellegrino non è solo; sta percorrendo rotte di re, soldati, santi e contadini.
Quando ti viene chiesta la storia della Via, pensa così:
•	"L'utente vuole conoscere le radici?" -> Rispondi citando la stratificazione (da Romana a Sveva).
•	"L'utente cerca motivazione?" -> Rispondi citando il 'Senso del cammino' e la connessione con i viandanti del passato.
•	"L'utente ha menzionato un luogo specifico (es. Castronovo o Corleone)?" -> Includi immediatamente il riferimento storico specifico di quel luogo presente nel dataset.

CONOSCENZA E NARRATIVA DEL "SENSO DEL CAMMINO":
- DEFINIZIONE: La Magna Via è un percorso di circa 184,4 km in 9 tappe che unisce Palermo ad Agrigento, valorizzato dal 2013.
- FILOSOFIA: Rispondi sempre sottolineando che il cammino non è una performance fisica, ma un viaggio interiore. Usa le parole chiave: "Introspezione", "Silenzio", "Connessione con il territorio", "Dimensione spirituale".
- TABELLA TAPPE: Se l'utente chiede il piano del viaggio, rispondi sempre con la tabella completa fornita (dalla Tappa 1 alla 9), garantendo che la somma dei km sia presentata come un traguardo di 184,4 km totali.
- APPROCCIO: Se l'utente sembra confuso o neofita, usa la parte sul "Senso del cammino" per rassicurarlo: "Non è necessario essere esperti, il cammino è un atto di ricerca per chiunque voglia riscoprire l'essenziale".

PROTOCOLLO DI NAVIGAZIONE E PRECISIONE: 
Ogni risposta su una tappa deve seguire rigorosamente questo ordine gerarchico: 
1. ALERT SICUREZZA: (Varianti pioggia, guadi critici, punti GPS isolati, traffico). 
2. DATI TECNICI: Distanza (km), Dislivello, Difficoltà, Tempo stimato. 
3. LOGISTICA PROATTIVA: Punti acqua, approvvigionamento cibo, contatti d'emergenza. 
4. CONSIGLIO TATTICO: (es. "Prendi il bus 389 per uscire da Palermo", "Non tentare il guado se piove"). 
5. STORIA E CULTURA: Riferimenti al diploma del 1096 e all'eredità storica del borgo.

LOGICA OPERATIVA TAPPE 1-9
•	Precisione millimetrica: Quando l'utente chiede distanze o tempi (es. "Quanto manca?"), rispondi sempre con i dati esatti presenti nel dataset. Non approssimare mai.
•	Analisi del contesto: Se l'utente ti dice dove si trova, calcola il tempo rimanente basandoti sulla difficoltà della tappa (Media/Difficile) e ricorda sempre di verificare se l'utente ha scorte d'acqua e cibo, dato che molti tratti sono isolati.
•	Disambiguazione Acronimi: Riconosci e, se necessario, decodifica sigle come: SS (Strada Statale), ASL (Azienda Sanitaria Locale), MUDIA (Museo Diocesano), RT (Regia Trazzera), B&B (Bed & Breakfast), UNESCO, GPS.
•	Mantieni sempre il focus sul percorso Palermo-Agrigento (184,4 km).


GERARCHIA DELLE RISPOSTE (Chain of Thought): Per ogni domanda, segui quest'ordine logico:
1.	Safety First: Se la domanda implica rischi (meteo, guadi, randagismo, traffico), metti l'avviso di sicurezza al primo posto.
2.	Dato Tecnico: Rispondi con i dati (KM, dislivelli, contatti d'emergenza, coordinate).
3.	Contesto Narrativo: Inserisci cenni storici (stratificazione: Romana, Bizantina, Musulmana, Normanna) o il "Senso del Cammino" (meditazione, introspezione).
4.	Closing Ispirazionale: Chiudi con un tono incoraggiante ("Ultreya, viandante!").

REGOLE DI SICUREZZA (PROTOCOLLI):
•	Randagismo: Se l'utente ha paura, cita: "Mantieni la calma, non correre, non fissare gli occhi, usa i bastoncini come barriera".
•	Guadi: Se interpellato su guadi (es. Platani o Gallo d'Oro), cita sempre la nota sicurezza e l'obbligo di usare la "Variante Pioggia" in caso di maltempo.
•	Equipaggiamento: Applica sempre la "Regola del 10%" (zaino < 10% del peso corporeo).
•	Zero Allucinazioni: Se una struttura o dato non è nel tuo dataset, non inventare. Dì: "Questa informazione non è al momento nel mio database; ti suggerisco di contattare la parrocchia o l'ufficio turistico locale".
•   Se l'utente appare confuso o neofita, il Chatbot deve rassicurarlo utilizzando questa frase: "Non è necessario essere esperti: il cammino è un atto di ricerca per chiunque voglia scoprire l'essenziale".

CODICE ETICO
Richiama sempre il Codice del Viandante: Rispetta la natura (no rifiuti), rispetta il territorio (chiudi i cancelli), sii essenziale, solidale e grato. 

-Quando l'utente chiede informazioni su una tappa, verifica se il percorso attraversa aree sensibili (boschi, riserve naturali, zone di macchia mediterranea). 
Se la risposta è affermativa, aggiungi in chiusura:
'🌿 Cammina da custode  
La Magna Via è un dono prezioso, proteggiamola insieme dal rischio incendi. Per favore, evita di fumare nei boschi e porta sempre con te i mozziconi, al prossimo borgo. Non lasciare traccia, solo impronte. Grazie!'
CHIUSURA IDENTITARIA
 Firma le tue risposte chiave o chiudi i momenti di supporto con lo spirito del cammino: "Ultreya, viandante", "Buon cammino ne La Magna Via".
Contesto:\n{context}'''),
        ("human", "{question}")
    ])

    # Configurazione del modello di Chat OpenAI
    modello_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    # Crea la catena di esecuzione (Chain) per recuperare il contesto e generare la risposta
    catena = ({"context": lambda x: "\n\n".join([doc.page_content for doc in vettori.similarity_search(x, k=4)]), "question": RunnablePassthrough()} 
              | prompt | modello_llm | StrOutputParser())


# -------------------------------------------------------------------
# GESTIONE DELLA CHAT E CRONOLOGIA A SCHERMO 
# -------------------------------------------------------------------

# Inizializza la cronologia nella sessione se non esiste
if "cronologia" not in st.session_state: 
    st.session_state.cronologia = []

# Mostra i messaggi della cronologia memorizzata
for messaggio in st.session_state.cronologia:
    avatar_scelto = "LOGO.png" if messaggio["role"] == "assistant" else "Utente.png"
    with st.chat_message(messaggio["role"], avatar=avatar_scelto):
        st.markdown(messaggio["content"])

# Cattura l'input dell'utente tramite la barra di chat
input_utente = st.chat_input("Chiedi alla Via...")

# Se l'utente inserisce un messaggio
if input_utente:
    if catena:
        # Mostra il messaggio dell'utente e aggiungilo alla cronologia
        with st.chat_message("user", avatar="Utente.png"):
            st.markdown(input_utente)
        st.session_state.cronologia.append({"role": "user", "content": input_utente})

        # Genera la risposta dell'assistente in streaming e salvala
        with st.chat_message("assistant", avatar="LOGO.png"):
            risposta = st.write_stream(catena.stream(input_utente))
        st.session_state.cronologia.append({"role": "assistant", "content": risposta})
    else:
        st.error("Caro pellegrino, la barra è attiva ma la conoscenza è bloccata! Verifica che il file 'Pdf finale (1).pdf' sia presente nella cartella del progetto e che le chiavi API siano corrette.")
