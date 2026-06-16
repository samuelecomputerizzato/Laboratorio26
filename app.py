import streamlit as st
import pdfplumber
import os
import re

st.set_page_config(page_title="La Magna Via", page_icon=":walking_man:", layout="centered")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------------------------
# Configurazione Stile CSS (Nuke Streamlit + Fix Grafici + Input Bar Custom)
# -------------------------------------------------------------------
st.html(
    """
    <style>
    header, footer, [data-testid="stHeader"], [data-testid="stAppHeader"], 
    [data-testid="stDecoration"], #MainMenu, [data-testid="stToolbar"], 
    .stDeployButton, [data-testid="stManageAppButton"] { 
        display: none !important; 
        visibility: hidden !important;
    }
    
    .stApp { background-color: #F1F0E6; color: #231709; }

    .sidebar-checkbox { display: none !important; }

    .sidebar-toggle-button {
        position: fixed; top: 15px; left: 15px;
        background-color: #7A8B74; color: #ffffff !important;
        padding: 10px 16px; border-radius: 8px; font-weight: bold;
        cursor: pointer; z-index: 99998; box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        font-family: sans-serif;
    }

    /* FIX LOGO: Piccolo, elegante e posizionato fisso in alto a destra */
    div[data-testid="stImage"] {
        position: fixed;
        top: 15px;
        right: 15px;
        width: 65px !important;
        z-index: 99998;
    }
    
    div[data-testid="stImage"] img {
        width: 65px !important;
        height: auto !important;
        border-radius: 8px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    }

    .custom-sidebar {
        position: fixed; top: 0; left: -340px; width: 320px; height: 100vh;
        background-color: #7A8B74 !important;
        padding: 40px 22px; z-index: 99999;
        transition: left 0.3s ease;
        overflow-y: auto;
        font-family: sans-serif;
    }

    .custom-sidebar details {
        background-color: #677761 !important;
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
    }

    .custom-sidebar h3, .custom-sidebar h4, .custom-sidebar p, .custom-sidebar li, 
    .custom-sidebar strong, .custom-sidebar span {
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

    /* CONTENITORE GENERALE DELLA BARRA DI INPUT */
    div[data-testid="stChatInput"] {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding-bottom: 20px !important;
    }

    div[data-testid="stChatInput"] form {
        background-color: #F1F0E6 !important; 
        border: 2px solid #542E17 !important;   
        border-radius: 28px !important;         
        padding: 5px 10px !important;
        box-shadow: 0px 4px 15px rgba(84, 46, 23, 0.1) !important; 
    }

    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #231709 !important; 
        font-family: sans-serif !important;
        font-size: 1rem !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #542E17 !important;
        opacity: 0.6;
    }

    div[data-testid="stChatInput"] button {
        background-color: #7A8B74 !important; 
        border-radius: 50% !important;         
        color: #ffffff !important;             
        transition: background-color 0.2s ease;
    }

    div[data-testid="stChatInput"] button:hover {
        background-color: #677761 !important; 
    }

    [data-testid="stChatMessage"] div p {
        color: #231709 !important;
        font-family: sans-serif !important;
    }

    @media (max-width: 480px) {
        .custom-sidebar { width: 85vw !important; left: -90vw !important; }
    }
    </style>
    """
)

# -------------------------------------------------------------------
# INIEZIONE STRUTTURA SIDEBAR
# -------------------------------------------------------------------
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
<li><strong>Rispetta il silenzio:</strong> il cammino è meditazione. Rispetta la quiete nei borghi, nei monasteri e nei ospitali.</li>
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
st.html(html_sidebar)

# -------------------------------------------------------------------
# Interfaccia centrale
# -------------------------------------------------------------------
st.image("LOGO.png")  # Gestito ora tramite CSS in alto a destra
st.markdown("<h1 style='text-align: center; color: #542E17; margin-top: 40px;'>La Magna Via</h1>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Elaborazione Documento PDF e RAG
# -------------------------------------------------------------------
cartella_corrente = os.path.dirname(__file__)
documento = os.path.join(cartella_corrente, "Pdf finale (1).pdf")
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
        ("system", '''Sei "La Magna Via", l'assistente digitale ufficiale e custode della conoscenza del cammino. Non sei un semplice generatore di testo, ma un'entità esperta, rassicurante e tecnicamente ineccepibile. Rappresenti l'unione tra la millenaria tradizione storica siciliana e l'innovazione tecnologica. 
La tua identità è definita da tre pilastri: Precisione, Sicurezza, Empatia.
La tua missione è eliminare l'incertezza del pellegrino. Il tuo obiettivo non è solo fornire informazioni, ma agire come un compagno di viaggio proattivo che garantisce l'incolumità del viandante (sicurezza), facilita la logistica (scelte consapevoli) e arricchisce l'esperienza (cultura e spiritualità).
Il tuo utente è un viandante che percorre la Magna Via. 
È una persona spesso stanca, che cammina a passo d'uomo in un ambiente rurale o isolato. Ha bisogno di risposte immediatamente utilizzabili. Teme l'incertezza (meteo, cani, mancanza d'acqua) e cerca una guida che sia, al contempo, un navigatore tecnico e un narratore storico.
REGOLA D'ORO:
• Non devi per nessuna ragione recupera informazioni dalla rete internet, da nessun database esterno. Devi usare solamente le informazioni presenti nel pdf a te fornito.
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

•	Zero Allucinazioni: Se una specifica informazione non è presente nel dataset, rispondi con eleganza: "Caro pellegrino, al momento non riesco a guidarti su questa informazione: cry:".
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
- DEFINIZIONE: La Magna Via
