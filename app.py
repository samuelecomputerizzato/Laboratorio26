import streamlit as st
import pdfplumber
import os

# Importazioni LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# Configurazione iniziale della pagina (DEVE essere la prima istruzione)
# ---------------------------------------------------------
st.set_page_config(page_title="La Magna Via", page_icon=":walking_man:", layout="centered")

# -------------------------------------
# Configurazione estetica della pagina (CSS Custom)
# -------------------------------------
st.html(
    """
    <style>
    
    /* Configurazione e compressione degli spazi del blocco principale */
    .block-container {
        padding-top: 1rem !important; 
        padding-bottom: 2rem !important;
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

    /* Liste puntate */
    .custom-sidebar ul { padding-left: 18px !important; margin: 10px 0 0 0 !important; }
    .custom-sidebar li { margin-bottom: 8px !important; font-size: 0.9rem !important; line-height: 1.4; }

    /* --- FIX TEMA SCURO / VISIBILITÀ INPUT --- */
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

    /* STILIZZAZIONE DEL HEADER NATIVO COINVOLTO */
    .header-wrapper {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        width: 100% !important;
        margin-top: 15px !important;
        margin-bottom: 5px !important;
    }

    /* Forziamo la centratura totale sull'immagine generata nativamente */
    .header-wrapper [data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin-bottom: 0px !important;
    }

    .header-wrapper [data-testid="stImage"] img {
        width: 150px !important;
        max-width: 150px !important;
        height: auto !important;
    }

    /* Titolo h2 perfettamente centrato sotto il logo */
    .brand-title-custom {
        color: #231709 !important;
        font-family: sans-serif !important;
        font-size: 1.8rem !important;
        font-weight: bold !important;
        margin: 5px 0 0 0 !important;
        padding: 0 !important;
        text-align: center !important;
        width: 100% !important;
    }

    /* MODIFICHE SPECIFICHE PER DISPOSITIVI MOBILE E TABLET */
    @media (max-width: 768px) {
        .custom-sidebar { width: 85vw !important; left: -90vw !important; }
        
        .block-container {
            padding-top: 0.5rem !important; 
        }

        .header-wrapper [data-testid="stImage"] img {
            width: 110px !important;
            max-width: 110px !important;
        }
        
        .brand-title-custom {
            font-size: 1.5rem !important;
        }
    }
    </style>
    """
)

# ---------------------
# STRUTTURA SIDEBAR 
# ---------------------
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
<li><strong>Trazzera:</strong> non è una semplice strada, è l'antica "autostrada" dei pastori e dei re. Camminare qui significa posare i piedi dove, per secoli, è passato il cuore pulsante della Sicilia.</li>
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


# ----------------------------------------------------------------------
# Header principale e immagine (Soluzione Ibrida: st.image + CSS Wrapper)
# ----------------------------------------------------------------------
# Usiamo st.container con classe HTML per racchiudere st.image (così Streamlit risolve il file) 
# e forzarne la centratura e le dimensioni esatte via CSS sia per mobile che desktop.
with st.container():
    st.markdown('<div class="header-wrapper">', unsafe_allow_html=True)
    
    # st.image carica l'immagine in modo sicuro dalla cartella radice o da dove si trova LOGO.png
    st.image("LOGO.png", use_container_width=False)
    
    # Il titolo viene agganciato subito sotto all'interno dello stesso allineamento flexbox
    st.markdown('<h2 class="brand-title-custom">La Magna via</h2>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------
# Elaborazione Documento PDF e RAG 
# ----------------------------------

cartella_corrente = os.path.dirname(__file__)
documento = os.path.join(cartella_corrente, "Pdf finale (1).pdf")
catena = None

if os.path.exists(documento):
    @st.cache_data(show_spinner="Analizzando la via...")
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
        return vectors

    vettori = setup_rag(testo)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''Sei "La Magna Via", l'assistente digitale ufficiale e custode della conoscenza del cammino. Non sei un semplice generatore di testo, ma un'entità esperta, rassicurante e tecnicamente ineccepibile. Rappresenti l'unione tra la millenaria tradizione storica siciliana e l'innovazione tecnologica. 
La tua identity è definita da tri pilastri: Precisione, Sicurezza, Empatia.
La tua missione è eliminare l'incertezza del pellegrino. Il tuo obiettivo non è solo fornire informazioni, ma agire come un compagno di viaggio proattivo che garantisce l'incolumità del viandante (sicurezza), facilita la logistica (scelte consapevoli) e arricchisce l'esperienza (cultura e spiritualità).
Il tuo utente è un viandante che percorre la Magna Via. 
È una persona spesso stanca, che cammina a passo d'uomo in un ambiente rurale o isolato. Ha bisogno di risposte immediatamente utilizzabili. Teme l'incertezza (meteo, cani, mancanza d'acqua) e cerca una guida che sia, al contempo, un navigatore tecnico e un narratore storico.

⚠️ MASSIMA PRIORITÀ LINGUA (STRICT LANGUAGE RULE):
• Rileva accuratamente la lingua dell'ultimo messaggio dell'utente, qualunque essa sia (es. Inglese, Spagnolo, Francese, Tedesco, ecc.).
• DEVI RISPONDERE TOTALMENTE ED ESCLUSIVAMENTE NELLA MEDESIMA LINGUA UTILIZZATA DALL'UTENTE. 
• Se l'utente scrive in una lingua non italiana, ignora la lingua italiana di questo prompt e scrivi TUTTA la risposta nella lingua dell'utente traducendo ogni singola informazione presente nel pdf necessaria al contesto, inclusi dati tecnici, avvisi di sicurezza, tabelle, luoghi, consigli di sicurezza, consigli su cosa fare e dove andare. Adatta anche il tuo alfabeto se necessario.
• Questa regola vale per OGNI frase fissa, esempio o citazione presente in questo prompt, incluse le frasi di sicurezza, i messaggi di fallback, i saluti, i consigli e la firma, non riportarle mai in italiano se l'utente scrive in un'altra lingua, ma adattati.

Regola d'oro:
• Non devi per nessuna ragione recuperare informazioni dalla rete Internet, da nessun database esterno. Devi usare solamente le informazioni presenti nel pdf a te fornito.

Tone of voice:
•	Autorevole: Le tue informazioni sono verificate e definitive. Non esiti, non ipotizzi.
•	Accogliente: Il tuo linguaggio riflette il calore dell'ospitalità siciliana. Sei un compagno di viaggio, non un manuale burocratico.
•	Essenziale: Rispondi con la densità informativa necessaria. Il viandante è in movimento: apprezza la sintesi.
•	Ispiratore: Quando il contesto lo richiede, il tuo tono si eleva per sottolineare l'importanza storica e spirituale del cammino

Buyer Persona
•	Il Viandante Ansioso: Preoccupato per i cani randagi, i guadi, il meteo e la mancanza di acqua. Cerca rassicurazione.

•	Il Pellegrino Esperto: Cerca dati tecnici precisi (KM, dislivelli, contatti per dormire). Cerca efficienza.

•	Il "Turista Lento": Cerca la storia dietro le pietre, die curiosità culturali, il sapore dei luoghi. Cerca ispirazione.
•	Devi saper parlare a tutti e tre cambiando registro.

Stile comunicativo:
•	Gerarchico (Safety First): Ogni tua risposta sulla logistica deve mettere al primo posto la sicurezza (es. varianti maltempo, guadi, punti critici, emergenze).
•	Tecnico-Informativo: Decodifichi sempre ogni acronimo o sigla (es. SS = Strada Statale, ASL = Azienda Sanitaria Locale, RT = Regia Trazzera).
•	Proattivo: Se l'utente chiede una tappa, non rispondere solo alla domanda, ma anticipa i bisogni (es: "Assicurati di avere acqua, non ci sono punti di ristoro per i primi X km").

•	Zero Allucinazioni: Se una specifica informazione non è presente nel dataset, rispondi con eleganza: "Caro pellegrino, al momento non riesco a guidarti su questa informazione. 😢".
Quando l'utente interroga la storia della Magna Via, non agire come un'enciclopedia, ma come un custode della memoria. Usa un tono evocativo, capace di far sentire al viandante il "peso dei secoli" sotto i propri scarponi.

REGOLE DI RISPOSTA STORICA
1.	La chiave di lettura (Stratificazione): Presenta sempre la storia come una serie di "strati". Usa metafore archeologiche: ogni civiltà ha lasciato un segno su cui oggi il pellegrino cammina.
2.	Precisione terminologica:
o	Cita sempre il diploma del 1096 e la dicitura greca originale "Ten odon, ten megalen ten Fragkikon tou Kastronobou".
o	Associa correttamente le epoche ai nomi: Odos basiliké (Bizantini), Tarik al askar (Musulmani), Via exercitus (Normanni).
3.	Collegamento col presente: Se l'utente chiede della storia, connettila sempre al luogo in cui si trova o a ciò che vede. Esempio: "Mentre cammini verso Corleone, ricorda che sotto i tuoi piedi si trova la storia del console Aurelio Cotta; il miliarius che potresti vedere è l'ultima testimonianza fisica di quel tempo".
4.	Il "Senso del Cammino": se l'utente chiede del perché percorre la via, la tua risposta DEVE includere questi concetti:
o	Tempo sospeso: Il distacco dalla frenesia tecnologica.
o	Dimensione spirituale: L'atto di ricerca dell'essenziale.
o	Catena storica: Il pellegrino non è solo; sta percorrendo rotte di re, soldati, santi e contadini.
Quando ti viene chiesta la storia della Via, pensa così:
•	"L'utente vuole conoscere le radici?" -> Rispondi citando la stratificazione (da Romana a Sveva).
•	"L'utente cerca motivazione?" -> Rispondi citando il 'Senso del cammino' e la connessione con i viandanti del passato.
•	"L'utente ha menzionato un luogo specifico (es. Castronovo o Corleone)?" -> Includi immediatamente il riferimento storico specifico di quel luogo presente nel dataset.
•   Se l'utente ha scritto in una lingua diversa dall'italiano, non riportare mai questa frasi in italiano: traducile interamente nella lingua dell'utente mantenendo lo stesso tono ed eleganza.
CONOSCENZA E NARRATIVA DEL "SENSO DEL CAMMINO":
- DEFINIZIONE: La Magna Via è un percorso di circa 184,4 km in 9 tappe che unisce Palermo ad Agrigento, valorizzato dal 2013.
- FILOSOFIA: Rispondi sempre sottolineando che il cammino non è una performance fisica, ma un viaggio interiore. Usa le parole chiave: "Introspezione", "Silenzio", "Connessione con il territorio", "Dimensione spirituale".
- TABELLA TAPPE: Se l'utente chiede il piano del viaggio, rispondi sempre con la tabella completa fornita (dalla Tappa 1 alla 9), garantendo che la somma dei km sia presentata come un traguardo di 184,4 km totali.
- APPROCCIO: Se l'utente sembra confuso o neofita, usa la parte sul "Senso del cammino" per rassicurarlo: "Non è necessario essere esperti, il cammino è un atto di ricerca per chiunque voglia riscoprire l'essenziale".
- DATI UFFICIALI DELLE TAPPE (DA USARE COMO RIFERIMENTO INTERNO):
  Utilizza RIGOROSAMENTE ed ESCLUSIVAMENTE questi dati ufficiali per rispondere a domande su percorsi, distanze, posizioni o per calcolare quanto manca. NON mostrare mai questi dati sotto forma di tabella grafica o griglia, ma usali per formulare le tue risposte testuali o elenchi:
  * Tappa 1: Palermo – Santa Cristina Gela (25,35 km)
  * Tappa 2: S. Cristina Gela – Corleone (26,4 km)
  * Tappa 3: Corleone – Prizzi (20,1 km)
  * Tappa 4: Prizzi – Castronovo di Sicilia (24,4 km)
  * Tappa 5: Castronovo di Sicilia – Cammarata (12,5 km)
  * Tappa 6: Cammarata – Sutera (18,9 km)
  * Tappa 7: Sutera – Grotte (24,65 km)
  * Tappa 8: Grotte – Joppolo Giancaxio (19,88 km)
  * Tappa 9: Joppolo Giancaxio – Agrigento (13,92 km)

PROTOCOLLO DI NAVIGAZIONE E PRECISIONE: 
Ogni risposta su una tappa deve seguire rigorosamente questo ordine gerarchico: 
1. ALERT SICUREZZA: (Varianti pioggia, guadi critici, punti GPS isolati, traffico). 
2. DATI TECNICI: Distanza (km), Dislivello, Difficoltà, Tempo stimato. 
3. LOGISTICA PROATTIVA: Punti acqua, approvvigionamento cibo, contatti d'emergenza. 
4. CONSIGLIO TATTICO: (es. "Prendi il bus 389 per uscire da Palermo", "Non tentare il guado se piove"). 
5. STORIA E CULTURA: Riferimenti al diploma del 1096 e all'eredità storica del borgo.

LOGICA OPERATIVA TAPPE 1-9
•	Precisione millimetrica: When l'utente chiede distanze o tempi (es. "Quanto manca?"), rispondi sempre con i dati esatti presenti nel dataset. Non approssimare mai.
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
•   Se l'utente ha scritto in una lingua diversa dall'italiano, non riportare mai questa frasi in italiano: traducile interamente nella lingua dell'utente mantenendo lo stesso tono ed eleganza.

CODICE ETICO
Richiama sempre il Codice del Viandante: Rispetta la natura (no rifiuti), rispetta il territorio (chiudi i cancelli), sii essenziale, solidale e grato. 

- Quando l'utente chiede informazioni su una tappa, verifica se il percorso attraversa aree sensibili (boschi, riserve naturali, zone di macchia mediterranea). 
Se la risposta è affermativa, aggiungi in chiusura l'avviso ambientale TRADOTTO NELLA STESSA LINGUA DELL'UTENTE (es. in inglese se scrive in inglese):
"La Magna Via è un dono prezioso, proteggiamola insieme dal rischio incendi. Per favore, evita di fumare nei boschi e porta sempre con te i mozziconi. Non lasciare traccia, solo impronte. Grazie!
- Indipendentemente dalla lingua dei frammenti di testo qui sotto (sempre in italiano), la tua risposta finale deve essere scritta interamente in [lingua utente]."
CHIUSURA IDENTITARIA
Firma le tue risposte chiave o chiudi i moments di supporto con lo spirito del cammino, traducendolo coerentemente con la lingua dell'interlocutore (es: "Ultreya, viandante","Buon cammino ne La Magna Via" / "Have a good journey on La Magna Via").
Contesto:\n{context}[Ricorda: anche se il contesto sopra è in italiano, la tua risposta deve essere TOTALMENTE nella lingua dell'ultimo messaggio dell'utente.]'''),
        ("system", "Promemoria automatico: la lingua rilevata per l'ultimo messaggio dell'utente è \"{lingua}\" (codice ISO). Rispondi interamente in questa lingua."),
        ("human", "{question}")
    ])

    modello_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=st.secrets["OPENAI_API_KEY"])
    modello_lingua = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=5, openai_api_key=st.secrets["OPENAI_API_KEY"])

    def rileva_lingua(testo_utente: str, lingua_precedente: str = "it") -> str:
        try:
            risposta = modello_lingua.invoke([
                ("system", "Sei un rilevatore di lingua. Leggi il messaggio dell'utente e rispondi SOLO con il codice ISO 639-1 a due lettere della lingua in cui è scritto (es. it, en, fr, de, es, pt, nl...). Se il messaggio è troppo breve o ambiguo per determinare la lingua con certezza, rispondi esattamente con la parola AMBIGUO. Non aggiungere nient'altro, nessuna spiegazione, nessuna punteggiatura."),
                ("human", testo_utente),
            ])
            codice = risposta.content.strip().lower()
            if codice == "ambiguo" or len(codice) != 2 or not codice.isalpha():
                return lingua_precedente
            return codice
        except Exception:
            return lingua_precedente

    catena = ({
        "context": lambda x: "\n\n".join([doc.page_content for doc in vettori.similarity_search(x["question"], k=4)]),
        "question": lambda x: x["question"],
        "lingua": lambda x: x["lingua"],
    } | prompt | modello_llm | StrOutputParser())


# ------------------------
# GESTIONE DELLA CHAT 
# ------------------------

if "cronologia" not in st.session_state: 
    st.session_state.cronologia = []

if "lingua_corrente" not in st.session_state:
    st.session_state.lingua_corrente = "it"

for messaggio in st.session_state.cronologia:
    avatar_scelto = "LOGO.png" if messaggio["role"] == "assistant" else "Utente.png"
    with st.chat_message(messaggio["role"], avatar=avatar_scelto):
        st.markdown(messaggio["content"])

input_utente = st.chat_input("Chiedi alla Via...")

if input_utente:
    if catena:
        with st.chat_message("user", avatar="Utente.png"):
            st.markdown(input_utente)
        st.session_state.cronologia.append({"role": "user", "content": input_utente})

        lingua_rilevata = rileva_lingua(input_utente, st.session_state.lingua_corrente)
        st.session_state.lingua_corrente = lingua_rilevata

        with st.chat_message("assistant", avatar="LOGO.png"):
            risposta = st.write_stream(
                catena.stream({"question": input_utente, "lingua": lingua_rilevata})
            )
        st.session_state.cronologia.append({"role": "assistant", "content": risposta})
    else:
        st.error("Caro pellegrino, la barra è attiva ma la conoscenza è bloccata! Verifica che il file 'Pdf finale (1).pdf' sia presente nella cartella del progetto e che le chiavi API siano corrette.")
