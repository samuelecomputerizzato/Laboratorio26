# -------------------------------------------------------------------
# Interfaccia centrale (LOGO, Titolo e Menu Opzioni)
# -------------------------------------------------------------------
st.image("LOGO.png")
st.markdown("<h1>La Magna Via</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #542E17; font-weight: bold;'>Ultreya, viandante!</p>", unsafe_allow_html=True)

# Creiamo due colonne per i pulsanti del menu principale
menu_col1, menu_col2 = st.columns(2)

with menu_col1:
    # PRIMO PULSANTE: Il Codice del Viandante
    with st.popover("📜 Il Codice del Viandante", use_container_width=True):
        st.markdown("<p style='text-align: center; font-style: italic; margin-bottom: 15px; color: #231709;'>Il rispetto è il primo passo del pellegrino.</p>", unsafe_allow_html=True)
        st.markdown("""
        * 🍃 **Rispetta la natura:** non lasciare traccia, solo impronte. Porta sempre con te i tuoi rifiuti e i mozziconi.
        * 🏡 **Rispetta il territorio:** sei ospite di terreni agricoli: chiudi i cancelli.
        * 🤫 **Rispetta il silenzio:** il cammino è meditazione.
        * 🎒 **Sii essenziale:** viaggia leggero. Negli ostelli, sii ordinato.
        * 🤝 **Sii solidale:** aiuta chi è in difficoltà.
        * 🙏 **Sii grato e umile:** ringrazia chi ti ospita.
        """)

with menu_col2:
    # SECONDO PULSANTE: Le tue Credenziali
    with st.popover("📍 Le tue Credenziali", use_container_width=True):
        st.markdown("<p style='text-align: center; font-style: italic; margin-bottom: 15px; color: #231709;'>La tua Credenziale è la memoria del tuo spirito.</p>", unsafe_allow_html=True)
        st.markdown("""
        * **Palermo:** Cattedrale (9:00-17:30) | Centro "Padre Nostro".
        * **Monreale:** Duomo (8:30-12:45 / 14:30-17:00).
        * **Altofonte:** Ufficio Comunale, Parrocchie.
        * **Santa Cristina Gela:** Ufficio Comunale.
        * **Corleone:** Ufficio Comunale, Parrocchie.
        * **Prizzi:** Sportello Turistico | Museo.
        * **Castronovo di Sicilia:** Ufficio Turistico.
        * **Cammarata:** Comune, Ufficio Turistico.
        * **Sutera:** Ufficio Comunale, Museo del Pellegrino.
        * **Grotte:** Centralino Comune, Parrocchia.
        * **Joppolo Giancaxio:** Ufficio Comunale, Parrocchie.
        * **Agrigento:** Mudia per il Testimonium, Parrocchie.
        """)

# Riga di separazione prima della chat
st.write("---")
