import streamlit as st
import openpyxl
import plotly.express as px
import pandas as pd
import os

st.set_page_config(page_title="Analisi dei dati",
                   page_icon=":bar_chart:",
                   layout="wide")
st.title(":bar_chart: Esempio di dati aziendali")

file = st.file_uploader(":file_folder: Carica un file",
                       type=("csv", "txt", "xlsx", "xls"))
if file is not None:
    filename = file.name
    st.write(filename)
    dati = pd.read_excel(file)
else:
    dati = pd.read_excel("./CORSO PYTHON/DATI/Supermarkt_sales.xlsx")

dati["Date"] = pd.to_datetime(dati["Date"])
startDate = pd.to_datetime(dati["Date"]).min()
endDate = pd.to_datetime(dati["Date"]).max()
col1, col2 = st.columns(2)
with col1:
    date1 = pd.to_datetime(st.date_input("Data iniziale", startDate))
with col2:
    date2 = pd.to_datetime(st.date_input("Data finale", endDate))
dati = dati[(dati["Date"] >= date1) & (dati["Date"] <= date2)].copy()

st.sidebar.header("Seleziona i tuoi filtri: ")
città_sel = st.sidebar.multiselect("Seleziona la città", 
                                   options=dati["City"].unique(),
                                   default=dati["City"].unique())

tipo_utente = st.sidebar.multiselect("Seleziona il tipo di utente", 
                                    options=dati["Customer_type"].unique(),
                                    default=dati["Customer_type"].unique())

dati_filtrati = dati.query(
    "City == @città_sel & Customer_type ==@tipo_utente")

st.dataframe(dati_filtrati)

st.title(":bar_chart: Dashboard vendite")
st.markdown("##")
vendite_totali = int(dati_filtrati["Total"].sum())
rating_medio = round(dati_filtrati["Rating"].mean(), 1)
star_rating = ":star:" * int(round(rating_medio, 0))
media_transazione = round(dati_filtrati["Total"].mean(), 2)

left_col, mid_col, right_col = st.columns(3)
with left_col:
    st.subheader("Vendite totali:")
    st.subheader(f"US $ {vendite_totali:,}")
with mid_col:
    st.subheader("Rating medio:")
    st.subheader(f"{rating_medio} {star_rating}")
with right_col:
    st.subheader("Spesa media:")
    st.subheader(f"US $ {media_transazione:,}")
st.markdown("---")

dati_cat = dati_filtrati.groupby(by=["Gender"], as_index = False)["Total"].sum().sort_values(by="Total")

side1, side2 = st.columns(2)

with side1:
    st.subheader("Vendite per genere")
    fig = px.bar(dati_cat, 
                 y = "Gender", 
                 x = "Total", 
                 text = [f"$ {x:.1f}" for x in dati_cat["Total"]],
                 orientation="h",
                 template = "seaborn" # anche "plotly_white"
                 )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=(dict(showgrid=False)))
    
    st.plotly_chart(fig, use_container_width=True, height=300)

with side2:
    st.subheader("Vendite per prodotto")
    fig = px.pie(dati_filtrati, values = "Total", names = "Product line", hole = 0)
    fig.update_traces(text = dati_filtrati["Product line"], textposition="inside")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Analisi delle serie storiche")

glinea = (
    dati_filtrati
    .groupby("Date", as_index=False)["Total"]
    .sum()
    .sort_values(by="Date"))

fig2 = px.line(
    glinea,
    x="Date",
    y="Total",
    labels={"Date": "Date", "Total": "Total"},
    height=500,
    template="gridon")

fig2.update_layout(xaxis=dict(tickmode="auto")) # auto, evita di mostrare tutte le date
                                                # con "linear" mostra tutto
st.plotly_chart(fig2, use_container_width=True)

# Tree map
st.subheader("Tree map:")
fig3 = px.treemap(dati_filtrati, 
                  path = ["City","Gender","Product line"], 
                  values = "Total", hover_data = ["Total"],
                  color = "Product line")
fig3.update_layout(width = 800, height = 650)
st.plotly_chart(fig3, use_container_width=True)

# Scatter plot
dispersione = px.scatter(dati_filtrati, 
                         x = "Unit price", 
                         y = "Total", 
                         size = "Quantity")

dispersione.update_layout(
    title="Relazione tra unità vendute e totale speso:",
    title_font_size=20,
    xaxis_title="Unit price",
    yaxis_title="Total",
    xaxis_title_font_size=19,
    yaxis_title_font_size=19)

st.plotly_chart(dispersione, use_container_width=True)