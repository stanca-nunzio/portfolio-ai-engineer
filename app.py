"""
Landing page del portfolio: elenca i progetti disponibili.
Streamlit riconosce automaticamente i file in pages/ come pagine
aggiuntive della sidebar (multi-page app nativa).
"""
import os
from pathlib import Path
import streamlit as st

if "ROOT_DIR" not in st.session_state:
    # Path(__file__).parent prende la cartella in cui si trova app.py
    st.session_state["ROOT_DIR"] = Path(__file__).parent.resolve()


# 1. Definisci gli oggetti pagina (st.Page)

home_page = st.Page("pages/0_Home.py", title="Home", default=True)
progetto_1_page = st.Page("pages/1_RAG_Demo.py", title="RAG Demo — Retrieval Augmented Generation")
progetto_2_page = st.Page("pages/2_Demo.py", title="Progetto Due")

pg = st.navigation([home_page,progetto_1_page, progetto_2_page])

pg.run()