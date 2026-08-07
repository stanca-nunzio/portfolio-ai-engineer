"""
Landing page del portfolio: elenca i progetti disponibili.
"""
import os
from pathlib import Path
import streamlit as st

if "ROOT_DIR" not in st.session_state:
    # Path(__file__).parent prende la cartella in cui si trova app.py
    st.session_state["ROOT_DIR"] = Path(__file__).parent.resolve()


# 1. Definisci gli oggetti pagina (st.Page)

home_page = st.Page("pages/0_Home.py", title="Home", default=True)
progetto_1_page = st.Page("pages/1_RAG.py", title="RAG: Retrieval Augmented Generation")

pg = st.navigation([home_page,progetto_1_page])

pg.run()