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
rag_system_page = st.Page("pages/1_RAG.py", title="RAG: Retrieval Augmented Generation")
eta_dashboard_page = st.Page("pages/2_ETA_Market.py", title="E-Commerce Sales: ETA, Dashboard")

pg = st.navigation([home_page,rag_system_page, eta_dashboard_page])

pg.run()