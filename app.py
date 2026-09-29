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
eu_ai_toolkit_page = st.Page("pages/3_EU_AI_Toolkit.py", title="Toolkit di Compliance GDPR / EU AI Act")
eta_dashboard_page = st.Page("pages/2_ETA_Market.py", title="E-Commerce Sales: ETA, Dashboard")

pg = st.navigation([home_page,rag_system_page, eu_ai_toolkit_page, eta_dashboard_page])

pg.run()