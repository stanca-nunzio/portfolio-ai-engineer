"""
Landing page del portfolio: elenca i progetti disponibili.
"""
import streamlit as st

st.title("Portfolio — AI Engineer / Data Scientist")
st.write(
    "Benvenuto! Questa è una raccolta di progetti dimostrativi che coprono "
    "diverse aree di AI Engineering e Data Science. Seleziona un progetto "
    "dalla sidebar a sinistra per aprirlo."
)

st.divider()
st.subheader("Progetti disponibili")

projects = [
    {
        "title": "RAG — Retrieval Augmented Generation",
        "description": (
            "Sistema di domande e risposte su documenti (PDF/Markdown) "
            "usando ricerca semantica su vector database ed un LLM per "
            "generare la risposta finale."
        ),
        "stack": "Streamlit - Qdrant (embedded) - sentence-transformers - Google Gemini",
        "page": "pages/1_RAG.py",
        "page_label": "RAG",
    },
    {
        "title": "LangGraph Agents - Assistenza scrittura",
        "description": (
            "Assistente di scrittura con scaletta interattiva: descrivi cosa "
            "vuoi scrivere, modifica la scaletta insieme al modello e genera "
            "il testo finale. Include controlli di sicurezza su input e output."
        ),
        "stack": "Streamlit - LangGraph - Gemini API",
        "page": "pages/4_LangGraph.py",
        "page_label": "LangGraph",
    },
    {
        "title": "EU AI ACT Toolkit",
        "description": (
            "Strumento didattico per una prima autovalutazione interna rispetto "
            "all'AI Act, tramite step guidati. Non costituisce parere legale."
        ),
        "stack": "Streamlit",
        "page": "pages/3_EU_AI_Toolkit.py",
        "page_label": "Toolkit",
    },
    {
        "title": "E-Commerce Sales Dashboard",
        "description": (
            "Pipeline end-to-end su dati di vendita e-commerce: pulizia del "
            "dataset, analisi statistica e dashboard interattiva per esplorare "
            "vendite, andamenti e metriche di business."
        ),
        "stack": "Streamlit - Pandas - Plotly",
        "page": "pages/2_ETA_Market.py",
        "page_label": "Dashboard",
    },
]

for project in projects:
    with st.container(border=True):
        st.markdown(f"### {project['title']}")
        st.write(project["description"])
        st.caption(f"**Stack:** {project['stack']}")
        st.page_link(project["page"], label=f"Apri {project['page_label']} →", icon="➡️")

st.divider()
st.caption(
    "Progetto open source — codice disponibile su GitHub. "
    "Costruito con Streamlit Community Cloud (hosting gratuito)."
)
