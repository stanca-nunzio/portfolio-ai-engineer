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
