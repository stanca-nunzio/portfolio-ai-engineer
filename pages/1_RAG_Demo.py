"""
Pagina progetto: RAG Demo.
L'indicizzazione dei documenti avviene una sola volta grazie a
st.cache_resource (persiste tra le interazioni dell'utente nella sessione
del server, evitando di ricalcolare embedding ad ogni domanda).
"""
import os
from pathlib import Path
import streamlit as st

from progetto_1.ingest import load_documents, chunk_text
from progetto_1.vectorstore import index_chunks, search, collection_count
from progetto_1.llm import generate_answer, get_models

st.title("RAG Demo")
st.write(
    "Fai una domanda sui documenti indicizzati. Il sistema cerca i passaggi "
    "più rilevanti tramite ricerca vettoriale e genera una risposta con un LLM."
)

with st.expander("Come funziona / stack tecnico"):
    st.markdown(
        """
        1. I documenti (PDF/Markdown) in `data/` vengono letti e spezzati in chunk
        2. Ogni chunk viene trasformato in un embedding con `all-MiniLM-L6-v2` (locale, gira su CPU)
        3. Gli embedding sono indicizzati in **Qdrant embedded** (in-process, salvato su disco locale — nessun server esterno)
        4. Alla domanda dell'utente, si recuperano i 15 chunk più simili per similarità coseno
        5. I 15 candidati vengono riordinati da un **cross-encoder** (`ms-marco-MiniLM-L-6-v2`), più preciso della sola similarità coseno, e si tengono i migliori 4
        6. I chunk finali vengono passati come contesto a **Google Gemini** per generare la risposta finale
        """
    )


@st.cache_resource(show_spinner="Indicizzazione documenti in corso (una tantum)...")
def setup_index():
    root_dir = st.session_state.get("ROOT_DIR")
    docs_dir = Path(st.secrets["DOCS_DIR"])
    chunk_size = int(st.secrets["CHUNK_SIZE"])
    chunk_overlap = int(st.secrets["CHUNK_OVERLAP"])

    docs_dir = Path(st.secrets["DOCS_DIR"])

    docs_path = st.session_state["DOCS_PATH"] = os.path.join(root_dir, docs_dir)

    print("root_dir", root_dir)
    print("DOCS_DIR", docs_dir)
    print("docs_path", docs_path)
    documents = load_documents(docs_path)
    all_chunks = []
    for doc in documents:
        pieces = chunk_text(
            doc["text"],
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            pages=doc.get("pages"),
        )
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "text": piece["text"],
                "source": doc["source"],
                "chunk_id": i,
                "page": piece["page"],
            })
    index_chunks(all_chunks)
    return len(documents), len(all_chunks)


n_docs, n_chunks = setup_index()
st.caption(f"{n_docs} documenti indicizzati -  {n_chunks} chunk - {collection_count()} vettori in Qdrant")

st.divider()

question = st.text_area(
    "La tua domanda",
    placeholder="Es: di cosa parla il documento di esempio?",
    height=100,
)

col1, col2 = st.columns([1, 4])
with col1:
    ask_clicked = st.button("Chiedi", type="primary")

if ask_clicked:
    #get_models()
    if not question.strip():
        st.warning("Scrivi prima una domanda.")
    else:
        with st.spinner("Cerco nei documenti e genero la risposta..."):
            try:
                contexts = search(question, top_k=4)
                if not contexts:
                    st.info("Nessun risultato rilevante trovato nei documenti indicizzati.")
                else:
                    answer = generate_answer(question, contexts)
                    st.markdown("### Risposta")
                    st.write(answer)

                    sources = sorted(set(c["source"] for c in contexts))
                    st.caption(f"**Fonti:** {', '.join(sources)}")

                    with st.expander("Chunk recuperati (debug)"):
                        for c in contexts:
                            page_info = f", pag. {c['page']}" if c.get("page") else ""
                            rerank_info = f" | re-rank: {c['rerank_score']:.3f}" if "rerank_score" in c else ""
                            st.markdown(f"**{c['source']}{page_info}** (qdrant score: {c['score']:.3f}{rerank_info})")
                            st.text(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))
                            st.divider()
            except RuntimeError as e:
                st.error(str(e))

st.divider()
st.page_link("pages/0_Home.py", label="Torna alla lista progetti")
