"""
Pagina progetto: RAG.
L'indicizzazione dei documenti avviene una sola volta grazie a
st.cache_resource (persiste tra le interazioni dell'utente nella sessione
del server, evitando di ricalcolare embedding ad ogni domanda).
"""
import os
from pathlib import Path
import pandas as pd
import streamlit as st

from rag_system.evaluate_retrieval import (
    DEFAULT_KS, check_known, evaluate, indexed_chunk_keys, metrics_rows, parse_questions,
)
from rag_system.ingest import load_documents, chunk_text, list_source_files, preview_file, ensure_seed_documents
from rag_system.vectorstore import index_chunks, search, collection_count
from rag_system.llm import generate_answer, get_models

EVAL_TEST_SET = "questions-test-set.csv"

st.title("RAG")
st.write(
    "Fai una domanda sui documenti indicizzati. Il sistema cerca i passaggi "
    "più rilevanti tramite ricerca vettoriale e genera una risposta con un LLM."
)

with st.expander("Come funziona / stack tecnico"):
    st.markdown(
        """
        1. I documenti (PDF/Markdown) in `data/` vengono letti e spezzati in chunk per paragrafi (i paragrafi troppo lunghi vengono ulteriormente divisi a finestra fissa)
        2. Ogni chunk viene trasformato in un embedding con `all-MiniLM-L6-v2` (locale, gira su CPU)
        3. Gli embedding sono indicizzati in **Qdrant embedded** (in-process, salvato su disco locale)
        4. Alla domanda dell'utente, si recuperano i chunk candidati più simili per similarità coseno (numero configurabile, default 15)
        5. I candidati vengono riordinati da un **cross-encoder** (`ms-marco-MiniLM-L-6-v2`), più preciso della sola similarità coseno, e si tengono i migliori 4
        6. I chunk finali vengono passati come contesto a **Google Gemini** per generare la risposta finale
        """
    )


@st.cache_resource(show_spinner="Indicizzazione documenti in corso (una tantum)...")
def setup_index():
    if "ROOT_DIR" not in st.session_state:
        st.session_state["ROOT_DIR"] = Path(__file__).resolve().parent.parent

    chunk_size = int(st.secrets["CHUNK_SIZE"])
    chunk_overlap = int(st.secrets["CHUNK_OVERLAP"])

    root_dir = st.session_state.get("ROOT_DIR")
    docs_dir = Path(st.secrets["DOCS_DIR"])
    res_dir = Path(st.secrets["RES_DIR"])

    docs_path = st.session_state["DOCS_PATH"] = os.path.join(root_dir, docs_dir)
    res_path = st.session_state["RES_PATH"] = os.path.join(root_dir, res_dir)
    seed_file = os.path.join(res_path, 'documents.json')

    ensure_seed_documents(docs_path, seed_file)

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

tab_chat, tab_files, tab_eval = st.tabs(["Chat", "Files", "Valutazione"])

with tab_chat:
    question = st.text_area(
        "La tua domanda",
        placeholder="Es: di cosa parla il documento di esempio?",
        height=100,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_clicked = st.button("Chiedi", type="primary")

    if ask_clicked:
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

with tab_files:
    if 'DOCS_PATH' not in st.session_state:
        root_dir = st.session_state.get("ROOT_DIR")
        docs_dir = Path(st.secrets["DOCS_DIR"])

        st.session_state["DOCS_PATH"] = os.path.join(root_dir, docs_dir)

    docs_path = Path(st.session_state["DOCS_PATH"])

    files = list_source_files(docs_path)

    if not files:
        st.info("Nessun file trovato nella cartella documenti.")
    else:
        st.caption(f"{len(files)} file trovati in `{docs_path.name}`")

        for f in files:
            col_name, col_size, col_btn = st.columns([5, 1, 1])
            col_name.write(f["rel_path"])
            col_size.caption(f"{f['size_kb']} KB")
            show = col_btn.button("Anteprima", key=f"preview_{f['rel_path']}")

            if show:
                st.session_state["files_preview_target"] = f["rel_path"]

        target = st.session_state.get("files_preview_target")
        if target:
            try:
                max_chars = int(st.secrets["PREVIEW_FILE_MAX_CHARS"])
            except Exception:
                max_chars = 2000

            st.divider()
            st.markdown(f"**Anteprima: {target}**")
            preview_text = preview_file(docs_path / target, max_chars = max_chars)
            if preview_text:
                st.text(preview_text)
            else:
                st.warning("Impossibile generare l'anteprima per questo file.")

with tab_eval:
    st.write(
        "Incolla il contenuto di un CSV con colonne `question`, `source`, `chunk_ids` "
        "(id separati da spazio). Le domande vengono eseguite sull'indice corrente, "
        "prima senza re-ranking e poi con il cross-encoder."
    )
    csv_text = st.text_area(
        "CSV",
        height=200,
        placeholder="question;source;chunk_ids\nDomanda di esempio;documento.pdf;12 13",
        key="eval_csv_text",
    )

    def run_evaluation(raw_csv: str, origin: str):
        try:
            questions = parse_questions(raw_csv)
            check_known(questions, indexed_chunk_keys())
            max_k = max(DEFAULT_KS)
            with st.spinner(f"Eseguo {len(questions)} domande, con e senza re-ranking..."):
                st.session_state["eval_result"] = evaluate(
                    questions,
                    lambda q, rerank: search(q, top_k=max_k, use_reranker=rerank),
                    DEFAULT_KS,
                )
            st.session_state["eval_origin"] = origin
        except ValueError as e:
            st.session_state.pop("eval_result", None)
            st.error(str(e))

    col_pasted, col_file, _ = st.columns([1, 2, 3])
    run_pasted = col_pasted.button("Esegui test", type="primary", key="eval_run")
    run_file = col_file.button(f"Usa {EVAL_TEST_SET}", key="eval_run_file")

    if run_pasted:
        if not csv_text.strip():
            st.warning("Incolla prima il contenuto del CSV (Formato 'question;source;chunk_ids')")
        else:
            run_evaluation(csv_text, "CSV incollato")
    elif run_file:
        test_set_path = Path(st.session_state["ROOT_DIR"]) / Path(st.secrets["RES_DIR"]) / EVAL_TEST_SET
        if not test_set_path.exists():
            st.error(f"File non trovato: {test_set_path}")
        else:
            run_evaluation(test_set_path.read_text(encoding="utf-8-sig"), EVAL_TEST_SET)

    eval_result = st.session_state.get("eval_result")
    if eval_result:
        plain, reranked, ks = eval_result["plain"], eval_result["reranked"], eval_result["ks"]

        st.subheader("Metriche")
        st.caption(
            f"Fonte: {st.session_state.get('eval_origin', '-')}. "
            "Valore con re-ranking; la variazione e' rispetto alla ricerca senza re-ranking."
        )
        headline = ["mrr", "hit@4", "recall@4", "ndcg@4"]
        for col, name in zip(st.columns(len(headline)), headline):
            col.metric(name, f"{reranked[name]:.3f}", f"{reranked[name] - plain[name]:+.3f}")

        metrics_df = pd.DataFrame(
            metrics_rows(plain, reranked, ks),
            columns=["metrica", "senza re-rank", "con re-rank", "delta"],
        )
        st.dataframe(metrics_df.round(3), hide_index=True, use_container_width=True)

        st.subheader("Riepilogo per domanda")
        summary_df = pd.DataFrame(eval_result["rows"])
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
        st.download_button(
            "Scarica rag_eval_results.csv",
            data=summary_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="rag_eval_results.csv",
            mime="text/csv",
        )

st.divider()
st.page_link("pages/0_Home.py", label="Torna alla lista progetti")
