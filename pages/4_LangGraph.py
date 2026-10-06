"""
Streamlit UI: assistente di scrittura con ricerca web e revisione delle fonti
(LangGraph + Google Gemini + Tavily).

Flusso:
  1. l'utente scrive un prompt -> controllo di sicurezza -> il modello pianifica le query e cerca sul web
  2. l'utente rivede le fonti: può deselezionarle, segnarle come prioritarie o chiedere altre ricerche
  3. "Approva e scrivi" -> il modello scrive con citazioni -> controllo di sicurezza -> testo finale
"""

import hashlib
import os
import uuid

import streamlit as st

from langgraph_agents.graph import (
    MAX_INPUT_CHARS,
    MAX_SEARCH_ROUNDS,
    MAX_SOURCES_SELECTED,
    build_graph,
    graph_to_dot,
    resume_session,
    start_session,
    visible_sources,
)

st.set_page_config(
    page_title="Assistente di scrittura con ricerca",
    layout="wide",
    initial_sidebar_state="expanded",
)

NEUTRAL_DARK = "#1B2430"
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #F7F8FA; }}
    h1, h2, h3 {{ color: {NEUTRAL_DARK}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

STAGES = ["Ricerca", "Fonti", "Testo"]
STAGE_OF_PHASE = {"start": 0, "blocked": 0, "sources": 1, "done": 2}


def _secret(name: str):
    """Legge un valore da st.secrets; ritorna None se il file secrets.toml non esiste o manca la chiave."""
    try:
        return st.secrets[name] if name in st.secrets else None
    except Exception:  # noqa: BLE001
        return None


def get_llm_models() -> list[str]:
    from_secrets = _secret("GEMINI_MODELS")
    if from_secrets:
        return list(from_secrets)

    env = os.environ.get("GEMINI_MODELS")
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]

    return ["gemini-2.0-flash"]


def get_api_key() -> str | None:
    return _secret("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def get_tavily_key() -> str | None:
    return _secret("TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY")


def _missing_keys() -> str | None:
    if not get_api_key():
        return "Nessuna API key Google trovata."
    if not get_tavily_key():
        return "Nessuna API key Tavily trovata (TAVILY_API_KEY)."
    return None


@st.cache_resource(show_spinner=False)
def get_app(api_key: str, tavily_key: str, model_name: str):
    """Il grafo (con il suo checkpointer in memoria) va creato una sola volta, non a ogni rerun."""
    return build_graph(api_key, tavily_key, model_name)


@st.cache_resource(show_spinner=False)
def get_graph_view(api_key: str, tavily_key: str, model_name: str):
    """
    Immagine del grafo. Prima prova il PNG di LangGraph (usa mermaid.ink, serve Internet);
    se non è raggiungibile ripiega su un disegno Graphviz.
    """
    app = get_app(api_key, tavily_key, model_name)
    try:
        return "png", app.get_graph().draw_mermaid_png()
    except Exception:  # noqa: BLE001
        return "dot", graph_to_dot(app)


for key, default in {
    "thread_id": None,
    "status": None,
    "session_model": None,
    "nonce": 0,
    "pending": None,
    "flash": None,
}.items():
    st.session_state.setdefault(key, default)


def _src_key(kind: str, url: str) -> str:
    return f"{kind}_" + hashlib.md5(url.encode()).hexdigest()[:10]


def reset_session():
    for k in [k for k in st.session_state if k.startswith(("use_", "pri_"))]:
        del st.session_state[k]
    st.session_state.thread_id = None
    st.session_state.status = None
    st.session_state.session_model = None
    st.session_state.pending = None
    st.session_state.flash = None
    st.session_state.nonce += 1


def _flash(kind: str, message: str):
    st.session_state.flash = (kind, message)


def _text(prefix: str) -> str:
    return st.session_state.get(f"{prefix}_{st.session_state.nonce}", "").strip()


def _md(text: str) -> str:
    """Rende sicuro per il markdown di Streamlit un testo preso dal web."""
    return text.replace("$", "\\$").replace("[", "(").replace("]", ")")


def _current_values() -> dict:
    return (st.session_state.status or {}).get("values", {})


def _read_selection() -> tuple[list[str], list[str]]:
    """Legge dai checkbox le fonti da usare e quelle prioritarie."""
    shown = visible_sources(_current_values())
    urls = [s["url"] for s in shown if st.session_state.get(_src_key("use", s["url"]), True)]
    priority = [u for u in urls if st.session_state.get(_src_key("pri", u), False)]
    return urls, priority


# I callback dei pulsanti validano e registrano l'azione; il lavoro vero avviene al rerun
# successivo, così la UI risulta già disabilitata mentre il grafo gira.
def request_start():
    text = _text("topic")
    err = _missing_keys()
    if err:
        _flash("error", err)
    elif not text:
        _flash("warning", "Scrivi prima un prompt.")
    else:
        st.session_state.pending = {"action": "start", "text": text}


def request_search():
    text = _text("more")
    err = _missing_keys()
    if err:
        _flash("error", err)
    elif not text:
        _flash("warning", "Scrivi cosa vuoi aggiungere o approfondire.")
    elif _current_values().get("search_round", 0) >= MAX_SEARCH_ROUNDS:
        _flash("warning", f"Limite di {MAX_SEARCH_ROUNDS} ricerche raggiunto: approva le fonti.")
    else:
        urls, priority = _read_selection()
        st.session_state.pending = {"action": "search", "text": text, "urls": urls, "priority_urls": priority}


def request_write():
    urls, priority = _read_selection()
    err = _missing_keys()
    if err:
        _flash("error", err)
    elif _text("more"):
        _flash(
            "warning",
            "Hai scritto una richiesta di ricerca non ancora inviata: premi «Cerca altre fonti» "
            "oppure svuota il campo, poi approva.",
        )
    elif not urls:
        _flash("warning", "Seleziona almeno una fonte.")
    elif len(urls) > MAX_SOURCES_SELECTED:
        _flash("warning", f"Puoi usare al massimo {MAX_SOURCES_SELECTED} fonti: deseleziona le meno utili.")
    else:
        st.session_state.pending = {"action": "write", "urls": urls, "priority_urls": priority}


def _next_label(node: str, update: dict) -> str | None:
    """Etichetta del passo successivo, mostrata nel riquadro di avanzamento."""
    if node == "guard_input" and update.get("input_safe"):
        return "Pianifico la ricerca…"
    if node == "planner":
        return "Cerco sul web con Tavily…"
    if node == "search":
        return "Valuto i risultati…"
    if node == "evaluator" and update.get("search_again"):
        return "Nuova ricerca mirata…"
    if node == "review_sources" and update.get("next_action") == "write":
        return "Scrivo il testo…"
    if node == "writer":
        return "Controllo di sicurezza sul testo finale…"
    return None


def run_with_status(initial_label: str, fn):
    """Esegue fn(on_step) mostrando in tempo reale i passaggi del grafo (compresi i controlli)."""
    with st.status(initial_label, expanded=True) as box:

        def on_step(node, update):
            for line in update.get("log", []):
                box.write(line)
            label = _next_label(node, update)
            if label:
                box.update(label=label)

        result = fn(on_step)
        box.update(label="Fatto", state="complete", expanded=False)
    return result


def _show_flash():
    flash = st.session_state.flash
    if flash:
        st.session_state.flash = None
        getattr(st, flash[0])(flash[1])


pending = st.session_state.pending
busy = pending is not None
status = st.session_state.status
phase = status["phase"] if status else "start"
values = status["values"] if status else {}
notice = values.get("notice", "")


with st.sidebar:
    st.title("Configurazione")

    model_name = st.selectbox(
        "Modello Gemini",
        options=get_llm_models(),
        index=0,
        # Bloccato mentre il grafo lavora e durante la revisione delle fonti.
        disabled=busy or phase == "sources",
        help=(
            "Si può cambiare prima di avviare la ricerca e dopo aver prodotto il testo. "
            "Durante una sessione è bloccato: usa «Nuova sessione» per sbloccarlo."
        ),
    )

    st.button("Nuova sessione", on_click=reset_session, use_container_width=True, disabled=busy)

    st.divider()
    st.markdown(
        f"""
        **Come funziona**

        1. Scrivi cosa vuoi produrre: il **Planner** genera le query e **Tavily** cerca sul web
        2. Rivedi le fonti trovate: deselezionale, segnale come **prioritarie** o chiedi altre
           ricerche (max **{MAX_SEARCH_ROUNDS}** in tutto)
        3. Approva: il **Writer** scrive il testo citando le fonti con [n]

        **Sicurezza**

        Una **Guardia** controlla ogni testo che scrivi prima che il modello lavori, e il testo
        finale prima che venga mostrato. Il contenuto delle pagine web è trattato come dato non
        fidato. Sono bloccati contenuti violenti, degradanti o d'odio, istruzioni per costruire
        oggetti pericolosi, contenuti sessuali espliciti, autolesionismo, attività illegali,
        dati personali di privati e tentativi di aggirare le regole.
        """
    )
    st.divider()
    st.caption("Progetto dimostrativo: LangGraph - Google Gemini - Tavily - Streamlit")


st.title("Assistente di scrittura con ricerca web")
stage = STAGE_OF_PHASE.get(phase, 0)
st.caption("  >  ".join(f"**{name}**" if i == stage else name for i, name in enumerate(STAGES)))

if phase == "sources":
    left, right = st.container(), None
else:
    left, right = st.columns([1, 1.3])


with left:
    if phase in ("start", "blocked"):
        st.subheader("1. Cosa vuoi scrivere?")
        _show_flash()
        if notice:
            st.warning(notice)

        with st.form("start_form", border=False):
            st.text_area(
                "Prompt",
                key=f"topic_{st.session_state.nonce}",
                max_chars=MAX_INPUT_CHARS,
                height=140,
                placeholder="Es. 'Un articolo sull'impatto dell'AI Act sulle piccole imprese italiane'",
                label_visibility="collapsed",
                disabled=busy,
            )
            st.form_submit_button("Cerca fonti", type="primary", on_click=request_start, disabled=busy)

    elif phase == "sources":
        shown = visible_sources(values)
        rounds = values.get("search_round", 1)
        limit_reached = rounds >= MAX_SEARCH_ROUNDS
        excluded = set(values.get("excluded", []))
        priority = set(values.get("priority", []))

        st.subheader("2. Rivedi le fonti")
        st.caption(
            "Togli la spunta alle fonti da non usare e segna come priorità quelle che vuoi "
            "più centrali nel testo. Per aggiungerne, descrivi cosa cercare."
        )
        _show_flash()
        if notice:
            st.warning(notice)
        if values.get("eval_note"):
            st.info(f"Valutazione automatica: {values['eval_note']}")

        with st.form("sources_form", border=False):
            for s in shown:
                with st.container(border=True):
                    c1, c2 = st.columns([5, 1.3])
                    c1.checkbox(
                        _md(s["title"]),
                        value=s["url"] not in excluded,
                        key=_src_key("use", s["url"]),
                        disabled=busy,
                    )
                    c1.caption(f"[{s['domain']}]({s['url']}) - {_md(s['summary'])}")
                    c2.checkbox(
                        "Priorità",
                        value=s["url"] in priority,
                        key=_src_key("pri", s["url"]),
                        disabled=busy,
                        help="Il testo darà più spazio a questa fonte.",
                    )

            st.text_area(
                "Aggiungi o approfondisci",
                key=f"more_{st.session_state.nonce}",
                max_chars=MAX_INPUT_CHARS,
                height=90,
                placeholder="Es. 'aggiungi fonti istituzionali' oppure 'cerca dati degli ultimi due anni'",
                disabled=limit_reached or busy,
            )
            st.caption(f"Ricerche effettuate: {rounds}/{MAX_SEARCH_ROUNDS}")

            c1, c2 = st.columns(2)
            with c1:
                st.form_submit_button(
                    "Cerca altre fonti",
                    on_click=request_search,
                    use_container_width=True,
                    disabled=limit_reached or busy,
                )
            with c2:
                st.form_submit_button(
                    "Approva e scrivi il testo",
                    type="primary",
                    on_click=request_write,
                    use_container_width=True,
                    disabled=busy,
                )

    elif phase == "done":
        st.subheader("3. Testo prodotto")
        _show_flash()
        st.success("Il testo è pronto. Per una nuova richiesta usa «Nuova sessione».")

    # qui compare il riquadro di avanzamento mentre il grafo lavora
    run_area = st.container()

    if values.get("log"):
        st.subheader("Attività")
        st.markdown("\n".join(f"- {line}" for line in values["log"]))

    if phase == "done":
        st.subheader("Fonti approvate")
        for s in values.get("selected", []):
            tag = " (prioritaria)" if s.get("priority") else ""
            st.markdown(f"{s['n']}. [{_md(s['title'])}]({s['url']}){tag}")


if right is not None:
    with right:
        if phase == "done":
            st.subheader("Testo")
            st.markdown(values["final_text"])
            st.caption("Testo generato da un modello di IA a partire da fonti web: verifica i dati prima di usarlo.")
            st.download_button(
                "Scarica il testo (Markdown)",
                data=values["final_text"],
                file_name="testo.md",
                mime="text/markdown",
            )
        elif phase == "blocked":
            st.subheader("Anteprima")
            st.info("Riformula la richiesta e riprova.")
        else:
            st.subheader("Anteprima")
            st.info("Le fonti trovate e poi il testo appariranno qui.")


with st.expander("Struttura del grafo"):
    _api, _tavily = get_api_key(), get_tavily_key()
    if _api and _tavily:
        kind, data = get_graph_view(_api, _tavily, model_name)
        if kind == "png":
            st.image(data)
        else:
            st.graphviz_chart(data)
    else:
        st.caption("Servono le API key per costruire e mostrare il grafo.")


if pending:
    st.session_state.pending = None  # azzerato subito: nessun rischio di ripetere l'azione
    action, text = pending["action"], pending.get("text", "")
    api_key, tavily_key = get_api_key(), get_tavily_key()
    new_status = None
    try:
        with run_area:
            if action == "start":
                st.session_state.session_model = model_name
                thread_id = uuid.uuid4().hex
                app = get_app(api_key, tavily_key, model_name)
                new_status = run_with_status(
                    "Controllo di sicurezza sulla richiesta…",
                    lambda cb: start_session(app, thread_id, text, on_step=cb),
                )
                st.session_state.thread_id = thread_id
                if new_status and new_status.get("phase") == "blocked":
                    st.session_state.nonce += 1  # svuota il prompt rifiutato
            else:
                app = get_app(api_key, tavily_key, st.session_state.session_model or model_name)
                thread_id = st.session_state.thread_id
                urls, priority_urls = pending.get("urls"), pending.get("priority_urls")
                if action == "search":
                    new_status = run_with_status(
                        "Controllo di sicurezza sulla richiesta…",
                        lambda cb: resume_session(
                            app, thread_id, "search", text, urls=urls, priority_urls=priority_urls, on_step=cb
                        ),
                    )
                    st.session_state.nonce += 1  # svuota il campo di ricerca
                else:
                    new_status = run_with_status(
                        "Preparo la scrittura…",
                        lambda cb: resume_session(
                            app, thread_id, "write", urls=urls, priority_urls=priority_urls, on_step=cb
                        ),
                    )
    except Exception as exc:  # noqa: BLE001
        _flash("error", f"Si è verificato un errore: {exc}")

    if new_status is not None:
        st.session_state.status = new_status
    st.rerun()
