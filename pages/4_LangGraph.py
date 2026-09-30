"""
Streamlit UI: assistente di scrittura con scaletta interattiva (LangGraph + Google Gemini).

Flusso:
  1. l'utente scrive un prompt -> controllo di sicurezza -> il modello crea la scaletta (a destra)
  2. l'utente chiede modifiche -> controllo di sicurezza -> la scaletta viene aggiornata
  3. "Produci il testo" -> il modello scrive -> controllo di sicurezza -> il testo viene mostrato
"""

import os
import uuid

import streamlit as st

from langgraph_agents.graph import (
    MAX_INPUT_CHARS,
    MAX_REFINEMENTS,
    build_graph,
    graph_to_dot,
    resume_session,
    start_session,
)

st.set_page_config(
    page_title="Assistente di scrittura",
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


def _secret(name: str):
    """Legge un valore da st.secrets; ritorna None se il file secrets.toml non esiste o manca la chiave."""
    try:
        return st.secrets[name] if name in st.secrets else None
    except Exception:  # noqa: BLE001  (es. StreamlitSecretNotFoundError)
        return None


def get_llm_models() -> list[str]:
    """Recupera la lista dei modelli da secrets, env var (separati da virgola) o fallback."""
    from_secrets = _secret("GEMINI_MODELS")
    if from_secrets:
        return list(from_secrets)

    env = os.environ.get("GEMINI_MODELS")
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]

    return ["gemini-2.0-flash"]


def get_api_key() -> str | None:
    """Recupera la API key da secrets o env var."""
    return _secret("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")


@st.cache_resource(show_spinner=False)
def get_app(api_key: str, model_name: str):
    """Il grafo (con il suo checkpointer in memoria) va creato una sola volta, non a ogni rerun."""
    return build_graph(api_key, model_name)


@st.cache_resource(show_spinner=False)
def get_graph_view(api_key: str, model_name: str):
    """
    Immagine del grafo. Prima prova il PNG generato da LangGraph (usa il servizio mermaid.ink,
    quindi serve Internet); se non è raggiungibile ripiega su un disegno vettoriale Graphviz.
    """
    app = get_app(api_key, model_name)
    try:
        return "png", app.get_graph().draw_mermaid_png()
    except Exception:  # noqa: BLE001
        return "dot", graph_to_dot(app)


for key, default in {
    "thread_id": None,      # id della sessione LangGraph in corso
    "status": None,         # ultima fotografia dello stato (fase + valori)
    "session_model": None,  # modello con cui è stata avviata la sessione
    "nonce": 0,             # cambia per svuotare i campi di testo
    "pending": None,        # azione richiesta da un click, eseguita al rerun successivo
    "flash": None,          # messaggio da mostrare (tipo, testo)
}.items():
    st.session_state.setdefault(key, default)


def reset_session():
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


# Callback dei pulsanti: validano e registrano l'azione. Il lavoro vero avviene al rerun
# successivo, così la UI (es. la scelta del modello) risulta già disabilitata mentre il grafo gira.
def request_start():
    text = _text("topic")
    if not get_api_key():
        _flash("error", "Nessuna API key trovata")
    elif not text:
        _flash("warning", "Scrivi prima un prompt.")
    else:
        st.session_state.pending = {"action": "start", "text": text}


def request_refine():
    text = _text("instr")
    if not get_api_key():
        _flash("error", "Nessuna API key trovata")
    elif not text:
        _flash("warning", "Scrivi cosa vuoi cambiare nella scaletta.")
    else:
        st.session_state.pending = {"action": "refine", "text": text}


def request_write():
    if not get_api_key():
        _flash("error", "Nessuna API key trovata")
    elif _text("instr"):
        _flash(
            "warning",
            "Hai scritto delle modifiche non ancora applicate: premi «Aggiorna scaletta» "
            "oppure svuota il campo, poi produci il testo.",
        )
    else:
        st.session_state.pending = {"action": "write", "text": ""}


def _next_label(node: str, update: dict) -> str | None:
    """Etichetta del passo successivo, mostrata nel riquadro di avanzamento."""
    if node == "guard_input" and update.get("input_safe"):
        return "Elaboro la scaletta…"
    if node == "review" and update.get("next_action") == "write":
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
        # Modificabile prima di creare la scaletta e a testo concluso. Bloccato mentre il grafo
        # lavora e durante la sessione (scaletta in modifica): per cambiarlo usa «Nuova sessione».
        disabled=busy or phase == "review",
        help=(
            "Si può cambiare prima di creare la scaletta e dopo aver prodotto il testo. "
            "Durante una sessione è bloccato: usa «Nuova sessione» per sbloccarlo."
        ),
    )

    st.button("Nuova sessione", on_click=reset_session, use_container_width=True, disabled=busy)

    st.divider()
    st.markdown(
        f"""
        **Come funziona**

        1. Scrivi cosa vuoi produrre: l'**Outliner** crea una scaletta
        2. Chiedi modifiche quante volte vuoi (max **{MAX_REFINEMENTS}**)
        3. Clicca **Produci il testo**: il **Writer** lo scrive seguendo la scaletta

        **Sicurezza**

        Una **Guardia** controlla ogni tuo intervento (prompt e modifiche) prima che il
        modello lavori, e il testo finale prima che venga mostrato. Sono bloccati contenuti
        violenti, degradanti o d'odio, istruzioni per costruire oggetti pericolosi,
        contenuti sessuali espliciti, autolesionismo, attività illegali, dati personali di
        privati e tentativi di aggirare le regole.
        """
    )
    st.divider()
    st.caption("Progetto dimostrativo: LangGraph - Google Gemini - Streamlit")


st.title("Assistente di scrittura con scaletta interattiva")

left, right = st.columns([1, 1.3])


with left:
    if phase in ("start", "blocked", "output_blocked"):
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
                placeholder="Es. 'Un articolo divulgativo sull'impatto del turismo di massa sulle città d'arte'",
                label_visibility="collapsed",
                disabled=busy,
            )
            st.form_submit_button(
                "Crea scaletta", type="primary", on_click=request_start, disabled=busy
            )
    elif phase == "review":
        used = values.get("outline_version", 1) - 1
        limit_reached = used >= MAX_REFINEMENTS

        st.subheader("2. Modifica la scaletta")
        _show_flash()
        if notice:
            st.warning(notice)

        with st.form("refine_form", border=False):
            st.text_area(
                "Modifiche",
                key=f"instr_{st.session_state.nonce}",
                max_chars=MAX_INPUT_CHARS,
                height=120,
                placeholder="Es. 'riassumi', 'sposta la sezione 3 all'inizio', 'aggiungi un paragrafo sui costi'",
                label_visibility="collapsed",
                disabled=limit_reached or busy,
            )
            st.caption(f"Modifiche usate: {used}/{MAX_REFINEMENTS}")

            c1, c2 = st.columns(2)
            with c1:
                st.form_submit_button(
                    "Aggiorna scaletta",
                    on_click=request_refine,
                    use_container_width=True,
                    disabled=limit_reached or busy,
                )
            with c2:
                st.form_submit_button(
                    "Produci il testo",
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
        st.subheader("Scaletta usata")
        st.markdown(values.get("outline", ""))


with right:
    if phase == "start":
        st.subheader("Anteprima")
        st.info("La scaletta e poi il testo appariranno qui.")

    elif phase == "review":
        st.subheader(f"Scaletta (versione {values.get('outline_version', 1)})")
        st.markdown(values.get("outline", ""))

    elif phase == "done":
        st.subheader("Testo")
        st.markdown(values["final_text"])
        st.caption("Testo generato da un modello di IA: verifica i dati prima di usarlo.")
        st.download_button(
            "Scarica il testo (Markdown)",
            data=values["final_text"],
            file_name="testo.md",
            mime="text/markdown",
        )

    elif phase == "output_blocked":
        st.subheader("Scaletta")
        st.markdown(values.get("outline", ""))

    else:  # blocked
        st.subheader("Anteprima")
        st.info("Riformula la richiesta e riprova.")

if pending:
    st.session_state.pending = None  # azzerato subito: nessun rischio di ripetere l'azione
    action, text = pending["action"], pending.get("text", "")
    api_key = get_api_key()
    new_status = None
    try:
        with run_area:
            if action == "start":
                st.session_state.session_model = model_name
                thread_id = uuid.uuid4().hex
                app = get_app(api_key, model_name)
                new_status = run_with_status(
                    "Controllo di sicurezza sulla richiesta…",
                    lambda cb: start_session(app, thread_id, text, on_step=cb),
                )
                st.session_state.thread_id = thread_id
                if new_status and new_status.get("phase") == "blocked":
                    st.session_state.nonce += 1  # svuota il prompt rifiutato
            else:
                app = get_app(api_key, st.session_state.session_model or model_name)
                thread_id = st.session_state.thread_id
                if action == "refine":
                    new_status = run_with_status(
                        "Controllo di sicurezza sulle modifiche…",
                        lambda cb: resume_session(app, thread_id, "refine", text, on_step=cb),
                    )
                    st.session_state.nonce += 1  # svuota il campo delle modifiche
                else:
                    new_status = run_with_status(
                        "Scrivo il testo…",
                        lambda cb: resume_session(app, thread_id, "write", on_step=cb),
                    )
    except Exception as exc:  # noqa: BLE001
        _flash("error", f"Si è verificato un errore: {exc}")

    if new_status is not None:
        st.session_state.status = new_status
    st.rerun()