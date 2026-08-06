"""
Chiamate all'LLM (Google Gemini) per generare la risposta finale a partire
dai chunk recuperati da Qdrant
"""
import os
from google import genai

_client = None


def _get_api_key() -> str | None:
    """
    Cerca la chiave prima nei secrets di Streamlit Cloud (st.secrets),
    poi come variabile d'ambiente classica (utile per test locali o altri host).
    """
    try:
        import streamlit as st
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY")

def _get_model_for_ask() -> str | None:
    """
    Cerca la chiave prima nei secrets di Streamlit Cloud (st.secrets),
    poi come variabile d'ambiente classica (utile per test locali o altri host).
    """
    try:
        import streamlit as st
        if "LLM_MODEL_ASK" in st.secrets:
            return st.secrets["LLM_MODEL_ASK"]
    except Exception:
        pass
    return os.environ.get("LLM_MODEL_ASK")


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = _get_api_key()
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY non impostata. "
                "Su Streamlit Cloud: Settings > Secrets. "
                "In locale: variabile d'ambiente o .streamlit/secrets.toml"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[Fonte: {c['source']}]\n{c['text']}" for c in contexts
    )
    return f"""Rispondi alla domanda usando SOLO le informazioni nel contesto fornito.
Se il contesto non contiene la risposta, dillo chiaramente invece di inventare.
Cita la fonte (nome file) quando possibile.

CONTESTO:
{context_block}

DOMANDA: {question}

RISPOSTA:"""


def get_models() -> str:
    client = get_client()

    res = ''

    for model in client.models.list():
        res += f"{model}\n"
        print(model.name)

    return res

def generate_answer(question: str, contexts: list[dict]) -> str:
    client = get_client()
    prompt = build_prompt(question, contexts)

    response = client.models.generate_content(
        model=_get_model_for_ask(),
        contents=prompt,
    )
    return response.text
