"""
Generazione della risposta con Google Gemini a partire dai chunk recuperati.
"""
import os

from google import genai

_client = None


def _get_setting(name: str) -> str | None:
    """st.secrets (Streamlit Cloud) con ripiego sulla variabile d'ambiente."""
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = _get_setting("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY non impostata. Su Streamlit Cloud: Settings > Secrets. "
                "In locale: variabile d'ambiente o .streamlit/secrets.toml"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join(f"[Fonte: {c['source']}]\n{c['text']}" for c in contexts)
    return f"""Rispondi alla domanda usando SOLO le informazioni nel contesto fornito.
Se il contesto non contiene la risposta, dillo chiaramente invece di inventare.
Cita la fonte (nome file) quando possibile.

CONTESTO:
{context_block}

DOMANDA: {question}

RISPOSTA:"""


def get_models() -> list[str]:
    return [m.name for m in get_client().models.list()]


def generate_answer(question: str, contexts: list[dict]) -> str:
    model = _get_setting("LLM_MODEL_ASK")
    if not model:
        raise RuntimeError("LLM_MODEL_ASK non impostata (secrets.toml o variabile d'ambiente).")

    response = get_client().models.generate_content(
        model=model,
        contents=build_prompt(question, contexts),
    )
    return response.text