"""
Lettura ed estrazione testo da file PDF e Markdown nella cartella data/.
Restituisce una lista di documenti grezzi con metadati minimi (fonte, titolo).
"""
from typing import Union
import os
from pathlib import Path
from pypdf import PdfReader

def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def read_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(dir:Union[Path|str]) -> list[dict]:
    """
    Scansiona dir e ritorna una lista di dict:
    {"source": nome_file, "text": testo_completo}
    """
    if isinstance(dir, str):
        dir = Path(dir)

    docs = []
    if not dir.exists():
        return docs

    for path in dir.rglob("*"):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                text = read_pdf(path)
            elif suffix in (".md", ".markdown", ".txt"):
                text = read_markdown(path)
            else:
                continue
        except Exception as e:
            print(f"[ingest] errore leggendo {path}: {e}")
            continue

        if text.strip():
            docs.append({"source": path.name, "text": text})

    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Chunking semplice a finestra scorrevole su caratteri.
    Per un portfolio va benissimo; in produzione si userebbero splitter
    più sofisticati (per frase/paragrafo, tokenizer-aware, ecc.)
    """
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
