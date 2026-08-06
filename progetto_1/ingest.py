"""
Lettura ed estrazione testo da file PDF e Markdown nella cartella data/.
Restituisce una lista di documenti grezzi con metadati minimi
"""
from typing import Union
import os
import re
from pathlib import Path
from pypdf import PdfReader

def read_pdf(path: Path) -> list[dict]:
    """
    Ritorna una lista di {"text": ..., "page": numero_pagina} per pagina,
    così il numero di pagina può poi propagarsi ai singoli chunk.
    """
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append({"text": page.extract_text() or "", "page": i})
    return pages


def read_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(dir:Union[Path|str]) -> list[dict]:
    """
    Scansiona dir e ritorna una lista di dict:
    {"source": nome_file, "text": testo_completo, "pages": [...] | None}

    "pages" è presente solo per i PDF (lista di {"text", "page"}); per i file
    di testo è None, dato che non esiste un concetto di pagina.
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
                pages = read_pdf(path)
                text = "\n".join(p["text"] for p in pages)
            elif suffix in (".md", ".markdown", ".txt"):
                pages = None
                text = read_markdown(path)
            else:
                continue
        except Exception as e:
            print(f"[ingest] errore leggendo {path}: {e}")
            continue

        if text.strip():
            docs.append({"source": path.name, "text": text, "pages": pages})

    return docs


def _split_paragraphs(text: str) -> list[str]:
    """Divide su righe vuote (uno o più \n consecutivi con solo spazi)."""
    raw = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in raw if p.strip()]


def _page_for_offset(offset: int, pages: list[dict]) -> int | None:
    """
    Dato un offset nel testo concatenato del documento, trova la pagina PDF
    corrispondente sommando le lunghezze dei testi di pagina in ordine.
    """
    if not pages:
        return None
    cursor = 0
    for p in pages:
        # +1 per il separatore "\n" usato in load_documents per unire le pagine
        cursor += len(p["text"]) + 1
        if offset < cursor:
            return p["page"]
    return pages[-1]["page"]


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150, pages: list[dict] | None = None) -> list[dict]:
    """
    Chunking per paragrafi: unisce paragrafi consecutivi fino a chunk_size
    caratteri. Un paragrafo più lungo di chunk_size viene spezzato a sua
    volta a finestra fissa, come fallback.

    Ritorna una lista di dict {"text", "start_offset", "page"}.
    "page" è None se il documento non ha informazioni di pagina (es. markdown).
    """
    text = text.strip()
    paragraphs = _split_paragraphs(text)

    chunks = []
    current_parts = []
    current_len = 0
    current_start = 0
    search_from = 0

    def flush():
        if not current_parts:
            return
        chunk_text_value = "\n\n".join(current_parts)
        chunks.append({
            "text": chunk_text_value,
            "start_offset": current_start,
            "page": _page_for_offset(current_start, pages),
        })

    for para in paragraphs:
        para_offset = text.find(para, search_from)
        if para_offset == -1:
            para_offset = search_from
        search_from = para_offset + len(para)

        if len(para) > chunk_size:
            # Paragrafo troppo lungo: prima svuota l'accumulo corrente,
            # poi spezza il paragrafo a finestra fissa.
            flush()
            current_parts, current_len = [], 0

            start = 0
            while start < len(para):
                end = start + chunk_size
                piece = para[start:end].strip()
                if piece:
                    chunks.append({
                        "text": piece,
                        "start_offset": para_offset + start,
                        "page": _page_for_offset(para_offset + start, pages),
                    })
                start += chunk_size - overlap
            continue

        if current_len + len(para) + 2 > chunk_size and current_parts:
            flush()
            current_parts, current_len = [], 0
            current_start = para_offset

        if not current_parts:
            current_start = para_offset

        current_parts.append(para)
        current_len += len(para) + 2

    flush()
    return chunks
