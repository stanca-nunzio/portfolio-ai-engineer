"""
Lettura ed estrazione testo da file PDF e Markdown nella cartella data/.
Restituisce una lista di documenti grezzi con metadati minimi
"""
from typing import Union
import os
import re
import json
import logging
import urllib.request
from pathlib import Path
from pypdf import PdfReader

# pypdf logga a livello WARNING quando incontra xref table malformate
# ("Ignoring wrong pointing object...")
logging.getLogger("pypdf").setLevel(logging.ERROR)

SUPPORTED_SUFFIXES = (".pdf", ".md", ".markdown", ".txt")

# File di stato scritto dal codice dentro data/: tiene traccia di quali file
# sono stati scaricati automaticamente e da dove, usato solo per mostrare
# l'attribuzione della fonte in preview_file.
SOURCES_METADATA_FILE = "sources.json"


def _load_seed_documents(seed_file_path) -> list[dict]:
    if not seed_file_path.exists():
        return []
    try:
        return json.loads(seed_file_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ingest] errore leggendo {seed_file_path}: {e}")
        return []


def _load_sources_metadata(dir: Path) -> dict:
    meta_path = dir / SOURCES_METADATA_FILE
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_sources_metadata(dir: Path, metadata: dict) -> None:
    meta_path = dir / SOURCES_METADATA_FILE
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_seed_documents(dir: Union[Path, str], seed_file: Union[Path, str]) -> None:
    """
    Se dir non contiene ancora nessun file supportato, scarica i documenti
    elencati in seed_file (documents.json) e ne registra url + fonte in sources.json
    (dentro dir), così la preview può mostrare l'attribuzione corretta.
    """
    if isinstance(dir, str):
        dir = Path(dir)

    if isinstance(seed_file, str):
        seed_file = Path(seed_file)

    dir.mkdir(parents=True, exist_ok=True)

    already_has_files = any(
        p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES for p in dir.rglob("*")
    )
    if already_has_files:
        return

    seed_documents = _load_seed_documents(seed_file)
    if not seed_documents:
        return

    metadata = _load_sources_metadata(dir)

    for seed in seed_documents:
        url = seed.get("url")
        source_label = seed.get("source", "")
        if not url:
            continue

        filename = url.rsplit("/", 1)[-1]
        dest = dir / filename
        try:
            urllib.request.urlretrieve(url, dest)
            metadata[filename] = {"url": url, "source": source_label}
        except Exception as e:
            print(f"[ingest] impossibile scaricare {url}: {e}")

    _save_sources_metadata(dir, metadata)

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


def list_source_files(dir: Union[Path, str]) -> list[dict]:
    """
    Scansiona dir (ricorsivamente) e ritorna solo i metadati dei file
    supportati, senza estrarne il contenuto: {"name", "rel_path", "suffix", "size_kb"}.
    Pensata per popolare la tab "Files" senza dover rileggere/parsare tutto.
    """
    if isinstance(dir, str):
        dir = Path(dir)

    files = []
    if not dir.exists():
        return files

    for path in sorted(dir.rglob("*")):
        if path.is_dir() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        files.append({
            "name": path.name,
            "rel_path": str(path.relative_to(dir)),
            "suffix": path.suffix.lower(),
            "size_kb": round(path.stat().st_size / 1024, 1),
        })
    return files


def preview_file(path: Path, max_chars: int = 2000) -> str:
    """
    Estrae il testo di un singolo file per sola anteprima (no chunking/indicizzazione).
    Per i PDF concatena le pagine finché non si supera max_chars.
    Se il file è tra quelli scaricati automaticamente (vedi sources.json),
    aggiunge in coda l'attribuzione della fonte.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        parts = []
        total_len = 0
        for page in reader.pages:
            page_text = page.extract_text() or ""
            parts.append(page_text)
            total_len += len(page_text)
            if total_len >= max_chars:
                break
        text = "\n\n".join(parts)
    elif suffix in (".md", ".markdown", ".txt"):
        text = read_markdown(path)
    else:
        return ""

    text = text.strip()
    if len(text) > max_chars:
        preview = text[:max_chars] + "\n\n[...anteprima troncata...]"
    else:
        preview = text

    metadata = _load_sources_metadata(path.parent)
    source_info = metadata.get(path.name)
    if source_info:
        preview += f"\n\n[Fonte: {source_info['url']} - © {source_info['source']}]"

    return preview


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
