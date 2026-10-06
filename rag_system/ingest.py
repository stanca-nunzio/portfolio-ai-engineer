"""
Lettura del testo da PDF e Markdown nella cartella dei documenti e chunking.
"""
import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Union

from pypdf import PdfReader

# pypdf segnala come WARNING le xref table malformate, rumore inutile nei log
logging.getLogger("pypdf").setLevel(logging.ERROR)

SUPPORTED_SUFFIXES = (".pdf", ".md", ".markdown", ".txt")
TEXT_SUFFIXES = (".md", ".markdown", ".txt")

# Scritto dentro la cartella documenti: ricorda url e fonte dei file scaricati
# dal seed, serve solo all'attribuzione nell'anteprima.
SOURCES_METADATA_FILE = "sources.json"


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ingest] errore leggendo {path}: {e}")
        return default


def _load_seed_documents(seed_file: Path) -> list[dict]:
    return _load_json(seed_file, [])


def _load_sources_metadata(folder: Path) -> dict:
    return _load_json(folder / SOURCES_METADATA_FILE, {})


def _save_sources_metadata(folder: Path, metadata: dict) -> None:
    (folder / SOURCES_METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def ensure_seed_documents(docs_dir: Union[Path, str], seed_file: Union[Path, str]) -> None:
    """Se docs_dir e' vuota scarica i documenti elencati in seed_file (documents.json)."""
    docs_dir, seed_file = Path(docs_dir), Path(seed_file)
    docs_dir.mkdir(parents=True, exist_ok=True)

    if any(p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES for p in docs_dir.rglob("*")):
        return

    seeds = _load_seed_documents(seed_file)
    if not seeds:
        return

    metadata = _load_sources_metadata(docs_dir)
    for seed in seeds:
        url = seed.get("url")
        if not url:
            continue
        filename = url.rsplit("/", 1)[-1]
        try:
            urllib.request.urlretrieve(url, docs_dir / filename)
            metadata[filename] = {"url": url, "source": seed.get("source", "")}
        except Exception as e:
            print(f"[ingest] download fallito per {url}: {e}")

    _save_sources_metadata(docs_dir, metadata)


def read_pdf(path: Path) -> list[dict]:
    """Una voce {"text", "page"} per pagina, cosi' la pagina arriva fino ai chunk."""
    reader = PdfReader(str(path))
    return [
        {"text": page.extract_text() or "", "page": n}
        for n, page in enumerate(reader.pages, start=1)
    ]


def read_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(docs_dir: Union[Path, str]) -> list[dict]:
    """
    Ritorna [{"source": nome_file, "text": testo, "pages": [...] | None}].
    "pages" esiste solo per i PDF.
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        return []

    docs = []
    for path in docs_dir.rglob("*"):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                pages = read_pdf(path)
                text = "\n".join(p["text"] for p in pages)
            elif suffix in TEXT_SUFFIXES:
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


def list_source_files(docs_dir: Union[Path, str]) -> list[dict]:
    """Metadati dei file supportati, senza leggerne il contenuto."""
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        return []

    return [
        {
            "name": p.name,
            "rel_path": str(p.relative_to(docs_dir)),
            "suffix": p.suffix.lower(),
            "size_kb": round(p.stat().st_size / 1024, 1),
        }
        for p in sorted(docs_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def preview_file(path: Path, max_chars: int = 2000) -> str:
    """Testo iniziale di un file, con l'attribuzione se il file e' stato scaricato dal seed."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        parts, total = [], 0
        for page in PdfReader(str(path)).pages:
            page_text = page.extract_text() or ""
            parts.append(page_text)
            total += len(page_text)
            if total >= max_chars:
                break
        text = "\n\n".join(parts)
    elif suffix in TEXT_SUFFIXES:
        text = read_markdown(path)
    else:
        return ""

    text = text.strip()
    preview = text[:max_chars] + "\n\n[...anteprima troncata...]" if len(text) > max_chars else text

    info = _load_sources_metadata(path.parent).get(path.name)
    if info:
        preview += f"\n\n[Fonte: {info['url']} - (c) {info['source']}]"
    return preview


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _page_for_offset(offset: int, pages: list[dict] | None) -> int | None:
    """Pagina PDF che contiene l'offset nel testo concatenato (pagine unite con '\\n')."""
    if not pages:
        return None
    cursor = 0
    for p in pages:
        cursor += len(p["text"]) + 1
        if offset < cursor:
            return p["page"]
    return pages[-1]["page"]


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 150,
    pages: list[dict] | None = None,
) -> list[dict]:
    """
    Unisce paragrafi consecutivi fino a chunk_size caratteri; un paragrafo piu'
    lungo di chunk_size viene spezzato a finestra fissa con sovrapposizione.
    Ritorna [{"text", "start_offset", "page"}].
    """
    text = text.strip()
    chunks = []
    parts: list[str] = []
    parts_len = 0
    parts_start = 0
    search_from = 0

    def flush():
        if parts:
            chunks.append({
                "text": "\n\n".join(parts),
                "start_offset": parts_start,
                "page": _page_for_offset(parts_start, pages),
            })

    for para in _split_paragraphs(text):
        offset = text.find(para, search_from)
        if offset == -1:
            offset = search_from
        search_from = offset + len(para)

        if len(para) > chunk_size:
            flush()
            parts, parts_len = [], 0
            step = max(1, chunk_size - overlap)
            for start in range(0, len(para), step):
                piece = para[start:start + chunk_size].strip()
                if piece:
                    chunks.append({
                        "text": piece,
                        "start_offset": offset + start,
                        "page": _page_for_offset(offset + start, pages),
                    })
            continue

        if parts and parts_len + len(para) + 2 > chunk_size:
            flush()
            parts, parts_len = [], 0

        if not parts:
            parts_start = offset
        parts.append(para)
        parts_len += len(para) + 2

    flush()
    return chunks


def build_chunks(docs_dir: Union[Path, str], chunk_size: int = 1200, overlap: int = 150) -> list[dict]:
    """
    Legge tutti i documenti e li spezza. Ogni chunk e' {"text", "source", "chunk_id", "page"},
    con chunk_id progressivo per documento: e' il formato atteso da index_chunks().
    """
    chunks = []
    for doc in load_documents(docs_dir):
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap, pages=doc.get("pages"))
        for chunk_id, piece in enumerate(pieces):
            chunks.append({
                "text": piece["text"],
                "source": doc["source"],
                "chunk_id": chunk_id,
                "page": piece["page"],
            })
    return chunks