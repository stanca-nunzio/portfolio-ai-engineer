"""
Vector store basato su Qdrant embedded (client Python in-process, NO server esterno)
"""
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, CrossEncoder

import streamlit as st

COLLECTION_NAME = "portfolio_docs"
# Modelli leggeri, run su CPU
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_embedder = None
_client = None
_reranker = None


def _get_qdrant_path() -> str:
    """
    Cerca il path prima nei secrets di Streamlit Cloud (st.secrets),
    poi come variabile d'ambiente, poi fallback su un default locale.
    """
    try:
        if "QDRANT_PATH" in st.secrets:
            return os.path.join(st.session_state["ROOT_DIR"], st.secrets["QDRANT_PATH"])
    except Exception:
        pass
    return os.environ.get("QDRANT_PATH", "qdrant_storage")

def _how_many_vectordb_candidates() -> int:
    """
    Cerca il path prima nei secrets di Streamlit Cloud (st.secrets),
    poi come variabile d'ambiente, poi fallback su un default locale.
    """
    try:
        if "RERANK_CANDIDATES" in st.secrets:
            return st.secrets["RERANK_CANDIDATES"]
    except Exception:
        pass
    return os.environ.get("RERANK_CANDIDATES", 15)


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        # path= -> modalità embedded, salva su disco locale, nessun server da avviare
        _client = QdrantClient(path=_get_qdrant_path())
    return _client


def ensure_collection(reset: bool = False):
    client = get_client()
    embedder = get_embedder()
    dim = embedder.get_embedding_dimension()

    existing = [c.name for c in client.get_collections().collections]
    if reset and COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        existing.remove(COLLECTION_NAME)

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

def index_chunks(chunks_with_meta: list[dict]):
    """
    chunks_with_meta: lista di {"text": str, "source": str, "chunk_id": int}
    Calcola embedding e fa upsert su Qdrant, salvando il testo come payload
    (così al retrieval abbiamo già il contenuto, senza riaprire i file originali).
    """
    if not chunks_with_meta:
        return

    ensure_collection()
    client = get_client()
    embedder = get_embedder()

    texts = [c["text"] for c in chunks_with_meta]
    vectors = embedder.encode(texts, show_progress_bar=False).tolist()

    points = []
    for chunk, vector in zip(chunks_with_meta, vectors):
        # id deterministico: stesso source+chunk_id -> stesso uuid.
        # Cosi' l'upsert sovrascrive il punto esistente invece di aggiungerne
        # uno nuovo ad ogni riavvio (il namespace UUID e' arbitrario ma fisso).
        point_key = f"{chunk['source']}::{chunk['chunk_id']}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, point_key))

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk.get("page"),
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)


def search(query: str, top_k: int = 4, use_reranker: bool = True) -> list[dict]:
    ensure_collection()
    client = get_client()
    embedder = get_embedder()

    query_vector = embedder.encode([query])[0].tolist()
    fetch_limit = _how_many_vectordb_candidates() if use_reranker else top_k
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=fetch_limit,
    )

    candidates = [
        {
            "text": p.payload["text"],
            "source": p.payload["source"],
            "page": p.payload.get("page"),
            "score": p.score,
        }
        for p in response.points
    ]

    if not use_reranker or not candidates:
        return candidates[:top_k]

    reranker = get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)

    for c, rerank_score in zip(candidates, rerank_scores):
        c["rerank_score"] = float(rerank_score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]


def collection_count() -> int:
    ensure_collection()
    client = get_client()
    info = client.get_collection(COLLECTION_NAME)
    return info.points_count or 0
