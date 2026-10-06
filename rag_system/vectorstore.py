"""
Vector store su Qdrant embedded (client in-process, nessun server esterno).
"""
import os
import uuid

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import CrossEncoder, SentenceTransformer

COLLECTION_NAME = "portfolio_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CANDIDATES = 15

_embedder = None
_client = None
_reranker = None


def _get_qdrant_path() -> str:
    """st.secrets (relativo a ROOT_DIR), poi variabile d'ambiente, poi 'qdrant_storage'."""
    try:
        if "QDRANT_PATH" in st.secrets:
            return os.path.join(st.session_state["ROOT_DIR"], st.secrets["QDRANT_PATH"])
    except Exception:
        pass
    return os.environ.get("QDRANT_PATH", "qdrant_storage")


def _how_many_vectordb_candidates() -> int:
    """Numero di candidati da passare al cross-encoder: st.secrets, poi env, poi 15."""
    try:
        if "RERANK_CANDIDATES" in st.secrets:
            return int(st.secrets["RERANK_CANDIDATES"])
    except Exception:
        pass
    return int(os.environ.get("RERANK_CANDIDATES", DEFAULT_CANDIDATES))


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
        _client = QdrantClient(path=_get_qdrant_path())
    return _client


def ensure_collection(reset: bool = False) -> None:
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]

    if reset and COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        existing.remove(COLLECTION_NAME)

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=get_embedder().get_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
        )


def index_chunks(chunks: list[dict]) -> None:
    """
    chunks: [{"text", "source", "chunk_id", "page"}]. Il testo resta nel payload,
    cosi' il retrieval non deve riaprire i file.
    L'id e' derivato da source+chunk_id: un riavvio sovrascrive i punti invece di duplicarli.
    I chunk di documenti rimossi o riscritti restano pero' in collezione: per ripulire
    serve ensure_collection(reset=True).
    """
    if not chunks:
        return

    ensure_collection()
    vectors = get_embedder().encode([c["text"] for c in chunks], show_progress_bar=False).tolist()

    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c['source']}::{c['chunk_id']}")),
            vector=vector,
            payload={
                "text": c["text"],
                "source": c["source"],
                "chunk_id": c["chunk_id"],
                "page": c.get("page"),
            },
        )
        for c, vector in zip(chunks, vectors)
    ]
    get_client().upsert(collection_name=COLLECTION_NAME, points=points)


def search(query: str, top_k: int = 4, use_reranker: bool = True) -> list[dict]:
    ensure_collection()
    query_vector = get_embedder().encode([query])[0].tolist()
    limit = max(_how_many_vectordb_candidates(), top_k) if use_reranker else top_k

    response = get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    )
    hits = [
        {
            "text": p.payload["text"],
            "source": p.payload["source"],
            "page": p.payload.get("page"),
            "chunk_id": p.payload.get("chunk_id"),
            "score": p.score,
        }
        for p in response.points
    ]

    if not use_reranker or not hits:
        return hits[:top_k]

    scores = get_reranker().predict([[query, h["text"]] for h in hits])
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)

    hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    return hits[:top_k]


def collection_count() -> int:
    ensure_collection()
    return get_client().get_collection(COLLECTION_NAME).points_count or 0