"""
Vector store basato su Qdrant embedded (client Python in-process, NO server esterno).
Persiste su disco locale nel container -> non serve alcun servizio esterno.
"""
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "portfolio_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_embedder = None
_client = None


def _get_qdrant_path() -> str:
    """
    Cerca il path prima nei secrets di Streamlit Cloud (st.secrets),
    poi come variabile d'ambiente, poi fallback su un default locale.
    Stesso pattern usato in llm.py per GOOGLE_API_KEY.
    """
    try:
        import streamlit as st
        if "QDRANT_PATH" in st.secrets:
            return os.path.join(st.session_state["ROOT_DIR"], st.secrets["QDRANT_PATH"])
    except Exception:
        pass
    return os.environ.get("QDRANT_PATH", "qdrant_storage")


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


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
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)


def search(query: str, top_k: int = 4) -> list[dict]:
    ensure_collection()
    client = get_client()
    embedder = get_embedder()

    query_vector = embedder.encode([query])[0].tolist()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )

    return [
        {
            "text": p.payload["text"],
            "source": p.payload["source"],
            "score": p.score,
        }
        for p in response.points
    ]


def collection_count() -> int:
    ensure_collection()
    client = get_client()
    info = client.get_collection(COLLECTION_NAME)
    return info.points_count or 0
