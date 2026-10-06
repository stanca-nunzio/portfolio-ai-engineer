"""
Valuta il retrieval su un set di domande con chunk di riferimento, prima con la
sola similarita' coseno di Qdrant e poi con il cross-encoder.

CSV in ingresso, una riga per domanda, colonne:
    question, source, chunk_ids
  - source: nome del file come in chunks_index.csv
  - chunk_ids: uno o piu' id separati da spazio o punto e virgola (es. "12 13")

Da riga di comando indicizza i documenti in un Qdrant temporaneo con lo stesso
chunking dell'app, quindi si puo' lanciare anche con l'app Streamlit aperta.
parse_questions e evaluate sono usate anche dal tab "Valutazione" di pages/1_RAG.py.

Uso, dalla root del progetto:
    python -m rag_system.evaluate_retrieval questions.csv
    python -m rag_system.evaluate_retrieval questions.csv --ks 1,4,10 --out risultati.csv
"""
import argparse
import csv
import io
import math
import re
import tempfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from rag_system.ingest import build_chunks

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KS = (1, 3, 4, 5, 10)


def parse_questions(text: str) -> list[dict]:
    text = text.lstrip("\ufeff").strip()
    if not text:
        raise ValueError("CSV vuoto")

    header = text.splitlines()[0]
    delimiter = max(",;\t", key=header.count)
    rows = csv.DictReader(io.StringIO(text), delimiter=delimiter, restkey="_extra")

    questions = []
    for n, row in enumerate(rows, start=2):
        # "12,13" non quotato in un CSV con virgola finisce in _extra: appartiene a chunk_ids
        extra = row.pop("_extra", [])
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        if extra:
            row["chunk_ids"] = " ".join([row.get("chunk_ids", "")] + [e.strip() for e in extra])
        if not row.get("question"):
            continue

        try:
            ids = [int(x) for x in re.split(r"[\s,;|]+", row.get("chunk_ids", "")) if x]
        except ValueError:
            raise ValueError(f"riga {n}: chunk_ids deve contenere solo numeri")
        if not row.get("source") or not ids:
            raise ValueError(f"riga {n}: servono source e chunk_ids")

        # accetta anche il nome del PDF annotato (<nome>_chunks.pdf)
        source = re.sub(r"_chunks(\.pdf)$", r"\1", row["source"], flags=re.IGNORECASE)
        questions.append({"question": row["question"], "relevant": {(source, i) for i in ids}})

    if not questions:
        raise ValueError("nessuna domanda trovata: servono le colonne question, source, chunk_ids")
    return questions


def check_known(questions: list[dict], known: set) -> None:
    for q in questions:
        missing = q["relevant"] - known
        if missing:
            sources = sorted({s for s, _ in known})
            raise ValueError(
                f"chunk inesistenti {sorted(missing)} per la domanda '{q['question']}'. Sorgenti: {sources}"
            )


def indexed_chunk_keys() -> set[tuple]:
    """(source, chunk_id) di ogni chunk presente nella collezione Qdrant corrente."""
    import rag_system.vectorstore as vs

    vs.ensure_collection()
    client = vs.get_client()
    keys, offset = set(), None
    while True:
        points, offset = client.scroll(
            vs.COLLECTION_NAME, limit=500, offset=offset,
            with_payload=["source", "chunk_id"], with_vectors=False,
        )
        keys.update((p.payload["source"], p.payload["chunk_id"]) for p in points)
        if offset is None:
            return keys


def question_metrics(ranked: list[tuple], relevant: set, ks: list[int]) -> dict:
    first_hit = next((r for r, key in enumerate(ranked, start=1) if key in relevant), None)
    out = {"mrr": 1 / first_hit if first_hit else 0.0}
    for k in ks:
        top = ranked[:k]
        found = sum(1 for key in top if key in relevant)
        dcg = sum(1 / math.log2(i + 2) for i, key in enumerate(top) if key in relevant)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        out[f"hit@{k}"] = float(found > 0)
        out[f"recall@{k}"] = found / len(relevant)
        out[f"precision@{k}"] = found / k
        out[f"ndcg@{k}"] = dcg / idcg
    return out


def average(per_question: list[dict]) -> dict:
    return {m: sum(q[m] for q in per_question) / len(per_question) for m in per_question[0]}


def evaluate(questions: list[dict], search, ks) -> dict:
    """
    search(question, use_reranker) -> hit con "source" e "chunk_id",
    gia' limitati a max(ks) risultati.
    """
    ks = sorted(ks)
    plain_metrics, rerank_metrics, rows = [], [], []

    for q in questions:
        plain = [(h["source"], h["chunk_id"]) for h in search(q["question"], False)]
        rerank = [(h["source"], h["chunk_id"]) for h in search(q["question"], True)]
        plain_metrics.append(question_metrics(plain, q["relevant"], ks))
        rerank_metrics.append(question_metrics(rerank, q["relevant"], ks))

        def rank_of(ranked, relevant=q["relevant"]):
            return next((r for r, key in enumerate(ranked, start=1) if key in relevant), None)

        def ids(ranked):
            return " ".join(str(c) for _, c in ranked)

        rows.append({
            "question": q["question"],
            "relevant": " ".join(str(c) for _, c in sorted(q["relevant"])),
            "rank_senza_rerank": rank_of(plain),
            "rank_con_rerank": rank_of(rerank),
            "top_senza_rerank": ids(plain),
            "top_con_rerank": ids(rerank),
        })

    return {"plain": average(plain_metrics), "reranked": average(rerank_metrics), "rows": rows, "ks": ks}


def metrics_rows(plain: dict, reranked: dict, ks) -> list[tuple]:
    names = ["mrr"] + [f"{m}@{k}" for k in ks for m in ("hit", "recall", "precision", "ndcg")]
    return [(n, plain[n], reranked[n], reranked[n] - plain[n]) for n in names]


def print_table(plain: dict, reranked: dict, ks) -> None:
    print(f"\n{'metrica':<16}{'senza re-rank':>15}{'con re-rank':>15}{'delta':>10}")
    print("-" * 56)
    for name, a, b, delta in metrics_rows(plain, reranked, ks):
        print(f"{name:<16}{a:>15.3f}{b:>15.3f}{delta:>+10.3f}")


def _read_secrets() -> dict:
    path = ROOT / ".streamlit" / "secrets.toml"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def main() -> None:
    secrets = _read_secrets()

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("--docs-dir", type=Path, default=ROOT / secrets.get("DOCS_DIR", "data"))
    ap.add_argument("--chunk-size", type=int, default=int(secrets.get("CHUNK_SIZE", 1200)))
    ap.add_argument("--chunk-overlap", type=int, default=int(secrets.get("CHUNK_OVERLAP", 150)))
    ap.add_argument("--candidates", type=int, default=int(secrets.get("RERANK_CANDIDATES", 15)),
                    help="candidati prelevati da Qdrant prima del re-ranking")
    ap.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)))
    ap.add_argument("--out", type=Path, default=ROOT / "rag_eval_results.csv")
    args = ap.parse_args()

    ks = sorted({int(k) for k in args.ks.split(",")})
    max_k = max(ks)
    if args.candidates < max_k:
        print(f"WARN: candidates={args.candidates} < max k={max_k}, il re-ranking vedra' meno di {max_k} chunk")

    questions = parse_questions(args.csv_path.read_text(encoding="utf-8-sig"))
    chunks = build_chunks(args.docs_dir, args.chunk_size, args.chunk_overlap)
    check_known(questions, {(c["source"], c["chunk_id"]) for c in chunks})

    # import qui: trascinano torch e sentence-transformers
    from qdrant_client import QdrantClient
    import rag_system.vectorstore as vs

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        vs._client = QdrantClient(path=tmp)
        vs._how_many_vectordb_candidates = lambda: args.candidates
        try:
            print(f"Indicizzo {len(chunks)} chunk...")
            vs.index_chunks(chunks)
            result = evaluate(
                questions,
                lambda q, rerank: vs.search(q, top_k=max_k, use_reranker=rerank),
                ks,
            )
        finally:
            vs._client.close()

    print(f"{len(questions)} domande, {len(chunks)} chunk")
    print_table(result["plain"], result["reranked"], ks)

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(f"\nDettaglio per domanda: {args.out}")


if __name__ == "__main__":
    main()