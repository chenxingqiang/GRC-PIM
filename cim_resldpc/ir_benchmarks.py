"""Bundled IR fixtures + optional HuggingFace MSMARCO/NQ loaders.

Offline-first: ships a mini MSMARCO-style and NQ-style JSON under data/.
Optional download when `datasets` is installed and network is allowed.
"""
from __future__ import annotations

import json
from pathlib import Path

from .datasets import Document, QAPair

DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_fixture(name: str) -> tuple[list[Document], list[QAPair]]:
    path = DATA_DIR / name
    raw = json.loads(path.read_text())
    docs = [Document(doc_id=d["doc_id"], text=d["text"], labels=d.get("labels", [])) for d in raw["documents"]]
    queries = [
        QAPair(
            qid=q["qid"],
            query=q["query"],
            relevant_doc_ids=list(q["relevant_doc_ids"]),
            answer=q.get("answer", ""),
        )
        for q in raw["queries"]
    ]
    return docs, queries


def load_msmarco_mini() -> tuple[list[Document], list[QAPair]]:
    """Passage-ranking mini fixture (MSMARCO-style schema)."""
    return _load_fixture("msmarco_mini.json")


def load_nq_mini() -> tuple[list[Document], list[QAPair]]:
    """Open-domain QA mini fixture (Natural Questions-style schema)."""
    return _load_fixture("nq_mini.json")


def try_load_hf_msmarco(n_queries: int = 32, seed: int = 0) -> tuple[list[Document], list[QAPair]] | None:
    """Best-effort HuggingFace MSMARCO subset; returns None if unavailable."""
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    try:
        ds = load_dataset("ms_marco", "v1.1", split="validation")
    except Exception:
        return None
    rng_idx = list(range(min(len(ds), max(n_queries * 4, n_queries))))
    docs: list[Document] = []
    queries: list[QAPair] = []
    doc_seen: set[str] = set()
    for i, row in enumerate(ds.select(rng_idx)):
        if len(queries) >= n_queries:
            break
        q = row.get("query") or row.get("query_text")
        passages = row.get("passages") or {}
        texts = passages.get("passage_text") or passages.get("text") or []
        is_sel = passages.get("is_selected") or []
        if not q or not texts:
            continue
        rel = []
        for j, text in enumerate(texts):
            did = f"hf_ms_{i}_{j}"
            if did not in doc_seen:
                docs.append(Document(did, str(text), labels=["msmarco"]))
                doc_seen.add(did)
            if j < len(is_sel) and int(is_sel[j]) == 1:
                rel.append(did)
        if not rel and texts:
            rel = [f"hf_ms_{i}_0"]
        queries.append(QAPair(qid=f"hf_ms_q_{i}", query=str(q), relevant_doc_ids=rel, answer=""))
    if not queries:
        return None
    return docs, queries


def try_load_hf_nq(n_queries: int = 32) -> tuple[list[Document], list[QAPair]] | None:
    """Best-effort HF NQ subset; returns None if unavailable."""
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    try:
        ds = load_dataset("nq_open", split="validation")
    except Exception:
        return None
    docs: list[Document] = []
    queries: list[QAPair] = []
    for i, row in enumerate(ds):
        if len(queries) >= n_queries:
            break
        q = row.get("question")
        answers = row.get("answer") or row.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]
        if not q or not answers:
            continue
        # NQ-open has no passages; synthesize a pseudo-doc from the answer
        did = f"hf_nq_{i}"
        ans = str(answers[0])
        docs.append(Document(did, f"Fact: {ans}. Related context for: {q}", labels=["nq"]))
        queries.append(QAPair(qid=f"hf_nq_q_{i}", query=str(q), relevant_doc_ids=[did], answer=ans))
    return (docs, queries) if queries else None
