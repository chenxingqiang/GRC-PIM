"""Build and cache HuggingFace IR subsets for offline re-runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cim_resldpc.datasets import Document, QAPair

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"


def _write_fixture(path: Path, name: str, docs: list[Document], queries: list[QAPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "documents": [
            {"doc_id": d.doc_id, "text": d.text, "labels": d.labels} for d in docs
        ],
        "queries": [
            {
                "qid": q.qid,
                "query": q.query,
                "relevant_doc_ids": q.relevant_doc_ids,
                "answer": q.answer,
            }
            for q in queries
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def build_msmarco_subset(n_queries: int = 64, seed: int = 0) -> tuple[list[Document], list[QAPair]]:
    from datasets import load_dataset

    ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
    # Deterministic stride sample over validation
    step = max(1, len(ds) // max(n_queries * 2, 1))
    idxs = list(range(0, len(ds), step))[: n_queries * 3]

    docs: list[Document] = []
    queries: list[QAPair] = []
    seen: set[str] = set()
    for i in idxs:
        if len(queries) >= n_queries:
            break
        row = ds[int(i)]
        qtext = str(row["query"]).strip()
        pas = row["passages"]
        texts = list(pas["passage_text"])
        selected = list(pas["is_selected"])
        answers = row.get("answers") or []
        answer = str(answers[0]) if answers else ""
        rel: list[str] = []
        for j, text in enumerate(texts):
            text = str(text).strip()
            if not text:
                continue
            did = f"ms_{row['query_id']}_{j}"
            if did not in seen:
                docs.append(Document(did, text, labels=["msmarco"]))
                seen.add(did)
            if j < len(selected) and int(selected[j]) == 1:
                rel.append(did)
        if not rel:
            # skip queries without a selected passage
            continue
        queries.append(
            QAPair(
                qid=f"ms_q_{row['query_id']}",
                query=qtext,
                relevant_doc_ids=rel,
                answer=answer,
            )
        )
    return docs, queries


def build_nq_subset(n_queries: int = 64, n_distractors: int = 8) -> tuple[list[Document], list[QAPair]]:
    """NQ-open has Q+A only; build a harder pseudo-passage corpus.

    Relevant doc is an encyclopedia-style fact that contains the answer but
    does **not** echo the question text (avoids trivial lexical shortcuts).
    Distractors are other answers from the pool.
    """
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/nq_open", split="validation")
    step = max(1, len(ds) // max(n_queries * 2, 1))
    idxs = list(range(0, len(ds), step))[:n_queries]

    docs: list[Document] = []
    queries: list[QAPair] = []
    all_answers = []
    for i in range(min(len(ds), 800)):
        ans = ds[i]["answer"]
        if isinstance(ans, list):
            ans = ans[0] if ans else ""
        if ans:
            all_answers.append(str(ans))

    for i in idxs:
        row = ds[int(i)]
        q = str(row["question"]).strip()
        ans = row["answer"]
        if isinstance(ans, list):
            ans = ans[0] if ans else ""
        ans = str(ans).strip()
        if not q or not ans:
            continue
        did = f"nq_{i}"
        text = (
            f"{ans} is a known entity or fact referenced in open-domain QA resources. "
            f"Additional context: {ans}."
        )
        docs.append(Document(did, text, labels=["nq"]))
        for k, other in enumerate(all_answers[i + 1 : i + 1 + n_distractors]):
            if other == ans:
                continue
            docs.append(
                Document(
                    f"nq_{i}_d{k}",
                    f"{other} appears in unrelated Wikipedia-style notes.",
                    labels=["nq_distractor"],
                )
            )
        queries.append(
            QAPair(qid=f"nq_q_{i}", query=q, relevant_doc_ids=[did], answer=ans)
        )
    return docs, queries


def load_cached_or_build(
    name: str,
    n_queries: int = 64,
    force: bool = False,
) -> tuple[list[Document], list[QAPair], Path]:
    path = CACHE_DIR / f"{name}_n{n_queries}.json"
    if path.exists() and not force:
        raw = json.loads(path.read_text())
        docs = [Document(**d) for d in raw["documents"]]
        queries = [QAPair(**q) for q in raw["queries"]]
        return docs, queries, path
    if name == "msmarco_hf":
        docs, queries = build_msmarco_subset(n_queries=n_queries)
    elif name == "nq_hf":
        docs, queries = build_nq_subset(n_queries=n_queries)
    else:
        raise ValueError(name)
    _write_fixture(path, name, docs, queries)
    return docs, queries, path


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_queries", type=int, default=64)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--corpus",
        default="all",
        choices=["all", "msmarco_hf", "nq_hf"],
        help="Which HF subset(s) to build (default: both).",
    )
    args = p.parse_args(argv)
    names = ("msmarco_hf", "nq_hf") if args.corpus == "all" else (args.corpus,)
    for name in names:
        docs, queries, path = load_cached_or_build(name, n_queries=args.n_queries, force=args.force)
        print(f"{name}: {len(docs)} docs, {len(queries)} queries -> {path}")


if __name__ == "__main__":
    main()
