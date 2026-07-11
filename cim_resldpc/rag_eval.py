"""Extractive RAG generation proxy + hallucination-oriented metrics.

No external LLM required: answers are extracted from retrieved passages.
This measures whether retrieval quality propagates to answerability —
the minimum RAG claim before plugging a real on-device LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .datasets import Document, QAPair


_TOKEN = re.compile(r"[a-z0-9]+", re.I)


def _norm(text: str) -> str:
    return " ".join(_TOKEN.findall(text.lower()))


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_toks = tokenize(prediction)
    gold_toks = tokenize(ground_truth)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gold_toks:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if _norm(prediction) == _norm(ground_truth) else 0.0


def support_overlap(answer: str, passages: list[str]) -> float:
    """Fraction of answer tokens that appear in retrieved passages (anti-hallucination)."""
    a = set(tokenize(answer))
    if not a:
        return 1.0
    ctx = set()
    for p in passages:
        ctx.update(tokenize(p))
    return len(a & ctx) / len(a)


def hallucination_rate(answer: str, passages: list[str], threshold: float = 0.5) -> float:
    """1 if support overlap < threshold (unsupported answer), else 0."""
    return 1.0 if support_overlap(answer, passages) < threshold else 0.0


@dataclass
class RAGResult:
    qid: str
    prediction: str
    gold: str
    f1: float
    em: float
    support: float
    hallucinated: float
    hit_relevant: float


def extractive_answer(query: str, passages: list[str], max_chars: int = 180) -> str:
    """Pick the passage sentence/span most overlapping the query as the answer proxy."""
    if not passages:
        return ""
    q = set(tokenize(query))
    best = passages[0]
    best_score = -1.0
    for p in passages:
        # sentence-ish splits
        parts = re.split(r"[.;\n]", p)
        candidates = [s.strip() for s in parts if s.strip()] or [p]
        for s in candidates:
            st = set(tokenize(s))
            if not st:
                continue
            score = len(q & st) / (len(q) + 1e-8) + 0.1 * len(st & q) / (len(st) + 1e-8)
            if score > best_score:
                best_score = score
                best = s
    return best[:max_chars]


def evaluate_rag(
    queries: list[QAPair],
    doc_by_id: dict[str, Document],
    retrieve_fn,
    k: int = 3,
) -> dict[str, float]:
    """retrieve_fn(query, k) -> list with .doc_id"""
    f1s, ems, supports, halls, hits = [], [], [], [], []
    details: list[RAGResult] = []
    for q in queries:
        hits_list = retrieve_fn(q.query, k)
        passages = []
        hit_rel = 0.0
        for h in hits_list:
            did = h.doc_id if hasattr(h, "doc_id") else h["doc_id"]
            if did in doc_by_id:
                passages.append(doc_by_id[did].text)
            if did in q.relevant_doc_ids:
                hit_rel = 1.0
        pred = extractive_answer(q.query, passages)
        gold = q.answer or ""
        # If gold empty, use first relevant doc text snippet
        if not gold and q.relevant_doc_ids:
            gold = doc_by_id[q.relevant_doc_ids[0]].text[:120]
        f1 = f1_score(pred, gold)
        em = exact_match(pred, gold)
        supp = support_overlap(pred, passages)
        hall = hallucination_rate(pred, passages)
        f1s.append(f1)
        ems.append(em)
        supports.append(supp)
        halls.append(hall)
        hits.append(hit_rel)
        details.append(
            RAGResult(q.qid, pred, gold, f1, em, supp, hall, hit_rel)
        )
    n = max(len(f1s), 1)
    return {
        "rag_f1": sum(f1s) / n,
        "rag_em": sum(ems) / n,
        "rag_support": sum(supports) / n,
        "rag_hallucination_rate": sum(halls) / n,
        "rag_retrieval_hit@k": sum(hits) / n,
        "n_queries": float(len(queries)),
        "k": float(k),
    }
