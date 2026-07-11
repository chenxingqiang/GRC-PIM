"""Pluggable RAG answer generators: extractive (default) and Flan-T5 (optional)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .datasets import Document, QAPair
from .rag_eval import (
    RAGResult,
    exact_match,
    extractive_answer,
    f1_score,
    hallucination_rate,
    support_overlap,
)


GeneratorFn = Callable[[str, list[str]], str]


@dataclass
class GeneratorInfo:
    name: str
    generate: GeneratorFn


def extractive_generator() -> GeneratorInfo:
    return GeneratorInfo(name="extractive", generate=extractive_answer)


def flan_t5_generator(
    model_name: str = "google/flan-t5-small",
    max_new_tokens: int = 32,
    device: str = "cpu",
) -> GeneratorInfo:
    """Seq2seq generative RAG (requires transformers)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    def generate(query: str, passages: list[str]) -> str:
        ctx = "\n".join(f"- {p}" for p in passages[:3])
        prompt = (
            "Answer the question using only the context. "
            "If the context is insufficient, say unknown.\n"
            f"Context:\n{ctx}\n"
            f"Question: {query}\n"
            "Answer:"
        )
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new_tokens)
        return tok.decode(out[0], skip_special_tokens=True).strip()

    return GeneratorInfo(name=f"flan:{model_name}", generate=generate)


def build_generator(kind: str = "extractive", **kwargs) -> GeneratorInfo:
    kind = kind.lower()
    if kind in {"extractive", "ext"}:
        return extractive_generator()
    if kind in {"flan", "flan-t5", "t5"}:
        return flan_t5_generator(**kwargs)
    raise ValueError(f"unknown generator kind={kind!r}")


def evaluate_rag_with_generator(
    queries: list[QAPair],
    doc_by_id: dict[str, Document],
    retrieve_fn,
    generator: GeneratorInfo,
    k: int = 3,
) -> dict[str, float]:
    f1s, ems, supports, halls, hits = [], [], [], [], []
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
        pred = generator.generate(q.query, passages)
        gold = q.answer or ""
        if not gold and q.relevant_doc_ids and q.relevant_doc_ids[0] in doc_by_id:
            gold = doc_by_id[q.relevant_doc_ids[0]].text[:120]
        f1s.append(f1_score(pred, gold))
        ems.append(exact_match(pred, gold))
        supports.append(support_overlap(pred, passages))
        halls.append(hallucination_rate(pred, passages))
        hits.append(hit_rel)
    n = max(len(f1s), 1)
    return {
        "generator": generator.name,
        "rag_f1": sum(f1s) / n,
        "rag_em": sum(ems) / n,
        "rag_support": sum(supports) / n,
        "rag_hallucination_rate": sum(halls) / n,
        "rag_retrieval_hit@k": sum(hits) / n,
        "n_queries": float(len(queries)),
        "k": float(k),
    }
