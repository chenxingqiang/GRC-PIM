"""v1.4 tests: Flan generator optional + scaled eval smoke."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cim_resldpc.datasets import Document, QAPair
from cim_resldpc.rag_generators import build_generator, evaluate_rag_with_generator


def test_extractive_generator_wrapper():
    gen = build_generator("extractive")
    assert gen.name == "extractive"
    ans = gen.generate("capital of France", ["Paris is the capital of France."])
    assert "paris" in ans.lower() or "france" in ans.lower()


def test_flan_generator_optional():
    pytest.importorskip("transformers")
    gen = build_generator("flan", model_name="google/flan-t5-small", max_new_tokens=8)
    ans = gen.generate(
        "What is the capital of France?",
        ["Paris is the capital of France on the Seine."],
    )
    assert isinstance(ans, str) and len(ans) > 0
    assert "paris" in ans.lower()


def test_evaluate_rag_with_flan_smoke():
    pytest.importorskip("transformers")
    docs = [
        Document("d1", "Paris is the capital of France.", ["geo"]),
        Document("d2", "Berlin is the capital of Germany.", ["geo"]),
    ]
    queries = [QAPair("q1", "What is the capital of France?", ["d1"], "Paris")]
    gen = build_generator("flan", max_new_tokens=8)

    class _R:
        def retrieve(self, query, k=3):
            return [type("H", (), {"doc_id": "d1"})()]

    metrics = evaluate_rag_with_generator(
        queries, {d.doc_id: d for d in docs}, _R().retrieve, gen, k=1,
    )
    assert metrics["generator"].startswith("flan:")
    assert 0.0 <= metrics["rag_f1"] <= 1.0
    assert metrics["rag_support"] >= 0.0


def test_scaled_eval_smoke_offline(tmp_path: Path):
    """Use bundled mini corpus path via run_hf_eval-compatible small call."""
    pytest.importorskip("sentence_transformers")
    from cim_resldpc.experiments.run_scaled_eval import eval_corpus
    from cim_resldpc.ir_benchmarks import load_msmarco_mini

    docs, queries = load_msmarco_mini()
    result = eval_corpus(
        "msmarco_mini",
        docs,
        queries[:6],
        encoder_kind="st",
        nor_name="nor_typical",
        use_gemv=False,
        seed=0,
        st_model="sentence-transformers/all-MiniLM-L6-v2",
        cim_dim=64,
        rag_generator="extractive",
    )
    assert "cim_ldpc_bp" in result["dense_only"]
    assert result["robustness"]["cim_dim"] == 64


def test_n200_cache_if_present():
    p = Path("cim_resldpc/data/cache/msmarco_hf_n200.json")
    if not p.exists():
        pytest.skip("n200 cache missing")
    raw = json.loads(p.read_text())
    assert len(raw["queries"]) == 200
    assert len(raw["documents"]) > 500
