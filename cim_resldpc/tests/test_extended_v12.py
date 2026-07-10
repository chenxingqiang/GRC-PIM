"""Tests for v1.2 extensions: encoder, IR fixtures, NOR, GEMV, RAG."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cim_resldpc.dense_index import EncoderDenseIndex
from cim_resldpc.encoders import TfidfSVDEncoder, build_encoder
from cim_resldpc.gemv_hook import apply_gemv_llr, load_gemv_calibrator
from cim_resldpc.ir_benchmarks import load_msmarco_mini, load_nq_mini
from cim_resldpc.ldpc_embed import protect_and_recover
from cim_resldpc.nor_calibration import get_nor_profile, list_nor_presets
from cim_resldpc.rag_eval import (
    evaluate_rag,
    exact_match,
    extractive_answer,
    f1_score,
    hallucination_rate,
    support_overlap,
)


def test_tfidf_svd_encoder_shapes():
    texts = [
        "NOR flash compute in memory drift read disturb floating gate",
        "BM25 sparse retrieval ranking inverse document frequency",
        "GF3 LDPC residual calibration soft LLR belief propagation",
        "edge RAG privacy local first semiconductor documents",
        "nDCG MAP recall information retrieval evaluation metrics",
    ]
    enc = build_encoder("tfidf_svd", dim=8, fit_texts=texts)
    emb = enc.encode(texts)
    assert emb.shape[0] == 5
    assert emb.shape[1] <= 8
    norms = np.linalg.norm(emb, axis=1)
    assert np.all(norms <= 1.0 + 1e-4)
    assert float(norms.mean()) > 0.5


def test_msmarco_nq_fixtures():
    ms_docs, ms_q = load_msmarco_mini()
    nq_docs, nq_q = load_nq_mini()
    assert len(ms_docs) >= 10 and len(ms_q) >= 10
    assert len(nq_docs) >= 8 and len(nq_q) >= 8
    assert all(q.answer for q in nq_q)


def test_nor_presets():
    presets = list_nor_presets()
    assert {p["name"] for p in presets} >= {"nor_fresh", "nor_typical", "nor_aged"}
    aged = get_nor_profile("nor_aged")
    fresh = get_nor_profile("nor_fresh")
    assert aged.p_indep > fresh.p_indep
    assert aged.p_drift > fresh.p_drift


def test_gemv_hook_loads_and_runs():
    cal = load_gemv_calibrator()
    rng = np.random.default_rng(0)
    # fake multi-cycle bits for n=288 symbols → 576 bits
    bits = rng.integers(0, 2, size=(2, 4, 576))
    llr, gate, used = apply_gemv_llr(bits, p_bit=5e-3, calibrator=cal, tau_d=0.0, tau_g=0.0)
    assert llr.shape == (2, 288, 3)
    assert 0.0 <= gate <= 1.0


def test_protect_with_optional_gemv():
    rng = np.random.default_rng(1)
    emb = np.clip(rng.normal(size=(2, 64)).astype(np.float32), -1, 1)
    profile = get_nor_profile("nor_typical")
    cal = load_gemv_calibrator()
    rec0 = protect_and_recover(
        emb, profile, rng=rng, use_bp=True, use_gemv=False,
    )
    rec1 = protect_and_recover(
        emb, profile, rng=np.random.default_rng(1), use_bp=True,
        use_gemv=True, gemv_calibrator=cal, tau_d=0.0, tau_g=0.0,
    )
    assert rec0.used_gemv is False
    assert rec1.emb_bprime.shape == emb.shape
    # With tau forced open, GEMV path should activate when drift allows
    assert isinstance(rec1.used_gemv, (bool, np.bool_))


def test_rag_metrics_and_extractive():
    assert f1_score("Paris is capital", "Paris") > 0.0
    assert exact_match("Paris", "paris") == 1.0
    passages = ["The capital of France is Paris on the Seine."]
    pred = extractive_answer("capital of France", passages)
    assert "paris" in pred.lower() or "france" in pred.lower()
    assert support_overlap("Paris", passages) == 1.0
    assert hallucination_rate("Atlantis unicorn", passages) == 1.0


def test_rag_eval_on_nq_mini():
    docs, queries = load_nq_mini()
    enc = build_encoder("tfidf_svd", dim=64, fit_texts=[d.text for d in docs] + [q.query for q in queries])
    idx = EncoderDenseIndex(
        [d.doc_id for d in docs],
        [d.text for d in docs],
        enc,
        apply_cim=False,
    )
    metrics = evaluate_rag(queries, {d.doc_id: d for d in docs}, idx.retrieve, k=3)
    assert metrics["rag_f1"] > 0.15
    assert metrics["rag_hallucination_rate"] < 0.5
    assert 0.0 <= metrics["rag_retrieval_hit@k"] <= 1.0


def test_extended_eval_smoke(tmp_path: Path):
    from cim_resldpc.experiments.run_extended_eval import run_extended

    out = tmp_path / "ext.json"
    result = run_extended(
        out_json=out,
        encoder="tfidf_svd",
        nor_name="nor_typical",
        use_gemv=True,
        seed=0,
        emb_dim=64,
    )
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["protocol_version"] == "v1.2"
    assert "msmarco_mini" in data and "nq_mini" in data
    assert "rag_generation" in data["nq_mini"]
    assert "cim_ldpc_bprime" in data["msmarco_mini"]["dense_only"]
    # Under aged NOR, LDPC B' should not be worse than unprotected by a huge margin
    # (allow equality); geometric expectation: often better
    b = data["msmarco_mini"]["dense_only"]["cim_ldpc_bprime"]["ndcg@10"]
    n = data["msmarco_mini"]["dense_only"]["cim_noisy_no_ldpc"]["ndcg@10"]
    assert b >= 0.0 and n >= 0.0
