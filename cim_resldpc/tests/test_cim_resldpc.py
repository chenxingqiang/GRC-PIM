"""Unit tests for CiM-ResLDPC (T1–T10 from design test plan)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cim_resldpc.config import CiMResLDPCConfig
from cim_resldpc.datasets import build_synthetic_ir, load_semi_corpus
from cim_resldpc.gf3_quantize import (
    dequantize_from_gf3,
    pack_symbols_for_ldpc,
    quantize_to_gf3,
)
from cim_resldpc.hybrid_score import fuse_scores
from cim_resldpc.metrics import average_precision, ndcg_at_k, noise_degradation, recall_at_k
from cim_resldpc.noise_bridge import (
    NoiseProfile,
    apply_embedding_cim_noise,
    residual_gate_weight,
    should_skip_gemv,
)
from cim_resldpc.pipeline import CiMResLDPCPipeline, Document
from cim_resldpc.tonel import TONELResEncoder


# ---------------------------------------------------------------------------
# T1 GF(3) quantize roundtrip
# ---------------------------------------------------------------------------
def test_t1_gf3_quantize_roundtrip():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(8, 64)).astype(np.float32)
    emb = np.clip(emb, -1.5, 1.5)
    sym = quantize_to_gf3(emb)
    assert sym.shape == emb.shape
    assert set(np.unique(sym)).issubset({0, 1, 2})
    rec = dequantize_from_gf3(sym)
    # Nearest-centroid reconstruction error bounded
    assert float(np.mean(np.abs(emb - rec))) < 0.6
    packed = pack_symbols_for_ldpc(sym, code_n=288)
    assert packed.shape == (8, 288)
    assert set(np.unique(packed)).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# T2 Noise bridge shapes + drift score range
# ---------------------------------------------------------------------------
def test_t2_noise_bridge_shapes():
    rng = np.random.default_rng(1)
    code = rng.integers(0, 3, size=(4, 64))
    profile = NoiseProfile(n_cycles=4, p_indep=5e-3, p_drift=5e-3, p_row_burst=5e-3)
    out = apply_embedding_cim_noise(code, profile, rng=rng)
    assert out.noisy_bits_3d.shape == (4, 4, 128)
    assert 0.0 <= out.drift_score <= 1.0
    assert out.recovered_symbols.shape == (4, 64)
    assert out.llr_bp.shape == (4, 64, 3)
    assert "global_drift" in out.scores


# ---------------------------------------------------------------------------
# T3 Residual weight monotone in drift
# ---------------------------------------------------------------------------
def test_t3_residual_weight_monotone_in_drift():
    w_low = residual_gate_weight(0.01, gate_mean=0.7, tau_d=0.025)
    w_high = residual_gate_weight(0.05, gate_mean=0.7, tau_d=0.025)
    assert w_high > w_low
    assert 0.0 <= w_low <= 1.0
    assert 0.0 <= w_high <= 1.0


# ---------------------------------------------------------------------------
# T4 TONEL train_step finite losses
# ---------------------------------------------------------------------------
def test_t4_tonel_train_step_finite():
    cfg = CiMResLDPCConfig(embed_dim=32, seed=0)
    enc = TONELResEncoder(cfg)
    queries = [
        "NOR flash CiM drift read disturb",
        "GF3 LDPC residual calibration",
    ]
    positives = [
        "floating-gate NOR compute-in-memory conductance drift",
        "non-binary LDPC GEMV residual soft LLR",
    ]
    stats = enc.train_step(queries, positives, rng=np.random.default_rng(0))
    for key in ("retrieval", "residual_align", "quant", "total", "residual_weight"):
        assert key in stats
        assert np.isfinite(stats[key])


# ---------------------------------------------------------------------------
# T5 Hybrid fusion ranks relevant above random on SyntheticIR
# ---------------------------------------------------------------------------
def test_t5_hybrid_ranks_relevant():
    docs, queries = build_synthetic_ir(n_topics=4, docs_per_topic=3)
    cfg = CiMResLDPCConfig(embed_dim=32, bm25_cutoff_m=16, top_k=5, seed=1)
    # Light training for signal
    enc = TONELResEncoder(cfg)
    for _ in range(5):
        for qa in queries:
            pos = next(d.text for d in docs if d.doc_id == qa.relevant_doc_ids[0])
            enc.train_step([qa.query], [pos], rng=np.random.default_rng(2))
    pipe = CiMResLDPCPipeline(cfg, encoder=enc)
    pipe.index(docs)
    qa = queries[0]
    hits = pipe.retrieve(qa.query, k=5)
    hit_ids = {h.doc_id for h in hits}
    assert hit_ids & set(qa.relevant_doc_ids), f"expected relevant in top-5, got {hit_ids}"


# ---------------------------------------------------------------------------
# T6 Decode prune skips GEMV when drift < tau_skip
# ---------------------------------------------------------------------------
def test_t6_decode_prune_skip():
    assert should_skip_gemv(0.005, tau_skip=0.015) is True
    assert should_skip_gemv(0.04, tau_skip=0.015) is False


# ---------------------------------------------------------------------------
# T7 Local-only privacy contract
# ---------------------------------------------------------------------------
def test_t7_local_only_privacy_contract():
    cfg = CiMResLDPCConfig(local_only=True)
    pipe = CiMResLDPCPipeline(cfg)
    assert pipe.uploads_raw_documents is False
    with pytest.raises(ValueError):
        CiMResLDPCPipeline(CiMResLDPCConfig(local_only=False))


# ---------------------------------------------------------------------------
# T8 Metrics ranges + robustness relative ordering
# ---------------------------------------------------------------------------
def test_t8_metrics_and_robustness():
    ranked = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    n = ndcg_at_k(ranked, k=10)
    assert 0.0 <= n <= 1.0
    ap = average_precision(ranked)
    assert 0.0 <= ap <= 1.0
    r = recall_at_k(ranked, n_relevant=3, k=5)
    assert 0.0 <= r <= 1.0
    # Synthetic: ResLDPC degrades less than naive noisy
    deg_naive = noise_degradation(0.80, 0.50)
    deg_ours = noise_degradation(0.80, 0.72)
    assert deg_ours < deg_naive


# ---------------------------------------------------------------------------
# T9 Eval JSON schema
# ---------------------------------------------------------------------------
def test_t9_eval_json_schema(tmp_path: Path):
    from cim_resldpc.experiments.run_sigir_eval import run_evaluation

    out = tmp_path / "eval.json"
    result = run_evaluation(
        out_json=out,
        n_topics=3,
        docs_per_topic=2,
        train_steps=2,
        seed=0,
        include_semi=True,
    )
    assert out.exists()
    data = json.loads(out.read_text())
    for key in ("config", "synthetic", "semi", "baselines", "cim_resldpc"):
        assert key in data
    assert "ndcg@10" in data["cim_resldpc"]["synthetic"]
    assert result["cim_resldpc"]["synthetic"]["ndcg@10"] >= 0.0


# ---------------------------------------------------------------------------
# T10 Import smoke / no circular import
# ---------------------------------------------------------------------------
def test_t10_import_smoke():
    import cim_resldpc
    from cim_resldpc import CiMResLDPCPipeline, CiMResLDPCConfig
    from pim_predecoder.core.noise3d import apply_pim_noise_3d
    from pim_predecoder.core.gemv_calibrator import global_drift_score

    assert cim_resldpc.__version__
    assert CiMResLDPCConfig().embed_dim == 64
    assert callable(apply_pim_noise_3d)
    assert callable(global_drift_score)


def test_fuse_scores_shape():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([0.1, 0.9, 0.2], dtype=np.float32)
    c = np.array([0.5, 0.5, 0.8], dtype=np.float32)
    f = fuse_scores(a, b, c)
    assert f.shape == (3,)
    assert np.isfinite(f).all()


def test_semi_corpus_nonempty():
    docs, queries = load_semi_corpus()
    assert len(docs) >= 8
    assert len(queries) >= 8
    assert all(q.relevant_doc_ids for q in queries)
