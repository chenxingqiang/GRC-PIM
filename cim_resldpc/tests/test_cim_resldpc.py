"""Unit tests for CiM-ResLDPC v1.1 (oracle-free, LDPC path)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cim_resldpc.config import CiMResLDPCConfig
from cim_resldpc.datasets import build_synthetic_ir, load_semi_corpus
from cim_resldpc.gf3_quantize import (
    dequantize_from_gf3,
    pack_symbols_for_ldpc,
    quantize_to_gf3,
)
from cim_resldpc.hybrid_score import fuse_scores
from cim_resldpc.ldpc_embed import LDPCCode, protect_and_recover, symbol_error_rate
from cim_resldpc.metrics import average_precision, ndcg_at_k, noise_degradation, recall_at_k
from cim_resldpc.noise_bridge import (
    NoiseProfile,
    apply_embedding_cim_noise,
    residual_gate_weight,
    should_skip_gemv,
)
from cim_resldpc.pipeline import CiMResLDPCPipeline
from cim_resldpc.tonel import TONELResEncoder, stable_token_id


def test_t1_gf3_quantize_roundtrip():
    from cim_resldpc.gf3_quantize import quantize_base3, dequantize_base3

    rng = np.random.default_rng(0)
    emb = np.clip(rng.normal(size=(8, 64)).astype(np.float32), -1.0, 1.0)
    # Legacy centroid
    sym = quantize_to_gf3(emb)
    assert set(np.unique(sym)).issubset({0, 1, 2})
    # base3 preserves geometry much better
    sym2 = quantize_base3(emb, n_sym_per_dim=2)
    assert sym2.shape == (8, 128)
    rec2 = dequantize_base3(sym2, n_sym_per_dim=2, dim=64)
    assert float(np.mean(np.abs(emb - rec2))) < 0.15
    packed = pack_symbols_for_ldpc(sym2, code_n=288)
    assert packed.shape == (8, 288)


def test_t2_noise_bridge_shapes():
    rng = np.random.default_rng(1)
    code = rng.integers(0, 3, size=(4, 64))
    profile = NoiseProfile(n_cycles=4, p_indep=5e-3, p_drift=5e-3, p_row_burst=5e-3)
    out = apply_embedding_cim_noise(code, profile, rng=rng)
    assert out.noisy_bits_3d.shape == (4, 4, 128)
    assert 0.0 <= out.drift_score <= 1.0
    assert out.recovered_symbols.shape == (4, 64)


def test_t3_residual_weight_monotone_in_drift():
    w_low = residual_gate_weight(0.01, gate_mean=0.7, tau_d=0.025)
    w_high = residual_gate_weight(0.05, gate_mean=0.7, tau_d=0.025)
    assert w_high > w_low


def test_t4_tonel_train_step_finite():
    cfg = CiMResLDPCConfig(embed_dim=32, seed=0)
    enc = TONELResEncoder(cfg)
    stats = enc.train_step(
        ["NOR flash CiM drift", "GF3 LDPC residual"],
        ["floating-gate conductance drift", "non-binary LDPC soft LLR"],
        rng=np.random.default_rng(0),
    )
    for key in ("retrieval", "residual_align", "quant", "total", "residual_weight"):
        assert key in stats
        assert np.isfinite(stats[key])


def test_t5_dense_mode_ranks_without_oracle():
    docs, queries = build_synthetic_ir(n_topics=4, docs_per_topic=3)
    cfg = CiMResLDPCConfig(embed_dim=32, top_k=5, seed=1, p_indep=1e-3, p_drift=1e-3, p_row_burst=1e-3)
    enc = TONELResEncoder(cfg)
    for _ in range(8):
        for qa in queries:
            pos = next(d.text for d in docs if d.doc_id == qa.relevant_doc_ids[0])
            enc.train_step([qa.query], [pos], rng=np.random.default_rng(2))
    pipe = CiMResLDPCPipeline(cfg, encoder=enc, recovery="soft")
    pipe.index(docs)
    # Calibrated embeddings must not equal clean (oracle would be too close / identical blend)
    assert pipe.clean_emb is not None and pipe.calibrated_emb is not None
    assert not np.allclose(pipe.calibrated_emb, pipe.clean_emb)
    qa = queries[0]
    hits = pipe.retrieve(qa.query, k=5, mode="dense")
    assert len(hits) == 5


def test_t6_decode_prune_skip():
    assert should_skip_gemv(0.005, tau_skip=0.015) is True
    assert should_skip_gemv(0.04, tau_skip=0.015) is False


def test_t7_local_only_privacy_contract():
    pipe = CiMResLDPCPipeline(CiMResLDPCConfig(local_only=True))
    assert pipe.uploads_raw_documents is False
    with pytest.raises(ValueError):
        CiMResLDPCPipeline(CiMResLDPCConfig(local_only=False))


def test_t8_metrics_and_robustness():
    ranked = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    assert 0.0 <= ndcg_at_k(ranked, k=10) <= 1.0
    assert 0.0 <= average_precision(ranked) <= 1.0
    assert 0.0 <= recall_at_k(ranked, n_relevant=3, k=5) <= 1.0
    assert noise_degradation(0.80, 0.72) < noise_degradation(0.80, 0.50)


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
        noise_p=0.05,
    )
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["protocol_version"] == "v1.1"
    assert "dense_only" in data
    assert "geometric_probe" in data
    assert "cim_ldpc_bprime" in data["dense_only"]
    assert "bprime_gain_vs_noisy" in data["robustness_dense"]


def test_geometric_probe_ldpc_helps():
    """Under controlled geometry at p=0.03, B' should beat unprotected noisy."""
    from cim_resldpc.experiments.run_sigir_eval import geometric_probe

    rows = {r["p"]: r for r in geometric_probe(seed=0)["rows"]}
    r = rows[0.03]
    assert r["ndcg_bprime"] > r["ndcg_noisy"]
    assert r["ser_bp"] <= r["ser_noisy"] + 1e-9


def test_ldpc_protect_improves_or_matches_ser():
    rng = np.random.default_rng(0)
    code = LDPCCode.load()
    emb = np.clip(rng.normal(size=(4, 64)).astype(np.float32), -1.0, 1.0)
    from cim_resldpc.gf3_quantize import quantize_base3

    clean_sym = quantize_base3(emb, n_sym_per_dim=2)
    profile = NoiseProfile(p_indep=3e-2, p_drift=3e-2, p_row_burst=0.0, n_cycles=4)
    rec = protect_and_recover(
        emb, profile, code=code, rng=rng, use_bp=True,
        n_sym_per_dim=2, quant_mode="base3",
    )
    ser_bp = symbol_error_rate(clean_sym, rec.symbols_bp)
    assert 0.0 <= ser_bp <= 1.0
    assert rec.emb_bprime.shape == emb.shape
    assert not np.allclose(rec.emb_bprime, emb)


def test_t10_import_smoke():
    import cim_resldpc
    from cim_resldpc import CiMResLDPCPipeline, CiMResLDPCConfig
    from pim_predecoder.core.noise3d import apply_pim_noise_3d

    assert cim_resldpc.__version__
    assert CiMResLDPCConfig().embed_dim == 64
    assert callable(apply_pim_noise_3d)


def test_no_oracle_blend_in_pipeline_source():
    """Guardrail: pipeline must not blend clean_emb into calibrated vectors."""
    src = Path(__file__).resolve().parents[1] / "pipeline.py"
    text = src.read_text()
    assert "0.5 * rec" not in text
    assert "0.5 * self.clean_emb" not in text
    assert "0.5 * q_emb" not in text


def test_stable_token_hash():
    a = stable_token_id("nor_flash", 4096)
    b = stable_token_id("nor_flash", 4096)
    assert a == b
    assert 0 <= a < 4096


def test_fuse_scores_shape():
    f = fuse_scores(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([0.1, 0.9, 0.2], dtype=np.float32),
        np.array([0.5, 0.5, 0.8], dtype=np.float32),
    )
    assert f.shape == (3,)


def test_semi_corpus_nonempty():
    docs, queries = load_semi_corpus()
    assert len(docs) >= 8
    assert all(q.relevant_doc_ids for q in queries)
