"""v1.3 tests: multi-block LDPC soft unpack + MiniLM optional + HF cache schema."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cim_resldpc.ldpc_embed import (
    LDPCCode,
    protect_and_recover,
    soft_expectation_from_llr_base3,
    unpack_llr_info_blocks,
)
from cim_resldpc.nor_calibration import get_nor_profile


def test_multi_block_soft_unpack_shapes():
    """384-d * 2 symbols needs 3 LDPC info blocks (k=256)."""
    rng = np.random.default_rng(0)
    emb = np.clip(rng.normal(size=(2, 384)).astype(np.float32), -1, 1)
    profile = get_nor_profile("nor_fresh")
    rec = protect_and_recover(
        emb, profile, rng=rng, use_bp=False, n_sym_per_dim=2, quant_mode="base3",
    )
    assert rec.emb_bprime.shape == (2, 384)
    assert rec.emb_soft.shape == (2, 384)
    assert rec.n_codewords == 2 * 3  # 768/256 = 3 blocks per row


def test_unpack_llr_info_blocks():
    B, n_blocks, k, n = 2, 3, 256, 288
    rng = np.random.default_rng(0)
    llr = rng.normal(size=(B * n_blocks, n, 3)).astype(np.float32)
    out = unpack_llr_info_blocks(llr, batch=B, sym_dim=500, n_blocks=n_blocks, k=k)
    assert out.shape == (B, 500, 3)


def test_soft_expectation_rejects_short_llr():
    llr = np.zeros((1, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        soft_expectation_from_llr_base3(llr, n_sym_per_dim=2, emb_dim=64)


def test_minilm_optional_encode():
    pytest.importorskip("sentence_transformers")
    from cim_resldpc.encoders import build_encoder

    enc = build_encoder("st")
    emb = enc.encode(["compute in memory", "information retrieval"])
    assert emb.shape == (2, 384)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)


def test_hf_cache_schema_if_present():
    cache = Path("cim_resldpc/data/cache/msmarco_hf_n48.json")
    if not cache.exists():
        pytest.skip("HF cache not built")
    raw = json.loads(cache.read_text())
    assert len(raw["documents"]) > 0
    assert len(raw["queries"]) == 48
    assert all(q["relevant_doc_ids"] for q in raw["queries"])


def test_hf_eval_smoke_minilm_projected(tmp_path: Path):
    pytest.importorskip("sentence_transformers")
    from cim_resldpc.experiments.run_hf_eval import run_hf_eval

    out = tmp_path / "hf.json"
    # Use bundled mini to keep smoke fast & offline-safe
    result = run_hf_eval(
        out_json=out,
        encoder="st",
        nor_name="nor_typical",
        use_gemv=False,
        n_queries=8,
        seed=0,
        allow_hf=False,
        corpora=["msmarco_mini"],
        cim_dim=64,
    )
    assert out.exists()
    assert result["protocol_version"] == "v1.3"
    block = result["msmarco_mini"]
    assert "cim_ldpc_bprime" in block["dense_only"]
    assert block["robustness"]["cim_dim"] == 64
    assert block["robustness"]["teacher_dim"] == 384
