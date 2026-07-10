"""v1.5 ablation smoke tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cim_resldpc.experiments.run_ablation_v15 import _project, run_condition


def test_project_dim_and_norm():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(8, 384)).astype(np.float32)
    out = _project(emb, cim_dim=128, seed=0)
    assert out.shape == (8, 128)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_ablation_json_schema_if_present():
    p = Path("cim_resldpc/results/sigir_eval_v15_ablation.json")
    if not p.exists():
        pytest.skip("ablation json missing")
    j = json.loads(p.read_text())
    assert j["protocol_version"] == "v1.5"
    assert "seed_summary" in j
    assert j["seed_summary"]["bp_gain_mean"] > 0.0
    assert "nor_sweep" in j and len(j["nor_sweep"]) >= 4


def test_run_condition_smoke_from_cache():
    cache_path = Path("cim_resldpc/data/cache/bge_msmarco_n200_emb.npz")
    if not cache_path.exists():
        pytest.skip("BGE emb cache missing")
    from cim_resldpc.experiments.run_ablation_v15 import load_emb_cache

    cache = load_emb_cache(cache_path)
    # Tiny slice for speed
    cache = dict(cache)
    cache["d_emb"] = cache["d_emb"][:80]
    cache["doc_ids"] = cache["doc_ids"][:80]
    cache["doc_texts"] = cache["doc_texts"][:80]
    # keep queries but filter rel ids that may be missing — use first 10 queries
    cache["queries"] = cache["queries"][:10]
    cache["q_emb"] = cache["q_emb"][:10]
    r = run_condition(cache, "nor_typical", cim_dim=64, seed=0, include_hybrid=True)
    assert "cim_ldpc_bp" in r["dense_only"]
    assert "hybrid_bm25_bp" in r["dense_only"]
