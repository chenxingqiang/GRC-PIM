"""v1.6 tests for 500q artifacts and paper docs presence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_msmarco500_cache_schema():
    p = Path("cim_resldpc/data/cache/msmarco_hf_n500.json")
    if not p.exists():
        pytest.skip("500q cache missing")
    j = json.loads(p.read_text())
    assert len(j["queries"]) == 500
    assert len(j["documents"]) > 2000
    assert all(q["relevant_doc_ids"] for q in j["queries"])


def test_v16_result_has_positive_main_config():
    p = Path("cim_resldpc/results/sigir_eval_v16_msmarco500.json")
    if not p.exists():
        pytest.skip("v16 result missing")
    j = json.loads(p.read_text())
    assert j["protocol_version"] == "v1.5" or "seed_summary" in j
    mains = j.get("paper_main_configs") or []
    # Find dim=128 seed=42 typical
    hit = None
    for r in mains:
        if r.get("nor") == "nor_typical" and r.get("cim_dim") == 128 and r.get("seed") == 42:
            hit = r
            break
    if hit is None:
        pytest.skip("paper_main_configs incomplete")
    assert hit["robustness"]["bp_gain_vs_noisy"] > 0.05
    assert hit["dense_only"]["hybrid_bm25_bp"]["ndcg@10"] > hit["dense_only"]["bm25"]["ndcg@10"]


def test_dim256_stability_block():
    p = Path("cim_resldpc/results/sigir_eval_v16_msmarco500.json")
    if not p.exists():
        pytest.skip("v16 result missing")
    j = json.loads(p.read_text())
    block = j.get("dim256_seed_gains")
    if not block:
        pytest.skip("dim256 block missing")
    assert block["mean"] > 0.1
    assert block["std"] < 0.05


def test_paper_drafts_exist():
    root = Path("docs/paper")
    for name in (
        "INTRODUCTION.md",
        "RELATED_WORK.md",
        "METHOD.md",
        "EXPERIMENTS.md",
        "PAPER_OUTLINE.md",
        "V16_MSMARCO500_REPORT.md",
    ):
        assert (root / name).exists(), name
