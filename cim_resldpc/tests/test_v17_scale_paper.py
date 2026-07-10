"""v1.7 tests for 1000q artifacts and paper docs presence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_msmarco1000_cache_schema():
    p = Path("cim_resldpc/data/cache/msmarco_hf_n1000.json")
    if not p.exists():
        pytest.skip("1000q cache missing")
    j = json.loads(p.read_text())
    assert len(j["queries"]) == 1000
    assert len(j["documents"]) > 5000
    assert all(q["relevant_doc_ids"] for q in j["queries"])


def test_v17_result_has_positive_main_config():
    p = Path("cim_resldpc/results/sigir_eval_v17_msmarco1000.json")
    if not p.exists():
        pytest.skip("v17 result missing")
    j = json.loads(p.read_text())
    assert j["n_queries"] == 1000
    assert j["n_docs"] == 8503
    mains = j.get("paper_main_configs") or []
    hit = None
    for r in mains:
        if r.get("nor") == "nor_typical" and r.get("cim_dim") == 128 and r.get("seed") == 42:
            hit = r
            break
    if hit is None:
        pytest.skip("paper_main_configs incomplete")
    assert hit["robustness"]["bp_gain_vs_noisy"] > 0.15
    assert hit["dense_only"]["hybrid_bm25_bp"]["ndcg@10"] > hit["dense_only"]["bm25"]["ndcg@10"]


def test_v17_seed_summary_positive():
    p = Path("cim_resldpc/results/sigir_eval_v17_msmarco1000.json")
    if not p.exists():
        pytest.skip("v17 result missing")
    j = json.loads(p.read_text())
    s = j["seed_summary"]
    assert s["bp_gain_mean"] > 0.1
    assert s["n_seeds"] >= 4


def test_dim256_stability_block_v17():
    p = Path("cim_resldpc/results/sigir_eval_v17_msmarco1000.json")
    if not p.exists():
        pytest.skip("v17 result missing")
    j = json.loads(p.read_text())
    block = j.get("dim256_seed_gains")
    if not block:
        pytest.skip("dim256 block missing")
    assert block["mean"] > 0.1
    assert block["std"] < 0.08
    assert len(block["seeds"]) == 4


def test_paper_drafts_v17_exist():
    root = Path("docs/paper")
    for name in (
        "INTRODUCTION.md",
        "RELATED_WORK.md",
        "METHOD.md",
        "EXPERIMENTS.md",
        "CONCLUSION.md",
        "PAPER_OUTLINE.md",
        "V17_MSMARCO1000_REPORT.md",
        "latex/main.tex",
    ):
        assert (root / name).exists(), name
