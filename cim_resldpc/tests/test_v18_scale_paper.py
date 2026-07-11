"""v1.8 tests for 2000q artifacts and paper docs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_msmarco2000_cache_schema():
    p = Path("cim_resldpc/data/cache/msmarco_hf_n2000.json")
    if not p.exists():
        pytest.skip("2000q cache missing")
    j = json.loads(p.read_text())
    assert len(j["queries"]) == 2000
    assert len(j["documents"]) > 10000
    assert all(q["relevant_doc_ids"] for q in j["queries"])


def test_v18_result_has_positive_main_config():
    p = Path("cim_resldpc/results/sigir_eval_v18_msmarco2000.json")
    if not p.exists():
        pytest.skip("v18 result missing")
    j = json.loads(p.read_text())
    assert j["protocol_version"] == "v1.8"
    assert j["n_queries"] == 2000
    assert j["n_docs"] == 16894
    hit = None
    for r in j.get("paper_main_configs") or []:
        if r.get("nor") == "nor_typical" and r.get("cim_dim") == 128 and r.get("seed") == 42:
            hit = r
            break
    if hit is None:
        pytest.skip("paper_main incomplete")
    assert hit["robustness"]["bp_gain_vs_noisy"] > 0.15
    assert hit["dense_only"]["hybrid_bm25_bp"]["ndcg@10"] > hit["dense_only"]["bm25"]["ndcg@10"]


def test_v18_seed_summary():
    p = Path("cim_resldpc/results/sigir_eval_v18_msmarco2000.json")
    if not p.exists():
        pytest.skip("v18 result missing")
    s = json.loads(p.read_text())["seed_summary"]
    assert s["bp_gain_mean"] > 0.08
    assert s["n_seeds"] >= 4
    assert s["bp_ndcg_std"] < 0.02


def test_v18_aged_operating_point_reported():
    p = Path("cim_resldpc/results/sigir_eval_v18_msmarco2000.json")
    if not p.exists():
        pytest.skip("v18 result missing")
    rows = json.loads(p.read_text()).get("nor_sweep_seed42") or []
    aged = next((r for r in rows if r.get("nor") == "nor_aged"), None)
    if aged is None:
        pytest.skip("aged row missing")
    # Aged corner: absolute BP remains very low — honesty check
    assert aged["dense_only"]["cim_ldpc_bp"]["ndcg@10"] < 0.15


def test_paper_docs_v18():
    root = Path("docs/paper")
    for name in (
        "V18_MSMARCO2000_REPORT.md",
        "latex/main.tex",
        "latex/references.bib",
        "CONCLUSION.md",
    ):
        assert (root / name).exists(), name


def test_run_paper_main_module_importable():
    from cim_resldpc.experiments import run_paper_main_v18

    assert hasattr(run_paper_main_v18, "main")
