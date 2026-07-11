"""Generate CiM-ResLDPC paper figures from committed results/*.json.

Reproduce: python3 docs/paper/latex/make_figs.py  (run from repo root).
All numbers are read from cim_resldpc/results/*.json (or, for the scale ladder,
the verified values tabulated in docs/paper/EXPERIMENTS.md). Figures are written
as vector PDF into docs/paper/latex/figs/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "cim_resldpc" / "results"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
NORS = ["fresh", "typical", "aged", "stress"]
KEY = {"fresh": "nor_fresh", "typical": "nor_typical",
       "aged": "nor_aged", "stress": "nor_stress_sweep"}


def _ndcg(block, k):
    return block[k]["ndcg@10"]


def fig_nor():
    bge = json.loads((RES / "sigir_eval_v21_bge_msmarco200_norsweep.json").read_text())["nor_sweep"]
    mini = json.loads((RES / "sigir_eval_v22_minilm_msmarco200_norsweep.json").read_text())["nor_sweep"]
    x = range(len(NORS))
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.plot(x, [_ndcg(bge[KEY[n]], "cim_noisy_no_ldpc") for n in NORS], "o--", color="#d62728", label="BGE noisy")
    ax.plot(x, [_ndcg(bge[KEY[n]], "cim_ldpc_bp") for n in NORS], "o-", color="#1f77b4", label="BGE BP")
    ax.plot(x, [_ndcg(mini[KEY[n]], "cim_noisy_no_ldpc") for n in NORS], "s--", color="#ff9896", label="MiniLM noisy")
    ax.plot(x, [_ndcg(mini[KEY[n]], "cim_ldpc_bp") for n in NORS], "s-", color="#aec7e8", label="MiniLM BP")
    ax.set_xticks(list(x)); ax.set_xticklabels(NORS)
    ax.set_xlabel("NOR operating point"); ax.set_ylabel("nDCG@10")
    ax.legend(fontsize=6, ncol=2)
    fig.savefig(OUT / "fig_nor.pdf"); plt.close(fig)


def fig_scale():
    # Verified scale-ladder values (BGE, dim=128, seed=42, nor_typical);
    # sources: results/sigir_eval_v14_bge_n200, _v16_msmarco500, _v17_msmarco1000, _v18_msmarco2000.
    q = [200, 500, 1000, 2000]
    noisy = [0.445, 0.423, 0.283, 0.254]
    bp = [0.531, 0.539, 0.496, 0.475]
    hybrid = [0.572, 0.565, 0.549, 0.557]
    bm25 = [0.528, 0.536, 0.520, 0.531]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.plot(q, noisy, "o--", color="#d62728", label="CiM noisy")
    ax.plot(q, bp, "o-", color="#1f77b4", label="LDPC BP")
    ax.plot(q, hybrid, "^-", color="#2ca02c", label="Hybrid BM25+BP")
    ax.plot(q, bm25, "s:", color="#7f7f7f", label="BM25")
    ax.set_xscale("log"); ax.set_xticks(q); ax.set_xticklabels(q)
    ax.set_xlabel("# queries (MSMARCO)"); ax.set_ylabel("nDCG@10")
    ax.legend(fontsize=6)
    fig.savefig(OUT / "fig_scale.pdf"); plt.close(fig)


def fig_geo():
    g = json.loads((RES / "sigir_eval.json").read_text())["geometric_probe"]
    rows = g.get("rows", g)
    p = [r["p"] for r in rows]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.plot(p, [r["ndcg_noisy"] for r in rows], "o--", color="#d62728", label="noisy nDCG")
    ax.plot(p, [r["ndcg_bprime"] for r in rows], "^-", color="#2ca02c", label="B$'$ nDCG")
    ax.plot(p, [r["ndcg_bp"] for r in rows], "o-", color="#1f77b4", label="BP nDCG")
    ax.set_xlabel("bit-flip $p$"); ax.set_ylabel("nDCG@10")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(p, [r["ser_noisy"] for r in rows], "x:", color="#9467bd", label="SER noisy")
    ax2.plot(p, [r["ser_bp"] for r in rows], "+:", color="#8c564b", label="SER BP")
    ax2.set_ylabel("symbol-error rate")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=5.5, loc="lower left")
    fig.savefig(OUT / "fig_geo.pdf"); plt.close(fig)


def fig_gate():
    d = json.loads((RES / "sigir_eval_v24_decode_prune_gate.json").read_text())["presets"]
    x = range(len(NORS)); w = 0.38
    skip = [d[KEY[n]]["skip_frac"] * 100 for n in NORS]
    opn = [d[KEY[n]]["gate_open_frac"] * 100 for n in NORS]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.bar([i - w / 2 for i in x], skip, w, color="#2ca02c", label="skip (prune)")
    ax.bar([i + w / 2 for i in x], opn, w, color="#1f77b4", label="gate open (GEMV)")
    ax.set_xticks(list(x)); ax.set_xticklabels(NORS)
    ax.set_xlabel("NOR operating point"); ax.set_ylabel("% of codewords")
    ax.legend(fontsize=6)
    fig.savefig(OUT / "fig_gate.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_nor(); fig_scale(); fig_geo(); fig_gate()
    print("wrote figures to", OUT)
