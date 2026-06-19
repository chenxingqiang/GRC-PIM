"""Aggregate the T-cycle sweep results (1024-symbol, E2 strong drift, p_bit=5e-3).

Reads pim_predecoder/results/supplementary/tsweep/T{T}_s{S}.json produced by
pim_predecoder.experiments.gemv and writes a mean+/-std summary across seeds for
each cycle count T, plus a compact LaTeX-ready table fragment to stdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pim_predecoder.paths import RESULTS

TSWEEP_DIR = RESULTS / "supplementary" / "tsweep"
OUT = RESULTS / "supplementary" / "tsweep_summary.json"

T_VALUES = [2, 3, 4, 5, 6]
SEEDS = [1, 2, 3]
P_BIT = 5e-3


def _row_for(path: Path) -> dict:
    data = json.loads(path.read_text())
    for row in data["rows"]:
        if abs(row["p_bit"] - P_BIT) < 1e-12:
            return row
    raise KeyError(f"p_bit={P_BIT} not found in {path}")


def main() -> None:
    summary = []
    for T in T_VALUES:
        bp, gemv, gain, drift, gate = [], [], [], [], []
        for S in SEEDS:
            path = TSWEEP_DIR / f"T{T}_s{S}.json"
            if not path.exists():
                print(f"[warn] missing {path}")
                continue
            r = _row_for(path)
            bp.append(r["Bp_post_BER"])
            gemv.append(r["GEMV_post_BER"])
            g = r["Bp_post_BER"] / r["GEMV_post_BER"] if r["GEMV_post_BER"] > 0 else float("nan")
            gain.append(g)
            drift.append(r["drift_score"])
            gate.append(r["GEMV_gate"])
        if not bp:
            continue
        summary.append(dict(
            T=T,
            n_seeds=len(bp),
            Bp_BER_mean=float(np.mean(bp)), Bp_BER_std=float(np.std(bp, ddof=1)),
            GEMV_BER_mean=float(np.mean(gemv)), GEMV_BER_std=float(np.std(gemv, ddof=1)),
            gain_mean=float(np.mean(gain)), gain_std=float(np.std(gain, ddof=1)),
            drift_mean=float(np.mean(drift)),
            gate_mean=float(np.mean(gate)),
        ))

    OUT.write_text(json.dumps(dict(
        description="T-cycle sweep, 1024-symbol rate 16/17, E2 strong drift, "
                    "p_bit=5e-3, gate forced open (tau_d=0.005, tau_g=0.5), "
                    "2000 codewords/seed, 3 seeds.",
        p_bit=P_BIT, seeds=SEEDS, rows=summary,
    ), indent=2))
    print(f"wrote {OUT}\n")

    # LaTeX-ready summary
    print(r"T & B' BER & GEMV BER & BER gain & drift & gate \\")
    for s in summary:
        print(
            f"{s['T']} & "
            f"${s['Bp_BER_mean']*1e3:.2f}\\times10^{{-3}}$ & "
            f"${s['GEMV_BER_mean']*1e3:.2f}\\times10^{{-3}}$ & "
            f"${s['gain_mean']:.2f}\\pm{s['gain_std']:.2f}\\times$ & "
            f"{s['drift_mean']:.3f} & {s['gate_mean']:.2f} \\\\"
        )


if __name__ == "__main__":
    main()
