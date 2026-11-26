#!/usr/bin/env python3
"""Exact tau_g sensitivity on the reported E2 test sweep.

The acceptance gate enters the reported result only through the indicator
``g-bar >= tau_g``, so once the per-point acceptance score and both error
rates are recorded the gain that any tau_g would have produced follows by
composing the stored decisions: no re-simulation and no interpolation is
involved. This is the tau_g companion of ``place_thresholds.py`` and
regenerates the acceptance-threshold table of the Supplementary Information.

Two things have to be read with care, and an earlier version of this script
got the first one wrong:

* A point the deployed tau_g closes stores B' in ``GEMV_post*`` (the gated
  fallback). Re-admitting it at a lower tau_g must therefore use the
  calibrator's *ungated* outcome, not that stored value, or the re-admitted
  point contributes a gain of exactly 1 and the table understates every row
  below the deployed threshold. ``result_data`` resolves the ungated
  outcome (``GEMV_ungated_post*``, or the forced-open companion run for the
  seeds that predate that field).
* Lowering tau_g also admits the mismatched models it exists to reject. The
  E5 column counts how many of the i.i.d.-trained, strong-drift-tested
  operating points (``nogate/``, 3 seeds x 5 p_bit) each tau_g would admit,
  so the two sides of the trade are on one line.

Usage:  python3 tau_g_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from result_data import (  # noqa: E402
    E2_SEEDS,
    GATE,
    aggregate_256_e2,
    aggregate_1024_e2,
    e5_admitted,
    e5_gate_points,
    gate_composition,
)

AGGREGATE = {1024: aggregate_1024_e2, 256: aggregate_256_e2}
TAUS = (0.0, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.99)


def compute() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for scale in (1024, 256):
        rows = AGGREGATE[scale]()
        # One entry per (operating point, seed).
        points = [
            dict(seed=seed, p_bit=r.p_bit, gate=r.gate_scores[i])
            for r in rows for i, seed in enumerate(E2_SEEDS)
        ]
        e5 = e5_gate_points(scale)
        per_tau = {}
        for tau in TAUS:
            mean_gain, opened, n_points = gate_composition(scale, tau_g=tau)
            admitted, n_e5 = e5_admitted(scale, tau)
            per_tau[f"{tau:g}"] = dict(
                mean_gain=mean_gain,
                open=f"{opened} of {n_points}",
                e5_admitted=f"{admitted} of {n_e5}")
        gates = [pt["gate"] for pt in points]
        by_seed = {
            str(seed): [round(pt["gate"], 3) for pt in points
                        if pt["seed"] == seed]
            for seed in E2_SEEDS}
        out[str(scale)] = dict(
            n_seeds=len(E2_SEEDS),
            gate_min=min(gates), gate_max=max(gates),
            gate_by_seed=by_seed,
            # Seeds whose acceptance score falls below the deployed threshold
            # at one operating point or more.
            seeds_with_closed_point=[
                seed for seed in E2_SEEDS
                if min(by_seed[str(seed)]) < GATE["tau_g"]],
            e5_gate_min=min(g for g, _ in e5),
            e5_gate_max=max(g for g, _ in e5),
            tau_g=per_tau,
        )
    return out


def main() -> None:
    out = compute()
    print(json.dumps(out, indent=2))
    (HERE / "tau_g_sensitivity.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
