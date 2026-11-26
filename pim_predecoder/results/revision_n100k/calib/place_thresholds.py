#!/usr/bin/env python3
"""Re-derive the drift-band thresholds from the held-out validation sweep.

Sec. 3.5 of the manuscript states a placement rule for the drift-gate lower
bound in prose. Prose is not reproducible: read literally, "the lowest drift
score at which the calibrator is beneficial" is 0.0019 on the 1024-symbol sweep
and would put tau_d near 0.0002, fifty times below the value we use. The rule
that actually yields 0.011 restricts the minimum to the matched channel -- the
one the residual was fitted on -- and subtracts twice that point's own
across-block s.d. This script is that rule, so the constants in the paper can be
regenerated instead of trusted.

It also answers the question the prose leaves open: how sensitive the reported
gain is to tau_d. The gate decision depends on tau_d only through the indicator
tau_d <= d, so for the test sweep the answer is exact rather than interpolated --
any tau_d at or below the smallest test drift score gives bit-identical results.

Run from the repo root:
    python3 pim_predecoder/results/revision_n100k/calib/place_thresholds.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))

from result_data import gate_composition  # noqa: E402

ROOT = "pim_predecoder/results/revision_n100k"
CALIB = os.path.join(ROOT, "calib")

# Deployed constants, quoted here only so that the script can check them.
BAND = {"1024": (0.011, 0.038), "256": (0.005, 0.045)}
TAU_G = 0.6


def _rows(pattern: str):
    """Yield (channel, row) for every operating point matching a glob."""
    for path in sorted(glob.glob(pattern)):
        blob = json.load(open(path))
        name = os.path.basename(path)
        for row in blob["rows"]:
            yield name, row


def _gain(row) -> float:
    gemv = row["GEMV_post_BER"]
    return row["Bp_post_BER"] / gemv if gemv > 0 else float("inf")


def _channel(name: str, scale: str) -> str:
    return name.replace(f"val_{scale}_", "").split("_")[0]


def place_lower_bound(scale: str) -> dict:
    """Apply the Sec. 3.5 rule to one code scale."""
    points = []
    for name, row in _rows(os.path.join(CALIB, f"val_{scale}_*.json")):
        points.append(
            {
                "channel": _channel(name, scale),
                "p_bit": row["p_bit"],
                "d": row["drift_score"],
                "sd": row["drift_score_sd"],
                "gain": _gain(row),
                "floor": row["resolution_floor"],
                "bp": row["Bp_post_BER"],
                "gemv": row["GEMV_post_BER"],
                "n_cw": row["n_codewords_eval"],
                "n_blocks": row["n_blocks"],
            }
        )
    if not points:
        raise SystemExit(f"no validation JSON found under {CALIB} for scale {scale}")

    matched = [p for p in points if p["channel"] == "matched"]
    tau_d, tau_max = BAND[scale]

    # The binding constraint: tau_d must clear every matched point by 2 s.d.
    binding = min(matched, key=lambda p: p["d"] - 2 * p["sd"])
    limit = binding["d"] - 2 * binding["sd"]
    margin = min((p["d"] - tau_d) / p["sd"] for p in matched)

    # Constraint (ii) in the published rule: exclude non-beneficial points.
    degrading = [p for p in points if p["gain"] < 1.0]

    admitted = [
        p for p in points
        if p["channel"] != "matched" and tau_d <= p["d"] <= tau_max
    ]
    rejected_weak = [
        p for p in points if p["channel"] == "weak" and p["d"] < tau_d
    ]

    return {
        "scale": scale,
        "n_points": len(points),
        "n_codewords": points[0]["n_cw"],
        "n_blocks": points[0]["n_blocks"],
        "floor": points[0]["floor"],
        "matched_d_min": min(p["d"] for p in matched),
        "matched_d_min_sd": min(matched, key=lambda p: p["d"])["sd"],
        "matched_d_max": max(p["d"] for p in matched),
        "matched_gain_min": min(p["gain"] for p in matched),
        "matched_gain_max": max(p["gain"] for p in matched),
        "weak_d_max": max(p["d"] for p in points if p["channel"] == "weak"),
        "binding_limit": limit,
        "tau_d": tau_d,
        "tau_d_ok": tau_d <= limit,
        "margin_sd": margin,
        "n_degrading": len(degrading),
        "gain_min_all": min(p["gain"] for p in points),
        "n_admitted_off_regime": len(admitted),
        "admitted_gain_min": min((p["gain"] for p in admitted), default=float("nan")),
        "n_rejected_weak": len(rejected_weak),
        "rejected_weak_gain_min": min((p["gain"] for p in rejected_weak), default=float("nan")),
        "rejected_weak_gain_max": max((p["gain"] for p in rejected_weak), default=float("nan")),
    }


TAU_D_GRID = (0.003, 0.005, 0.008, 0.011, 0.015, 0.016, 0.018, 0.020, 0.025, 0.030)


def tau_d_sensitivity(scale: str) -> list[tuple[float, float, int, int]]:
    """Effective gain on the test sweep as a function of tau_d.

    The gate admits a point iff tau_d <= d <= tau_d_max and gbar >= tau_g, so
    composing the per-point decisions gives the gain any tau_d would have
    produced without re-running the channel (``result_data
    .gate_composition``, shared with the tau_g table). Below the smallest test
    drift score the result is exactly flat. SER is the metric at 256 symbols
    and BER at 1024, as in the main tables.
    """
    return [(tau, *gate_composition(int(scale), tau_d=tau)) for tau in TAU_D_GRID]


def main() -> None:
    for scale in ("1024", "256"):
        r = place_lower_bound(scale)
        print(f"=== scale {r['scale']}: {r['n_points']} points, "
              f"{r['n_codewords']} codewords in {r['n_blocks']} blocks, "
              f"floor {r['floor']:.3e} ===")
        print(f"  matched drift score   {r['matched_d_min']:.4f} "
              f"(s.d. {r['matched_d_min_sd']:.4f}) .. {r['matched_d_max']:.4f}")
        print(f"  matched gain          {r['matched_gain_min']:.3f} .. {r['matched_gain_max']:.3f}")
        print(f"  weak-drift score max  {r['weak_d_max']:.4f}")
        print(f"  min_i(d_i - 2 s_i)    {r['binding_limit']:.4f}")
        print(f"  tau_d = {r['tau_d']}            "
              f"{'OK' if r['tau_d_ok'] else 'VIOLATES THE RULE'}, "
              f"clears every matched point by {r['margin_sd']:.2f} s.d.")
        print(f"  degrading points      {r['n_degrading']} "
              f"(smallest gain over all points {r['gain_min_all']:.3f})")
        print(f"  off-regime admitted   {r['n_admitted_off_regime']} "
              f"(min gain {r['admitted_gain_min']:.3f})")
        print(f"  weak points rejected  {r['n_rejected_weak']} "
              f"(gains {r['rejected_weak_gain_min']:.2f} .. {r['rejected_weak_gain_max']:.2f})")
        print()

    for scale in ("1024", "256"):
        rows = tau_d_sensitivity(scale)
        n_points = rows[0][3]
        print(f"=== tau_d sensitivity on the {scale}-symbol test sweep "
              f"({n_points} point x seed combinations) ===")
        print("  tau_d    mean gain   gate open")
        for tau, gain, opened, _ in rows:
            print(f"  {tau:<8.3f} {gain:<11.3f} {opened}/{n_points}")
        print()


if __name__ == "__main__":
    main()
