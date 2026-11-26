"""Area-projection sensitivity for the GEMV calibrator (Config B).

The manuscript projects the yosys cell area from Nangate45 to 28 nm by
converting the whole cell area through the NAND2 area ratio, dividing by a
placement utilisation, and adding the weight bank. That treats flip-flops as
scaling like combinational cells, which is optimistic: flip-flop area scales
roughly with feature size (28/45 = 0.622), not with NAND2 area
(0.32/0.798 = 0.401).

This script quantifies the resulting uncertainty over three axes:
  * placement utilisation u in {0.40, 0.60, 0.80};
  * cell-area uncertainty delta in {-20%, 0, +20%};
  * sequential share s in {0.30, 0.40, 0.50} (measured: 39.9%),
and reports, for each cell, the Config B area (199 time-shared units) and the
largest MAC array that still fits the 0.05 mm^2 budget.

Run from the repo root:
  python3 pim_predecoder/results/area_sensitivity/compute_area_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from result_data import SYNTH  # noqa: E402

RTL = ROOT / "rtl" / "synth_area.json"
BUDGET_MM2 = 0.05
N_UNITS = 199
FEATURE_RATIO = 28.0 / 45.0


def load() -> dict:
    return json.loads(RTL.read_text())


def area_mm2(hw: dict, n_macs: int, u: float, delta: float,
             seq_share: float, seq_ratio: float) -> float:
    """Projected 28 nm area with a separate scaling for the sequential share."""
    a45 = hw["area_um2_liberty"] + (n_macs - 1) * SYNTH["mac_area_um2_ng45"]
    r_c = hw["nand2_um2_target"] / hw["nand2_um2_liberty"]
    cells = a45 * ((1.0 - seq_share) * r_c + seq_share * seq_ratio)
    return (cells * (1.0 + delta) / u + hw["weight_sram_um2"]) / 1e6


def max_macs(hw: dict, u: float, delta: float, seq_share: float,
             seq_ratio: float, budget: float = BUDGET_MM2) -> int:
    best = 0
    for n in range(1, 398):
        if area_mm2(hw, n, u, delta, seq_share, seq_ratio) <= budget:
            best = n
    return best


def reference(hw: dict) -> None:
    """Config B area and MAC ceiling under the two scaling assumptions."""
    r_c = hw["nand2_um2_target"] / hw["nand2_um2_liberty"]
    seq_share = hw["sequential_um2"] / hw["area_um2_liberty"]
    all_nand2 = area_mm2(hw, N_UNITS, 0.60, 0.0, 0.0, r_c)
    ff_aware = area_mm2(hw, N_UNITS, 0.60, 0.0, seq_share, FEATURE_RATIO)
    print(f"measured sequential share s = {seq_share:.3f}")
    print(f"all-NAND2 reference  : {all_nand2:.3f} mm^2, "
          f"maxN = {max_macs(hw, 0.60, 0.0, 0.0, r_c)}")
    print(f"FF-aware reference   : {ff_aware:.3f} mm^2, "
          f"maxN = {max_macs(hw, 0.60, 0.0, seq_share, FEATURE_RATIO)}")
    print()


def main() -> None:
    hw = load()
    reference(hw)
    print("u    delta | s=30% area/maxN | s=40% area/maxN | s=50% area/maxN")
    for u in (0.40, 0.60, 0.80):
        for d in (-0.20, 0.0, 0.20):
            cells = []
            for s in (0.30, 0.40, 0.50):
                a = area_mm2(hw, N_UNITS, u, d, s, FEATURE_RATIO)
                cells.append(f"{a:.3f}/{max_macs(hw, u, d, s, FEATURE_RATIO):>3d}")
            print(f"{int(u * 100):>3d}% {int(d * 100):+4d}% | " + " | ".join(cells))


if __name__ == "__main__":
    main()
