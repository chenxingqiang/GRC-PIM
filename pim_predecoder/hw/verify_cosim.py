"""Cosimulation helpers: Python float pipeline vs hardware-aligned LLR / Q16 gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pim_predecoder.core.codec import bp_decode, encode, load_HG
from pim_predecoder.core.gemv_calibrator import GEMVResidualCalibrator, soft_llr_product
from pim_predecoder.core.noise3d import apply_pim_noise_3d
from pim_predecoder.hw.llr import (
    compare_llr_frontends,
    hw_drift_score_q16,
    soft_llr_product_hw,
)
from pim_predecoder.paths import CACHE, RTL


def run_verify(
    *,
    info_length: int = 1024,
    rate: str = "16_17",
    p_bit: float = 5e-3,
    n_codewords: int = 32,
    n_cycles: int = 4,
    seed: int = 1,
    checkpoint: Path | None = None,
) -> dict:
    H, G = load_HG(".", info_length=info_length, rate=rate)
    rng = np.random.default_rng(seed)
    info = rng.integers(0, 3, size=(n_codewords, G.shape[0]))
    clean = encode(info, G)
    noisy = apply_pim_noise_3d(
        clean, n_cycles=n_cycles, p_indep=p_bit,
        p_drift=5e-3, p_row_burst=5e-3, rng=rng,
    )
    bits = noisy.reshape(n_codewords, n_cycles, -1)

    cmp = compare_llr_frontends(bits, p_bit)
    llr_sw = soft_llr_product(bits, p_bit)
    llr_hw = soft_llr_product_hw(bits)

    x_sw, _, _ = bp_decode(llr_sw, H, n_iter=3, early_stop=True)
    x_hw, _, _ = bp_decode(llr_hw, H, n_iter=3, early_stop=True)
    ber_sw = float(np.not_equal(x_sw, clean).mean())
    ber_hw = float(np.not_equal(x_hw, clean).mean())

    out = {
        "info_length": info_length,
        "rate": rate,
        "p_bit": p_bit,
        "n_codewords": n_codewords,
        "llr_frontend": cmp,
        "ber_sw_llr": ber_sw,
        "ber_hw_llr": ber_hw,
        "drift_q16_mean": float(np.mean([hw_drift_score_q16(bits[i:i+1]) for i in range(n_codewords)])),
    }

    if checkpoint and checkpoint.exists():
        import torch
        model = GEMVResidualCalibrator(hidden=24)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        llr_gemv, gate = model.soft_llr(bits, p_bit)
        x_g, _, _ = bp_decode(llr_gemv, H, n_iter=3, early_stop=True)
        out["ber_gemv_float"] = float(np.not_equal(x_g, clean).mean())
        out["gate_mean"] = gate

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--info-length", type=int, default=1024)
    ap.add_argument("--rate", default="16_17")
    ap.add_argument("--p-bit", type=float, default=5e-3)
    ap.add_argument("--n-codewords", type=int, default=32)
    ap.add_argument("--checkpoint", type=Path,
                    default=CACHE / "gemv_k1024_n1088_r16_17_T4_d5e-03_b5e-03_s600_bs48_h24_seed1.pt")
    ap.add_argument("--out", type=Path, default=RTL / "cosim_report.json")
    args = ap.parse_args()
    report = run_verify(
        info_length=args.info_length, rate=args.rate, p_bit=args.p_bit,
        n_codewords=args.n_codewords, checkpoint=args.checkpoint,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
