"""Optional GRC-PIM GEMV residual LLR calibration hook."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pim_predecoder.core.gemv_calibrator import (
    GEMVResidualCalibrator,
    global_drift_score,
)
from pim_predecoder.paths import CACHE


DEFAULT_GEMV_CKPT = CACHE / (
    "gemv_k256_n288_r8_9_T4_d5e-03_b5e-03_s900_bs128_h24_seed1.pt"
)


def load_gemv_calibrator(
    checkpoint: Path | str | None = None,
    device: str = "cpu",
) -> GEMVResidualCalibrator:
    path = Path(checkpoint) if checkpoint else DEFAULT_GEMV_CKPT
    if not path.exists():
        # fall back to any k256 T4 checkpoint
        cands = sorted(CACHE.glob("gemv_k256_n288_r8_9_T4_*h24_seed1.pt"))
        if not cands:
            raise FileNotFoundError(f"no GEMV checkpoint at {path} or under {CACHE}")
        path = cands[0]
    model = GEMVResidualCalibrator()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def apply_gemv_llr(
    noisy_bits_3d: np.ndarray,
    p_bit: float,
    calibrator: GEMVResidualCalibrator,
    tau_d: float = 0.025,
    tau_g: float = 0.6,
    device: str = "cpu",
) -> tuple[np.ndarray, float, bool]:
    """Return calibrated LLR, mean gate, and whether GEMV path was used.

    Dual-gate policy matches GRC-PIM: use GEMV only if drift>=tau_d and gate>=tau_g;
    otherwise return B'-equivalent soft_llr from the calibrator anchor path by
    calling soft_llr then optionally ignoring delta when gate closed.
    """
    drift = global_drift_score(noisy_bits_3d)
    llr, gate_mean = calibrator.soft_llr(noisy_bits_3d, p_bit=p_bit, device=device)
    use = (drift >= tau_d) and (gate_mean >= tau_g)
    if not use:
        # Recompute pure B' by disabling residual: soft_llr always adds gated delta;
        # when gate is low the delta is already small; for strict fallback rebuild B'.
        from pim_predecoder.core.gemv_calibrator import soft_llr_product

        llr = soft_llr_product(noisy_bits_3d, p_bit=p_bit)
    return llr.astype(np.float32), float(gate_mean), bool(use)
