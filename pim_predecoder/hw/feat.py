"""Quantize GEMV feature vectors for GRC_PIM_TOP FEAT_IN[103:0] bus.

Feature extraction (13-D per symbol) runs on CPU / digital front-end; this module
defines the fixed-point contract between software and RTL. See FEATURE_OFFLOAD.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pim_predecoder.core.gemv_calibrator import build_gemv_features

# Per-dimension scale applied before INT8 saturation (matches training ranges).
FEAT_DIVISORS = np.array([
    20.0, 20.0, 20.0,          # llr_scale (B' / 20)
    1.0, 1.0, 1.0, 1.0,        # symbol histogram (already in [0,1])
    0.5, 0.5,                  # bit_mean in [0,1] -> centered via (x-0.5)/0.5
    1.0, 1.0,                  # bit_disagree in [0,1]
    1.1,                       # entropy in [0, ln 3]
    1.0,                       # column index in [-1,1]
], dtype=np.float32)

FEAT_OFFSETS = np.array([
    0, 0, 0,
    0, 0, 0, 0,
    -0.5, -0.5,
    0, 0,
    0,
    0,
], dtype=np.float32)


def float_feat_to_int8(feat_f32: np.ndarray) -> np.ndarray:
    """Map one symbol's float feature row to signed INT8 for FEAT_IN."""
    x = (feat_f32.astype(np.float32) + FEAT_OFFSETS) / FEAT_DIVISORS
    return np.clip(np.round(x * 127.0), -128, 127).astype(np.int8)


def pack_feat_bus(feat_i8: np.ndarray) -> int:
    """Pack 13 INT8 values into 104-bit FEAT_IN (LSB = feat[0])."""
    assert feat_i8.shape == (13,)
    word = 0
    for i, v in enumerate(feat_i8):
        word |= (int(v) & 0xFF) << (8 * i)
    return word


def features_from_readout(
    noisy_bits_3d: np.ndarray,
    p_bit: float,
    *,
    batch_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build float + INT8 features from (B, T, 2n) readout."""
    feats_f, llr_bp = build_gemv_features(noisy_bits_3d, p_bit)
    feats_i8 = np.stack(
        [float_feat_to_int8(feats_f[b, s]) for b in range(feats_f.shape[0])
         for s in range(feats_f.shape[1])],
        axis=0,
    ).reshape(feats_f.shape[0], feats_f.shape[1], 13)
    return feats_i8[batch_idx], llr_bp[batch_idx]


def export_feat_golden(
    noisy_bits_3d: np.ndarray,
    p_bit: float,
    out_path: Path,
) -> dict:
    """Write reference INT8 features for cosim / regression."""
    feats_i8, llr_bp = features_from_readout(noisy_bits_3d, p_bit, batch_idx=0)
    payload = {
        "p_bit": p_bit,
        "n_symbols": int(feats_i8.shape[0]),
        "features_i8": feats_i8.tolist(),
        "llr_bp": llr_bp.tolist(),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return payload
