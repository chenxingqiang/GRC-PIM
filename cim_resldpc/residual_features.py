"""Residual soft-evidence features for retrieval reranking."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .noise_bridge import NoisyReadout, residual_gate_weight, should_skip_gemv


@dataclass
class ResidualEvidence:
    drift_score: float
    gate_mean: float
    delta_mag: float
    residual_conf: float
    used_gemv: bool
    weight: float


def evidence_from_readout(
    readout: NoisyReadout,
    gate_mean: float = 0.5,
    delta_mag: float = 0.0,
    tau_d: float = 0.025,
    tau_g: float = 0.6,
    tau_skip: float = 0.015,
) -> ResidualEvidence:
    """Build residual confidence used in hybrid rerank.

    residual_conf high when calibrated recovery is trusted:
      - moderate/high gate when GEMV is used
      - or low drift (clean channel) when GEMV is skipped
    """
    skip = should_skip_gemv(readout.drift_score, tau_skip=tau_skip)
    used = (not skip) and (readout.drift_score >= tau_d) and (gate_mean >= tau_g)
    w = residual_gate_weight(
        readout.drift_score, gate_mean, tau_d=tau_d, tau_g=tau_g,
    )
    if skip:
        # Clean channel: high confidence without GEMV
        conf = float(np.clip(1.0 - readout.drift_score / max(tau_skip, 1e-8), 0.0, 1.0))
    elif used:
        conf = float(np.clip(0.5 * gate_mean + 0.5 * (1.0 - min(delta_mag, 2.0) / 2.0), 0.0, 1.0))
    else:
        # Drift present but GEMV closed → lower confidence
        conf = float(np.clip(0.3 * (1.0 - w), 0.0, 1.0))
    return ResidualEvidence(
        drift_score=float(readout.drift_score),
        gate_mean=float(gate_mean),
        delta_mag=float(delta_mag),
        residual_conf=conf,
        used_gemv=bool(used),
        weight=w,
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise or broadcast cosine. a (Q,d), b (N,d) -> (Q,N) or a (d,), b (N,d)->(N,)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
        squeeze = True
    else:
        squeeze = False
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    sim = a_n @ b_n.T
    if squeeze:
        return sim[0]
    return sim
