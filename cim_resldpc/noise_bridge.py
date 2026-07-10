"""Bridge GRC-PIM multi-cycle noise and drift scores to embedding codes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pim_predecoder.core.noise3d import apply_pim_noise_3d
from pim_predecoder.core.gemv_calibrator import (
    enhanced_drift_scores,
    global_drift_score,
    soft_llr_product,
)
from pim_predecoder.core.noise import bits_to_symbols


@dataclass
class NoiseProfile:
    p_indep: float = 5e-3
    p_drift: float = 5e-3
    p_row_burst: float = 5e-3
    n_cycles: int = 4
    drift_persist: int = 3


@dataclass
class NoisyReadout:
    noisy_bits_3d: np.ndarray          # (B, T, 2n)
    drift_score: float
    scores: dict[str, float]
    recovered_symbols: np.ndarray      # (B, n) majority / soft argmax proxy
    llr_bp: np.ndarray                 # (B, n, 3)


def apply_embedding_cim_noise(
    code_sym: np.ndarray,
    profile: NoiseProfile,
    rng: np.random.Generator | None = None,
    H: np.ndarray | None = None,
) -> NoisyReadout:
    """Pass GF(3) embedding codes through multi-cycle PIM noise channel.

    Args:
      code_sym: (B, n) clean GF(3) symbols stored in CiM.
      profile: noise hyperparameters.
      rng: optional RNG.
      H: optional parity-check matrix for syndrome instability score.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    noisy = apply_pim_noise_3d(
        code_sym,
        n_cycles=profile.n_cycles,
        p_indep=profile.p_indep,
        p_drift=profile.p_drift,
        drift_persist=profile.drift_persist,
        p_row_burst=profile.p_row_burst,
        rng=rng,
    )
    drift = global_drift_score(noisy)
    scores = enhanced_drift_scores(noisy, H=H)
    llr_bp = soft_llr_product(noisy, p_bit=profile.p_indep)
    # Soft recovery: argmax of B' LLR
    recovered = llr_bp.argmax(axis=-1).astype(np.int64)
    return NoisyReadout(
        noisy_bits_3d=noisy,
        drift_score=drift,
        scores=scores,
        recovered_symbols=recovered,
        llr_bp=llr_bp,
    )


def residual_gate_weight(
    drift_score: float,
    gate_mean: float,
    tau_d: float = 0.025,
    tau_g: float = 0.6,
) -> float:
    """Dynamic residual regularizer weight in [0, 1].

    High drift and high model-confidence gate → stronger residual alignment.
    """
    drift_term = float(np.clip(drift_score / max(tau_d, 1e-8), 0.0, 2.0)) / 2.0
    gate_term = float(np.clip(gate_mean, 0.0, 1.0))
    return float(np.clip(0.5 * drift_term + 0.5 * gate_term, 0.0, 1.0))


def should_skip_gemv(drift_score: float, tau_skip: float = 0.015) -> bool:
    """Latency prune: skip full GEMV residual path under low drift."""
    return float(drift_score) < float(tau_skip)


def majority_symbols_from_bits(noisy_bits_3d: np.ndarray) -> np.ndarray:
    """Majority-vote symbols across cycles: (B, T, 2n) -> (B, n)."""
    from pim_predecoder.core.noise3d import collapse_majority

    bits = collapse_majority(noisy_bits_3d)
    return bits_to_symbols(bits) % 3
