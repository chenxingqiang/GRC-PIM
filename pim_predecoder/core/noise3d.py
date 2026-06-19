"""Multi-cycle PIM noise model with temporal & spatial correlation.

Adds a temporal axis to the bit-flip channel so a 3D CNN has something
non-trivial to exploit. The same code grid is "read" T times in a row;
the noise model is the sum of:

  - independent per-(row, time, bit) flips at rate p_indep,
  - column-correlated drift: a per-column bias that persists for K cycles,
    then resamples (mimics IR-drop / thermal drift),
  - row-correlated burst: occasionally a whole row flips one bit
    (mimics common-mode read disturbance).

The clean GF(3) symbols are constant across the T cycles (we are reading
the *same* MAC result T times to enable cycle averaging); only the noise
varies.
"""
from __future__ import annotations

import numpy as np

from .noise import bits_to_symbols, symbols_to_bits


def apply_pim_noise_3d(
    code_sym: np.ndarray,
    n_cycles: int,
    p_indep: float,
    p_drift: float = 0.0,
    drift_persist: int = 3,
    p_row_burst: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate (B, T, 2*n) noisy bit reads.

    Args:
      code_sym:        (B, n) clean GF(3) codewords.
      n_cycles:        T, number of consecutive MAC cycles to simulate.
      p_indep:         per-bit independent flip probability.
      p_drift:         probability that a column has a persistent biased
                       flip during any cycle.
      drift_persist:   how many consecutive cycles a drift event lasts.
      p_row_burst:     per-row, per-cycle probability of a 1-bit burst.
    """
    if rng is None:
        rng = np.random.default_rng()
    B, n = code_sym.shape
    bits = symbols_to_bits(code_sym).astype(np.int64)         # (B, 2n)
    out = np.broadcast_to(bits[:, None, :], (B, n_cycles, 2 * n)).copy()

    # 1) independent flips
    flip = rng.random(out.shape) < p_indep
    out ^= flip.astype(np.int64)

    # 2) column drift: pick events (col, start_t) and flip that column
    #    in [start_t, start_t + drift_persist) cycles.
    if p_drift > 0:
        n_events = rng.binomial(2 * n * n_cycles, p_drift)
        if n_events:
            cols = rng.integers(0, 2 * n, size=n_events)
            starts = rng.integers(0, n_cycles, size=n_events)
            for c, s in zip(cols, starts):
                end = min(n_cycles, s + drift_persist)
                out[:, int(s):int(end), int(c)] ^= 1

    # 3) row burst: per-row, per-cycle, with prob p_row_burst flip 1 bit
    if p_row_burst > 0:
        burst = rng.random((B, n_cycles)) < p_row_burst
        if burst.any():
            idx = np.where(burst)
            cols = rng.integers(0, 2 * n, size=idx[0].shape[0])
            out[idx[0], idx[1], cols] ^= 1

    return out


def collapse_majority(noisy_bits_3d: np.ndarray) -> np.ndarray:
    """Bit-wise majority vote across cycles -> (B, 2n) bits.

    This is the classical "average T reads" defence and is the right
    baseline to compare a 3D-CNN against.
    """
    return (noisy_bits_3d.mean(axis=1) >= 0.5).astype(np.int64)
