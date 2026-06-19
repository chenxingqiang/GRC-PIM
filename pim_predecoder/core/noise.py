"""PIM bit-flip noise model used by the simulation.

Each GF(3) symbol s is mapped to its 2-bit binary representation
(00, 01, 10). The hardware reads the bits with independent flip probability
p_bit. The composite noise model also supports a column-correlated
component to mimic IR-drop / common-mode noise.
"""
from __future__ import annotations

import numpy as np


def symbols_to_bits(sym: np.ndarray) -> np.ndarray:
    """(B, n) GF(3) symbols -> (B, 2*n) binary bits, MSB first."""
    bit1 = (sym >> 1) & 1
    bit0 = sym & 1
    out = np.stack([bit1, bit0], axis=-1)
    return out.reshape(sym.shape[0], -1)


def bits_to_symbols(bits: np.ndarray) -> np.ndarray:
    """(B, 2*n) binary bits -> (B, n) integers in {0,1,2,3}.

    Note: 3 is illegal in GF(3) and indicates a multi-bit error.
    """
    bits = bits.reshape(bits.shape[0], -1, 2)
    return bits[..., 0] * 2 + bits[..., 1]


def apply_pim_noise(
    code_sym: np.ndarray,
    p_bit: float,
    p_col_corr: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply realistic PIM bit-flip noise to GF(3) codewords.

    Args:
      code_sym:    (B, n) original GF(3) codewords.
      p_bit:       per-bit flip probability (independent component).
      p_col_corr:  probability that an entire column flips one bit
                    (models IR-drop / common-mode noise across rows).
      rng:         numpy random generator.
    Returns:
      noisy_bits:  (B, 2*n) noisy binary readout.
    """
    if rng is None:
        rng = np.random.default_rng()
    B, n = code_sym.shape
    bits = symbols_to_bits(code_sym).astype(np.int64)

    # Independent bit flips
    flip = rng.random((B, 2 * n)) < p_bit
    bits ^= flip.astype(np.int64)

    # Column-correlated noise: for each column, with prob p_col_corr
    # flip a random bit of that column on every row in the batch.
    if p_col_corr > 0:
        col_hit = rng.random(2 * n) < p_col_corr
        if col_hit.any():
            bits[:, col_hit] ^= 1
    return bits


def estimate_ber(noisy_bits: np.ndarray, clean_sym: np.ndarray) -> float:
    """Bit-level BER between noisy bits and the clean symbols' encoding."""
    clean_bits = symbols_to_bits(clean_sym)
    return float((noisy_bits != clean_bits).mean())
