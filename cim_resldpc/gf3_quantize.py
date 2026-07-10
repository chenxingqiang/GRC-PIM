"""Float embedding <-> GF(3) symbol quantization for CiM storage."""
from __future__ import annotations

import numpy as np


def quantize_to_gf3(
    emb: np.ndarray,
    levels: tuple[float, float, float] = (-1.0, 0.0, 1.0),
) -> np.ndarray:
    """Map float embeddings to GF(3) symbols in {0, 1, 2}.

    Args:
      emb: (..., d) float array.
      levels: reconstruction centroids for symbols 0,1,2.

    Returns:
      symbols: (..., d) int64 in {0,1,2}.
    """
    emb = np.asarray(emb, dtype=np.float32)
    levels_arr = np.asarray(levels, dtype=np.float32)
    # Nearest centroid assignment
    # dist shape (..., d, 3)
    dist = np.abs(emb[..., None] - levels_arr)
    return dist.argmin(axis=-1).astype(np.int64)


def dequantize_from_gf3(
    symbols: np.ndarray,
    levels: tuple[float, float, float] = (-1.0, 0.0, 1.0),
) -> np.ndarray:
    """Map GF(3) symbols back to float centroids."""
    symbols = np.asarray(symbols, dtype=np.int64)
    levels_arr = np.asarray(levels, dtype=np.float32)
    if symbols.min() < 0 or symbols.max() > 2:
        raise ValueError("GF(3) symbols must be in {0,1,2}")
    return levels_arr[symbols]


def soft_ternary_targets(emb: np.ndarray) -> np.ndarray:
    """Soft assignment probabilities over 3 levels for quantization loss.

    Returns (..., d, 3) probabilities via softmax(-|x-level|).
    """
    emb = np.asarray(emb, dtype=np.float32)
    levels = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    logits = -np.abs(emb[..., None] - levels)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs.astype(np.float32)


def pack_symbols_for_ldpc(symbols: np.ndarray, code_n: int) -> np.ndarray:
    """Pad / truncate symbol vectors to LDPC codeword length n.

    Args:
      symbols: (B, d) GF(3) symbols.
      code_n: target codeword length (e.g. 288 for k=256 rate 8/9).

    Returns:
      (B, code_n) GF(3) array (zero-padded or truncated).
    """
    symbols = np.asarray(symbols, dtype=np.int64) % 3
    b, d = symbols.shape
    out = np.zeros((b, code_n), dtype=np.int64)
    n_copy = min(d, code_n)
    out[:, :n_copy] = symbols[:, :n_copy]
    return out
