"""Float embedding <-> GF(3) symbol quantization for CiM storage.

v1.1 finding (see SCIENTIFIC_REFLECTION.md): single-centroid ternary
quantization ({-1,0,1} per dim) collapses embedding geometry and can hide
LDPC gains. Default is **base-3 multi-symbol** packing (n_sym_per_dim=2 →
9 reconstruction levels), which preserves cosine structure while remaining
GF(3)-alphabet compatible with GRC-PIM LDPC.
"""
from __future__ import annotations

import numpy as np


def quantize_to_gf3(
    emb: np.ndarray,
    levels: tuple[float, float, float] = (-1.0, 0.0, 1.0),
) -> np.ndarray:
    """Legacy nearest-centroid map to one GF(3) symbol per dim (lossy)."""
    emb = np.asarray(emb, dtype=np.float32)
    levels_arr = np.asarray(levels, dtype=np.float32)
    dist = np.abs(emb[..., None] - levels_arr)
    return dist.argmin(axis=-1).astype(np.int64)


def dequantize_from_gf3(
    symbols: np.ndarray,
    levels: tuple[float, float, float] = (-1.0, 0.0, 1.0),
) -> np.ndarray:
    """Legacy inverse of nearest-centroid quantization."""
    symbols = np.asarray(symbols, dtype=np.int64)
    levels_arr = np.asarray(levels, dtype=np.float32)
    if symbols.min() < 0 or symbols.max() > 2:
        raise ValueError("GF(3) symbols must be in {0,1,2}")
    return levels_arr[symbols]


def quantize_base3(
    emb: np.ndarray,
    n_sym_per_dim: int = 2,
) -> np.ndarray:
    """Map each float dim in ~[-1,1] to `n_sym_per_dim` GF(3) digits (MSB first).

    Effective reconstruction levels = 3**n_sym_per_dim (9 for default).
    Returns symbols with shape (..., d * n_sym_per_dim).
    """
    if n_sym_per_dim < 1:
        raise ValueError("n_sym_per_dim must be >= 1")
    emb = np.asarray(emb, dtype=np.float32)
    n_levels = 3 ** n_sym_per_dim
    x = np.clip((emb + 1.0) * 0.5, 0.0, 1.0)
    idx = np.clip(np.rint(x * (n_levels - 1)), 0, n_levels - 1).astype(np.int64)
    digits = []
    tmp = idx.copy()
    for _ in range(n_sym_per_dim):
        digits.append(tmp % 3)
        tmp //= 3
    # MSB first
    return np.concatenate(digits[::-1], axis=-1).astype(np.int64)


def dequantize_base3(
    symbols: np.ndarray,
    n_sym_per_dim: int = 2,
    dim: int | None = None,
) -> np.ndarray:
    """Inverse of quantize_base3 → float in [-1, 1]."""
    symbols = np.asarray(symbols, dtype=np.int64)
    if symbols.min() < 0 or symbols.max() > 2:
        raise ValueError("GF(3) symbols must be in {0,1,2}")
    if dim is None:
        if symbols.shape[-1] % n_sym_per_dim != 0:
            raise ValueError("symbol length must be divisible by n_sym_per_dim")
        dim = symbols.shape[-1] // n_sym_per_dim
    expected = dim * n_sym_per_dim
    if symbols.shape[-1] != expected:
        raise ValueError(f"expected last dim {expected}, got {symbols.shape[-1]}")
    n_levels = 3 ** n_sym_per_dim
    blocks = np.split(symbols, n_sym_per_dim, axis=-1)
    idx = np.zeros(blocks[0].shape, dtype=np.int64)
    for block in blocks:
        idx = idx * 3 + block
    x = idx.astype(np.float32) / float(n_levels - 1)
    return (x * 2.0 - 1.0).astype(np.float32)


def soft_ternary_targets(emb: np.ndarray) -> np.ndarray:
    """Soft assignment over {-1,0,1} for legacy quant loss."""
    emb = np.asarray(emb, dtype=np.float32)
    levels = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    logits = -np.abs(emb[..., None] - levels)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs.astype(np.float32)


def pack_symbols_for_ldpc(symbols: np.ndarray, code_n: int) -> np.ndarray:
    """Pad / truncate symbol vectors to LDPC codeword length n."""
    symbols = np.asarray(symbols, dtype=np.int64) % 3
    b, d = symbols.shape
    out = np.zeros((b, code_n), dtype=np.int64)
    n_copy = min(d, code_n)
    out[:, :n_copy] = symbols[:, :n_copy]
    return out
