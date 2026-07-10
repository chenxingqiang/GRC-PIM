"""GF(3) LDPC protect / recover path for ternary embedding symbols.

Uses base-3 multi-symbol quantization by default so embedding geometry
survives the GF(3) alphabet constraint, then LDPC-protects the symbol
stream with the shipped NBLDPC matrices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pim_predecoder.core.codec import (
    GF,
    bp_decode,
    encode,
    load_HG,
)
from pim_predecoder.paths import REPO

from .gf3_quantize import (
    dequantize_base3,
    dequantize_from_gf3,
    quantize_base3,
    quantize_to_gf3,
)
from .noise_bridge import NoiseProfile, apply_embedding_cim_noise


@dataclass
class LDPCCode:
    H: np.ndarray
    G: np.ndarray
    k: int
    n: int

    @classmethod
    def load(cls, info_length: int = 256, rate: str = "8_9") -> "LDPCCode":
        H, G = load_HG(REPO, info_length=info_length, rate=rate)
        return cls(H=H, G=G, k=info_length, n=G.shape[1])


def pack_info_blocks(symbols: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    symbols = np.asarray(symbols, dtype=np.int64) % GF
    b, d = symbols.shape
    n_blocks = int(np.ceil(d / k))
    pad = n_blocks * k - d
    if pad:
        symbols = np.pad(symbols, ((0, 0), (0, pad)), mode="constant")
    blocks = symbols.reshape(b * n_blocks, k)
    return blocks, n_blocks


def unpack_info_blocks(blocks: np.ndarray, batch: int, dim: int, n_blocks: int) -> np.ndarray:
    flat = blocks.reshape(batch, n_blocks * blocks.shape[-1])
    return flat[:, :dim]


def soft_expectation_from_llr_base3(
    llr_info: np.ndarray,
    n_sym_per_dim: int,
    emb_dim: int,
) -> np.ndarray:
    """Soft-reconstruct floats from per-symbol LLRs over base-3 digits.

    For each embedding dim, digits form a base-3 integer; we compute the
    expected index under independent digit posteriors, then map to [-1,1].
    """
    # llr_info: (B, n_sym_total, 3) for info symbols (may include pad)
    B = llr_info.shape[0]
    need = emb_dim * n_sym_per_dim
    llr = llr_info[:, :need, :]
    m = llr.max(axis=-1, keepdims=True)
    probs = np.exp(llr - m)
    probs /= probs.sum(axis=-1, keepdims=True) + 1e-8  # (B, need, 3)

    # Reshape to (B, emb_dim, n_sym, 3) MSB-first
    probs = probs.reshape(B, emb_dim, n_sym_per_dim, 3)
    # Expected digit values
    digit_vals = np.arange(3, dtype=np.float32)
    exp_digits = (probs * digit_vals).sum(axis=-1)  # (B, emb_dim, n_sym)
    # Expected index
    idx = np.zeros((B, emb_dim), dtype=np.float32)
    for s in range(n_sym_per_dim):
        idx = idx * 3.0 + exp_digits[:, :, s]
    n_levels = 3 ** n_sym_per_dim
    x = idx / float(n_levels - 1)
    return (x * 2.0 - 1.0).astype(np.float32)


@dataclass
class RecoveredEmbedding:
    symbols_bprime: np.ndarray
    symbols_bp: np.ndarray
    emb_soft: np.ndarray
    emb_bp: np.ndarray
    emb_bprime: np.ndarray
    drift_score: float
    gate_proxy: float
    used_bp: bool
    n_codewords: int
    n_sym_per_dim: int
    emb_dim: int


def protect_and_recover(
    emb: np.ndarray,
    profile: NoiseProfile,
    code: LDPCCode | None = None,
    levels: tuple[float, float, float] = (-1.0, 0.0, 1.0),
    rng: np.random.Generator | None = None,
    use_bp: bool = True,
    bp_iters: int = 5,
    n_sym_per_dim: int = 2,
    quant_mode: str = "base3",
) -> RecoveredEmbedding:
    """Quantize → LDPC encode → CiM noise → B'/BP recover → float embedding.

    No clean-embedding oracle. Default quant_mode='base3' preserves geometry.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if code is None:
        code = LDPCCode.load()
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb[None, :]
    batch, dim = emb.shape

    if quant_mode == "base3":
        symbols = quantize_base3(emb, n_sym_per_dim=n_sym_per_dim)
        sym_dim = dim * n_sym_per_dim
    elif quant_mode == "centroid":
        symbols = quantize_to_gf3(emb, levels=levels)
        sym_dim = dim
        n_sym_per_dim = 1
    else:
        raise ValueError(f"unknown quant_mode={quant_mode!r}")

    info_blocks, n_blocks = pack_info_blocks(symbols, code.k)
    codewords = encode(info_blocks, code.G)
    readout = apply_embedding_cim_noise(codewords, profile, rng=rng, H=code.H)
    llr = readout.llr_bp

    hard_bprime = llr.argmax(axis=-1)
    info_bprime = hard_bprime[:, : code.k]

    used_bp = False
    if use_bp:
        x_hat, post_llr, _ = bp_decode(llr, code.H, n_iter=bp_iters)
        info_bp = x_hat[:, : code.k]
        used_bp = True
    else:
        info_bp = info_bprime
        post_llr = llr

    sym_bprime = unpack_info_blocks(info_bprime, batch, sym_dim, n_blocks)
    sym_bp = unpack_info_blocks(info_bp, batch, sym_dim, n_blocks)

    if quant_mode == "base3":
        emb_bprime = dequantize_base3(sym_bprime, n_sym_per_dim=n_sym_per_dim, dim=dim)
        emb_bp = dequantize_base3(sym_bp, n_sym_per_dim=n_sym_per_dim, dim=dim)
        emb_soft = soft_expectation_from_llr_base3(
            llr[:, : code.k, :], n_sym_per_dim=n_sym_per_dim, emb_dim=dim,
        )
        # If multiple blocks, soft_expectation only used first codeword's info
        # for dim*n_sym <= k (true for dim<=128 with n_sym=2, k=256).
        if sym_dim > code.k:
            # Fall back: hard B' soft-less reconstruction for overflow dims
            emb_soft = emb_bprime
    else:
        emb_bprime = dequantize_from_gf3(sym_bprime, levels=levels)
        emb_bp = dequantize_from_gf3(sym_bp, levels=levels)
        # Soft centroid expectation
        levels_arr = np.asarray(levels, dtype=np.float32)
        llr_i = llr[:, :dim, :]
        m = llr_i.max(axis=-1, keepdims=True)
        probs = np.exp(llr_i - m)
        probs /= probs.sum(axis=-1, keepdims=True) + 1e-8
        emb_soft = (probs * levels_arr).sum(axis=-1).astype(np.float32)

    probs_g = np.exp(llr[:, : code.k, :] - llr[:, : code.k, :].max(axis=-1, keepdims=True))
    probs_g /= probs_g.sum(axis=-1, keepdims=True) + 1e-8
    ent = -(probs_g * np.log(probs_g + 1e-8)).sum(axis=-1).mean()
    gate_proxy = float(np.clip(1.0 - ent / np.log(3.0), 0.0, 1.0))

    return RecoveredEmbedding(
        symbols_bprime=sym_bprime,
        symbols_bp=sym_bp,
        emb_soft=emb_soft.astype(np.float32),
        emb_bp=emb_bp.astype(np.float32),
        emb_bprime=emb_bprime.astype(np.float32),
        drift_score=float(readout.drift_score),
        gate_proxy=gate_proxy,
        used_bp=used_bp,
        n_codewords=int(info_blocks.shape[0]),
        n_sym_per_dim=n_sym_per_dim,
        emb_dim=dim,
    )


def symbol_error_rate(clean_symbols: np.ndarray, recovered: np.ndarray) -> float:
    return float(np.mean(np.asarray(clean_symbols) != np.asarray(recovered)))
