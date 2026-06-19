"""Hardware-aligned LLR front-end for GRC-PIM.

Maps 3-bit ADC words to signed integer LLRs using the same case tables as
``deps/NBLDPC_PIM/NBLDPC_Verilog/LLR_TRANS.v`` (MESSAGE_TRANS / CHECK_TRANS).
Multi-cycle B' is accumulated in the integer domain, then converted to float
for ``bp_decode()``.
"""
from __future__ import annotations

import numpy as np

from pim_predecoder.core.codec import init_llr_from_bits

# Signed 3-bit LLR values from MESSAGE_TRANS in LLR_TRANS.v
# Keys: (adc_word, cim_e) -> [llr_sym0, llr_sym1, llr_sym2]
_MESSAGE_LUT: dict[tuple[int, bool], list[int]] = {
    (0b000, False): [2, 1, 0],
    (0b000, True): [2, 1, 0],
    (0b001, False): [1, 2, 0],
    (0b001, True): [1, 2, 1],
    (0b010, False): [1, 1, 2],
    (0b010, True): [1, 1, 2],
    (0b011, False): [2, 1, 1],
    (0b011, True): [2, 1, 1],
    (0b100, False): [1, 2, 0],
    (0b100, True): [1, 2, 0],
}


def _sign3(x: int) -> int:
    """Interpret a 3-bit pattern as signed two's complement."""
    x &= 0x7
    return x - 8 if x >= 4 else x


def message_llr_int(adc_word: int, *, cim_e: bool = True) -> np.ndarray:
    """Return integer LLR vector (3,) for one 3-bit ADC codeword."""
    key = (adc_word & 0x7, cim_e)
    if key not in _MESSAGE_LUT:
        return np.zeros(3, dtype=np.int16)
    return np.array([_sign3(v) for v in _MESSAGE_LUT[key]], dtype=np.int16)


def bits_to_adc_word(bit0: int, bit1: int, *, msb_first: bool = True) -> int:
    """Pack two readout bits into a 3-bit ADC-style word (MSB..LSB)."""
    if msb_first:
        return ((bit1 & 1) << 2) | ((bit0 & 1) << 1) | 0
    return ((bit0 & 1) << 2) | ((bit1 & 1) << 1) | 0


def init_llr_from_adc_hw(
    noisy_bits: np.ndarray,
    *,
    cim_e: bool = True,
    llr_scale: float = 1.0,
) -> np.ndarray:
    """Hardware-aligned per-symbol LLR from 2-bit GF(3) readout.

    Args:
      noisy_bits: (B, 2*n) binary readout.
      cim_e: CIM enable flag (matches ``CIM_E`` in Verilog).
      llr_scale: multiply integer LLR before float conversion.
    Returns:
      llr: (B, n, 3) float32, max-normalised per symbol.
    """
    bits = noisy_bits.reshape(noisy_bits.shape[0], -1, 2)
    batch, n, _ = bits.shape
    out = np.zeros((batch, n, 3), dtype=np.float32)
    for b in range(batch):
        for i in range(n):
            adc = bits_to_adc_word(int(bits[b, i, 0]), int(bits[b, i, 1]))
            out[b, i] = message_llr_int(adc, cim_e=cim_e).astype(np.float32) * llr_scale
    out -= out.max(axis=-1, keepdims=True)
    return out


def soft_llr_product_hw(
    noisy_bits_3d: np.ndarray,
    *,
    cim_e: bool = True,
    llr_scale: float = 1.0,
) -> np.ndarray:
    """T-cycle B' using hardware MESSAGE_TRANS LLR accumulation."""
    acc = None
    for t in range(noisy_bits_3d.shape[1]):
        llr_t = init_llr_from_adc_hw(noisy_bits_3d[:, t, :], cim_e=cim_e, llr_scale=llr_scale)
        acc = llr_t if acc is None else acc + llr_t
    assert acc is not None
    acc -= acc.max(axis=-1, keepdims=True)
    return acc.astype(np.float32)


def hw_drift_score_q16(noisy_bits_3d: np.ndarray) -> int:
    """Global drift score in Q16 (0..65535 ≈ 0..1), matching DRIFT_DETECTOR.v."""
    if noisy_bits_3d.shape[1] < 2:
        return 0
    disagree = 0
    total = 0
    for t in range(1, noisy_bits_3d.shape[1]):
        xor = np.not_equal(noisy_bits_3d[:, t], noisy_bits_3d[:, t - 1])
        disagree += int(xor.sum())
        total += xor.size
    if total == 0:
        return 0
    frac = disagree / total
    return int(round(min(1.0, frac) * 65535))


def int8_to_llr3(v: int) -> int:
    """Saturate GRC int8 LLR to ECC 3-bit signed prior (LLR_INT8_TO_ECC3.v)."""
    v = int(v)
    if v > 3:
        return 3
    if v < -4:
        return -4
    return v


def int8_vec_to_ecc9(llr_i8: np.ndarray) -> np.ndarray:
    """Quantize (3,) int8 GRC output to (3,) 3-bit signed LLRs, flat ECC order c0,c1,c2."""
    llr_i8 = np.asarray(llr_i8, dtype=np.int32).reshape(3)
    return np.array([int8_to_llr3(x) for x in llr_i8], dtype=np.int32)


def pack_ecc_input_llr(
    info_llr_i8: np.ndarray,
    buffer_llr9: np.ndarray,
    *,
    info_num: int,
) -> np.ndarray:
    """Build per-symbol (SYMBOL_NUM, 3) ECC prior LLRs (3-bit signed as int)."""
    symbol_num = buffer_llr9.shape[0]
    out = buffer_llr9.copy()
    n = min(info_num, info_llr_i8.shape[0])
    for s in range(n):
        out[s] = int8_vec_to_ecc9(info_llr_i8[s])
    return out


def llr9_to_bp_llr(llr9: np.ndarray) -> np.ndarray:
    """Convert (n, 3) hardware int3 LLRs to (1, n, 3) float priors for bp_decode."""
    x = llr9.astype(np.float32)
    x -= x.max(axis=-1, keepdims=True)
    return x[None, ...]


def pack_input_symbol_bus(info_symbols: np.ndarray, *, symbol_bit: int = 3) -> list[int]:
    """Pack info GF(3) symbols into flat INPUT_SYMBOL bit bus (LSB-first per symbol)."""
    flat: list[int] = []
    for sym in info_symbols.reshape(-1):
        v = int(sym) & ((1 << symbol_bit) - 1)
        for b in range(symbol_bit):
            flat.append((v >> b) & 1)
    return flat


def codeword_llr9(codeword: np.ndarray, *, strong: int = 3, weak: int = -3) -> np.ndarray:
    """Strong 3-bit priors from hard GF(3) codeword symbols."""
    llr9 = np.full((codeword.shape[0], 3), weak, dtype=np.int32)
    for s, sym in enumerate(codeword):
        llr9[s, int(sym)] = strong
    return llr9


def compare_llr_frontends(
    noisy_bits_3d: np.ndarray,
    p_bit: float,
    *,
    cim_e: bool = True,
) -> dict[str, float]:
    """Compare probabilistic vs hardware LLR on the same readout batch."""
    llr_prob = sum(
        init_llr_from_bits(noisy_bits_3d[:, t, :], p_bit=p_bit)
        for t in range(noisy_bits_3d.shape[1])
    )
    llr_prob -= llr_prob.max(axis=-1, keepdims=True)
    llr_hw = soft_llr_product_hw(noisy_bits_3d, cim_e=cim_e)
    diff = np.abs(llr_prob - llr_hw)
    return {
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "prob_argmax_agree": float(
            np.mean(np.argmax(llr_prob, axis=-1) == np.argmax(llr_hw, axis=-1))
        ),
    }
