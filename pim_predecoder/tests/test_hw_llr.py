"""Tests for pim_predecoder.hw.llr."""
from __future__ import annotations

import numpy as np

from pim_predecoder.core.codec import init_llr_from_bits
from pim_predecoder.core.noise3d import apply_pim_noise_3d
from pim_predecoder.hw.llr import (
    bits_to_adc_word,
    codeword_llr9,
    compare_llr_frontends,
    hw_drift_score_q16,
    init_llr_from_adc_hw,
    int8_to_llr3,
    int8_vec_to_ecc9,
    llr9_to_bp_llr,
    message_llr_int,
    pack_ecc_input_llr,
    pack_input_symbol_bus,
    soft_llr_product_hw,
)


def test_message_llr_int_lut():
    llr = message_llr_int(0b010, cim_e=True)
    assert llr.tolist() == [1, 1, 2]


def test_bits_to_adc_word():
    assert bits_to_adc_word(1, 0) == 0b010
    assert bits_to_adc_word(0, 1) == 0b100


def test_int8_to_llr3_saturation():
    assert int8_to_llr3(10) == 3
    assert int8_to_llr3(-10) == -4
    assert int8_to_llr3(2) == 2


def test_int8_vec_to_ecc9():
    out = int8_vec_to_ecc9(np.array([1, -2, 3], dtype=np.int8))
    assert out.tolist() == [1, -2, 3]


def test_hw_drift_score_q16():
    identical = np.zeros((1, 4, 4), dtype=np.int8)
    assert hw_drift_score_q16(identical) == 0
    flip = np.array([[[0, 0, 0, 0], [1, 1, 1, 1]]], dtype=np.int8)
    assert hw_drift_score_q16(flip) == 65535


def test_init_llr_from_adc_hw_shape(rng):
    bits = rng.integers(0, 2, size=(2, 64))
    llr = init_llr_from_adc_hw(bits)
    assert llr.shape == (2, 32, 3)


def test_soft_llr_product_hw(rng):
    sym = rng.integers(0, 3, size=(2, 16))
    noisy = apply_pim_noise_3d(sym, 3, 0.01, rng=rng)
    llr = soft_llr_product_hw(noisy)
    assert llr.shape == (2, 16, 3)


def test_compare_llr_frontends(rng):
    sym = rng.integers(0, 3, size=(4, 32))
    noisy = apply_pim_noise_3d(sym, 4, 0.01, rng=rng)
    cmp = compare_llr_frontends(noisy, 0.01)
    assert "prob_argmax_agree" in cmp
    assert 0.0 <= cmp["prob_argmax_agree"] <= 1.0


def test_pack_ecc_input_llr():
    info = np.array([[1, 2, 3]], dtype=np.int8)
    buf = np.full((4, 3), -1, dtype=np.int32)
    merged = pack_ecc_input_llr(info, buf, info_num=1)
    assert merged[0].tolist() == [1, 2, 3]
    assert merged[1].tolist() == [-1, -1, -1]


def test_llr9_to_bp_llr():
    llr9 = np.array([[3, -1, -3], [0, 2, -2]], dtype=np.int32)
    bp = llr9_to_bp_llr(llr9)
    assert bp.shape == (1, 2, 3)


def test_pack_input_symbol_bus():
    syms = np.array([0, 1, 2], dtype=np.int64)
    flat = pack_input_symbol_bus(syms, symbol_bit=3)
    assert len(flat) == 9
    assert all(b in (0, 1) for b in flat)


def test_codeword_llr9():
    cw = np.array([0, 1, 2], dtype=np.int64)
    llr9 = codeword_llr9(cw)
    assert llr9[0, 0] == 3
    assert llr9[1, 1] == 3
    assert llr9[2, 2] == 3


def test_hw_vs_prob_llr_same_bits(rng):
    bits = rng.integers(0, 2, size=(1, 64))
    hw = init_llr_from_adc_hw(bits)
    prob = init_llr_from_bits(bits, p_bit=5e-3)
    # Same argmax on clean readout should often agree; just check shapes
    assert hw.shape == prob.shape
