"""Tests for pim_predecoder.core.noise."""
from __future__ import annotations

import numpy as np
import pytest

from pim_predecoder.core.noise import (
    apply_pim_noise,
    bits_to_symbols,
    estimate_ber,
    symbols_to_bits,
)


def test_symbols_bits_roundtrip(rng):
    sym = rng.integers(0, 3, size=(4, 32))
    bits = symbols_to_bits(sym)
    assert bits.shape == (4, 64)
    back = bits_to_symbols(bits)
    assert np.array_equal(back, sym)


def test_bits_to_symbols_illegal():
    bits = np.array([[1, 1]], dtype=np.int64)
    sym = bits_to_symbols(bits)
    assert sym[0, 0] == 3  # 11 -> illegal


def test_apply_pim_noise_shape(rng):
    sym = rng.integers(0, 3, size=(3, 16))
    noisy = apply_pim_noise(sym, p_bit=0.1, rng=rng)
    assert noisy.shape == (3, 32)
    assert set(np.unique(noisy)).issubset({0, 1})


def test_apply_pim_noise_zero_p(rng):
    sym = rng.integers(0, 3, size=(2, 8))
    noisy = apply_pim_noise(sym, p_bit=0.0, rng=rng)
    clean_bits = symbols_to_bits(sym)
    assert np.array_equal(noisy, clean_bits)


def test_estimate_ber(rng):
    sym = rng.integers(0, 3, size=(1, 8))
    clean_bits = symbols_to_bits(sym)
    noisy = clean_bits.copy()
    noisy[0, 0] ^= 1
    ber = estimate_ber(noisy, sym)
    assert ber == pytest.approx(1 / noisy.size)
