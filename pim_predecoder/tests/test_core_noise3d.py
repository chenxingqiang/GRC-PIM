"""Tests for pim_predecoder.core.noise3d."""
from __future__ import annotations

import numpy as np

from pim_predecoder.core.noise3d import apply_pim_noise_3d, collapse_majority
from pim_predecoder.core.noise import symbols_to_bits


def test_apply_pim_noise_3d_shape(rng):
    sym = rng.integers(0, 3, size=(2, 16))
    out = apply_pim_noise_3d(sym, n_cycles=4, p_indep=0.05, rng=rng)
    assert out.shape == (2, 4, 32)


def test_collapse_majority_identical():
    bits = np.ones((1, 5, 4), dtype=np.int64)
    maj = collapse_majority(bits)
    assert maj.shape == (1, 4)
    assert np.all(maj == 1)


def test_collapse_majority_tie_break():
    bits = np.array([[[0, 1], [1, 0]]], dtype=np.int64)  # (1, 2, 2)
    maj = collapse_majority(bits)
    assert maj.shape == (1, 2)
    # mean >= 0.5 -> 1
    assert maj[0, 0] == 1


def test_zero_noise_matches_clean(rng):
    sym = rng.integers(0, 3, size=(1, 8))
    clean_bits = symbols_to_bits(sym)
    out = apply_pim_noise_3d(sym, n_cycles=3, p_indep=0.0, rng=rng)
    for t in range(3):
        assert np.array_equal(out[0, t], clean_bits[0])
