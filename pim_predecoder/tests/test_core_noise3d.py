"""Tests for pim_predecoder.core.noise3d."""
from __future__ import annotations

import numpy as np

from pim_predecoder.core.noise3d import (
    apply_pim_noise_3d,
    apply_pim_noise_conductance,
    collapse_majority,
)
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


# --- conductance-domain channel (out-of-distribution robustness) -----------
#
# This channel injects impairments in the analog conductance domain and lets
# bit errors emerge from ADC threshold crossings, rather than flipping bits
# with a Bernoulli model. It is therefore structurally different from the
# channel the calibrator is trained on.


def test_conductance_channel_shape_and_dtype(rng):
    sym = rng.integers(0, 3, size=(3, 16))
    out = apply_pim_noise_conductance(
        sym, n_cycles=4, sigma_ln=0.1, drift_nu=0.05, rng=rng)
    assert out.shape == (3, 4, 32)
    assert set(np.unique(out)).issubset({0, 1})


def test_conductance_channel_noiseless_is_lossless(rng):
    """With no variability, drift or RTN the ADC must recover every symbol."""
    sym = rng.integers(0, 3, size=(4, 32))
    clean_bits = symbols_to_bits(sym)
    out = apply_pim_noise_conductance(sym, n_cycles=3, rng=rng)
    for t in range(3):
        assert np.array_equal(out[:, t, :], clean_bits)


def test_conductance_drift_increases_error_rate(rng):
    """A larger retention-drift exponent must degrade the readout."""
    sym = rng.integers(0, 3, size=(64, 64))
    clean_bits = symbols_to_bits(sym)

    def err(nu):
        out = apply_pim_noise_conductance(
            sym, n_cycles=4, sigma_ln=0.05, drift_nu=nu,
            rng=np.random.default_rng(7))
        return float((out != clean_bits[:, None, :]).mean())

    assert err(0.30) > err(0.10) > err(0.0)


def test_conductance_drift_is_monotonically_persistent(rng):
    """Retention drift must be irreversible, RTN must not be.

    This is the structural signature the drift detector keys on: under a
    monotonic conductance decay a cell that has slipped below an ADC
    threshold stays slipped for every later read, whereas a telegraph trap
    repeatedly returns the cell to its correct level.
    """
    sym = rng.integers(0, 3, size=(128, 64))
    clean = symbols_to_bits(sym)

    def monotone_fraction(bits):
        err = (bits != clean[:, None, :])            # (B, T, 2n)
        # Non-decreasing along the cycle axis for every bit position.
        return float(np.all(np.diff(err.astype(np.int8), axis=1) >= 0,
                            axis=1).mean())

    drift_only = apply_pim_noise_conductance(
        sym, n_cycles=4, sigma_ln=0.05, drift_nu=0.30,
        rng=np.random.default_rng(11))
    rtn_only = apply_pim_noise_conductance(
        sym, n_cycles=4, sigma_ln=0.05, rtn_traps=2, rtn_amp=0.18,
        rtn_switch=0.5, rng=np.random.default_rng(11))

    assert monotone_fraction(drift_only) == 1.0
    assert monotone_fraction(rtn_only) < 0.99


def test_conductance_rtn_is_multilevel(rng):
    """Several traps must move a symbol across more than one ADC level."""
    sym = np.full((256, 8), 2, dtype=np.int64)
    out = apply_pim_noise_conductance(
        sym, n_cycles=6, rtn_traps=3, rtn_amp=0.18, rtn_switch=0.4,
        rng=np.random.default_rng(3))
    pairs = out.reshape(256, 6, 8, 2)
    syms = pairs[..., 0] * 2 + pairs[..., 1]
    assert len(np.unique(syms)) >= 3
