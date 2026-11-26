"""Tests for pim_predecoder.core.gemv_calibrator."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pim_predecoder.core.codec import encode, init_llr_from_bits
from pim_predecoder.core.gemv_calibrator import (
    GEMV_FEATURES,
    GEMVResidualCalibrator,
    LinearResidualCalibrator,
    PerColumnAffineCalibrator,
    build_gemv_features,
    column_persistence_score,
    composite_drift_score,
    enhanced_drift_scores,
    global_drift_score,
    make_training_batch_gemv,
    row_burst_score,
    soft_llr_product,
    syndrome_instability,
    train_affine_calibrator,
    train_gemv_calibrator,
    train_linear_calibrator,
    weighted_soft_llr_product,
)
from pim_predecoder.core.noise3d import (
    apply_pim_noise_3d,
    apply_pim_noise_conductance,
)


# --- drift score as an out-of-distribution signal --------------------------
#
# The calibrator is only valid for drift strengths it was calibrated at. The
# conductance-domain channel can produce a global drift score far outside that
# range, which is what an upper gate bound has to catch. These tests pin the
# two directions the score can leave the in-distribution band.


def test_drift_score_zero_when_reads_are_identical(hg_default, rng):
    """Static conductance spread gives identical reads, hence no drift signal.

    There is then no multi-cycle evidence for the calibrator to exploit, even
    though the readout is badly corrupted, so the lower gate bound must close.
    """
    _, G = hg_default
    clean = encode(rng.integers(0, 3, size=(64, G.shape[0])), G)
    noisy = apply_pim_noise_conductance(
        clean, n_cycles=4, sigma_ln=0.20, drift_nu=0.0, rng=rng)
    assert np.array_equal(noisy[:, 0, :], noisy[:, 3, :])
    assert global_drift_score(noisy) == 0.0


def test_drift_score_far_above_training_range_under_strong_retention(
        hg_default, rng):
    """A steep retention exponent pushes the score well past the Bernoulli range.

    The synthetic channel the model is trained on tops out near 0.031; strong
    conductance-domain drift reaches an order of magnitude higher, which is the
    condition the upper gate bound exists to reject.
    """
    _, G = hg_default
    clean = encode(rng.integers(0, 3, size=(64, G.shape[0])), G)
    trained = apply_pim_noise_3d(clean, n_cycles=4, p_indep=5e-3,
                                 p_drift=5e-3, p_row_burst=5e-3, rng=rng)
    ood = apply_pim_noise_conductance(
        clean, n_cycles=4, sigma_ln=0.05, drift_nu=0.15, rng=rng)
    assert global_drift_score(trained) < 0.05
    assert global_drift_score(ood) > 0.15


def test_global_drift_score_bounds(rng):
    sym = rng.integers(0, 3, size=(4, 32))
    noisy = apply_pim_noise_3d(sym, 4, 0.05, rng=rng)
    score = global_drift_score(noisy)
    assert 0.0 <= score <= 1.0


def test_enhanced_drift_scores(hg_default, rng):
    _, G = hg_default
    H, _ = hg_default
    sym = rng.integers(0, 3, size=(8, G.shape[1]))
    noisy = apply_pim_noise_3d(sym, 4, 0.01, p_drift=0.01, rng=rng)
    scores = enhanced_drift_scores(noisy, H)
    assert "global_drift" in scores
    assert "composite" in scores
    assert 0.0 <= scores["composite"] <= 1.0


def test_composite_drift_score():
    scores = {
        "global_drift": 0.5,
        "column_persistence": 0.5,
        "row_burst": 0.5,
        "syndrome_instability": 0.5,
    }
    c = composite_drift_score(scores)
    assert 0.0 <= c <= 1.0


def test_syndrome_instability_single_cycle(hg_default, rng):
    H, G = hg_default
    clean = encode(rng.integers(0, 3, (2, G.shape[0])), G)
    noisy = apply_pim_noise_3d(clean, 1, 0.01, rng=rng)
    s = syndrome_instability(noisy, H)
    assert 0.0 <= s <= 1.0


def test_soft_llr_product_shape(rng):
    sym = rng.integers(0, 3, size=(2, 16))
    noisy = apply_pim_noise_3d(sym, 4, 0.01, rng=rng)
    llr = soft_llr_product(noisy, 0.01)
    assert llr.shape == (2, 16, 3)
    assert np.allclose(llr.max(axis=-1), 0.0)


# --- weighted-BP baseline ----------------------------------------------------
#
# B' sums the per-cycle LLRs with equal weights, which is exact only if the
# T reads are conditionally independent. Drift persistence violates that
# assumption (a drifted column stays corrupted for consecutive reads), so a
# natural stronger baseline discounts later cycles geometrically before the
# same BP decoder runs. These tests pin the weighting exactly.


def test_weighted_soft_llr_product_rho_one_equals_b_prime(rng):
    """rho=1 must reduce exactly to the plain B' product."""
    sym = rng.integers(0, 3, size=(2, 16))
    noisy = apply_pim_noise_3d(sym, 4, 0.01, rng=rng)
    assert np.allclose(
        weighted_soft_llr_product(noisy, 0.01, rho=1.0),
        soft_llr_product(noisy, 0.01),
    )


def test_weighted_soft_llr_product_rho_zero_keeps_first_cycle(rng):
    """rho=0 keeps only cycle 0's evidence, normalised the same way."""
    sym = rng.integers(0, 3, size=(2, 16))
    noisy = apply_pim_noise_3d(sym, 4, 0.01, rng=rng)
    first = init_llr_from_bits(noisy[:, 0, :], p_bit=0.01)
    first = first - first.max(axis=-1, keepdims=True)
    assert np.allclose(weighted_soft_llr_product(noisy, 0.01, rho=0.0), first)


def test_build_gemv_features_raw_bits_expose_cycle_order(rng):
    """The default features are invariant to the order of the T reads; the
    raw-bit extension appends the 2T reads per symbol and is not."""
    sym = rng.integers(0, 3, size=(2, 16))
    noisy = apply_pim_noise_3d(sym, 4, 0.05, p_drift=0.05, rng=rng)
    perm = noisy[:, [3, 1, 0, 2], :]
    base, _ = build_gemv_features(noisy, 0.05)
    base_p, _ = build_gemv_features(perm, 0.05)
    assert np.allclose(base, base_p)
    ext, _ = build_gemv_features(noisy, 0.05, raw_bits=True)
    ext_p, _ = build_gemv_features(perm, 0.05, raw_bits=True)
    assert ext.shape == (2, 16, GEMV_FEATURES + 8)
    assert np.allclose(ext[..., :GEMV_FEATURES], base)
    assert not np.allclose(ext, ext_p)
    # Symbol j, cycle t, bit b sits at column GEMV_FEATURES + 2t + b, in +-1.
    assert ext[0, 5, GEMV_FEATURES + 2 * 2 + 1] == 2.0 * noisy[0, 2, 11] - 1.0


def test_gemv_calibrator_raw_bit_variant_soft_llr(rng):
    model = GEMVResidualCalibrator(hidden=8, raw_bit_cycles=4)
    assert model.fc1.in_features == GEMV_FEATURES + 8
    sym = rng.integers(0, 3, size=(2, 16))
    noisy = apply_pim_noise_3d(sym, 4, 0.01, rng=rng)
    llr, gate = model.soft_llr(noisy, 0.01)
    assert llr.shape == (2, 16, 3)
    assert 0.0 <= gate <= 1.0


def test_train_gemv_calibrator_raw_bits_persist_mixture(hg_default):
    _, G = hg_default
    model = train_gemv_calibrator(
        G, n_cycles=4, n_steps=4, batch=8, hidden=8, log_every=100, seed=0,
        raw_bits=True, drift_persist_choices=(1, 2, 3, 4),
    )
    assert model.fc1.in_features == GEMV_FEATURES + 8


def test_weighted_soft_llr_product_geometric_discount():
    """Cycle k (0-based) carries weight rho**k exactly.

    With cycle 0 reading symbol 0 (bits 00) and cycle 1 reading symbol 1
    (bits 01), the accumulated margin between the two candidates is
    (1 - rho) * log((1-p)/p).
    """
    noisy = np.zeros((1, 2, 4), dtype=np.int64)
    noisy[0, 1, 1] = 1  # cycle 1: b0 of symbol 0 flips, readout 01
    p = 0.1
    for rho in (1.0, 0.5, 0.25):
        llr = weighted_soft_llr_product(noisy, p, rho=rho)
        expected = (1.0 - rho) * np.log((1 - p) / p)
        assert np.isclose(llr[0, 0, 0] - llr[0, 0, 1], expected)
        assert np.allclose(llr.max(axis=-1), 0.0)
        assert llr.shape == (1, 2, 3)


def test_weighted_soft_llr_product_rejects_bad_rho(rng):
    sym = rng.integers(0, 3, size=(1, 16))
    noisy = apply_pim_noise_3d(sym, 4, 0.01, rng=rng)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            weighted_soft_llr_product(noisy, 0.01, rho=bad)


def test_build_gemv_features_shape(rng):
    sym = rng.integers(0, 3, size=(3, 32))
    noisy = apply_pim_noise_3d(sym, 4, 0.01, rng=rng)
    feats, llr_bp = build_gemv_features(noisy, 0.01)
    assert feats.shape == (3, 32, GEMV_FEATURES)
    assert llr_bp.shape == (3, 32, 3)


def test_gemv_residual_calibrator_forward():
    model = GEMVResidualCalibrator(hidden=8)
    x = torch.randn(2, 4, GEMV_FEATURES)
    anchor = torch.randn(2, 4, 3)
    out, gate = model(x, anchor)
    assert out.shape == (2, 4, 3)
    assert gate.shape == (2, 4, 1)
    assert torch.all(gate >= 0) and torch.all(gate <= 1)


def test_per_column_affine_forward():
    model = PerColumnAffineCalibrator(n_columns=16)
    anchor = torch.randn(2, 16, 3)
    out = model(anchor)
    assert out.shape == anchor.shape


# --- hidden-layer-free linear baseline (Sec. baseline comparison) -----------

def test_linear_residual_calibrator_forward():
    model = LinearResidualCalibrator()
    x = torch.randn(2, 4, GEMV_FEATURES)
    anchor = torch.randn(2, 4, 3)
    out, gate = model(x, anchor)
    assert out.shape == (2, 4, 3)
    assert gate.shape == (2, 4, 1)
    assert torch.all(gate >= 0) and torch.all(gate <= 1)


def test_linear_residual_calibrator_identity_at_init():
    """A zero-initialised correction head must leave the B' anchor unchanged."""
    model = LinearResidualCalibrator()
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
    x = torch.randn(2, 4, GEMV_FEATURES)
    anchor = torch.randn(2, 4, 3)
    out, _ = model(x, anchor)
    assert torch.allclose(out, anchor)


def test_linear_residual_calibrator_param_count():
    """13x3+3 delta head plus 13x1+1 gate head: 56 parameters vs GEMV's 425."""
    model = LinearResidualCalibrator()
    n = sum(p.numel() for p in model.parameters())
    assert n == GEMV_FEATURES * 3 + 3 + GEMV_FEATURES + 1


def test_linear_residual_calibrator_delta_affine_in_features():
    """With the clip wide open the correction is affine in the features:
    o(x1) + o(x2) == o(x1+x2) + o(0). A hidden ReLU layer would fail this."""
    model = LinearResidualCalibrator(clip=1e9)
    with torch.no_grad():
        model.gate.weight.zero_()
        model.gate.bias.fill_(5.0)   # constant gate ~ 0.993
    anchor = torch.zeros(1, 4, 3)
    xa = torch.randn(1, 4, GEMV_FEATURES)
    xb = torch.randn(1, 4, GEMV_FEATURES)
    oa, _ = model(xa, anchor)
    ob, _ = model(xb, anchor)
    oab, _ = model(xa + xb, anchor)
    o_zero, _ = model(torch.zeros_like(xa), anchor)
    assert torch.allclose(oa + ob, oab + o_zero, atol=1e-4)


def test_linear_residual_calibrator_soft_llr(rng):
    model = LinearResidualCalibrator()
    sym = rng.integers(0, 3, size=(2, 16))
    noisy = apply_pim_noise_3d(sym, 4, 0.01, rng=rng)
    llr, gate = model.soft_llr(noisy, 0.01)
    assert llr.shape == (2, 16, 3)
    assert 0.0 <= gate <= 1.0
    # LLRs are shifted so the best candidate is at 0, like every other path.
    assert np.allclose(llr.max(axis=-1), 0.0)


def test_train_linear_calibrator_short(hg_default):
    _, G = hg_default
    model = train_linear_calibrator(
        G, n_cycles=2, n_steps=3, batch=8, log_every=100, seed=0,
    )
    assert isinstance(model, LinearResidualCalibrator)


def test_make_training_batch_gemv(hg_default, rng):
    _, G = hg_default
    feats, llr_bp, clean = make_training_batch_gemv(
        G, batch_size=4, n_cycles=4, p_indep=0.01,
        p_drift=0.01, p_row_burst=0.01, rng=rng,
    )
    assert feats.shape[0] == 4
    assert llr_bp.shape[0] == 4
    assert clean.shape == (4, G.shape[1])


def test_train_gemv_calibrator_short(hg_default):
    _, G = hg_default
    model = train_gemv_calibrator(
        G, n_cycles=2, n_steps=3, batch=8, hidden=8,
        log_every=100, seed=0,
    )
    assert isinstance(model, GEMVResidualCalibrator)


def test_train_affine_calibrator_short(hg_default):
    _, G = hg_default
    model = train_affine_calibrator(
        G, n_cycles=2, n_steps=3, batch=8, log_every=100, seed=0,
    )
    assert isinstance(model, PerColumnAffineCalibrator)


def test_column_and_row_scores(rng):
    sym = rng.integers(0, 3, size=(16, 32))
    noisy = apply_pim_noise_3d(sym, 4, 0.05, p_row_burst=0.05, rng=rng)
    cp = column_persistence_score(noisy)
    rb = row_burst_score(noisy)
    assert 0.0 <= cp <= 1.0
    assert 0.0 <= rb <= 1.0
