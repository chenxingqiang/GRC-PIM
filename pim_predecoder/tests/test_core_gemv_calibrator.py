"""Tests for pim_predecoder.core.gemv_calibrator."""
from __future__ import annotations

import numpy as np
import torch

from pim_predecoder.core.codec import encode
from pim_predecoder.core.gemv_calibrator import (
    GEMV_FEATURES,
    GEMVResidualCalibrator,
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
)
from pim_predecoder.core.noise3d import apply_pim_noise_3d


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
