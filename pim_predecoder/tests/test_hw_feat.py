"""Tests for pim_predecoder.hw.feat."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pim_predecoder.core.noise3d import apply_pim_noise_3d
from pim_predecoder.hw.feat import (
    FEAT_DIVISORS,
    FEAT_OFFSETS,
    export_feat_golden,
    features_from_readout,
    float_feat_to_int8,
    pack_feat_bus,
)


def test_feat_constants_length():
    assert len(FEAT_DIVISORS) == 13
    assert len(FEAT_OFFSETS) == 13


def test_float_feat_to_int8_range():
    feat = np.linspace(-1, 1, 13, dtype=np.float32)
    i8 = float_feat_to_int8(feat)
    assert i8.shape == (13,)
    assert i8.dtype == np.int8
    assert i8.min() >= -128 and i8.max() <= 127


def test_pack_feat_bus_roundtrip():
    feat = np.arange(13, dtype=np.int8)
    word = pack_feat_bus(feat)
    recovered = []
    for i in range(13):
        v = (word >> (8 * i)) & 0xFF
        if v >= 128:
            v -= 256
        recovered.append(v)
    assert recovered == feat.tolist()


def test_features_from_readout(rng):
    sym = rng.integers(0, 3, size=(2, 16))
    noisy = apply_pim_noise_3d(sym, 4, 0.01, rng=rng)
    feats_i8, llr_bp = features_from_readout(noisy, 0.01, batch_idx=0)
    assert feats_i8.shape == (16, 13)
    assert llr_bp.shape == (16, 3)


def test_export_feat_golden(tmp_path: Path, rng):
    sym = rng.integers(0, 3, size=(1, 8))
    noisy = apply_pim_noise_3d(sym, 4, 0.01, rng=rng)
    out = tmp_path / "feat.json"
    payload = export_feat_golden(noisy, 0.01, out)
    assert out.is_file()
    loaded = json.loads(out.read_text())
    assert loaded["n_symbols"] == 8
    assert len(payload["features_i8"]) == 8
