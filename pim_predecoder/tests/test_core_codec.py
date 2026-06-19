"""Tests for pim_predecoder.core.codec."""
from __future__ import annotations

import numpy as np
import pytest

from pim_predecoder.core.codec import (
    GF,
    bp_decode,
    encode,
    init_llr_from_bits,
    load_HG,
    load_default_HG,
    load_matrix,
    syndrome,
)


def test_load_default_hg_shapes(hg_default):
    H, G = hg_default
    assert H.shape == (32, 288)
    assert G.shape == (256, 288)
    assert set(np.unique(H)).issubset({0, 1, 2})


def test_load_hg_1024(hg_1024):
    H, G = hg_1024
    assert H.shape[0] == 64
    assert G.shape[0] == 1024
    assert G.shape[1] == H.shape[1] == 1088


def test_load_hg_bad_rate(repo_root):
    with pytest.raises(ValueError):
        load_HG(repo_root, info_length=256, rate="bad")


def test_encode_syndrome_zero(hg_default, rng):
    H, G = hg_default
    info = rng.integers(0, GF, size=(8, G.shape[0]))
    code = encode(info, G)
    syn = syndrome(code, H)
    assert syn.shape == (8, H.shape[0])
    assert np.all(syn == 0)


def test_bp_decode_clean_llr(hg_default, rng):
    H, G = hg_default
    info = rng.integers(0, GF, size=(4, G.shape[0]))
    code = encode(info, G)
    llr = np.full((4, code.shape[1], GF), -20.0, dtype=np.float32)
    for b in range(4):
        for i, sym in enumerate(code[b]):
            llr[b, i, sym] = 0.0
    x_hat, _, iters = bp_decode(llr, H, n_iter=3, early_stop=True)
    assert np.all((x_hat - code) % GF == 0)
    assert 1 <= iters <= 3


def test_init_llr_from_bits_shape(rng):
    bits = rng.integers(0, 2, size=(5, 576))
    llr = init_llr_from_bits(bits, p_bit=1e-3)
    assert llr.shape == (5, 288, 3)
    assert np.allclose(llr.max(axis=-1), 0.0)


def test_load_matrix_roundtrip(tmp_path):
    p = tmp_path / "M.txt"
    p.write_text("0 1 2\n1 0 2\n")
    M = load_matrix(p, (2, 3))
    assert M.shape == (2, 3)
    assert M[0, 2] == 2
