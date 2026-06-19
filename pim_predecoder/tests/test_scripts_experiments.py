"""Tests for pim_predecoder.scripts and experiments helpers."""
from __future__ import annotations

import subprocess
import sys

import numpy as np

from pim_predecoder.core.codec import bp_decode, encode, init_llr_from_bits, load_default_HG
from pim_predecoder.core.noise import apply_pim_noise
from pim_predecoder.experiments import ablation, gemv


def test_smoke_pipeline(repo_root, rng):
    H, G = load_default_HG(repo_root)
    info = rng.integers(0, 3, size=(4, G.shape[0]))
    code = encode(info, G)
    nb = apply_pim_noise(code, p_bit=1e-3, rng=rng)
    llr = init_llr_from_bits(nb, p_bit=1e-3)
    xh, _, it = bp_decode(llr, H, n_iter=3)
    assert xh.shape == code.shape
    assert it >= 1


def test_smoke_module_runs(repo_root):
    proc = subprocess.run(
        [sys.executable, "-m", "pim_predecoder.scripts.smoke"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "post-ECC mismatches" in proc.stdout


def test_ablation_ber_to_ser():
    ser = ablation._ber_to_ser(0.01)
    assert 0.0 < ser < 0.02


def test_ablation_fmt():
    assert ablation._fmt(0.005) == "0.005"
    assert ablation._fmt(1.0) == "1"


def test_ablation_aggregate_across_seeds():
    row = {"p_bit": 0.005, "Bp_post": 0.1, "GEMV_post": 0.2}
    results = [
        {"rows": [row, {"p_bit": 0.003, "Bp_post": 0.2, "GEMV_post": 0.3}]},
        {"rows": [row, {"p_bit": 0.003, "Bp_post": 0.3, "GEMV_post": 0.4}]},
    ]
    agg = ablation._aggregate_across_seeds(results)
    assert len(agg) == 2
    p005 = next(a for a in agg if a["p_bit"] == 0.005)
    assert p005["n_seeds"] == 2
    assert p005["Bp_post_mean"] == 0.1


def test_gemv_ser_ber(rng):
    ref = rng.integers(0, 3, size=(4, 16))
    x = ref.copy()
    x[0, 0] = (x[0, 0] + 1) % 3
    assert gemv._ser(x, ref) == 1.0 / (4 * 16)
    assert gemv._ber(x, ref) > 0.0


def test_gemv_post_metrics(hg_default, rng):
    H, G = hg_default
    info = rng.integers(0, 3, size=(2, G.shape[0]))
    clean = encode(info, G)
    llr = np.zeros((2, clean.shape[1], 3), dtype=np.float32)
    for b in range(2):
        for i, s in enumerate(clean[b]):
            llr[b, i, s] = 0.0
            llr[b, i, (s + 1) % 3] = -5.0
            llr[b, i, (s + 2) % 3] = -5.0
    ser, ber, iters = gemv._post_metrics(llr, H, clean, bp_iters=3)
    assert ser == 0.0
    assert iters >= 1.0
