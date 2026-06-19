"""Tests for pim_predecoder.hw.simulate and verify_cosim."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pim_predecoder.hw.export_weights import export_checkpoint
from pim_predecoder.hw.simulate import (
    GemvMlpHw,
    GrcPimSim,
    run_self_test,
    _sigmoid_lut_q16,
)
from pim_predecoder.hw.verify_cosim import run_verify


def test_sigmoid_lut_monotonic():
    vals = [_sigmoid_lut_q16(x) for x in [-16, -8, 0, 8, 16]]
    assert vals == sorted(vals)


def test_gemv_mlp_hw_forward(rtl_weights_dir):
    mlp = GemvMlpHw.from_weights_dir(rtl_weights_dir)
    feat = np.zeros(13, dtype=np.int8)
    anchor = np.array([1, 0, -1], dtype=np.int8)
    llr, gate = mlp.forward(feat, anchor)
    assert llr.shape == (3,)
    assert 0 <= gate <= 65535


def test_grc_pim_sim_accum():
    mlp = GemvMlpHw.from_weights_dir()
    sim = GrcPimSim(n_cycles=2, mlp=mlp)
    sim.feed_cycle(0b010, cycle_last=True)
    sim.feed_cycle(0b011, cycle_last=True)
    r = sim.finish_symbol(
        np.full(13, 4, dtype=np.int8),
        noisy_bits_cycles=[(1, 0), (1, 1)],
    )
    assert r["accum_llr"] == [3, 2, 3]
    assert "llr_out" in r


def test_run_self_test():
    st = run_self_test()
    assert st["all_pass"] is True
    assert len(st["tests"]) >= 3


def test_run_verify_small(hg_1024):
    H, _ = hg_1024
    rep = run_verify(
        info_length=1024, rate="16_17", p_bit=5e-3,
        n_codewords=4, n_cycles=4, seed=1, checkpoint=None,
    )
    assert rep["n_codewords"] == 4
    assert rep["llr_frontend"]["prob_argmax_agree"] > 0.9


def test_run_verify_with_checkpoint(gemv_checkpoint):
    rep = run_verify(
        info_length=1024, rate="16_17", p_bit=5e-3,
        n_codewords=4, seed=1, checkpoint=gemv_checkpoint,
    )
    assert "ber_gemv_float" in rep
    assert "gate_mean" in rep


def test_export_checkpoint(tmp_path: Path, gemv_checkpoint):
    out = tmp_path / "weights"
    meta = export_checkpoint(gemv_checkpoint, out, hidden=24)
    assert (out / "fc1_w.hex").is_file()
    assert meta["hidden"] == 24
    mlp = GemvMlpHw.from_weights_dir(out)
    feat = np.zeros(13, dtype=np.int8)
    llr, _ = mlp.forward(feat, np.zeros(3, dtype=np.int8))
    assert llr.shape == (3,)
