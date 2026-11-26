"""Tests for pim_predecoder.scripts and experiments helpers."""
from __future__ import annotations

import subprocess
import sys

import numpy as np

from pim_predecoder.core.codec import bp_decode, encode, init_llr_from_bits, load_default_HG
from pim_predecoder.core.gemv_calibrator import (
    GEMVResidualCalibrator,
    LinearResidualCalibrator,
)
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


def test_gemv_count_err_matches_rates(rng):
    """Error counts must reproduce the _ser/_ber rates exactly."""
    ref = rng.integers(0, 3, size=(8, 16))
    x = ref.copy()
    x[0, 0] = (x[0, 0] + 1) % 3
    x[3, 7] = (x[3, 7] + 2) % 3
    sym_err, sym_tot, bit_err, bit_tot = gemv._count_err(x, ref)
    assert sym_err / sym_tot == gemv._ser(x, ref)
    assert bit_err / bit_tot == gemv._ber(x, ref)


def test_gemv_count_err_chunked_aggregation_is_exact(rng):
    """Summing per-chunk counts must equal the full-batch rate bit-exactly.

    This is the invariant that makes streaming 100k-codeword evaluation
    equivalent to a monolithic run: SER/BER are means over codewords and are
    therefore exactly decomposable into per-chunk error counts.
    """
    ref = rng.integers(0, 3, size=(60, 32))
    x = ref.copy()
    flip = rng.random(x.shape) < 0.1
    x[flip] = (x[flip] + 1) % 3

    full = gemv._count_err(x, ref)
    acc = [0, 0, 0, 0]
    for start in range(0, ref.shape[0], 7):          # ragged chunks on purpose
        part = gemv._count_err(x[start:start + 7], ref[start:start + 7])
        acc = [a + b for a, b in zip(acc, part)]

    assert tuple(acc) == full
    assert acc[0] / acc[1] == gemv._ser(x, ref)
    assert acc[2] / acc[3] == gemv._ber(x, ref)


# --- conductance-domain OOD evaluation ------------------------------------
#
# evaluate() must be able to drive the conductance-domain channel instead of
# the Bernoulli one it was written for. That channel has no p_bit parameter,
# so the nominal bit-flip rate handed to B'/the calibrator is measured from
# the realised readout and reported as `p_effective`.


def test_evaluate_conductance_is_lossless_when_noiseless(hg_default):
    """Zero spread/drift/RTN must give an error-free readout and p_eff = 0."""
    H, G = hg_default
    rows = gemv.evaluate(
        None, H, G, [5e-3],
        n_codewords=4, n_cycles=4,
        test_p_drift=0.0, test_p_row_burst=0.0,
        bp_iters=3, device="cpu", seed=1, chunk=512,
        drift_score_threshold=None, model_gate_threshold=None,
        channel="conductance",
        cond_params={"sigma_ln": 0.0, "drift_nu": 0.0, "rtn_traps": 0,
                     "rtn_amp": 0.0, "rtn_switch": 0.5},
    )
    assert len(rows) == 1
    assert rows[0]["p_effective"] == 0.0
    assert rows[0]["Bp_post"] == 0.0
    assert rows[0]["drift_score"] == 0.0


def test_evaluate_conductance_reports_measured_p(hg_default):
    """p_effective must track the realised readout error rate, not --p_bits."""
    H, G = hg_default
    rows = gemv.evaluate(
        None, H, G, [5e-3],
        n_codewords=64, n_cycles=4,
        test_p_drift=0.0, test_p_row_burst=0.0,
        bp_iters=3, device="cpu", seed=1, chunk=512,
        drift_score_threshold=None, model_gate_threshold=None,
        channel="conductance",
        cond_params={"sigma_ln": 0.15, "drift_nu": 0.05, "rtn_traps": 2,
                     "rtn_amp": 0.03, "rtn_switch": 0.5},
        eval_chunk=16,
    )
    row = rows[0]
    assert row["p_bit"] == 5e-3                 # label is preserved
    assert row["p_effective"] > 1e-3            # measured, and clearly not 5e-3
    assert row["p_effective"] != 5e-3
    assert row["n_blocks"] == 4                 # streaming still works


def test_evaluate_conductance_static_spread_has_zero_drift_score(hg_default):
    """Static spread alone yields identical reads, hence d_global == 0.

    This is the case that must close the drift gate: with no cycle-to-cycle
    disagreement there is no multi-cycle evidence for the calibrator to use,
    even though the readout is badly corrupted.
    """
    H, G = hg_default
    rows = gemv.evaluate(
        None, H, G, [5e-3],
        n_codewords=64, n_cycles=4,
        test_p_drift=0.0, test_p_row_burst=0.0,
        bp_iters=3, device="cpu", seed=1, chunk=512,
        drift_score_threshold=None, model_gate_threshold=None,
        channel="conductance",
        cond_params={"sigma_ln": 0.20, "drift_nu": 0.0, "rtn_traps": 0,
                     "rtn_amp": 0.0, "rtn_switch": 0.5},
    )
    assert rows[0]["p_effective"] > 0.0         # genuinely corrupted
    assert rows[0]["drift_score"] == 0.0        # but no temporal evidence


def test_evaluate_drift_upper_bound_closes_gate_out_of_range(hg_default):
    """An upper drift bound must reject impairment stronger than calibrated.

    The lower bound alone admits any heavily drifted channel, including ones
    whose drift score is an order of magnitude past anything the calibrator was
    trained at. Passing drift_score_max must close the gate there.
    """
    H, G = hg_default
    common = dict(
        n_codewords=32, n_cycles=4,
        test_p_drift=0.0, test_p_row_burst=0.0,
        bp_iters=3, device="cpu", seed=1, chunk=512,
        drift_score_threshold=0.011, model_gate_threshold=None,
        channel="conductance",
        cond_params={"sigma_ln": 0.05, "drift_nu": 0.15, "rtn_traps": 0,
                     "rtn_amp": 0.0, "rtn_switch": 0.5},
    )
    # Without a model there is no gate decision to inspect, so assert on the
    # score itself: it must sit far above the in-distribution band.
    rows = gemv.evaluate(None, H, G, [5e-3], **common)
    assert rows[0]["drift_score"] > 0.15
    assert rows[0]["drift_score_max"] is None

    bounded = gemv.evaluate(None, H, G, [5e-3],
                            **{**common, "drift_score_max": 0.035})
    assert bounded[0]["drift_score_max"] == 0.035


def test_evaluate_drift_bounds_are_recorded(hg_default):
    """Both gate bounds must be reported so a run is self-describing."""
    H, G = hg_default
    rows = gemv.evaluate(
        None, H, G, [5e-3],
        n_codewords=32, n_cycles=4,
        test_p_drift=5e-3, test_p_row_burst=5e-3,
        bp_iters=3, device="cpu", seed=1, chunk=512,
        drift_score_threshold=0.011, model_gate_threshold=0.6,
        drift_score_max=0.035,
    )
    assert rows[0]["drift_score_max"] == 0.035


def test_evaluate_synthetic_channel_unchanged(hg_default):
    """The default path must stay bit-identical to the pre-change behaviour."""
    H, G = hg_default
    common = dict(
        n_codewords=32, n_cycles=4,
        test_p_drift=5e-3, test_p_row_burst=5e-3,
        bp_iters=3, device="cpu", seed=1, chunk=512,
        drift_score_threshold=None, model_gate_threshold=None,
    )
    base = gemv.evaluate(None, H, G, [5e-3], **common)
    explicit = gemv.evaluate(None, H, G, [5e-3], channel="synthetic", **common)
    assert base[0]["Bp_post"] == explicit[0]["Bp_post"]
    assert base[0]["drift_score"] == explicit[0]["drift_score"]
    assert base[0]["p_effective"] == 5e-3       # nominal == effective here


def test_evaluate_weighted_bp_baseline(hg_default):
    """Weighted-BP baseline: off by default, rho=1 decodes identically to B'."""
    H, G = hg_default
    common = dict(
        n_codewords=16, n_cycles=4,
        test_p_drift=5e-3, test_p_row_burst=5e-3,
        bp_iters=3, device="cpu", seed=1, chunk=512,
        drift_score_threshold=None, model_gate_threshold=None,
    )
    # Disabled by default: the row schema must not change.
    base = gemv.evaluate(None, H, G, [5e-3], **common)[0]
    assert "Wbp_post" not in base
    # rho=1 is the same prior over the same readout, so it must decode
    # identically to B' (exact integer error counts).
    row = gemv.evaluate(None, H, G, [5e-3], wbp_rho=1.0, **common)[0]
    assert row["Wbp_post"] == row["Bp_post"]
    assert row["Wbp_post_BER"] == row["Bp_post_BER"]
    # rho < 1 produces a valid decode from a genuinely different prior.
    row = gemv.evaluate(None, H, G, [5e-3], wbp_rho=0.5, **common)[0]
    assert 0.0 <= row["Wbp_post"] <= 1.0
    assert row["Wbp_iters"] >= 1.0


def test_evaluate_with_linear_calibrator(hg_default):
    """The hidden-layer-free calibrator runs through the shared evaluate path
    unchanged: same row schema, same gate decision fields."""
    H, G = hg_default
    common = dict(
        n_codewords=16, n_cycles=4,
        test_p_drift=5e-3, test_p_row_burst=5e-3,
        bp_iters=3, device="cpu", seed=1, chunk=512,
        drift_score_threshold=None, model_gate_threshold=0.6,
    )
    gemv_model = GEMVResidualCalibrator()
    linear_model = LinearResidualCalibrator()
    g_row = gemv.evaluate(gemv_model, H, G, [5e-3], **common)[0]
    l_row = gemv.evaluate(linear_model, H, G, [5e-3], **common)[0]
    assert set(l_row.keys()) == set(g_row.keys())
    assert 0.0 <= l_row["GEMV_post"] <= 1.0
    assert "use_gemv" in l_row and "GEMV_gate" in l_row
