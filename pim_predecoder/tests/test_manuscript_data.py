"""Tests for the result_data aggregates backing figures and audits."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import pytest

from result_data import (
    E2_SEEDS,
    WbpCalibPoint,
    aggregate_256_e2,
    aggregate_1024_e2,
    gate_scenarios,
    linear_formal,
    wbp_calibration,
    wbp_formal,
)


def _write_calib_file(directory: Path, scale: int, seed: int, rho: float,
                      rows: list[dict]) -> None:
    tag = f"rho{str(rho).replace('.', '')}"
    path = directory / f"wbp_calib_{scale}_s{seed}_{tag}.json"
    payload = {
        "wbp_rho": rho,
        "label": f"wbp_calib_{scale}_s{seed}",
        "n_codewords": 10000,
        "rows": rows,
    }
    path.write_text(json.dumps(payload))


def _row(p_bit: float, bp_ser: float, wbp_ser: float,
         bp_ber: float, wbp_ber: float) -> dict:
    """One operating point carrying both SER and BER fields.

    The two field sets carry deliberately different numbers so the tests
    can assert that the 256-scale aggregation reads SER and the 1024-scale
    aggregation reads BER.
    """
    return {
        "p_bit": p_bit,
        "Bp_post": bp_ser, "Wbp_post": wbp_ser,
        "Bp_post_BER": bp_ber, "Wbp_post_BER": wbp_ber,
    }


def test_wbp_calibration_reads_scale_metric(tmp_path: Path) -> None:
    """256 reads Wbp_post/Bp_post; 1024 reads Wbp_post_BER/Bp_post_BER."""
    ser_rows = [_row(5e-3, 1e-3, 2e-3, 1.0, 1.0),
                _row(1e-3, 1e-3, 2e-3, 1.0, 1.0)]
    ber_rows = [_row(5e-3, 1.0, 1.0, 4e-3, 8e-3),
                _row(1e-3, 1.0, 1.0, 4e-3, 8e-3)]
    for seed in (1, 2, 3):
        _write_calib_file(tmp_path, 256, seed, 0.5, ser_rows)
        _write_calib_file(tmp_path, 1024, seed, 0.5, ber_rows)

    points = wbp_calibration(tmp_path)
    assert [(p.scale, p.rho) for p in points] == [(256, 0.5), (1024, 0.5)]
    assert points[0].per_seed == (2.0, 2.0, 2.0)
    assert points[0].mean_ratio == pytest.approx(2.0)
    assert points[0].std_ratio == pytest.approx(0.0)
    assert points[1].mean_ratio == pytest.approx(2.0)


def test_wbp_calibration_seed_spread(tmp_path: Path) -> None:
    """Per-seed ratios 2, 3, 4 give mean 3 and s.d. 1."""
    for seed, ratio in zip((1, 2, 3), (2.0, 3.0, 4.0)):
        _write_calib_file(
            tmp_path, 256, seed, 0.75,
            [_row(5e-3, 1e-3, 1e-3 * ratio, 1.0, 1.0)],
        )
    point = wbp_calibration(tmp_path)[0]
    assert point.rho == pytest.approx(0.75)
    assert point.mean_ratio == pytest.approx(3.0)
    assert point.std_ratio == pytest.approx(1.0)
    assert point.excess == pytest.approx(200.0)


def test_calibration_averages_pointwise_gains_before_seeds(tmp_path: Path) -> None:
    # Inverting a seed-mean error ratio would reverse the conclusion here.
    for seed in (1, 2, 3):
        rows = [_row(5e-3, 0.02, 0.01, 1, 1),
                _row(1e-3, 0.01, 0.02 * seed, 1, 1)]
        _write_calib_file(tmp_path, 256, seed, 0.9, rows)
    p = wbp_calibration(tmp_path)[0]
    gains = [(2 + 0.5 / seed) / 2 for seed in (1, 2, 3)]
    assert p.per_seed_gains == pytest.approx(gains)
    assert p.mean_gain == pytest.approx(mean(gains))
    assert p.std_gain == pytest.approx(stdev(gains))


def test_diagnostic_gate_uses_bit_floor_and_two_conditions() -> None:
    rows = gate_scenarios()
    assert all(r.ber_floor == pytest.approx(3 / (800 * 1088 * 2)) for r in rows)
    assert [r.scenario for r in rows if r.use_gemv] == ["E2"]
    assert [r.scenario for r in rows if 0.011 <= r.drift_score <= 0.038
            and r.acceptance_score >= 0.6] == ["E1", "E2", "E3"]


def test_wbp_calibration_missing_seed_raises(tmp_path: Path) -> None:
    _write_calib_file(tmp_path, 256, 1, 0.9, [_row(5e-3, 1e-3, 2e-3, 1.0, 1.0)])
    with pytest.raises(RuntimeError, match="256.*rho=0.9"):
        wbp_calibration(tmp_path)


def test_wbp_calibration_skips_zero_error_points(tmp_path: Path) -> None:
    """Operating points where B' decoded with zero errors carry no ratio."""
    rows = [
        _row(5e-3, 1e-3, 2e-3, 1.0, 1.0),
        _row(1e-4, 0.0, 0.0, 1.0, 1.0),   # resolution floor; skipped
    ]
    for seed in (1, 2, 3):
        _write_calib_file(tmp_path, 256, seed, 0.5, rows)
    point = wbp_calibration(tmp_path)[0]
    assert point.mean_ratio == pytest.approx(2.0)


def test_wbp_calibration_all_zero_raises(tmp_path: Path) -> None:
    for seed in (1, 2, 3):
        _write_calib_file(tmp_path, 256, seed, 0.5,
                          [_row(5e-3, 0.0, 0.0, 1.0, 1.0)])
    with pytest.raises(RuntimeError, match="no usable"):
        wbp_calibration(tmp_path)


def test_wbp_point_is_frozen() -> None:
    point = WbpCalibPoint(scale=256, rho=0.5, mean_ratio=2.0,
                          std_ratio=0.0, per_seed=(2.0, 2.0, 2.0),
                          per_seed_gains=(0.5, 0.5, 0.5))
    with pytest.raises(AttributeError):
        point.rho = 0.9


def _write_formal_file(directory: Path, scale: int, seed: int, rows: list[dict],
                       rho: float = 0.9) -> None:
    """Write one formal-run JSON; the filename is fixed, the payload rho is
    the variable under test (a wrong rho must be caught from the payload)."""
    path = directory / f"E2_{scale}_s{seed}_n100k_rho09.json"
    payload = {
        "wbp_rho": rho,
        "label": f"wbp_{scale}_s{seed}",
        "n_codewords": 100000,
        "rows": rows,
    }
    path.write_text(json.dumps(payload))


def test_wbp_formal_reads_scale_metric(tmp_path: Path) -> None:
    """256 reads Wbp_post SER cells; 1024 reads Wbp_post_BER cells."""
    ser_rows = [_row(5e-3, 1e-3, 2e-3, 1.0, 1.0),
                _row(3e-3, 3e-3, 6e-3, 1.0, 1.0)]
    ber_rows = [_row(5e-3, 1.0, 1.0, 4e-3, 8e-3),
                _row(3e-3, 1.0, 1.0, 2e-3, 4e-3)]
    for seed in (1, 2, 3):
        _write_formal_file(tmp_path, 256, seed, ser_rows)
        _write_formal_file(tmp_path, 1024, seed, ber_rows)

    formal = wbp_formal(tmp_path, seeds=(1, 2, 3))
    assert formal[256]["cells"] == [(2e-3, 0.0), (6e-3, 0.0)]
    assert formal[1024]["cells"] == [(8e-3, 0.0)]
    assert formal[256]["ratio_pt"] == (2.0, 2.0)
    assert formal[1024]["ratio_pt"] == (2.0,)
    assert formal[256]["ratio_mean"] == pytest.approx(2.0)
    assert formal[1024]["ratio_mean"] == pytest.approx(2.0)
    assert formal[256]["ratio_std"] == pytest.approx(0.0)


def test_wbp_formal_seed_spread(tmp_path: Path) -> None:
    """Per-seed SER values 1, 2, 3 (x1e-3) give cell mean 2e-3, s.d. 1e-3."""
    for seed, wbp in zip((1, 2, 3), (1e-3, 2e-3, 3e-3)):
        _write_formal_file(tmp_path, 256, seed,
                           [_row(5e-3, 1e-3, wbp, 1.0, 1.0),
                            _row(3e-3, 1e-3, wbp, 1.0, 1.0)])
        _write_formal_file(tmp_path, 1024, seed,
                           [_row(5e-3, 1.0, 1.0, 4e-3, 8e-3),
                            _row(3e-3, 1.0, 1.0, 2e-3, 4e-3)])
    formal = wbp_formal(tmp_path, seeds=(1, 2, 3))
    assert formal[256]["cells"][0] == (pytest.approx(2e-3), pytest.approx(1e-3))
    # Per-seed mean ratios 1, 2, 3 give mean 2 and s.d. 1.
    assert formal[256]["ratio_mean"] == pytest.approx(2.0)
    assert formal[256]["ratio_std"] == pytest.approx(1.0)


def test_wbp_formal_wrong_rho_raises(tmp_path: Path) -> None:
    for seed in (1, 2, 3):
        _write_formal_file(tmp_path, 256, seed,
                           [_row(5e-3, 1e-3, 2e-3, 1.0, 1.0)], rho=0.75)
    with pytest.raises(RuntimeError, match="wbp_rho"):
        wbp_formal(tmp_path, seeds=(1, 2, 3))


def test_wbp_formal_missing_seed_raises(tmp_path: Path) -> None:
    _write_formal_file(tmp_path, 256, 1, [_row(5e-3, 1e-3, 2e-3, 1.0, 1.0)])
    with pytest.raises(FileNotFoundError):
        wbp_formal(tmp_path, seeds=(1, 2, 3))


def _linear_row(p_bit: float, bp_ser: float, lin_ser: float,
                bp_ber: float, lin_ber: float) -> dict:
    """One operating point carrying both SER and BER fields.

    The linear head reuses the GEMV row schema (GEMV_post / GEMV_post_BER),
    so the two field sets carry deliberately different numbers to assert
    that the 256-scale aggregation reads SER and the 1024-scale reads BER.
    """
    return {
        "p_bit": p_bit,
        "Bp_post": bp_ser, "GEMV_post": lin_ser,
        "Bp_post_BER": bp_ber, "GEMV_post_BER": lin_ber,
    }


def _write_linear_file(directory: Path, scale: int, seed: int,
                       rows: list[dict],
                       model: str = "LinearResidualCalibrator") -> None:
    """Write one formal linear-run JSON; the filename is fixed, the payload
    model is the variable under test (a wrong model must be caught from
    the payload)."""
    path = directory / f"E2_{scale}_s{seed}_n100k_linear.json"
    payload = {
        "model": model,
        "label": f"linear_{scale}_s{seed}",
        "n_codewords": 100000,
        "rows": rows,
    }
    path.write_text(json.dumps(payload))


def test_linear_formal_reads_scale_metric(tmp_path: Path) -> None:
    """256 reads GEMV_post SER cells; 1024 reads GEMV_post_BER cells."""
    ser_rows = [_linear_row(5e-3, 1e-3, 2e-3, 1.0, 1.0),
                _linear_row(3e-3, 3e-3, 6e-3, 1.0, 1.0)]
    ber_rows = [_linear_row(5e-3, 1.0, 1.0, 4e-3, 8e-3),
                _linear_row(3e-3, 1.0, 1.0, 2e-3, 4e-3)]
    for seed in (1, 2, 3):
        _write_linear_file(tmp_path, 256, seed, ser_rows)
        _write_linear_file(tmp_path, 1024, seed, ber_rows)

    formal = linear_formal(tmp_path)
    assert formal[256]["cells"] == [(2e-3, 0.0), (6e-3, 0.0)]
    assert formal[1024]["cells"] == [(8e-3, 0.0)]
    assert formal[256]["ratio_pt"] == (2.0, 2.0)
    assert formal[1024]["ratio_pt"] == (2.0,)
    assert formal[256]["ratio_mean"] == pytest.approx(2.0)
    assert formal[1024]["ratio_mean"] == pytest.approx(2.0)
    assert formal[256]["ratio_std"] == pytest.approx(0.0)


def test_linear_formal_seed_spread(tmp_path: Path) -> None:
    """Per-seed SER values 1, 2, 3 (x1e-3) give cell mean 2e-3, s.d. 1e-3."""
    for seed, lin in zip((1, 2, 3), (1e-3, 2e-3, 3e-3)):
        _write_linear_file(tmp_path, 256, seed,
                           [_linear_row(5e-3, 1e-3, lin, 1.0, 1.0),
                            _linear_row(3e-3, 1e-3, lin, 1.0, 1.0)])
        _write_linear_file(tmp_path, 1024, seed,
                           [_linear_row(5e-3, 1.0, 1.0, 4e-3, 8e-3),
                            _linear_row(3e-3, 1.0, 1.0, 2e-3, 4e-3)])
    formal = linear_formal(tmp_path)
    assert formal[256]["cells"][0] == (pytest.approx(2e-3), pytest.approx(1e-3))
    # Per-seed mean ratios 1, 2, 3 give mean 2 and s.d. 1.
    assert formal[256]["ratio_mean"] == pytest.approx(2.0)
    assert formal[256]["ratio_std"] == pytest.approx(1.0)


def test_linear_formal_wrong_model_raises(tmp_path: Path) -> None:
    for seed in (1, 2, 3):
        _write_linear_file(tmp_path, 256, seed,
                           [_linear_row(5e-3, 1e-3, 2e-3, 1.0, 1.0)],
                           model="GEMVResidualCalibrator")
    with pytest.raises(RuntimeError, match="model"):
        linear_formal(tmp_path)


def test_linear_formal_missing_seed_raises(tmp_path: Path) -> None:
    _write_linear_file(tmp_path, 256, 1, [_linear_row(5e-3, 1e-3, 2e-3,
                                                      1.0, 1.0)])
    with pytest.raises(FileNotFoundError):
        linear_formal(tmp_path)


def _write_ungated_file(directory: Path, scale: int, seed: int,
                        rows: list[dict]) -> None:
    """One ungated diagnostic run: same schema, tau_g disabled in the run."""
    path = directory / f"E2_{scale}_s{seed}_n100k_linear_ungated.json"
    payload = {
        "model": "LinearResidualCalibrator",
        "label": f"linear_ungated_{scale}_s{seed}",
        "model_gate_threshold": None,
        "n_codewords": 100000,
        "rows": rows,
    }
    path.write_text(json.dumps(payload))


def test_linear_formal_ungated_reads_distinct_files(tmp_path: Path) -> None:
    """ungated=True reads the *_linear_ungated.json files (tau_g disabled)."""
    gated_rows = [_linear_row(5e-3, 1e-3, 1e-3, 1.0, 1.0),
                  _linear_row(3e-3, 1e-3, 1e-3, 1.0, 1.0)]
    for seed in (1, 2, 3):
        # Gated files exist but must NOT be read when ungated=True.
        _write_linear_file(tmp_path, 256, seed, gated_rows)
        _write_linear_file(tmp_path, 1024, seed, gated_rows)
        _write_ungated_file(tmp_path, 256, seed,
                            [_linear_row(5e-3, 1e-3, 5e-3, 1.0, 1.0),
                             _linear_row(3e-3, 1e-3, 5e-3, 1.0, 1.0)])
        _write_ungated_file(tmp_path, 1024, seed,
                            [_linear_row(5e-3, 1.0, 1.0, 4e-3, 2e-2),
                             _linear_row(3e-3, 1.0, 1.0, 2e-3, 1e-2)])

    formal = linear_formal(tmp_path, ungated=True, seeds=(1, 2, 3))
    # The gated files (ratio 1.0 everywhere) must not have been picked up.
    assert formal[256]["cells"] == [(5e-3, 0.0), (5e-3, 0.0)]
    assert formal[1024]["cells"] == [(2e-2, 0.0)]
    assert formal[256]["ratio_pt"] == (5.0, 5.0)
    assert formal[1024]["ratio_pt"] == (5.0,)


def test_linear_formal_ungated_missing_file_raises(tmp_path: Path) -> None:
    _write_ungated_file(tmp_path, 256, 1, [_linear_row(5e-3, 1e-3, 2e-3,
                                                       1.0, 1.0)])
    with pytest.raises(FileNotFoundError):
        linear_formal(tmp_path, ungated=True, seeds=(1, 2, 3))


def _write_e2_file(directory: Path, scale: int, seed: int, tag: str,
                   rows: list[dict]) -> None:
    (directory / f"E2_{scale}_s{seed}_n100k_{tag}.json").write_text(
        json.dumps({"label": f"recal_{scale}_s{seed}", "rows": rows}))


def test_headline_tables_use_ten_seeds() -> None:
    """The manuscript states 10 seeds for Tables 1-2; the default must agree."""
    assert E2_SEEDS == tuple(range(1, 11))


def test_aggregate_e2_per_seed_gain_and_fallback(tmp_path: Path) -> None:
    """Gain is the mean of per-seed ratios; a closed gate contributes B'."""
    for seed, (bp, gemv, use) in zip(
            (1, 2, 3), ((4e-3, 2e-3, True), (6e-3, 2e-3, True),
                        (5e-3, 1e-3, False))):
        # As gemv.py writes it: GEMV_post_BER carries the gated fallback and
        # GEMV_ungated_post_BER the calibrator's own outcome.
        row = {"p_bit": 5e-3, "use_gemv": use, "drift_score": 0.03,
               "GEMV_gate": 0.9 if use else 0.5,
               "Bp_post": 9.0, "GEMV_post": 9.0,
               "Bp_post_BER": bp,
               "GEMV_post_BER": gemv if use else bp,
               "GEMV_ungated_post_BER": gemv}
        _write_e2_file(tmp_path, 1024, seed, "td011", [row])
    (r,) = aggregate_1024_e2(seeds=(1, 2, 3), directory=tmp_path)
    assert r.gains == (2.0, 3.0, 1.0)          # seed 3 falls back to B'
    assert r.ungated_gains == (2.0, 3.0, 5.0)  # what the closed gate forgoes
    assert r.open_flags == (True, True, False)
    assert r.ungated_mean == pytest.approx((2e-3 + 2e-3 + 1e-3) / 3)
    assert r.gain_seed_mean == pytest.approx(2.0)
    assert r.gain_seed_std == pytest.approx(1.0)
    assert r.gemv_mean == pytest.approx((2e-3 + 2e-3 + 5e-3) / 3)
    assert (r.n_open, r.n_seeds) == (2, 3)
    # The ratio of means differs from the mean of ratios; the tables quote
    # the latter.
    assert r.gain_mean == pytest.approx(5e-3 / 3e-3)


def test_aggregate_e2_reads_ser_at_256(tmp_path: Path) -> None:
    for seed in (1, 2):
        _write_e2_file(tmp_path, 256, seed, "td005", [{
            "p_bit": 5e-3, "use_gemv": True,
            "Bp_post": 2e-3, "GEMV_post": 1e-3,
            "Bp_post_BER": 9.0, "GEMV_post_BER": 9.0}])
    (r,) = aggregate_256_e2(seeds=(1, 2), directory=tmp_path)
    assert (r.bp_mean, r.gemv_mean) == (2e-3, 1e-3)


def test_aggregate_e2_missing_seed_raises(tmp_path: Path) -> None:
    """Averaging fewer seeds than stated must fail loudly, not silently."""
    _write_e2_file(tmp_path, 1024, 1, "td011", [{
        "p_bit": 5e-3, "use_gemv": True,
        "Bp_post_BER": 2e-3, "GEMV_post_BER": 1e-3}])
    with pytest.raises(FileNotFoundError):
        aggregate_1024_e2(seeds=(1, 2), directory=tmp_path)


def test_aggregate_e2_legacy_closed_gate_needs_companion(tmp_path: Path) -> None:
    """A pre-``GEMV_ungated_*`` run with a closed gate hides the correction
    it rejected; the forced-open companion supplies it, and its absence or a
    different data stream must raise rather than be papered over."""
    closed = {"p_bit": 1e-4, "use_gemv": False,
              "Bp_post_BER": 4e-3, "GEMV_post_BER": 4e-3}
    _write_e2_file(tmp_path, 1024, 3, "td011", [closed])
    with pytest.raises(FileNotFoundError, match="ungated_full"):
        aggregate_1024_e2(seeds=(3,), directory=tmp_path)

    forced = {"p_bit": 1e-4, "use_gemv": True,
              "Bp_post_BER": 4e-3, "GEMV_post_BER": 2e-3}
    _write_e2_file(tmp_path, 1024, 3, "td011_ungated_full", [forced])
    (r,) = aggregate_1024_e2(seeds=(3,), directory=tmp_path)
    assert r.gains == (1.0,) and r.ungated_gains == (2.0,)

    _write_e2_file(tmp_path, 1024, 3, "td011_ungated_full",
                   [dict(forced, Bp_post_BER=5e-3)])
    with pytest.raises(RuntimeError, match="not the same data stream"):
        aggregate_1024_e2(seeds=(3,), directory=tmp_path)
