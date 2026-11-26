"""Tests for pim_predecoder.cosim batch/regression/cascade/decode."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pim_predecoder.cosim.batch import (
    CosimConfig,
    compare_logs,
    generate_trace,
    run_verilog,
    write_golden,
    write_stimulus,
)
from pim_predecoder.cosim.cascade import (
    CascadeConfig,
    _flat_to_llr9,
    _llr9_to_flat,
    compare_cascade_log,
    generate_cascade_golden,
    write_cascade_stimulus,
)
from pim_predecoder.cosim.regression import run_regression_python
from pim_predecoder.paths import RTL


K1024_DIGEST = (
    "583751fff848d8d024750b4b22d25856059729f5ea7b39ccdd7ab40cf018124d"
)


def test_generate_trace_defaults():
    trace = generate_trace()
    assert trace.cfg.n_symbols == 8
    assert trace.cfg.n_cycles == 4
    assert len(trace.cycle_rows) == 32
    assert len(trace.symbol_rows) == 8
    assert "drift_q16" in trace.batch_row


def test_write_stimulus_and_golden(tmp_path: Path):
    trace = generate_trace(CosimConfig(n_symbols=4, n_cycles=2, seed=1))
    stim = write_stimulus(trace, tmp_path)
    gold = write_golden(trace, tmp_path)
    assert stim.is_file()
    assert gold.is_file()
    data = json.loads(gold.read_text())
    assert len(data["cycle_rows"]) == 8


def test_compare_logs_pass_with_golden(tmp_path: Path):
    trace = generate_trace(CosimConfig(n_symbols=2, n_cycles=2, seed=0))
    gold_path = write_golden(trace, tmp_path)
    # Synthesize rtl.log from golden (perfect match)
    golden = json.loads(gold_path.read_text())
    lines = []
    for row in golden["cycle_rows"]:
        cl = " ".join(str(x) for x in row["cllr"])
        ac = " ".join(str(x) for x in row["accum"])
        lines.append(f"CYCLE {row['cy']} {row['sym']} {row['adc']} {cl} {ac}")
    for row in golden["symbol_rows"]:
        out = " ".join(str(x) for x in row["llr_out"])
        lines.append(f"OUT {row['sym']} {out} {row['gate_q16']}")
    br = golden["batch_row"]
    lines.append(f"BATCH {br['drift_q16']} {br['gate_mean_q16']} {int(br['use_gemv'])}")
    rtl_path = tmp_path / "rtl.log"
    rtl_path.write_text("\n".join(lines) + "\n")
    result = compare_logs(gold_path, rtl_path)
    assert result["pass"] is True


def test_llr9_flat_roundtrip():
    llr9 = np.array([[-3, 0, 2], [1, -1, 3]], dtype=np.int32)
    flat = _llr9_to_flat(llr9)
    back = _flat_to_llr9(flat, 2)
    assert np.array_equal(back, llr9)


def test_generate_cascade_golden():
    golden, trace = generate_cascade_golden(CascadeConfig())
    assert "ecc_input_flat" in golden
    assert "merged_llr9" in golden
    assert trace.cfg.n_symbols == 8


def test_write_cascade_stimulus(tmp_path: Path):
    golden, trace = generate_cascade_golden(CascadeConfig(n_grc_symbols=4))
    write_cascade_stimulus(golden, trace, tmp_path)
    assert (tmp_path / "stimulus.txt").is_file()
    assert (tmp_path / "buffer_llr.txt").is_file()
    assert (tmp_path / "golden.json").is_file()


def test_compare_cascade_log_pass(tmp_path: Path):
    golden, _ = generate_cascade_golden(CascadeConfig())
    gold_path = tmp_path / "golden.json"
    gold_path.write_text(json.dumps(golden))
    flat = golden["ecc_input_flat"]
    rtl_path = tmp_path / "rtl.log"
    rtl_path.write_text("PRIORS " + " ".join(str(b) for b in flat) + "\n")
    result = compare_cascade_log(gold_path, rtl_path)
    assert result["pass"] is True


def test_regression_python_k1024_digest():
    rep = run_regression_python(n_symbols=1024, n_cycles=4, seed=42)
    assert rep["digest"] == K1024_DIGEST
    assert rep["n_symbol_rows"] == 1024


@pytest.mark.rtl
def test_drift_bounds_tb(require_iverilog, tmp_path):
    """Q16 drift scaling and the band-pass gate bounds, in RTL.

    Covers the overflow the widened product fixes: at 3998 accumulated
    comparisons the intermediate disagree*65535 exceeds 2^32, and the original
    32-bit expression reported 15981 instead of 32767.
    """
    exe = tmp_path / "tb_drift"
    build = subprocess.run(
        ["iverilog", "-g2005", "-o", str(exe), "-s", "tb_drift_bounds",
         str(RTL / "tb_drift_bounds.v"),
         str(RTL / "DRIFT_DETECTOR_CODEWORD.v"),
         str(RTL / "DRIFT_DETECTOR.v")],
        capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "ALL CHECKS PASSED" in run.stdout, run.stdout
    assert "FAIL" not in run.stdout, run.stdout


@pytest.mark.rtl
def test_cosim_batch_rtl(require_iverilog):
    trace = generate_trace(CosimConfig(n_symbols=8, n_cycles=4, seed=42))
    out_dir = RTL / "cosim"
    write_stimulus(trace, out_dir)
    write_golden(trace, out_dir)
    rtl_log = run_verilog(out_dir, trace.cfg)
    result = compare_logs(out_dir / "golden.json", rtl_log)
    assert result["pass"] is True


@pytest.mark.rtl
def test_cosim_modules_main(require_iverilog):
    modules = [
        "pim_predecoder.cosim.batch",
        "pim_predecoder.cosim.cascade",
        "pim_predecoder.cosim.decode",
    ]
    for mod in modules:
        proc = subprocess.run(
            [sys.executable, "-m", mod],
            cwd=str(RTL.parent.parent),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"{mod} failed:\n{proc.stderr}\n{proc.stdout}"
