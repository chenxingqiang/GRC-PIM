"""Python golden regression at k=1024 (or arbitrary N_SYMBOLS) + optional RTL spot-check."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from pim_predecoder.cosim.batch import (
    CosimConfig,
    compare_logs,
    generate_trace,
    write_golden,
    write_stimulus,
)
from pim_predecoder.paths import REPO, RTL

ROOT = REPO
COSIM_DIR = RTL / "cosim"
VDIR = RTL


def _iverilog_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + env.get("PATH", "")
    return env


def _sym_idx_w(n_symbols: int) -> int:
    return max(1, math.ceil(math.log2(n_symbols))) if n_symbols > 1 else 1


def _digest_trace(trace) -> str:
    payload = {
        "n_symbols": trace.cfg.n_symbols,
        "batch": trace.batch_row,
        "llr_out": [r["llr_out"] for r in trace.symbol_rows],
        "gates": [r["gate_q16"] for r in trace.symbol_rows],
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def run_regression_python(n_symbols: int, n_cycles: int, seed: int) -> dict:
    cfg = CosimConfig(n_symbols=n_symbols, n_cycles=n_cycles, seed=seed)
    trace = generate_trace(cfg)
    digest = _digest_trace(trace)
    return {
        "mode": "python",
        "n_symbols": n_symbols,
        "n_cycles": n_cycles,
        "seed": seed,
        "digest": digest,
        "batch": trace.batch_row,
        "n_symbol_rows": len(trace.symbol_rows),
    }


def run_regression_rtl(n_symbols: int, n_cycles: int, seed: int) -> dict:
    cfg = CosimConfig(n_symbols=n_symbols, n_cycles=n_cycles, seed=seed)
    trace = generate_trace(cfg)
    write_stimulus(trace, COSIM_DIR)
    write_golden(trace, COSIM_DIR)

    n_bits = n_symbols * 2
    sym_w = _sym_idx_w(n_symbols)
    cyc_w = _sym_idx_w(n_cycles)
    out_bin = COSIM_DIR / "regression.o"
    cmd = [
        "iverilog", "-g2012", "-o", str(out_bin),
        f"-DGRC_N_SYMBOLS={n_symbols}",
        f"-DGRC_N_CYCLES={n_cycles}",
        f"-DGRC_N_BITS={n_bits}",
        f"-DGRC_SYM_IDX_W={sym_w}",
        f"-DGRC_CYC_IDX_W={cyc_w}",
        str(VDIR / "GRC_LLR_LUT.v"),
        str(VDIR / "DRIFT_DETECTOR_CODEWORD.v"),
        str(VDIR / "INT8_GEMV_MLP.v"),
        str(VDIR / "GRC_PIM_TOP.v"),
        str(VDIR / "TestBench/GRC_PIM_TOP_COSIM_TB.v"),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, env=_iverilog_env())
    subprocess.run(["vvp", str(out_bin)], cwd=ROOT, check=True, env=_iverilog_env())

    result = compare_logs(COSIM_DIR / "golden.json", COSIM_DIR / "rtl.log")
    result.update({
        "mode": "rtl",
        "n_symbols": n_symbols,
        "digest_py": _digest_trace(trace),
    })
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="GRC-PIM golden regression")
    ap.add_argument("--n-symbols", type=int, default=1024)
    ap.add_argument("--n-cycles", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rtl", action="store_true", help="Run iverilog spot-check (use <=32 symbols)")
    args = ap.parse_args()

    if args.rtl:
        if args.n_symbols > 32:
            print("[regression] RTL spot-check: clamping n_symbols 32 (full 1024 is Python-only)")
            args.n_symbols = 32
        result = run_regression_rtl(args.n_symbols, args.n_cycles, args.seed)
        print(json.dumps(result, indent=2))
        if not result.get("pass"):
            raise SystemExit(1)
        print("REGRESSION RTL PASS")
        return

    result = run_regression_python(args.n_symbols, args.n_cycles, args.seed)
    print(json.dumps(result, indent=2))
    print(f"REGRESSION PYTHON PASS digest={result['digest'][:16]}...")


if __name__ == "__main__":
    main()
