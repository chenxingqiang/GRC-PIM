"""Generate cosim vectors and compare Verilog RTL log against Python golden model."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pim_predecoder.hw.llr import hw_drift_score_q16, message_llr_int
from pim_predecoder.hw.simulate import (
    TAU_D_Q16,
    TAU_G_Q16,
    GemvMlpHw,
    _sigmoid_lut_q16,
)
from pim_predecoder.paths import REPO, RTL

ROOT = REPO
COSIM_DIR = RTL / "cosim"
VDIR = RTL


@dataclass
class CosimConfig:
    n_symbols: int = 8
    n_cycles: int = 4
    seed: int = 42


@dataclass
class CosimTrace:
    cfg: CosimConfig
    adc: np.ndarray          # (n_cyc, n_sym) int
    bits: np.ndarray         # (n_cyc, n_sym, 2) int
    feats: np.ndarray        # (n_sym, 13) int8
    cycle_rows: list[dict] = field(default_factory=list)
    symbol_rows: list[dict] = field(default_factory=list)
    batch_row: dict = field(default_factory=dict)


def _adc_from_bits(b0: int, b1: int) -> int:
    return ((b1 & 1) << 2) | ((b0 & 1) << 1)


def generate_trace(cfg: CosimConfig | None = None) -> CosimTrace:
    cfg = cfg or CosimConfig()
    rng = np.random.default_rng(cfg.seed)
    bits = rng.integers(0, 2, size=(cfg.n_cycles, cfg.n_symbols, 2), dtype=np.int8)
    adc = np.zeros((cfg.n_cycles, cfg.n_symbols), dtype=np.int32)
    for cy in range(cfg.n_cycles):
        for sym in range(cfg.n_symbols):
            b0, b1 = int(bits[cy, sym, 0]), int(bits[cy, sym, 1])
            adc[cy, sym] = _adc_from_bits(b0, b1)
    feats = rng.integers(-32, 33, size=(cfg.n_symbols, 13), dtype=np.int8)
    trace = CosimTrace(cfg=cfg, adc=adc, bits=bits, feats=feats)

    mlp = GemvMlpHw.from_weights_dir()
    accums = np.zeros((cfg.n_symbols, 3), dtype=np.int32)
    anchors = np.zeros((cfg.n_symbols, 3), dtype=np.int32)
    gemv_out = np.zeros((cfg.n_symbols, 3), dtype=np.int32)
    sym_gates = np.zeros(cfg.n_symbols, dtype=np.int32)

    # Cycle-major order (matches Verilog TB).
    for cy in range(cfg.n_cycles):
        for sym in range(cfg.n_symbols):
            word = int(adc[cy, sym])
            cllr = message_llr_int(word, cim_e=True).astype(np.int32)
            if cy == 0:
                accums[sym] = cllr
            else:
                accums[sym] += cllr
            trace.cycle_rows.append({
                "cy": cy, "sym": sym, "adc": word,
                "cllr": cllr.tolist(),
                "accum": accums[sym].tolist(),
                "cycle_row_last": sym == cfg.n_symbols - 1,
                "batch_last": cy == cfg.n_cycles - 1 and sym == cfg.n_symbols - 1,
            })
            if cy == cfg.n_cycles - 1:
                anchors[sym] = accums[sym]

    # Drift over full bit tensor (batch, cyc, 2*n_sym).
    bit_tensor = bits.reshape(1, cfg.n_cycles, cfg.n_symbols * 2)
    drift_q16 = hw_drift_score_q16(bit_tensor)

    gate_sum = 0
    for sym in range(cfg.n_symbols):
        anchor_i8 = np.clip(anchors[sym], -128, 127).astype(np.int8)
        llr_out, gate_q16 = mlp.forward(feats[sym], anchor_i8)
        gemv_out[sym] = llr_out.astype(np.int32)
        sym_gates[sym] = gate_q16
        gate_sum += gate_q16
        trace.symbol_rows.append({
            "sym": sym,
            "anchor": anchor_i8.tolist(),
            "gemv": llr_out.tolist(),
            "gate_q16": int(gate_q16),
        })

    gate_mean = gate_sum // cfg.n_symbols
    use_gemv = bool(drift_q16 >= TAU_D_Q16 and gate_mean >= TAU_G_Q16)
    for row in trace.symbol_rows:
        sym = row["sym"]
        row["llr_out"] = (gemv_out[sym] if use_gemv else anchors[sym]).tolist()

    trace.batch_row = {
        "drift_q16": int(drift_q16),
        "gate_mean_q16": int(gate_mean),
        "use_gemv": use_gemv,
    }
    return trace


def write_stimulus(trace: CosimTrace, out_dir: Path = COSIM_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stimulus.txt"
    lines = [
        f"{trace.cfg.n_symbols} {trace.cfg.n_cycles}",
    ]
    for cy in range(trace.cfg.n_cycles):
        for sym in range(trace.cfg.n_symbols):
            feat = trace.feats[sym]
            feat_dec = " ".join(str(int(v)) for v in feat)
            row_last = int(sym == trace.cfg.n_symbols - 1)
            batch_last = int(
                cy == trace.cfg.n_cycles - 1 and sym == trace.cfg.n_symbols - 1
            )
            lines.append(
                f"{cy} {sym} {trace.adc[cy, sym]} 1 {feat_dec} {row_last} {batch_last}"
            )
    path.write_text("\n".join(lines) + "\n")
    return path


def write_golden(trace: CosimTrace, out_dir: Path = COSIM_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "golden.json"
    path.write_text(json.dumps({
        "cycle_rows": trace.cycle_rows,
        "symbol_rows": trace.symbol_rows,
        "batch_row": trace.batch_row,
    }, indent=2))
    return path


def compare_logs(golden_path: Path, rtl_path: Path, *, accum_tol: int = 0) -> dict:
    golden = json.loads(golden_path.read_text())
    rtl_lines = rtl_path.read_text().splitlines()
    rtl_cycle = [ln for ln in rtl_lines if ln.startswith("CYCLE ")]
    rtl_out = [ln for ln in rtl_lines if ln.startswith("OUT ")]
    rtl_batch = next((ln for ln in rtl_lines if ln.startswith("BATCH ")), None)

    errors: list[str] = []
    for i, (g, r) in enumerate(zip(golden["cycle_rows"], rtl_cycle)):
        gp = g
        parts = r.split()
        # CYCLE cy sym adc cl0 cl1 cl2 ac0 ac1 ac2
        r_adc = int(parts[3])
        r_cl = list(map(int, parts[4:7]))
        r_ac = list(map(int, parts[7:10]))
        if r_adc != gp["adc"]:
            errors.append(f"cycle[{i}] adc {r_adc} != {gp['adc']}")
        if r_cl != gp["cllr"]:
            errors.append(f"cycle[{i}] cllr {r_cl} != {gp['cllr']}")
        if any(abs(a - b) > accum_tol for a, b in zip(r_ac, gp["accum"])):
            errors.append(f"cycle[{i}] accum rtl={r_ac} py={gp['accum']}")

    for g, r in zip(golden["symbol_rows"], rtl_out):
        parts = r.split()
        # OUT sym o0 o1 o2 gate
        if len(parts) < 6:
            errors.append(f"malformed OUT line: {r}")
            continue
        r_out = list(map(int, parts[2:5]))
        r_gate = int(parts[5])
        if r_out != g["llr_out"]:
            errors.append(f"sym {g['sym']} llr_out rtl={r_out} py={g['llr_out']}")
        if r_gate != g["gate_q16"]:
            errors.append(f"sym {g['sym']} gate rtl={r_gate} py={g['gate_q16']}")

    if rtl_batch:
        parts = rtl_batch.split()
        gb = golden["batch_row"]
        r_drift, r_gate, r_use = int(parts[1]), int(parts[2]), int(parts[3])
        if abs(r_drift - gb["drift_q16"]) > 1:
            errors.append(f"drift rtl={r_drift} py={gb['drift_q16']}")
        if r_gate != gb["gate_mean_q16"]:
            errors.append(f"gate_mean rtl={r_gate} py={gb['gate_mean_q16']}")
        if r_use != int(gb["use_gemv"]):
            errors.append(f"use_gemv rtl={r_use} py={gb['use_gemv']}")

    return {
        "n_cycle_checks": len(golden["cycle_rows"]),
        "n_symbol_checks": len(golden["symbol_rows"]),
        "errors": errors,
        "pass": len(errors) == 0,
    }


def _iverilog_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + env.get("PATH", "")
    return env

def run_verilog(out_dir: Path = COSIM_DIR, cfg: CosimConfig | None = None) -> Path:
    cfg = cfg or CosimConfig()
    import math
    n_bits = cfg.n_symbols * 2
    sym_w = max(1, math.ceil(math.log2(cfg.n_symbols))) if cfg.n_symbols > 1 else 1
    cyc_w = max(1, math.ceil(math.log2(cfg.n_cycles))) if cfg.n_cycles > 1 else 1
    out_bin = out_dir / "cosim.o"
    cmd = [
        "iverilog", "-g2012", "-o", str(out_bin),
        f"-DGRC_N_SYMBOLS={cfg.n_symbols}",
        f"-DGRC_N_CYCLES={cfg.n_cycles}",
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
    return out_dir / "rtl.log"


def main() -> None:
    trace = generate_trace()
    write_stimulus(trace)
    write_golden(trace)
    print(f"[cosim] wrote {COSIM_DIR}/stimulus.txt and golden.json")
    print(f"[cosim] batch: {trace.batch_row}")

    rtl_log = run_verilog(cfg=trace.cfg)
    result = compare_logs(COSIM_DIR / "golden.json", rtl_log)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        print("COSIM FAILED", file=sys.stderr)
        for e in result["errors"][:20]:
            print(f"  {e}", file=sys.stderr)
        raise SystemExit(1)
    print("COSIM PASS")


if __name__ == "__main__":
    main()
