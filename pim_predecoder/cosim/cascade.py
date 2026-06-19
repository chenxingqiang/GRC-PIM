"""Bus-level cascade cosim: GRC OUT_LLR -> GRC_ECC_PRIOR_MUX -> ECC INPUT_LLR."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pim_predecoder.cosim.batch import CosimConfig, generate_trace
from pim_predecoder.hw.llr import int8_vec_to_ecc9
from pim_predecoder.paths import REPO, RTL

ROOT = REPO
VDIR = RTL
CASCADE_DIR = VDIR / "cosim_cascade"


def _iverilog_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + env.get("PATH", "")
    return env


@dataclass
class CascadeConfig:
    n_grc_symbols: int = 8
    n_cycles: int = 4
    ecc_symbol_num: int = 288
    info_num: int = 256
    seed: int = 42


def _sym_idx_w(n: int) -> int:
    return max(1, math.ceil(math.log2(n))) if n > 1 else 1


def _llr9_to_flat(llr9: np.ndarray) -> list[int]:
    """Pack (SYMBOL_NUM, 3) signed LLRs into ECC flat bit bus (MSB field2)."""
    bits: list[int] = []
    for s in range(llr9.shape[0]):
        for f in range(3):
            v = int(llr9[s, f])
            if v < 0:
                v = (v + 8) & 0x7
            for b in range(3):
                bits.append((v >> b) & 1)
    return bits


def _flat_to_llr9(flat: list[int], symbol_num: int) -> np.ndarray:
    out = np.zeros((symbol_num, 3), dtype=np.int32)
    idx = 0
    for s in range(symbol_num):
        for f in range(3):
            v = 0
            for b in range(3):
                v |= (flat[idx] & 1) << b
                idx += 1
            if v >= 4:
                v -= 8
            out[s, f] = v
    return out


def generate_cascade_golden(cfg: CascadeConfig) -> tuple[dict, object]:
    grc_cfg = CosimConfig(
        n_symbols=cfg.n_grc_symbols, n_cycles=cfg.n_cycles, seed=cfg.seed
    )
    trace = generate_trace(grc_cfg)
    rng = np.random.default_rng(cfg.seed + 1)

    buffer_llr9 = rng.integers(-3, 4, size=(cfg.ecc_symbol_num, 3), dtype=np.int32)
    info_llr = np.array([row["llr_out"] for row in trace.symbol_rows], dtype=np.int32)
    use_grc = trace.batch_row["use_gemv"]

    merged = buffer_llr9.copy()
    if use_grc:
        for s in range(min(cfg.n_grc_symbols, cfg.info_num)):
            merged[s] = int8_vec_to_ecc9(info_llr[s])

    flat = _llr9_to_flat(merged)
    buffer_flat = _llr9_to_flat(buffer_llr9)

    golden = {
        "cfg": cfg.__dict__,
        "use_grc": use_grc,
        "grc_llr_out": info_llr.tolist(),
        "buffer_flat": buffer_flat,
        "ecc_input_flat": flat,
        "merged_llr9": merged.tolist(),
    }
    return golden, trace


def write_cascade_stimulus(golden: dict, trace, out_dir: Path = CASCADE_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = golden["cfg"]
    grc_cfg = CosimConfig(
        n_symbols=cfg["n_grc_symbols"],
        n_cycles=cfg["n_cycles"],
        seed=cfg["seed"],
    )

    lines = [f"{grc_cfg.n_symbols} {grc_cfg.n_cycles}"]
    for cy in range(grc_cfg.n_cycles):
        for sym in range(grc_cfg.n_symbols):
            feat_dec = " ".join(str(int(v)) for v in trace.feats[sym])
            row_last = int(sym == grc_cfg.n_symbols - 1)
            batch_last = int(
                cy == grc_cfg.n_cycles - 1 and sym == grc_cfg.n_symbols - 1
            )
            lines.append(
                f"{cy} {sym} {trace.adc[cy, sym]} 1 {feat_dec} {row_last} {batch_last}"
            )
    (out_dir / "stimulus.txt").write_text("\n".join(lines) + "\n")

    buf_lines = [" ".join(str(b) for b in golden["buffer_flat"])]
    (out_dir / "buffer_llr.txt").write_text("\n".join(buf_lines) + "\n")
    (out_dir / "golden.json").write_text(json.dumps(golden, indent=2))


def compare_cascade_log(golden_path: Path, rtl_path: Path) -> dict:
    golden = json.loads(golden_path.read_text())
    rtl_lines = rtl_path.read_text().splitlines()
    priors_line = next((ln for ln in rtl_lines if ln.startswith("PRIORS ")), None)
    errors: list[str] = []
    if not priors_line:
        errors.append("missing PRIORS line in rtl.log")
        return {"pass": False, "errors": errors}

    rtl_flat = list(map(int, priors_line.split()[1:]))
    py_flat = golden["ecc_input_flat"]
    if len(rtl_flat) != len(py_flat):
        errors.append(f"flat len rtl={len(rtl_flat)} py={len(py_flat)}")
    else:
        sym_num = golden["cfg"]["ecc_symbol_num"]
        rtl_llr9 = _flat_to_llr9(rtl_flat, sym_num)
        py_llr9 = np.array(golden["merged_llr9"], dtype=np.int32)
        for s in range(sym_num):
            if not np.array_equal(rtl_llr9[s], py_llr9[s]):
                errors.append(f"sym {s} rtl={rtl_llr9[s].tolist()} py={py_llr9[s].tolist()}")
                if len(errors) >= 10:
                    break
    return {"pass": len(errors) == 0, "errors": errors}


def run_verilog_cascade(cfg: CascadeConfig) -> Path:
    n_bits = cfg.n_grc_symbols * 2
    sym_w = _sym_idx_w(cfg.n_grc_symbols)
    cyc_w = _sym_idx_w(cfg.n_cycles)
    ecc_w = cfg.ecc_symbol_num * 9
    out_bin = CASCADE_DIR / "cascade.o"
    cmd = [
        "iverilog", "-g2012", "-o", str(out_bin),
        f"-DGRC_N_SYMBOLS={cfg.n_grc_symbols}",
        f"-DGRC_N_CYCLES={cfg.n_cycles}",
        f"-DGRC_N_BITS={n_bits}",
        f"-DGRC_SYM_IDX_W={sym_w}",
        f"-DGRC_CYC_IDX_W={cyc_w}",
        f"-DGRC_ECC_SYMBOL_NUM={cfg.ecc_symbol_num}",
        f"-DGRC_INFO_NUM={cfg.info_num}",
        str(VDIR / "GRC_LLR_LUT.v"),
        str(VDIR / "DRIFT_DETECTOR_CODEWORD.v"),
        str(VDIR / "INT8_GEMV_MLP.v"),
        str(VDIR / "GRC_PIM_TOP.v"),
        str(VDIR / "LLR_INT8_TO_ECC3.v"),
        str(VDIR / "GRC_ECC_PRIOR_MUX.v"),
        str(VDIR / "TestBench/GRC_ECC_CASCADE_TB.v"),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, env=_iverilog_env())
    subprocess.run(["vvp", str(out_bin)], cwd=ROOT, check=True, env=_iverilog_env())
    return CASCADE_DIR / "rtl.log"


def main() -> None:
    cfg = CascadeConfig()
    golden, trace = generate_cascade_golden(cfg)
    write_cascade_stimulus(golden, trace)
    print(f"[cascade] wrote {CASCADE_DIR}/golden.json use_grc={golden['use_grc']}")

    rtl_log = run_verilog_cascade(cfg)
    result = compare_cascade_log(CASCADE_DIR / "golden.json", rtl_log)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        for e in result["errors"]:
            print(f"  {e}", file=sys.stderr)
        raise SystemExit(1)
    print("CASCADE COSIM PASS")


if __name__ == "__main__":
    main()
