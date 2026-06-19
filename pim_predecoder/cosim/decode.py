"""End-to-end smoke: GRC -> PRIOR_MUX -> ECC_TOP decode vs Python bp_decode."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pim_predecoder.core.codec import bp_decode, encode, load_default_HG
from pim_predecoder.cosim.batch import CosimConfig, generate_trace
from pim_predecoder.cosim.cascade import (
    CascadeConfig,
    _flat_to_llr9,
    _iverilog_env,
    _llr9_to_flat,
    _sym_idx_w,
)
from pim_predecoder.hw.llr import (
    codeword_llr9,
    llr9_to_bp_llr,
    pack_input_symbol_bus,
)
from pim_predecoder.paths import REPO, RTL

ROOT = REPO
VDIR = RTL
NBDIR = ROOT / "deps" / "NBLDPC_PIM" / "NBLDPC_Verilog"
DECODE_DIR = VDIR / "cosim_decode"


@dataclass
class DecodeConfig(CascadeConfig):
    decode_cycles: int = 4000


def generate_decode_golden(cfg: DecodeConfig) -> tuple[dict, object]:
    grc_cfg = CosimConfig(
        n_symbols=cfg.n_grc_symbols, n_cycles=cfg.n_cycles, seed=cfg.seed
    )
    trace = generate_trace(grc_cfg)
    H, G = load_default_HG(ROOT)
    rng = np.random.default_rng(cfg.seed + 7)
    info = rng.integers(0, 3, size=(cfg.info_num,), dtype=np.int32)
    codeword = encode(info[None], G)[0].astype(np.int32)

    buffer_llr9 = codeword_llr9(codeword)
    merged = buffer_llr9.copy()
    input_symbol = codeword[: cfg.info_num]
    llr_f = llr9_to_bp_llr(merged)
    x_hat, _, iters = bp_decode(llr_f, H, n_iter=3, early_stop=True)
    expected_info = x_hat[0, : cfg.info_num].astype(np.int32)

    golden = {
        "cfg": cfg.__dict__,
        "use_grc": bool(trace.batch_row["use_gemv"]),
        "force_buffer": True,
        "bp_iters": int(iters),
        "info_seed": int(cfg.seed + 7),
        "input_symbol": input_symbol.tolist(),
        "expected_info": expected_info.tolist(),
        "buffer_flat": _llr9_to_flat(buffer_llr9),
        "ecc_input_flat": _llr9_to_flat(merged),
        "merged_llr9": merged.tolist(),
    }
    return golden, trace


def write_decode_stimulus(golden: dict, trace, out_dir: Path = DECODE_DIR) -> None:
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
    (out_dir / "buffer_llr.txt").write_text(
        " ".join(str(b) for b in golden["buffer_flat"]) + "\n"
    )
    (out_dir / "input_symbol.txt").write_text(
        " ".join(str(b) for b in pack_input_symbol_bus(np.array(golden["input_symbol"])))
        + "\n"
    )
    (out_dir / "golden.json").write_text(json.dumps(golden, indent=2))


def _decode_symbols_from_bus(flat_bits: list[int], info_num: int) -> np.ndarray:
    out = np.zeros(info_num, dtype=np.int32)
    idx = 0
    for s in range(info_num):
        v = 0
        for b in range(3):
            v |= (flat_bits[idx] & 1) << b
            idx += 1
        out[s] = v
    return out


def compare_decode_log(golden_path: Path, rtl_path: Path) -> dict:
    golden = json.loads(golden_path.read_text())
    rtl_lines = rtl_path.read_text().splitlines()
    decode_line = next((ln for ln in rtl_lines if ln.startswith("DECODE ")), None)
    errors: list[str] = []
    if not decode_line:
        errors.append("missing DECODE line in rtl.log")
        return {"pass": False, "errors": errors}

    rtl_bits = list(map(int, decode_line.split()[1:]))
    rtl_info = _decode_symbols_from_bus(rtl_bits, golden["cfg"]["info_num"])
    py_info = np.array(golden["expected_info"], dtype=np.int32)
    mism = np.where(rtl_info != py_info)[0]
    for s in mism[:10]:
        errors.append(f"info[{s}] rtl={int(rtl_info[s])} py={int(py_info[s])}")
    return {
        "pass": len(errors) == 0,
        "errors": errors,
        "n_info": golden["cfg"]["info_num"],
        "n_mismatch": int(len(mism)),
        "bp_iters_py": golden.get("bp_iters"),
    }


def run_verilog_decode(cfg: DecodeConfig) -> Path:
    n_bits = cfg.n_grc_symbols * 2
    sym_w = _sym_idx_w(cfg.n_grc_symbols)
    cyc_w = _sym_idx_w(cfg.n_cycles)
    out_bin = DECODE_DIR / "decode.o"
    nb_sources = [
        NBDIR / "LLR_TRANS.v",
        NBDIR / "LLR_REFLECT.v",
        NBDIR / "Signed_Quant.v",
        NBDIR / "Signed_Propagate.v",
        NBDIR / "Signed_Comparator.v",
        NBDIR / "Signed_CheckNode_onlypos.v",
        NBDIR / "Signed_Variable.v",
        NBDIR / "Serial_ECC_TOP_FSM.v",
    ]
    cmd = [
        "iverilog", "-g2012", "-o", str(out_bin),
        f"-DGRC_N_SYMBOLS={cfg.n_grc_symbols}",
        f"-DGRC_N_CYCLES={cfg.n_cycles}",
        f"-DGRC_N_BITS={n_bits}",
        f"-DGRC_SYM_IDX_W={sym_w}",
        f"-DGRC_CYC_IDX_W={cyc_w}",
        f"-DGRC_ECC_SYMBOL_NUM={cfg.ecc_symbol_num}",
        f"-DGRC_INFO_NUM={cfg.info_num}",
        f"-DGRC_ECC_DECODE_CYCLES={cfg.decode_cycles}",
        *[str(p) for p in nb_sources],
        str(VDIR / "GRC_LLR_LUT.v"),
        str(VDIR / "DRIFT_DETECTOR_CODEWORD.v"),
        str(VDIR / "INT8_GEMV_MLP.v"),
        str(VDIR / "GRC_PIM_TOP.v"),
        str(VDIR / "LLR_INT8_TO_ECC3.v"),
        str(VDIR / "GRC_ECC_PRIOR_MUX.v"),
        str(VDIR / "TestBench/GRC_ECC_DECODE_TB.v"),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, env=_iverilog_env())
    subprocess.run(["vvp", str(out_bin)], cwd=ROOT, check=True, env=_iverilog_env())
    return DECODE_DIR / "rtl.log"


def main() -> None:
    ap = argparse.ArgumentParser(description="GRC + ECC_TOP decode smoke cosim")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decode-cycles", type=int, default=4000)
    args = ap.parse_args()

    cfg = DecodeConfig(seed=args.seed, decode_cycles=args.decode_cycles)
    golden, trace = generate_decode_golden(cfg)
    write_decode_stimulus(golden, trace)
    print(
        f"[decode] wrote {DECODE_DIR}/golden.json "
        f"bp_iters={golden['bp_iters']} use_grc={golden['use_grc']}"
    )

    rtl_log = run_verilog_decode(cfg)
    result = compare_decode_log(DECODE_DIR / "golden.json", rtl_log)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        for e in result["errors"]:
            print(f"  {e}", file=sys.stderr)
        raise SystemExit(1)
    print("DECODE COSIM PASS")


if __name__ == "__main__":
    main()
