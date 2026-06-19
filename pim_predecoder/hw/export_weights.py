"""Export trained GEMV calibrator weights to INT8 hex ROMs for Verilog.

Usage:
  python3 -m pim_predecoder.hw.export_weights \\
      --checkpoint pim_predecoder/_cache/gemv_k1024_n1088_r16_17_T4_d5e-03_b5e-03_s600_bs48_h24_seed1.pt \\
      --out-dir pim_predecoder/rtl/Weights
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pim_predecoder.core.gemv_calibrator import GEMVResidualCalibrator
from pim_predecoder.paths import RTL_WEIGHTS


def _quantize_int8(arr: np.ndarray) -> tuple[np.ndarray, float]:
    arr = arr.astype(np.float64)
    peak = max(float(np.abs(arr).max()), 1e-12)
    scale = 127.0 / peak
    q = np.clip(np.round(arr * scale), -128, 127).astype(np.int8)
    return q, scale


def _write_hex(path: Path, values: np.ndarray) -> None:
    flat = values.reshape(-1)
    lines = []
    for v in flat:
        u = int(v) & 0xFF
        lines.append(f"{u:02x}")
    path.write_text("\n".join(lines) + "\n")


def export_checkpoint(checkpoint: Path, out_dir: Path, hidden: int = 24) -> dict:
    model = GEMVResidualCalibrator(hidden=hidden)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict = {"checkpoint": str(checkpoint), "hidden": hidden, "layers": {}}
    layers = {
        "fc1_w": model.fc1.weight.detach().cpu().numpy(),   # (24, 13)
        "fc1_b": model.fc1.bias.detach().cpu().numpy(),     # (24,)
        "fc2_w": model.fc2.weight.detach().cpu().numpy(),   # (3, 24)
        "fc2_b": model.fc2.bias.detach().cpu().numpy(),       # (3,)
        "gate_w": model.gate.weight.detach().cpu().numpy(),   # (1, 13)
        "gate_b": model.gate.bias.detach().cpu().numpy(),     # (1,)
    }
    for name, arr in layers.items():
        q, scale = _quantize_int8(arr)
        _write_hex(out_dir / f"{name}.hex", q)
        meta["layers"][name] = {
            "shape": list(arr.shape),
            "scale": scale,
            "hex_file": f"{name}.hex",
        }

    meta["clip"] = float(model.clip)
    (out_dir / "quant_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[export] wrote INT8 ROMs to {out_dir}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=RTL_WEIGHTS)
    ap.add_argument("--hidden", type=int, default=24)
    args = ap.parse_args()
    export_checkpoint(args.checkpoint, args.out_dir, hidden=args.hidden)


if __name__ == "__main__":
    main()
