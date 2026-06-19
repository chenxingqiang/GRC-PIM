"""Cycle-accurate Python model of GRC_PIM Verilog for Mac cosim without iverilog.

Mirrors:
  - GRC_MESSAGE_TRANS (hw_llr.message_llr_int)
  - B' integer accumulation
  - DRIFT_DETECTOR Q16 score
  - GRC_DUAL_GATE
  - GEMV_MLP_CORE INT8 MAC + piecewise sigmoid LUT
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pim_predecoder.hw.llr import hw_drift_score_q16, message_llr_int
from pim_predecoder.paths import RTL_WEIGHTS

WEIGHTS_DIR = RTL_WEIGHTS
TAU_D_Q16 = 1638   # 0.025
TAU_G_Q16 = 39321  # 0.6


def _read_hex_int8(path: Path) -> np.ndarray:
    vals = [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]
    u = np.array(vals, dtype=np.uint8)
    return u.view(np.int8)


def _gate_pre_hw(acc: int) -> int:
    v = (acc >> 4) & 0xFF
    return v - 256 if v >= 128 else v


def _relu_hw(acc: int) -> int:
    y = _gate_pre_hw(acc)  # same [11:4] slice
    return max(0, y)


def _sat8(x: int) -> int:
    return max(-128, min(127, x))


def _sigmoid_lut_q16(x: int) -> int:
    if x <= -8:
        return 2621
    if x <= 0:
        return 19660
    if x <= 8:
        return 45875
    return 62914


@dataclass
class GemvMlpHw:
    fc1_w: np.ndarray  # (24, 13) int8
    fc1_b: np.ndarray  # (24,)
    fc2_w: np.ndarray  # (3, 24)
    fc2_b: np.ndarray  # (3,)
    gate_w: np.ndarray  # (13,)
    gate_b: np.ndarray  # (1,)
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_weights_dir(cls, d: Path = WEIGHTS_DIR) -> GemvMlpHw:
        meta = json.loads((d / "quant_meta.json").read_text())
        return cls(
            fc1_w=_read_hex_int8(d / "fc1_w.hex").reshape(24, 13),
            fc1_b=_read_hex_int8(d / "fc1_b.hex").reshape(24),
            fc2_w=_read_hex_int8(d / "fc2_w.hex").reshape(3, 24),
            fc2_b=_read_hex_int8(d / "fc2_b.hex").reshape(3),
            gate_w=_read_hex_int8(d / "gate_w.hex").reshape(13),
            gate_b=_read_hex_int8(d / "gate_b.hex").reshape(1),
            meta=meta,
        )

    def forward(self, feat_i8: np.ndarray, llr_anchor_i8: np.ndarray) -> tuple[np.ndarray, int]:
        hidden = np.zeros(24, dtype=np.int8)
        for h in range(24):
            s = int(self.fc1_b[h])
            for i in range(13):
                s += int(feat_i8[i]) * int(self.fc1_w[h, i])
            hidden[h] = np.int8(_relu_hw(s))
        delta = np.zeros(3, dtype=np.int8)
        for o in range(3):
            s = int(self.fc2_b[o])
            for h in range(24):
                s += int(hidden[h]) * int(self.fc2_w[o, h])
            delta[o] = np.int8(_gate_pre_hw(s))
        gs = int(self.gate_b[0])
        for i in range(13):
            gs += int(feat_i8[i]) * int(self.gate_w[i])
        gate_q16 = _sigmoid_lut_q16(_gate_pre_hw(gs))
        llr_out = llr_anchor_i8.astype(np.int16) + delta.astype(np.int16)
        return llr_out.astype(np.int8), gate_q16


@dataclass
class GrcPimSim:
    n_cycles: int = 4
    tau_d_q16: int = TAU_D_Q16
    tau_g_q16: int = TAU_G_Q16
    cim_e: bool = True
    accum_llr: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int32))
    cycle_cnt: int = 0
    gate_sum: int = 0
    gate_cnt: int = 0
    drift_q16: int = 0
    mlp: GemvMlpHw | None = None

    def reset_symbol(self) -> None:
        self.accum_llr[:] = 0
        self.cycle_cnt = 0

    def feed_cycle(self, adc_word: int, *, cycle_last: bool) -> None:
        llr = message_llr_int(adc_word, cim_e=self.cim_e).astype(np.int32)
        self.accum_llr += llr
        if cycle_last:
            self.cycle_cnt += 1

    def finish_symbol(
        self,
        feat_i8: np.ndarray | None,
        *,
        noisy_bits_cycles: list[tuple[int, int]] | None = None,
    ) -> dict:
        """After n_cycles, run MLP if features provided; compute drift from bit pairs."""
        if noisy_bits_cycles is not None and len(noisy_bits_cycles) >= 2:
            bits_3d = np.array(noisy_bits_cycles, dtype=np.int8)[None, :, :]
            self.drift_q16 = hw_drift_score_q16(bits_3d)
        anchor_i8 = np.clip(self.accum_llr, -128, 127).astype(np.int8)
        gate_mean_q16 = 0
        llr_out = anchor_i8.copy()
        if feat_i8 is not None and self.mlp is not None:
            llr_gemv, gate_q16 = self.mlp.forward(feat_i8, anchor_i8)
            self.gate_sum += gate_q16
            self.gate_cnt += 1
            gate_mean_q16 = self.gate_sum // max(1, self.gate_cnt)
            use_gemv = (self.drift_q16 >= self.tau_d_q16) and (gate_mean_q16 >= self.tau_g_q16)
            if use_gemv:
                llr_out = llr_gemv
        else:
            use_gemv = False
            gate_mean_q16 = 0
        return {
            "accum_llr": anchor_i8.tolist(),
            "llr_out": llr_out.tolist(),
            "drift_q16": self.drift_q16,
            "gate_mean_q16": gate_mean_q16,
            "use_gemv": use_gemv,
        }


def run_self_test() -> dict:
    """Sanity checks against known LUT values and gate thresholds."""
    results: dict = {"tests": []}

    # LUT: ADC 010 -> LLR [1,1,2] signed
    llr = message_llr_int(0b010, cim_e=True)
    ok_lut = llr.tolist() == [1, 1, 2]
    results["tests"].append({"name": "message_lut_010", "pass": ok_lut, "got": llr.tolist()})

    # B' accumulate two cycles
    sim = GrcPimSim(n_cycles=2, mlp=GemvMlpHw.from_weights_dir())
    sim.feed_cycle(0b010, cycle_last=True)
    sim.feed_cycle(0b011, cycle_last=True)
    bits_cycles = [(1, 0), (1, 1)]  # approx from adc mapping
    r = sim.finish_symbol(np.full(13, 4, dtype=np.int8), noisy_bits_cycles=bits_cycles)
    results["b_prime_accum"] = r["accum_llr"]
    results["drift_q16"] = r["drift_q16"]
    results["use_gemv"] = r["use_gemv"]
    results["tests"].append({
        "name": "accum_two_cycles",
        "pass": r["accum_llr"] == [3, 2, 3],
        "got": r["accum_llr"],
        "expect": [3, 2, 3],
    })

    # Drift: identical cycles -> score 0
    identical = np.zeros((1, 4, 4), dtype=np.int8)
    d0 = hw_drift_score_q16(identical)
    results["tests"].append({"name": "drift_identical_zero", "pass": d0 == 0, "got": d0})

    # Drift: all flip -> score 65535
    flip = np.array([[[0, 0, 0, 0], [1, 1, 1, 1]]], dtype=np.int8)
    d1 = hw_drift_score_q16(flip)
    results["tests"].append({"name": "drift_all_flip_high", "pass": d1 == 65535, "got": d1})

    results["all_pass"] = all(t["pass"] for t in results["tests"])
    return results


def compare_with_float_pipeline(n_codewords: int = 8, seed: int = 1) -> dict:
    import torch
    from pim_predecoder.core.codec import bp_decode, encode, load_HG
    from pim_predecoder.core.gemv_calibrator import GEMVResidualCalibrator, build_gemv_features, soft_llr_product
    from pim_predecoder.core.noise3d import apply_pim_noise_3d
    from pim_predecoder.paths import CACHE

    H, G = load_HG(".", 1024, "16_17")
    rng = np.random.default_rng(seed)
    info = rng.integers(0, 3, (n_codewords, G.shape[0]))
    clean = encode(info, G)
    noisy = apply_pim_noise_3d(clean, 4, 5e-3, 5e-3, 5e-3, rng=rng)
    bits = noisy.reshape(n_codewords, 4, -1)

    ckpt = CACHE / "gemv_k1024_n1088_r16_17_T4_d5e-03_b5e-03_s600_bs48_h24_seed1.pt"
    model = GEMVResidualCalibrator(hidden=24)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

    mlp_hw = GemvMlpHw.from_weights_dir()
    sim = GrcPimSim(n_cycles=4, mlp=mlp_hw)

    # Single-symbol spot check (first codeword, symbol 0)
    sym_bits = bits[0, :, 0:2]  # 4 cycles x 2 bits for symbol 0
    sim.reset_symbol()
    adc_words = []
    for t in range(4):
        b0, b1 = int(sym_bits[t, 0]), int(sym_bits[t, 1])
        adc = (b1 << 2) | (b0 << 1)
        adc_words.append(adc)
        sim.feed_cycle(adc, cycle_last=True)
    feats, llr_bp = build_gemv_features(bits[0:1], 5e-3)
    feat_q = np.clip(np.round(feats[0, 0] * 127 / max(np.abs(feats[0, 0]).max(), 1e-6)), -128, 127).astype(np.int8)
    cycles_bits = [(int(sym_bits[t, 0]), int(sym_bits[t, 1])) for t in range(4)]
    hw_res = sim.finish_symbol(feat_q, noisy_bits_cycles=cycles_bits)

    with torch.no_grad():
        llr_f, gate_f = model.forward(
            torch.from_numpy(feats[0:1, 0:1]),
            torch.from_numpy(llr_bp[0:1, 0:1]),
        )
        llr_f = llr_f - llr_f.max(dim=-1, keepdim=True).values
    float_out = llr_f[0, 0].numpy()

    llr_hw_f = np.array(hw_res["llr_out"], dtype=np.float32)
    drift_ok = hw_res["drift_q16"] >= TAU_D_Q16

    x_hw, _, _ = bp_decode(
        soft_llr_product(bits, 5e-3), H, 3, True  # use sw B' for batch BER baseline
    )
    return {
        "symbol0_hw_llr": hw_res["llr_out"],
        "symbol0_float_llr": float_out.tolist(),
        "symbol0_gate_hw_q16": hw_res.get("gate_mean_q16", 0),
        "drift_q16": hw_res["drift_q16"],
        "drift_gate_would_open": drift_ok,
        "use_gemv_hw": hw_res["use_gemv"],
        "note": "INT8 HW MLP is approximate vs float; check drift/gate/LUT primarily",
    }


def main() -> None:
    print("=" * 60)
    print("GRC-PIM Python RTL self-test")
    st = run_self_test()
    for t in st["tests"]:
        status = "PASS" if t["pass"] else "FAIL"
        print(f"  [{status}] {t['name']}: {t.get('got')}")
    print(f"  all_pass={st['all_pass']}")

    print("=" * 60)
    print("Float vs INT8 HW spot check (symbol 0)")
    cmp = compare_with_float_pipeline()
    print(json.dumps(cmp, indent=2))

    print("=" * 60)
    print("Full-batch LLR frontend + BP (verify_hw_cosim)")
    from pim_predecoder.hw.verify_cosim import run_verify
    rep = run_verify(n_codewords=32, seed=1)
    print(json.dumps(rep, indent=2))
    ok = st["all_pass"] and rep["llr_frontend"]["prob_argmax_agree"] > 0.99
    print(f"\nOverall: {'OK' if ok else 'CHECK FAILURES'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
