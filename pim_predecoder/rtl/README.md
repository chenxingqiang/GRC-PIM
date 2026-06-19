# GRC-PIM Hardware Reference (Verilog)

Pre-silicon RTL aligned with `deps/NBLDPC_PIM` tape-out conventions:

| Module | Role |
|--------|------|
| `GRC_LLR_LUT.v` | `GRC_MESSAGE_TRANS` + `B_PRIME_ACCUM` — same ADC→LLR case table as `LLR_TRANS.v` |
| `DRIFT_DETECTOR.v` | Global temporal bit-disagreement score (Q16) + `GRC_DUAL_GATE` |
| `INT8_GEMV_MLP.v` | 425-param INT8 sequential GEMV MLP (13→24→3 + gate) |
| `GRC_PIM_TOP.v` | Top integration: B′ accumulate → optional GEMV → NB-LDPC prior LLR |

Python companions:

| File | Role |
|------|------|
| `pim_predecoder/hw/llr.py` | Simulation LLR front-end matching Verilog LUT |
| `pim_predecoder/hw/export_weights.py` | PyTorch checkpoint → INT8 `.hex` ROMs |

## Generate weight ROMs

```bash
cd /path/to/cim-corection
python3 -m pim_predecoder.hw.export_weights \
  --checkpoint pim_predecoder/_cache/gemv_k1024_n1088_r16_17_T4_d5e-03_b5e-03_s600_bs48_h24_seed1.pt \
  --out-dir pim_predecoder/rtl/Weights
```

Outputs: `fc1_w.hex`, `fc1_b.hex`, `fc2_w.hex`, `fc2_b.hex`, `gate_w.hex`, `gate_b.hex`, `quant_meta.json`.

## Simulate (Icarus Verilog)

```bash
cd pim_predecoder/rtl/TestBench
iverilog -o grc_pim_tb.o GRC_PIM_TOP_TB.v && vvp grc_pim_tb.o
```

Run from repository root so `$readmemh` paths resolve.

## Integration with NB-LDPC decoder

1. **Upstream:** Reuse `NBLDPC_Verilog/ECC_BUFFER.v` multi-cycle ADC capture.
2. **LLR path:** Replace per-cycle `MESSAGE_TRANS` output with `GRC_PIM_TOP` accumulated/corrected LLR bus (`OUT_LLR`, 3×3-bit signed).
3. **Downstream:** Feed `OUT_LLR` into `Serial_ECC_TOP_FSM` variable-node prior input (same width as `PRIOR_LLR`).

Default thresholds (1024-scale): `TAU_D_Q16 = 1638` (0.025), `TAU_G_Q16 = 39321` (0.6).

## Simulation cross-check (Python)

```python
from pim_predecoder.hw.llr import soft_llr_product_hw, compare_llr_frontends
# compare against init_llr_from_bits / soft_llr_product in core/gemv_calibrator.py
```

## Notes

- This is a **reference RTL** for architecture alignment and cosim, not a placed-and-routed tape-out block.
- Feature extraction (13-D per symbol) is assumed to run in the digital front-end; the testbench supplies `FEAT_IN` directly.
- Sigmoid/tanh in `GEMV_MLP_CORE` use piecewise LUT / saturation approximations suitable for fixed-point hardware; re-quantize after training export for bit-exact match if needed.
