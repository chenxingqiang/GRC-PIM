# GRC-PIM Hardware Reference (Verilog)

Pre-silicon RTL aligned with `deps/NBLDPC_PIM` tape-out conventions:

| Module | Role |
|--------|------|
| `GRC_LLR_LUT.v` | `GRC_MESSAGE_TRANS` + `B_PRIME_ACCUM` — same ADC→LLR case table as `LLR_TRANS.v` |
| `DRIFT_DETECTOR.v` | Global temporal bit-disagreement score (Q16) + `GRC_DUAL_GATE` |
| `INT8_GEMV_MLP.v` | 425-param INT8 GEMV MLP (13→24→3 + gate): `GEMV_MLP_CORE` (sequential, `LANES`/`PIPE`) and the 3-stage symbol pipeline `GEMV_MLP_PIPE` (`ROWS_PER_CYCLE`) |
| `GRC_PIM_TOP.v` | Top integration: B′ accumulate → optional GEMV → NB-LDPC prior LLR |

Python companions:

| File | Role |
|------|------|
| `pim_predecoder/hw/llr.py` | Simulation LLR front-end matching Verilog LUT |
| `pim_predecoder/hw/export_weights.py` | PyTorch checkpoint → INT8 `.hex` ROMs |

## Generate weight ROMs

```bash
python3 -m pim_predecoder.hw.export_weights \
  --checkpoint pim_predecoder/_cache/gemv_k1024_n1088_r16_17_T4_d5e-03_b5e-03_s600_bs48_h24_seed1.pt \
  --out-dir pim_predecoder/rtl/Weights
```

Outputs: `fc1_w.hex`, `fc1_b.hex`, `fc2_w.hex`, `fc2_b.hex`, `gate_w.hex`, `gate_b.hex`, `quant_meta.json`.

## Simulate (Icarus Verilog)

```bash
bash pim_predecoder/rtl/run_sim.sh
```

Run from repository root so `$readmemh` paths resolve.

Throughput regression: the sequential FSM (401 cycles/symbol) against the three-stage symbol
pipeline (cold start 4, steady state 2 cycles/symbol), including a bit-exactness check against
the sequential core over eight symbols.

```bash
iverilog -g2012 -DROWS_PER_CYCLE=15 -DNSYM=8 -o /tmp/gemv_pipe_tb.o \
  pim_predecoder/rtl/INT8_GEMV_MLP.v pim_predecoder/rtl/TestBench/GEMV_PIPE_TB.v
vvp /tmp/gemv_pipe_tb.o   # prints PIPE_RESULT ... checks=ok
```

## Integration with NB-LDPC decoder

1. **Upstream:** Reuse `NBLDPC_Verilog/ECC_BUFFER.v` multi-cycle ADC capture.
2. **LLR path:** Replace per-cycle `MESSAGE_TRANS` output with `GRC_PIM_TOP` accumulated/corrected LLR bus (`OUT_LLR`, 3×8-bit signed).
3. **Downstream:** Use the INT8-to-ECC9 adapter to form the decoder priors. See the [integration boundary](../hw/FEATURE_OFFLOAD.md).

Default thresholds (1024-scale): `TAU_D_Q16 = 721` (0.011), `TAU_G_Q16 = 39321` (0.6).
Both are module parameters; the 256-scale configuration uses `TAU_D_Q16 = 328` (0.005).
See `research/sn-article.tex` Sec. 4.4 for how these were calibrated.

## Simulation cross-check (Python)

```python
from pim_predecoder.hw.llr import soft_llr_product_hw, compare_llr_frontends
# compare against init_llr_from_bits / soft_llr_product in core/gemv_calibrator.py
```

## Notes

- This is a **reference RTL** for architecture alignment and cosim, not a placed-and-routed tape-out block.
- Feature extraction (13-D per symbol) is assumed to run in the digital front-end; the testbench supplies `FEAT_IN` directly.
- Sigmoid/tanh in `GEMV_MLP_CORE` use piecewise LUT / saturation approximations suitable for fixed-point hardware; re-quantize after training export for bit-exact match if needed.
