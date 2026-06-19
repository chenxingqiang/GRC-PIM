# Feature extraction offload boundary (GRC-PIM)

## Summary

The **13-D per-symbol GEMV feature vector is not implemented in GRC RTL**.
It is computed on a **CPU / system digital front-end** (or dedicated small core)
and streamed into `GRC_PIM_TOP.FEAT_IN[103:0]` when `FEAT_VALID=1`.

On-chip GRC RTL owns:

| Block | Responsibility |
|-------|----------------|
| `GRC_MESSAGE_TRANS` | 3-bit ADC → 3×3-bit cycle LLR |
| B′ accumulator | Multi-cycle int8 sum per symbol |
| `DRIFT_DETECTOR_CODEWORD` | Cross-cycle bit drift (Q16) |
| `GEMV_MLP_CORE` | INT8 MLP + sigmoid gate on **preloaded** features |
| Dual gate | `USE_GEMV` from drift + mean gate |
| LLR emit | Anchor vs GEMV-corrected int8 LLR |

Off-chip / CPU owns:

| Step | Source | Notes |
|------|--------|-------|
| Multi-cycle readout buffer | PIM array / ADC | `(T, 2n)` bits |
| B′ soft product (float) | `build_gemv_features()` | Training anchor; optional cross-check |
| Symbol histogram | `bits_to_symbols()` over T cycles | 4-D normalized counts |
| Bit mean / disagree | Per-symbol over cycles | Stability features |
| Entropy | From B′ distribution | 1-D |
| Column index | `linspace(-1,1,n)` | Position prior |
| INT8 quantize | `hw.feat.float_feat_to_int8()` | Maps to `FEAT_IN` bus |

## FEAT_IN bus format

- Width: `13 × 8 = 104` bits, **little-endian by feature index**
- `FEAT_IN[8*i +: 8]` = signed INT8 feature `i` (two's complement)
- Loaded once per symbol on the last cycle row when `FEAT_VALID=1`

Quantization (see `pim_predecoder/hw/feat.py`):

```
feat_i8[i] = sat8( round( 127 * (feat_f32[i] + offset[i]) / divisor[i] ) )
```

Divisors match training ranges: LLR/20, histogram [0,1], bit mean centered, etc.

## ECC integration boundary

```
PIM readout (CPU) ──► FEAT_IN ──► GRC_PIM_TOP ──► OUT_LLR (int8×3)
                              │                        │
                              │                        ▼
ECC_BUFFER CHECK_TRANS ──► BUFFER_LLR ──► GRC_ECC_PRIOR_MUX ──► INPUT_LLR[2591:0]
                                                              └──► ECC_TOP (deps/NBLDPC_PIM)
```

- **Info symbols** `[0:255]`: GRC corrected priors via `LLR_I8_VEC_TO_ECC9`
- **Parity symbols** `[256:287]`: unchanged from `ECC_BUFFER` (`CHECK_TRANS`)
- Flat bus order matches `Serial_ECC_TOP_FSM.v`:
  `INPUT_LLR[s*9 + f*3 + b]` = symbol `s`, field `f`, bit `b`

## When to add fixed-point feature RTL

Add on-chip feature extraction only if:

1. FEAT bandwidth exceeds CPU→GRC streaming budget at target symbol rate, or
2. Tight coupling with ADC timing requires single-pass feature build.

A minimal RTL slice would implement: cycle counter, 4-way symbol histogram,
running bit mean/variance, and entropy LUT — **not** the full float `build_gemv_features`.
Until then, document latency budget: CPU must deliver `FEAT_IN` before `BATCH_LAST`
for each symbol (same cycle as final ADC row is acceptable with prefetch FIFO).

## Regression

| Test | Command |
|------|---------|
| Python golden k=1024 | `python3 -m pim_predecoder.cosim.regression --n-symbols 1024` |
| RTL spot-check (≤32 sym) | `python3 -m pim_predecoder.cosim.regression --rtl --n-symbols 32` |
| PRIOR_LLR cascade | `python3 -m pim_predecoder.cosim.cascade` |
| ECC_TOP decode smoke | `python3 -m pim_predecoder.cosim.decode` |
| Original 8×4 cosim | `python3 -m pim_predecoder.cosim.batch` |
