# GRC-PIM — `pim_predecoder`

GF(3) NB-LDPC simulation with multi-cycle PIM noise, gated GEMV residual calibration,
and hardware-aligned Verilog cosim for the [GRC-PIM](../README.md) public release.

> Run commands from the **repository root** (the directory that contains `pim_predecoder/` and `figures/`).  
> Paper figure PNGs: [`../figures/`](../figures/). Data index: [`../DATA_MANIFEST.md`](../DATA_MANIFEST.md).

## Pipeline

**Simulation (Python):**

```
clean info → GF(3) encode → multi-cycle PIM noise → B′ soft-LLR
  → (optional) gated GEMV residual → NB-LDPC BP decode
```

**Hardware integration:**

```
PIM readout → FEAT_IN (CPU) → GRC_PIM_TOP → PRIOR_MUX → ECC_TOP
```

The 13-D GEMV features are built **off-chip** and streamed on `FEAT_IN`; see [`hw/FEATURE_OFFLOAD.md`](hw/FEATURE_OFFLOAD.md).

## Layout

```
pim_predecoder/
├── paths.py              # PKG, REPO, RTL, CACHE, RESULTS
├── core/                 # codec, noise, gemv calibrator
├── hw/                   # LLR/feat models, weight export, RTL golden model
├── cosim/                # Verilog cosim drivers
├── experiments/          # gemv.py, ablation.py
├── scripts/              # smoke tests
├── rtl/                  # reference Verilog + Weights/*.hex
├── results/              # bundled paper JSON (84 files)
├── results/supplementary/
└── _cache/               # trained GEMV + affine checkpoints
```

| Path | Role |
|------|------|
| `core/codec.py` | H/G matrices, encode, vectorised GF(3) BP decoder |
| `core/noise.py` | Single-cycle PIM bit-flip channel |
| `core/noise3d.py` | Multi-cycle drift / burst noise (E1–E5) |
| `core/gemv_calibrator.py` | GEMV MLP + dual gate (drift + model confidence) |
| `experiments/gemv.py` | Train/evaluate one gated experiment → JSON |
| `experiments/ablation.py` | Multi-seed + threshold + bp_iters sweep |
| `hw/llr.py` | LLR front-end aligned with `LLR_TRANS.v` |
| `hw/feat.py` | 13-D feature quantisation for `FEAT_IN` bus |
| `hw/export_weights.py` | Checkpoint → INT8 Verilog weight ROMs |
| `hw/simulate.py` | Cycle-accurate Python model of GRC RTL |
| `cosim/batch.py` | Verilog cosim vs Python golden (small batch) |
| `cosim/regression.py` | k=1024 golden + optional RTL spot-check |
| `cosim/cascade.py` | GRC → `INPUT_LLR` bus cosim |
| `cosim/decode.py` | GRC → MUX → `ECC_TOP` decode smoke |
| `hw/verify_cosim.py` | HW LLR vs probabilistic LLR + BP BER |
| `scripts/smoke.py` | Quick BP decoder sanity check |
| `hw/FEATURE_OFFLOAD.md` | CPU vs on-chip feature boundary |
| `rtl/README.md` | RTL module map |

Matrices live in [`../deps/NBLDPC_PIM/`](../deps/NBLDPC_PIM/) (`DAC_TEST/H_*.txt`, `G_*.txt`).

## Quick start

```bash
pip install -r pim_predecoder/requirements-dev.txt

python3 -m pim_predecoder.scripts.smoke
make -C pim_predecoder test-fast
```

## Experiments

### Single E2 run (1024-symbol)

`--out_json` is **required**. Uses cached checkpoint in `_cache/` when available.

```bash
python3 -m pim_predecoder.experiments.gemv \
  --matrix_info 1024 --matrix_rate 16_17 \
  --train_p_drift 5e-3 --train_p_row_burst 5e-3 \
  --test_p_drift 5e-3 --test_p_row_burst 5e-3 \
  --train_steps 600 --n_codewords 2000 --seed 1 \
  --p_bits 5e-3 3e-3 1e-3 \
  --drift_score_threshold 0.025 \
  --out_json pim_predecoder/results/E2_gemv_repro.json \
  --label E2_repro
```

### Multi-seed ablation (paper Tables)

```bash
python3 -m pim_predecoder.experiments.ablation \
  --matrix_info 1024 --matrix_rate 16_17 --n_codewords 2000

python3 -m pim_predecoder.experiments.ablation \
  --matrix_info 256 --matrix_rate 8_9 --n_codewords 2000
```

Default output: `results/ablation/ablation_gemv_1024.json` (and 256 variant). Pass `--force_train` to retrain all checkpoints.

### Inspect bundled results (no re-run)

JSON rows use the key **`rows`**, not `results`. Example — gate cases E1–E5:

```bash
python3 - <<'PY'
import json
from pathlib import Path
for s in ["E1", "E2", "E3", "E4", "E5"]:
    j = json.loads(Path(f"pim_predecoder/results/paper1024_{s}_gemv_gated.json").read_text())
    r = next(x for x in j["rows"] if x["p_bit"] == 5e-3)
    print(f"{s}: drift={r['drift_score']:.4f} gate={r['use_gemv']} "
          f"BER {r['Bp_post_BER']:.4g} → {r['GEMV_post_BER']:.4g}")
PY
```

See [`../DATA_MANIFEST.md`](../DATA_MANIFEST.md) for the full figure/table → file mapping.

## Make targets

| Target | Action |
|--------|--------|
| `make -C pim_predecoder test-fast` | Python unit tests (no Verilog) |
| `make -C pim_predecoder test` | Python + RTL cosim tests |
| `make -C pim_predecoder test-rtl` | RTL cosim only |
| `make -C pim_predecoder install-dev` | `pip install -r requirements-dev.txt` |

## Hardware cosim

```bash
python3 -m pim_predecoder.hw.export_weights \
  --checkpoint pim_predecoder/_cache/gemv_k1024_n1088_r16_17_T4_d5e-03_b5e-03_s600_bs48_h24_seed1.pt \
  --out-dir pim_predecoder/rtl/Weights

python3 -m pim_predecoder.cosim.regression --n-symbols 1024
python3 -m pim_predecoder.cosim.regression --rtl --n-symbols 32   # needs iverilog
python3 -m pim_predecoder.cosim.cascade
python3 -m pim_predecoder.cosim.decode
python3 -m pim_predecoder.cosim.batch
python3 -m pim_predecoder.hw.simulate
```

Integration details: [`rtl/README.md`](rtl/README.md), [`hw/FEATURE_OFFLOAD.md`](hw/FEATURE_OFFLOAD.md).

## Dependencies

| File | Purpose |
|------|---------|
| `requirements.txt` | `numpy`, `torch` (runtime) |
| `requirements-dev.txt` | + `pytest` (tests) |
| `pyproject.toml` / `uv.lock` | Optional locked env (Python 3.14) |

## See also

- Project overview & Fig. 1–5 walkthrough: [`../README.md`](../README.md)
- License: [`../LICENSE`](../LICENSE) (MIT)
