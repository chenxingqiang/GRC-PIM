# `pim_predecoder`

GF(3) NB-LDPC simulation with multi-cycle PIM noise, gated GEMV residual calibration, and Verilog cosimulation for the [GRC-PIM](../README.md) release.

Run every command from the **repository root**, the directory that contains `pim_predecoder/` and `figures/`. Installation is described in [`../README.md`](../README.md); the data index is [`../DATA_MANIFEST.md`](../DATA_MANIFEST.md).

## Pipeline

Simulation:

```
clean info → GF(3) encode → multi-cycle PIM noise → B′ soft-LLR
  → gated GEMV residual (optional) → NB-LDPC BP decode
```

Hardware integration:

```
PIM readout → FEAT_IN (off-chip features) → GRC_PIM_TOP → prior mux → ECC_TOP
```

The 13 per-symbol features are built off-chip and streamed on `FEAT_IN`; see [`hw/FEATURE_OFFLOAD.md`](hw/FEATURE_OFFLOAD.md).

## Modules

| Path | Role |
|------|------|
| `core/codec.py` | H/G matrices, encode, vectorised GF(3) BP decoder |
| `core/noise.py` | Single-cycle bit-flip channel |
| `core/noise3d.py` | Multi-cycle drift and burst channel (E1–E5); optional conductance and `aihwkit` channels |
| `core/gemv_calibrator.py` | B′ soft-LLR, GEMV calibrator, drift band and acceptance gate |
| `experiments/gemv.py` | Train and evaluate one experiment, write JSON |
| `experiments/ablation.py` | Small-budget multi-seed, threshold and BP-iteration sweep |
| `experiments/tsweep_summary.py` | Aggregate the read-depth sweep |
| `bench_inference.py` | ResNet-18 / CIFAR-10 accuracy translation |
| `result_data.py` | Aggregate the bundled result files; gate thresholds and synthesis inputs |
| `hw/llr.py`, `hw/feat.py` | LLR front-end and feature quantisation matching the RTL |
| `hw/export_weights.py` | Checkpoint → INT8 Verilog weight ROMs |
| `hw/simulate.py` | Python model of the GRC RTL |
| `hw/verify_cosim.py` | Hardware LLR against probabilistic LLR, with BP error rates |
| `cosim/` | Verilog cosimulation drivers (`batch`, `regression`, `cascade`, `decode`) |
| `rtl/` | Reference Verilog, testbenches, weight ROMs, synthesis script |
| `results/` | Result JSON and the scripts that produced them |
| `_cache/` | Trained checkpoints |
| `tests/` | pytest suite; RTL tests carry the `rtl` marker |

Matrices are read from [`../deps/NBLDPC_PIM/`](../deps/NBLDPC_PIM/) (`DAC_TEST/H_*.txt`, `G_*.txt`).

## Tests

```bash
python3 -m pim_predecoder.scripts.smoke
python3 -m pytest pim_predecoder/tests -m "not rtl"
python3 -m pytest pim_predecoder/tests -m rtl        # needs iverilog and vvp
```

| Make target | Action |
|-------------|--------|
| `make -C pim_predecoder install-dev` | Install both requirements files |
| `make -C pim_predecoder test-fast` | Python tests without Verilog |
| `make -C pim_predecoder test-rtl` | RTL cosimulation tests only |
| `make -C pim_predecoder test` | Both |

Make uses the active environment’s `python3`; set `PY` to override it.

## Experiments

### One E2 point

`--out_json` is required. Defaults are the 256-symbol scale; pass the code scale and training settings explicitly.

```bash
python3 -m pim_predecoder.experiments.gemv \
  --matrix_info 1024 --matrix_rate 16_17 \
  --train_steps 600 --train_batch 48 --seed 1 \
  --n_codewords 2000 --p_bits 5e-3 \
  --drift_score_threshold 0.011 --drift_score_max 0.038 --model_gate_threshold 0.6 \
  --out_json E2_1024_s1_check.json --label E2_check
```

Settings used for the reported results:

| Scale | `--matrix_info` / `--matrix_rate` | `--train_steps` / `--train_batch` | Drift band | Acceptance threshold |
|-------|-----------------------------------|-----------------------------------|------------|----------------------|
| 1024-symbol | `1024` / `16_17` | `600` / `48` | `0.011` to `0.038` | `0.6` |
| 256-symbol | `256` / `8_9` | `900` / `128` | `0.005` to `0.045` | `0.6` |

- The channel defaults to E2 (`--train_p_drift`, `--train_p_row_burst`, `--test_p_drift`, `--test_p_row_burst` all `5e-3`). Change them for E1 and E3–E5.
- A checkpoint in `_cache/` is reused when the scale, channel, training settings and seed all match. Otherwise a new model is trained and cached; `--force_train` always retrains.
- Omitting the three threshold options evaluates the calibrator ungated while still recording the drift score and acceptance score.
- `--model linear` selects the linear-head baseline and `--wbp_rho` adds cycle-discounted fusion.
- The reported sweeps use 100,000 codewords per seed, evaluated in blocks of 2,000 (`--eval_chunk 2000`). The exact settings of each bundled run are stored at the top level of its JSON, and the `run_*.sh` scripts under `results/` show complete invocations.

### Small sweep

```bash
python3 -m pim_predecoder.experiments.ablation \
  --matrix_info 1024 --matrix_rate 16_17 --out_json ablation_check.json
```

By default this is a three-seed, 800-codeword sweep. It is an exploration tool and does not regenerate the ten-seed, 100,000-codeword tables. Keep `--out_json`: without it the output goes under `results/ablation/` and can overwrite a bundled file.

### Threshold placement

These read the bundled JSON and need no channel simulation:

```bash
python3 pim_predecoder/results/revision_n100k/calib/place_thresholds.py
python3 pim_predecoder/results/area_sensitivity/compute_area_sensitivity.py
```

### ResNet-18 accuracy translation

```bash
python3 -m pim_predecoder.bench_inference --n_seeds 10
```

Uses `_cache/resnet18_cifar10_int8.pt` and reads the ten-seed 1024-symbol post-decoding BER from `results/revision_n100k/recal/`. CIFAR-10 is downloaded by torchvision into `data/` at the repository root on first use. The run rewrites `results/inference_accuracy.json` and `.csv`. `--force_train` retrains the network instead of loading the checkpoint. This is a two-stage translation with independent weight flips, not an end-to-end inference run through the channel and decoder.

## Hardware

Export weight ROMs (the bundled `rtl/Weights/` already holds the export of this checkpoint):

```bash
python3 -m pim_predecoder.hw.export_weights \
  --checkpoint pim_predecoder/_cache/gemv_k1024_n1088_r16_17_T4_d5e-03_b5e-03_s600_bs48_h24_seed1.pt \
  --out-dir pim_predecoder/rtl/Weights
```

Pipelined-datapath throughput testbench (needs `iverilog`); it reports the cold-start and steady-state cycles per symbol and checks bit-exactness against the sequential core:

```bash
iverilog -g2012 -DROWS_PER_CYCLE=15 -DNSYM=8 -o gemv_pipe_tb.o \
  pim_predecoder/rtl/INT8_GEMV_MLP.v pim_predecoder/rtl/TestBench/GEMV_PIPE_TB.v
vvp gemv_pipe_tb.o
```

Cosimulation drivers:

```bash
python3 -m pim_predecoder.cosim.regression --n-symbols 1024
python3 -m pim_predecoder.cosim.regression --rtl --n-symbols 32   # needs iverilog
python3 -m pim_predecoder.cosim.batch
python3 -m pim_predecoder.cosim.cascade
python3 -m pim_predecoder.cosim.decode
```

Area synthesis is `rtl/run_synth.sh`. It needs `yosys` and the Nangate45 liberty file (the script header explains how to obtain it and which options set the path, target cell area and utilisation) and writes `rtl/synth_area.json`. The result is a cell-area projection without place-and-route or timing closure.

Integration details: [`rtl/README.md`](rtl/README.md), [`hw/FEATURE_OFFLOAD.md`](hw/FEATURE_OFFLOAD.md).

## Dependencies

| File | Purpose |
|------|---------|
| `requirements.txt` | Runtime: NumPy, PyTorch, torchvision (pinned) |
| `requirements-dev.txt` | pytest and matplotlib only; install together with `requirements.txt` |
| `pyproject.toml`, `uv.lock` | Optional locked environment (Python 3.14) |

## See also

- Overview and figures: [`../README.md`](../README.md)
- Data index: [`../DATA_MANIFEST.md`](../DATA_MANIFEST.md)
- License: [`../LICENSE`](../LICENSE) (MIT)
