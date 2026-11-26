# GRC-PIM data manifest

Index of the artifacts bundled with this release: figure PNGs, result JSON, the scripts that produced them, trained checkpoints and RTL weight ROMs. For an overview and commands, see [`README.md`](README.md).

Paths below are relative to `pim_predecoder/results/` unless they start with another directory. `{…}` lists alternatives and `s{N}` is the seed.

## Figures (`figures/`)

File numbers follow the manuscript.

| File | Content | Underlying data |
|------|---------|-----------------|
| `fig1.png` | System architecture | Drawing, no data |
| `fig2.png` | BER/SER under E2 strong drift, both code scales | `revision_n100k/recal/` |
| `fig3.png` | Held-out calibration of cycle-discounted fusion | `revision_n100k/wbp/calib/` |
| `fig4.png` | Gate behaviour across E1–E5 (single-seed diagnostic) | `paper1024_E{1..5}_gemv_gated.json` |
| `fig5.png` | Hardware cost at two design points | `pim_predecoder/rtl/synth_area.json`, RTL cycle counts |
| `fig6.png` | ResNet-18 CIFAR-10 accuracy translation | `inference_accuracy.json` |

The scripts that draw the manuscript figures are not part of this release.

## Result file format

Experiment JSON files written by `experiments/gemv.py` share one shape: run settings at the top level (channel, seed, code scale, codeword budget, thresholds) and one entry per p_bit operating point under `rows`.

| Field in `rows[]` | Meaning |
|-------------------|---------|
| `p_bit` | Independent bit-flip probability of the operating point |
| `Bp_post`, `GEMV_post` | Post-decoding **symbol** error rate of B′ and of GRC-PIM as gated |
| `Bp_post_BER`, `GEMV_post_BER` | The same as **bit** error rates |
| `use_gemv` | Whether the gates admitted the correction |
| `drift_score`, `drift_score_sd` | Temporal-disagreement score and its across-block s.d. |
| `GEMV_gate` | Mean acceptance score ḡ |
| `GEMV_ungated_post`, `GEMV_ungated_post_BER` | Calibrator with both gates forced open; recorded by the runs for seeds 4–10 |
| `Maj_post*`, `Affine_post*` | Majority-vote and per-column LLR-scaling baselines |
| `resolution_floor` | Symbol-level counting floor; divide by two for BER (two bits per symbol) |

Older and smaller runs carry a subset of these fields. The manuscript reports BER at the 1024-symbol scale and SER at the 256-symbol scale. For seeds 1–3 an open gate means the gated value is also the ungated one; the single closed point among them (1024-symbol, seed 3, p_bit = 10⁻⁴) has its ungated value in `revision_n100k/recal/E2_1024_s3_n100k_td011_ungated_full.json`.

`pim_predecoder/result_data.py` aggregates these files across seeds; the README shows a short example.

## Headline E2 results (10 seeds × 100,000 codewords)

| Content | Files |
|---------|-------|
| 1024-symbol sweep, deployed thresholds | `revision_n100k/recal/E2_1024_s{1..10}_n100k_td011.json` |
| 256-symbol sweep, deployed thresholds | `revision_n100k/recal/E2_256_s{1..10}_n100k_td005.json` |
| Majority-vote and per-column scaling baselines | Columns of the two file sets above |
| Cycle-discounted fusion, ρ = 0.9, test sweep | `revision_n100k/wbp/E2_{256,1024}_s{N}_n100k_rho09.json`; see note |
| Linear head on the same 13 features, ungated | `revision_n100k/linear/E2_{256,1024}_s{1..10}_n100k_linear_ungated.json` |
| Linear head, gated diagnostic (seeds 1–3) | `revision_n100k/linear/E2_{256,1024}_s{1..3}_n100k_linear.json` |
| ResNet-18 accuracy translation | `inference_accuracy.json`, `inference_accuracy.csv` |

Note: dedicated ρ = 0.9 files exist for seeds 1–3 at the 1024-symbol scale and seeds 1–5 at the 256-symbol scale. For the other seeds there is no dedicated file: the discounted path is recorded in `revision_n100k/linear/E2_{scale}_s{N}_n100k_linear_ungated.json` as `Wbp_*` columns, with `wbp_rho` at the top level. `wbp_formal()` in `pim_predecoder/result_data.py` resolves the file for each seed.

## Threshold calibration and gate analyses

| Content | Files |
|---------|-------|
| Held-out validation sweep (evaluation seeds 101–103; matched, weak and i.i.d. channels) | `revision_n100k/calib/val_{256,1024}_{matched,weak,iid}_s{1..3}.json`, `run_calib.sh` |
| Drift-band placement and exact τ_d sensitivity | `revision_n100k/calib/place_thresholds.py` |
| τ_g sensitivity | `revision_n100k/calib/tau_g_sensitivity.py`, `tau_g_sensitivity.json` |
| Held-out cycle-discount grid (ρ ∈ {0.9, 0.75, 0.5, 0.25}) | `revision_n100k/wbp/calib/wbp_calib_{256,1024}_s{1..3}_rho*.json` |
| Drift-score spread at per-codeword granularity | `revision_n100k/ood/granularity/` |
| E1–E5 gate diagnostic (seed 1, 800 codewords, drift threshold 0.030) | `paper1024_E{1..5}_gemv_gated.json` |
| E5 with both gates forced open (3 seeds, both scales) | `revision_n100k/nogate/nogate_{256,1024}_s{1..3}.json`, `run_nogate.sh` |
| Drift-rate sweep at the 1024-symbol scale (3 seeds) | `revision_n100k/drift_sweep/`, `run_drift_sweep.sh` |
| Model seed × data-stream seed cross evaluation | `revision_n100k/crossed/`, `run_crossed.sh` |
| Component-level gate ablation | `ablation/` |

## Robustness checks

| Content | Budget | Files |
|---------|--------|-------|
| Decoder strength: 3, 8 and 15 BP iterations | 10,000 codewords per seed, seeds 1–3 | `decoder_strength/`, `run_decoder_strength.sh` |
| Device-calibrated channel cross-check (needs `aihwkit`) | 20,000 codewords per seed, seeds 1–3 | `revision_n100k/ood/aihwkit/`, `run_aihwkit.sh` |
| Read-depth sweep, T = 2…6 | 2,000 codewords per seed, seeds 1–3 | `supplementary/tsweep/T{T}_s{S}.json`, `supplementary/tsweep_summary.json` |
| Feature-group dropout | 2,000 codewords, seed 1 | `supplementary/feature_dropout_1024.json` |
| Area-projection sensitivity | From synthesis output | `area_sensitivity/compute_area_sensitivity.py` |

The device-calibrated cross-check replaces the noise physics with a third-party model. It is a model-to-model transfer and does not substitute for measured device data.

## Earlier and exploratory runs

The remaining files under `pim_predecoder/results/` are kept for provenance and are **not** behind the numbers summarised in the README. They use smaller codeword budgets, three seeds or earlier threshold settings:

- top-level `E1*.json`, `E2_gemv_gated*.json`, `E5_*.json` and `paper1024_E2_s{2,3}_gemv_gated.json`;
- `baselines/`;
- `revision_n100k/E2_{256,1024}_s{1..3}_n100k.json` (100,000 codewords before the drift band was recalibrated);
- `revision_n100k/ood/ood_1024_C*` and `id_1024_bandpass_s*` (a conductance-channel gate study);
- `supplementary/threshold_sensitivity_1024.json` (a 2,000-codeword threshold sweep).

Do not mix their values with the ten-seed results.

## Hardware data

| Content | Files |
|---------|-------|
| Synthesis area report and the script that produces it | `pim_predecoder/rtl/synth_area.json`, `pim_predecoder/rtl/run_synth.sh` |
| Throughput testbenches (sequential and pipelined datapath) | `pim_predecoder/rtl/TestBench/GEMV_LANES_TB.v`, `GEMV_PIPE_TB.v` |
| Exported INT8 weight ROMs | `pim_predecoder/rtl/Weights/*.hex`, `quant_meta.json` |
| Cosimulation stimulus and golden vectors | `pim_predecoder/rtl/cosim*/` |

The bundled ROMs are exported from the 1024-symbol seed-1 E2 checkpoint. Area figures are projections from synthesis cell area; no place-and-route, timing closure or switching-activity power analysis was performed.

## Checkpoints (`pim_predecoder/_cache/`)

| Pattern | Content |
|---------|---------|
| `gemv_k1024_*_T4_d5e-03_b5e-03_s600_bs48_h24_seed{1..10}.pt` | 1024-symbol GEMV calibrators behind the headline sweep |
| `gemv_k256_*_T4_d5e-03_b5e-03_s900_bs128_h24_seed{1..10}.pt` | 256-symbol GEMV calibrators behind the headline sweep |
| `affine_*`, `linear_*` | Per-column scaling and linear-head baselines |
| `*_d0e+00_b0e+00_*`, `*_d5e-04_b1e-03_*` | Models trained on the i.i.d. and weak-drift channels (E1, E3–E5, validation) |
| `*_T{2,3,5,6}_*` | Read-depth sweep |
| `resnet18_cifar10_int8.pt` | ResNet-18 used by `bench_inference.py` |

Naming: `{model}_k{K}_n{N}_r{rate}_T{cycles}_d{p_drift}_b{p_row_burst}_s{steps}_bs{batch}[_h{hidden}]_seed{S}.pt`. An experiment reuses a checkpoint only when all of these settings match; otherwise it trains and caches a new one.

## Required to re-run

- `deps/NBLDPC_PIM/DAC_TEST/`: the GF(3) parity-check and generator matrices, vendored.
- `pim_predecoder/_cache/`: checkpoints, or pass `--force_train` to retrain.
- CIFAR-10 for `bench_inference.py`: downloaded by torchvision on first use; not bundled.

## Not bundled

Manuscript sources and PDFs, manuscript plotting scripts, patent materials, peer-review correspondence, run logs, and the CIFAR-10 dataset.

## License

[MIT](LICENSE). `deps/NBLDPC_PIM/` follows its upstream terms.
