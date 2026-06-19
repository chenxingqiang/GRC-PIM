# GRC-PIM Data Manifest

Index of **bundled artifacts** in this public release: figure PNGs, experiment JSON, trained checkpoints, and RTL weight ROMs.

Companion doc: [`README.md`](README.md) (Fig. 1–5 walkthrough, 0→1 reproduction ladder, use directions).

This bundle ships **simulation code + JSON/checkpoints + figure PNGs**. It does **not** ship manuscript LaTeX, plotting scripts, or ResNet inference drivers.

---

## Repository layout (data-bearing paths)

```
GRC-PIM/
├── figures/fig{1..5}.png          # paper figures (README embeds these)
├── pim_predecoder/
│   ├── results/                   # 84 experiment JSON files
│   ├── results/supplementary/     # threshold / T-sweep / feature dropout
│   ├── results/ablation/          # aggregated ablation outputs
│   ├── results/baselines/         # multi-seed E2 + Maj/Affine baselines
│   ├── _cache/*.pt                # trained GEMV + affine checkpoints
│   └── rtl/Weights/*.hex          # exported INT8 weight ROMs (optional path)
└── deps/NBLDPC_PIM/DAC_TEST/      # H/G parity-check matrices (required to re-run)
```

Refresh bundle from the development repo: `scripts/prepare_grc_pim_bundle.sh` (outside this tree).

**License:** [MIT](LICENSE) — Copyright (c) 2026 Hangzhou Zhicun Technology Co., Ltd. and Xingqiang Chen

---

## Figure PNGs (`figures/`)

| File | Manuscript | Source |
|------|------------|--------|
| `fig1.png` | Fig. 1 — system architecture | `research/fig1.png` (copied at bundle time) |
| `fig2.png` | Fig. 2 — BER/SER vs \(p_\mathrm{bit}\) | 300 dpi PNG from dev `plot_figures.py` |
| `fig3.png` | Fig. 3 — gate behavior E1–E5 | same |
| `fig4.png` | Fig. 4 — hardware cost | same (`manuscript_data.HARDWARE` constants) |
| `fig5.png` | Fig. 5 — ResNet-18 Top-1 | same (`inference_accuracy.json`) |

---

## JSON file format

All experiment outputs share a common shape:

| Field | Meaning |
|-------|---------|
| `rows[]` | One dict per \(p_\mathrm{bit}\) operating point (**not** `results`) |
| `Bp_post_BER` | B′ baseline bit error rate (or SER at 256-scale files) |
| `GEMV_post_BER` | Gated GRC-PIM post-decode rate |
| `use_gemv` | Whether GEMV path activated (0/1) |
| `drift_score` | Global drift score \(d_\mathrm{global}\) |
| `resolution_floor` | Statistical BER floor for batch size |

Inspect a row:

```bash
python3 - <<'PY'
import json
from pathlib import Path
j = json.loads(Path("pim_predecoder/results/paper1024_E2_gemv_gated.json").read_text())
r = next(x for x in j["rows"] if x["p_bit"] == 5e-3)
print({k: r[k] for k in ("p_bit", "drift_score", "use_gemv", "Bp_post_BER", "GEMV_post_BER")})
PY
```

---

## Published figures → JSON

| Figure | Primary JSON | Headline numbers @ \(p_\mathrm{bit}=5\times10^{-3}\) |
|--------|--------------|--------------------------------------------------------|
| **Fig. 2a** (1024 BER) | `baselines/E2_baselines_1024_s{1,2,3}_n10000.json`, `*_n2000*.json` | B′ \(4.18\times10^{-3}\) → GEMV \(2.00\times10^{-3}\), gain **2.10×** (3-seed mean) |
| **Fig. 2b** (256 SER) | `E2_gemv_gated_s{1,2,3}.json` | SER gain **2.49×** / **2.48×** @ \(5\times10^{-3}\) / \(3\times10^{-3}\) |
| **Fig. 3** | `paper1024_E{1,2,3,4,5}_gemv_gated.json` | E2 gate on, 2.08× BER; E5 GRC = B′ |
| **Fig. 4** | *(no JSON)* — analytical hardware table | area <0.05 mm², power ~41 μW, latency <3 μs |
| **Fig. 5** | `inference_accuracy.json` | raw 71.8% → B′+dec 79.9% → GRC **88.0%** Top-1 |

Fig. 4 numbers are **not** stored as JSON; they match the paper hardware table and the dev-only `manuscript_data.HARDWARE` constants used at figure-generation time.

---

## Main manuscript tables → JSON

| Table | Label | Primary JSON |
|-------|-------|--------------|
| Ablation 1024 | `tab:ablation_1024` | `baselines/E2_baselines_1024_s*_n10000.json`, `*_n2000*.json` |
| Ablation 256 | `tab:ablation_256` | `E2_gemv_gated_s{1,2,3}.json` |
| Baseline comparison | `tab:baseline_comparison` | `baselines/E2_baselines_1024_s*.json`, `baselines/E2_baselines_256_s*.json` |
| Gate cases | `tab:gate_cases` | `paper1024_E{1..5}_gemv_gated.json` |
| Component ablation | `tab:component_ablation` | `ablation/ablation_gemv_256_components.json` |
| E5 without gates | `tab:e5_no_gates` | `E5_mlp_only_no_gates.json` |
| Hardware | `tab:hardware` | *(constants only; see Fig. 4)* |

Aggregated ablation summaries: `ablation/ablation_gemv_1024.json`, `ablation/ablation_gemv_256.json`.

---

## Supplementary tables → JSON

| Section | Content | JSON |
|---------|---------|------|
| S1 | Per-seed 1024 breakdown | `baselines/E2_baselines_1024_s*.json` |
| S3 | Gate threshold sensitivity | `supplementary/threshold_sensitivity_1024.json` |
| S4 | Read-depth \(T\)-sweep | `supplementary/tsweep/T{T}_s{S}.json` (\(T=2..6\), seeds 1–3) |
| S4 summary | Aggregated T-sweep | `supplementary/tsweep_summary.json` |
| Feature dropout | Feature-group ablation | `supplementary/feature_dropout_1024.json` |

---

## Gate scenarios E1–E5 (`paper1024_*.json`)

1024-symbol, rate 16/17, 800 codewords, seed 1, \(p_\mathrm{bit}=5\times10^{-3}\) unless noted.

| Scenario | Files | Train drift / burst | Test drift / burst | Expected gate |
|----------|-------|---------------------|--------------------|---------------|
| **E1** weak / weak | `paper1024_E1_gemv_gated.json` | 5e-4 / 1e-3 | 5e-4 / 1e-3 | off |
| **E2** strong / strong | `paper1024_E2_gemv_gated.json` | 5e-3 / 5e-3 | 5e-3 / 5e-3 | **on** |
| **E3** weak → i.i.d. | `paper1024_E3_gemv_gated.json` | 5e-4 / 1e-3 | 0 / 0 | off |
| **E4** i.i.d. / i.i.d. | `paper1024_E4_gemv_gated.json` | 0 / 0 | 0 / 0 | off |
| **E5** i.i.d. → strong | `paper1024_E5_gemv_gated.json` | 0 / 0 | 5e-3 / 5e-3 | off (confidence rejects) |

Drift threshold in these runs: \(\tau_d=0.030\); model gate: \(\tau_g=0.6\).  
1024-scale E2 ablation table uses recalibrated \(\tau_d=0.025\) (see `baselines/*_t025.json`).

Inspect all five:

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

---

## Drift calibration

| File | Use |
|------|-----|
| `results/drift_calibration.json` | Threshold calibration traces (paper Sec. drift detector) |

---

## Checkpoints (`pim_predecoder/_cache/`)

~49 PyTorch `.pt` files, including:

| Pattern | Content |
|---------|---------|
| `gemv_k1024_*_T4_*_seed{1,2,3}.pt` | Paper-default 1024-symbol GEMV (T=4) |
| `affine_k1024_*_T4_*_seed{1,2,3}.pt` | Per-column affine baseline models |
| `gemv_k256_*_T4_*` | 256-symbol scale |
| `*_T{2,3,5,6}_*` | Supplementary read-depth sweep |

Naming: `{gemv\|affine}_k{K}_n{N}_r{rate}_T{n_cycles}_d{p_drift}_b{p_burst}_s{steps}_..._seed{S}.pt`

Export to Verilog ROMs:

```bash
python3 -m pim_predecoder.hw.export_weights \
  --checkpoint pim_predecoder/_cache/gemv_k1024_n1088_r16_17_T4_d5e-03_b5e-03_s600_bs48_h24_seed1.pt \
  --out-dir pim_predecoder/rtl/Weights
```

Pre-exported hex files may already exist under `rtl/Weights/`.

---

## Hardware / integration docs

| Doc | Audience |
|-----|----------|
| [`pim_predecoder/rtl/README.md`](pim_predecoder/rtl/README.md) | RTL module map, cosim entry points |
| [`pim_predecoder/hw/FEATURE_OFFLOAD.md`](pim_predecoder/hw/FEATURE_OFFLOAD.md) | CPU vs on-chip boundary for 13-D `FEAT_IN` features |

---

## Minimal subset (figures + main tables only)

```
figures/fig{1..5}.png
pim_predecoder/results/baselines/E2_baselines_1024_s{1,2,3}_n10000.json
pim_predecoder/results/baselines/E2_baselines_1024_s{1,2,3}_n2000*.json
pim_predecoder/results/E2_gemv_gated_s{1,2,3}.json
pim_predecoder/results/paper1024_E{1..5}_gemv_gated.json
pim_predecoder/results/inference_accuracy.json
pim_predecoder/results/ablation/ablation_gemv_256_components.json
pim_predecoder/results/E5_mlp_only_no_gates.json
pim_predecoder/results/supplementary/
```

Full ECC **re-runs** additionally need `deps/NBLDPC_PIM/DAC_TEST/` matrices and `_cache/*.pt` (or `--force_train`).

---

## Not bundled

| Item | Where |
|------|--------|
| Manuscript LaTeX / PDF | Separate manuscript repo |
| `plot_figures.py`, `manuscript_data.py` | Dev repo only |
| ResNet-18 inference driver | External; only `inference_accuracy.json` here |
| `audit_data_figures.py` | Dev `research/` tree |
