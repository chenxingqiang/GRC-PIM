# v1.8 Report — MSMARCO 2000q focused paper-main

## Scale

| Corpus | Queries | Docs |
|--------|---------|------|
| MSMARCO HF | **2000** | **16894** |

Teacher: **BGE-small-en-v1.5** → project to `cim_dim` → base-3 GF(3) → LDPC.  
Runner: `cim_resldpc/experiments/run_paper_main_v18.py`  
Artifact: `cim_resldpc/results/sigir_eval_v18_msmarco2000.json`

## Main IR table (MSMARCO 2000q, `nor_typical`, dim=128, seed=42)

| Method | nDCG@10 |
|--------|---------|
| BM25 | 0.531 |
| Dense clean | 0.636 |
| CiM noisy | 0.254 |
| **LDPC BP** | **0.475 (+0.221 vs noisy)** |
| **Hybrid BM25+BP** | **0.557** (> BM25) |

## Multi-seed (dim=128, `nor_typical`, seeds 0/1/2/42)

| Seed | Noisy | BP | BP gain | Hybrid |
|------|-------|-----|---------|--------|
| 0 | 0.375 | 0.482 | +0.106 | 0.564 |
| 1 | 0.324 | 0.484 | +0.160 | 0.548 |
| 2 | 0.386 | 0.479 | +0.093 | 0.559 |
| 42 | 0.254 | 0.475 | +0.221 | 0.557 |

**Summary:** BP gain **0.145 ± 0.050**; BP nDCG **0.480 ± 0.003** (very stable absolute BP).

## Spot-checks

| Setting | Noisy | BP | Gain |
|---------|-------|-----|------|
| dim=256, seed=42, typical | 0.137 | 0.345 | **+0.208** |
| nor_fresh, dim=128, seed=42 | 0.465 | 0.475 | +0.010 |
| nor_aged, dim=128, seed=42 | 0.055 | 0.061 | +0.006 |

Aged corner: BP barely helps ranking—report operating point.

## Scale ladder (dim=128, seed=42, `nor_typical`)

| Scale | Noisy → BP | Hybrid vs BM25 |
|-------|------------|----------------|
| 200q | 0.445 → 0.531 (+0.086) | 0.572 > 0.528 |
| 500q | 0.423 → 0.539 (+0.116) | 0.565 > 0.536 |
| 1000q | 0.283 → 0.496 (+0.214) | 0.549 > 0.520 |
| **2000q** | **0.254 → 0.475 (+0.221)** | **0.557 > 0.531** |

Absolute BP nDCG settles near ~0.48–0.50 while unprotected noise worsens with corpus size; hybrid remains above BM25.

## Writing / LaTeX

- Expanded `docs/paper/latex/main.tex` + `references.bib`
- Updated Intro / Experiments / Conclusion / Outline with 2000q
