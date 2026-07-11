# v1.7 Report — MSMARCO 1000q + Conclusion + ACM LaTeX

## Scale

| Corpus | Queries | Docs |
|--------|---------|------|
| MSMARCO HF | **1000** | **8503** |

Teacher: **BGE-small-en-v1.5** → project to `cim_dim` → base-3 GF(3) → LDPC.  
Artifact: `cim_resldpc/results/sigir_eval_v17_msmarco1000.json`

## Main IR table (MSMARCO 1000q, `nor_typical`, dim=128, seed=42)

| Method | nDCG@10 |
|--------|---------|
| BM25 | 0.520 |
| Dense clean | 0.641 |
| CiM noisy | 0.283 |
| LDPC B′ | 0.381 |
| **LDPC BP** | **0.496 (+0.214 vs noisy)** |
| **Hybrid BM25+BP** | **0.549** (> BM25) |

## Multi-seed (dim=128, `nor_typical`, seeds 0/1/2/42)

| Seed | Noisy | BP | BP gain | Hybrid |
|------|-------|-----|---------|--------|
| 0 | 0.357 | 0.501 | +0.143 | 0.551 |
| 1 | 0.304 | 0.514 | +0.210 | 0.555 |
| 2 | 0.365 | 0.525 | +0.160 | 0.559 |
| 42 | 0.283 | 0.496 | +0.214 | 0.549 |

**Summary:** BP gain **0.182 ± 0.031**; BP nDCG **0.509 ± 0.011**.  
Unlike 500q (where seed=0 was pathological), all four seeds are positive at 1000q / dim=128.

## Stability (dim=256, `nor_typical`, seeds 0/1/2/42)

| Seed | Noisy | BP | BP gain |
|------|-------|-----|---------|
| 0 | 0.169 | 0.397 | +0.229 |
| 1 | 0.196 | 0.347 | +0.151 |
| 2 | 0.210 | 0.379 | +0.170 |
| 42 | 0.203 | 0.376 | +0.173 |

**dim=256 BP gain: 0.180 ± 0.029.** Absolute BP nDCG is lower than dim=128 (more noise exposure on wider vectors under the same per-symbol channel), but recovery headroom vs noisy is large and stable.

## NOR (dim=128, seed=0)

| NOR | Noisy | BP | Gain |
|-----|-------|----|------|
| fresh | 0.461 | 0.501 | +0.040 |
| typical | 0.357 | 0.501 | +0.143 |

## Writing landed

- `docs/paper/CONCLUSION.md` (updated with 1000q)
- `docs/paper/EXPERIMENTS.md` / `INTRODUCTION.md` / `PAPER_OUTLINE.md`
- `docs/paper/latex/main.tex` ACM skeleton

## Claim discipline

- Primary system claim: **1000q dim=128 seed=42** hybrid 0.549 > BM25 0.520; BP +0.214 vs noisy.  
- Stability claim: multi-seed dim=128 **0.182±0.031** or dim=256 **0.180±0.029**.  
- Still a stratified validation subset, not full MSMARCO Dev leaderboard.
