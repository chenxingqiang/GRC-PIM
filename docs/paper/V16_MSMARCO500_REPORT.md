# v1.6 Report — MSMARCO 500q + Intro/Related Work

## Scale

| Corpus | Queries | Docs |
|--------|---------|------|
| MSMARCO HF | **500** | **4201** |
| NQ-open hard | 500 | 2624 (cache built; IR focus this round = MSMARCO) |

Teacher: **BGE-small-en-v1.5** → project to `cim_dim` → base-3 GF(3) → LDPC.

## Main IR table (MSMARCO 500q, `nor_typical`, dim=128, seed=42)

| Method | nDCG@10 |
|--------|---------|
| BM25 | 0.536 |
| Dense clean | 0.648 |
| CiM noisy | 0.423 |
| LDPC B′ | 0.482 |
| **LDPC BP** | **0.539 (+0.116 vs noisy)** |
| **Hybrid BM25+BP** | **0.565** (> BM25) |

## Multi-seed notes (important)

**dim=128, nor_typical:**

| Seed | BP gain |
|------|---------|
| 0 | **−0.194** (pathological; BP underperforms noisy) |
| 1 | +0.130 |
| 2 | +0.066 |
| 42 | +0.116 |

Mean over {1,2,42}: **+0.104**; including seed 0 collapses the mean.  
**Interpretation:** at 128-D, a bad projection×noise draw can make BP hurt ranking. Papers must report multi-seed **and** flag failures.

**dim=256, nor_typical (seeds 0/1/2/42):** BP gain **0.157 ± 0.013** — much more stable. Prefer **cim_dim=256** when storage budget allows.

## NOR (dim=128, seed=0 from sweep)

| NOR | Noisy | BP | Gain |
|-----|-------|----|------|
| fresh | 0.531 | 0.530 | ≈0 |
| typical | 0.456 | 0.262 | −0.194 (same pathological seed) |
| aged | 0.194 | 0.060 | −0.134 |

Use seed=42 typical as the **primary** 500q claim; use dim=256 multi-seed as the **stability** claim.

## Writing landed

- `docs/paper/INTRODUCTION.md`
- `docs/paper/RELATED_WORK.md` (rewritten)
- This report + updated Experiments pointers

## Artifact

`cim_resldpc/results/sigir_eval_v16_msmarco500.json`
