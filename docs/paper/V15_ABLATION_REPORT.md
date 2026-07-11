# v1.5 Ablation Report

## Seed stability (BGE, MSMARCO 200q, nor_typical, cim_dim=128)

| Seed | Noisy | BP | BP gain |
|------|-------|----|---------|
| 0 | 0.486 | 0.540 | +0.055 |
| 1 | — | — | +0.092 |
| 2 | — | — | +0.078 |
| **Mean±Std** | **0.492±0.017** | **0.567±0.022** | **+0.075±0.015** |

## Paper main (seed=42, dim=128, nor_typical)

noisy **0.445** → BP **0.531** (+0.086); hybrid **0.572** > BM25 **0.528**.

## NOR / dim

See `EXPERIMENTS.md` §5.5–5.6. Typical helps; aged/stress can hurt. dim=256 increases BP gain headroom.

## Artifacts

- `cim_resldpc/results/sigir_eval_v15_ablation.json`
- `cim_resldpc/data/cache/bge_msmarco_n200_emb.npz` (local encode cache; optional in git if large)
