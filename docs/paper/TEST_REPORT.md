# Test Report — CiM-ResLDPC (Testing Phase)

**Date:** 2026-07-10  
**Branch:** `cursor/cim-resldpc-edge-rag-a594`  
**Design:** `docs/design/CIM_RESLDPC_DESIGN.md`  
**Phase:** Testing → Verification (no design change required)

## Commands

```bash
export PATH="$HOME/.local/bin:$PATH"
PYTHONPATH=/workspace python3 -m pytest cim_resldpc/tests -v --tb=short
PYTHONPATH=/workspace python3 -m cim_resldpc.experiments.run_sigir_eval \
  --out_json cim_resldpc/results/sigir_eval.json --train_steps 30 --n_topics 6
```

## Unit / integration tests (T1–T10)

| Result | Detail |
|--------|--------|
| **12 passed** | 1.60s, Python 3.12.3 |

All design-plan cases passed: GF(3) roundtrip, noise bridge, residual weight monotonicity, TONEL finite losses, hybrid ranking, decode prune, local-only privacy, metrics ranges, eval JSON schema, import smoke.

## Offline SIGIR eval (`cim_resldpc/results/sigir_eval.json`)

| Method | Split | nDCG@10 | MAP | Notes |
|--------|-------|---------|-----|-------|
| BM25 | synthetic | 0.936 | — | strong lexical overlap on toy topics |
| Naive dense (clean) | synthetic | 0.785 | — | float, no CiM noise |
| CiM noisy (no residual) | synthetic | 0.365 | — | **53.5%** relative nDCG drop vs clean |
| **CiM-ResLDPC** | synthetic | **0.933** | **0.861** | hybrid + residual; mean latency **0.52 ms** |
| **CiM-ResLDPC** | semi corpus | **0.862** | **0.797** | Recall@10 = 1.0; latency **0.45 ms** |

**Robustness takeaway:** Under the default PIM noise profile, uncalibrated CiM dense retrieval collapses; CiM-ResLDPC recovers ranking via residual-calibrated dense features + BM25 fusion, matching the paper narrative (hardware noise destroys dense IR; residual soft-info + hybrid restores it).

## Verification

No design modification required. Implementation matches `CIM_RESLDPC_DESIGN.md` APIs and test plan. Optional follow-ups (out of v1 scope): MSMARCO/NQ scale-up, external Contriever/BGE/RoCR checkpoints, full GRC GEMV checkpoint wiring per document instead of LLR-proxy blend.

## Issues

None blocking.
