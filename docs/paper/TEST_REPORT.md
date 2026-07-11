# Test Report — CiM-ResLDPC (Verification / v1.1)

**Date:** 2026-07-10  
**Branch:** `cursor/cim-resldpc-edge-rag-a594`  
**Reflection:** `docs/design/SCIENTIFIC_REFLECTION.md`

## What changed after fact-check

| Issue | Fix |
|-------|-----|
| Clean-embedding oracle blend | Removed; recover from B′/BP only |
| No real LDPC on embeddings | `ldpc_embed.protect_and_recover` |
| BM25 confounding “robustness” | Dense-only scientific table + separate hybrid |
| Centroid ternary geometry collapse | Default **base-3** multi-symbol packing |
| Symmetric query+doc noise | Default **asymmetric** (digital query, CiM docs) |
| Unstable `hash()` | `blake2b` stable token ids |

## Tests

- `cim_resldpc/tests`: **16 passed**
- Includes `test_no_oracle_blend_in_pipeline_source`, `test_geometric_probe_ldpc_helps`

## Honest eval highlights (`sigir_eval.json` v1.1)

**Dense-only SyntheticIR @ p=0.05 (asymmetric):**

| Method | nDCG@10 |
|--------|---------|
| Clean dense | 0.630 |
| CiM noisy (no LDPC) | 0.134 |
| CiM LDPC B′ | **0.308** (+0.175 vs noisy) |
| CiM LDPC soft | **0.323** |
| CiM LDPC BP | 0.213 |

**Geometric probe (controlled clusters):** at p=0.03, noisy 0.905 → B′ 0.952; SER 0.086→0.038.

**Note:** BP hard decisions can underperform B′ on *ranking* even when SER improves — report both; do not claim BP always wins IR metrics.

## Verdict

Design direction is **viable and now empirically grounded** under an honest protocol. Ready for next iteration: stronger encoders / public IR benchmarks / optional GRC GEMV hook.
