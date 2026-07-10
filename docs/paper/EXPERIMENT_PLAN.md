# Experiment Plan (SIGIR-AP evaluation)

## Datasets

| Split | Description | Status in repo |
|-------|-------------|----------------|
| SyntheticIR | Topic-structured docs/queries | `cim_resldpc/datasets.py` |
| SemiCorpusMini | Semiconductor / CiM handbook snippets | bundled |
| Noise sweep | Vary `p_indep`, `p_drift`, `p_row_burst` (E1–E5-like) | via `NoiseProfile` |
| MSMARCO / NQ | Standard IR (optional scale-up) | not bundled (download later) |

## Baselines

| ID | Meaning |
|----|---------|
| `bm25` | Sparse only |
| `naive_dense` | Float dense, no CiM noise |
| `cim_noisy` | Quantize + PIM noise, no residual calibration |
| `cim_resldpc` | Full proposed hybrid + residual confidence |

*(Paper can add Contriever/BGE/RoCR/CiMRAG when external checkpoints are available.)*

## Metrics

- **IR (scientific core):** dense-only nDCG@10, MAP, Recall@k under CiM doc-store noise  
- **Robustness:** nDCG drop vs clean dense **within dense-only family**; gain vs unprotected noisy  
- **Geometric probe:** controlled cluster embeddings (validates LDPC ranking when geometry exists)  
- **System (separate):** hybrid BM25+dense — not used as robustness denominator  
- **Hardware proxy:** latency_ms, decode skip rate  
- **Privacy:** `local_only` contract tests  

## Commands

```bash
PYTHONPATH=. python3 -m pytest cim_resldpc/tests -v
PYTHONPATH=. python3 -m cim_resldpc.experiments.run_sigir_eval \
  --out_json cim_resldpc/results/sigir_eval.json --noise_p 0.05
```

## Reported artifacts

- `cim_resldpc/results/sigir_eval.json` — v1.1 protocol  
- `docs/design/SCIENTIFIC_REFLECTION.md` — claim integrity  
- GRC-PIM `pim_predecoder/results/paper1024_E*.json` — ECC channel evidence  

## Ablations

1. No LDPC vs B′ vs BP vs soft  
2. Centroid vs base-3 quantization  
3. Symmetric vs asymmetric query noise  
4. Noise strength sweep (geometric probe rows)  
5. Hybrid vs dense-only (system vs science)  
