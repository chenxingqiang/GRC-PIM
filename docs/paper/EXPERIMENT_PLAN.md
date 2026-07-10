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

- **IR:** nDCG@10, MAP, Recall@k  
- **Robustness:** relative nDCG drop under noise vs clean dense  
- **Hardware proxy:** mean latency_ms, decode skip rate, BM25 cutoff size  
- **Privacy:** `local_only` / `uploads_raw_documents=False` contract tests  

## Commands

```bash
python -m cim_resldpc.experiments.run_sigir_eval \
  --out_json cim_resldpc/results/sigir_eval.json \
  --train_steps 30 --n_topics 6

python -m pytest cim_resldpc/tests -v
```

## Reported artifacts

- `cim_resldpc/results/sigir_eval.json` — primary offline table source  
- `docs/design/CIM_RESLDPC_DESIGN.md` — method specification  
- GRC-PIM `pim_predecoder/results/paper1024_E*.json` — hardware ECC evidence to cite for noise realism  

## Ablations (paper tables)

1. w/o residual regularizer (`lambda_res=0`)  
2. w/o hybrid BM25 (`alpha=0`)  
3. w/o decode prune (`tau_skip=0`)  
4. Noise strength sweep  
5. Embedding dim / GF(3) levels sensitivity  
