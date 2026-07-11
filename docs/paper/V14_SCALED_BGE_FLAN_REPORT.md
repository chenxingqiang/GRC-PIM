# v1.4 Report — Scaled HF (200q) + BGE + Flan-T5 RAG

**Date:** 2026-07-10  
**Protocol:** teacher encoder → `cim_dim=128` → base-3 GF(3) → LDPC / B′ / BP / GEMV; asymmetric digital query

## What this step adds

| Item | Purpose |
|------|---------|
| 200-query MSMARCO / NQ caches | Closer to paper-scale IR evidence |
| BGE-small encoder | Stronger dense IR teacher than MiniLM |
| Flan-T5-small generative RAG | Real generation + measurable hallucination rate |

## MSMARCO HF n=200 (1720 passages), `nor_typical`

### MiniLM-L6-v2

| Method | nDCG@10 | MAP | RAG F1 (extractive) |
|--------|---------|-----|---------------------|
| BM25 | 0.528 | 0.402 | 0.178 |
| dense clean | 0.667 | 0.559 | 0.221 |
| CiM noisy | 0.570 | 0.448 | 0.202 |
| **LDPC B′** | **0.613** | **0.497** | 0.206 |
| LDPC BP | 0.605 | 0.486 | 0.207 |

B′ gain vs noisy: **+0.042** nDCG (stable vs 48q result +0.042).

### BGE-small-en-v1.5

| Method | nDCG@10 | MAP | RAG F1 |
|--------|---------|-----|--------|
| dense clean | 0.673 | 0.570 | 0.206 |
| CiM noisy | 0.407 | 0.301 | 0.167 |
| LDPC B′ | 0.493 | 0.382 | 0.187 |
| **LDPC BP** | **0.573** | **0.452** | **0.191** |

BP gain vs noisy: **+0.166** nDCG — strongest IR recovery so far.

## NQ-open hard n=200 (911 docs), MiniLM

| Method | nDCG@10 | B′/BP gain vs noisy |
|--------|---------|---------------------|
| clean | 0.145 | — |
| noisy | 0.106 | — |
| B′ | 0.118 | **+0.012** |
| BP | 0.111 | +0.005 |

## Generative RAG (Flan-T5-small, MSMARCO n=48, MiniLM)

| Retriever | RAG F1 | Hallucination rate |
|-----------|--------|--------------------|
| clean | 0.293 | 0.104 |
| noisy | 0.279 | 0.062 |
| B′ | 0.278 | 0.104 |
| BP | 0.250 | 0.083 |
| BM25 | 0.240 | 0.062 |

Unlike extractive RAG (hallu≈0 by construction), Flan produces **non-zero unsupported answers**, enabling a real RAG hallucination metric.

## Commands

```bash
pip install -r cim_resldpc/requirements-ir.txt
PYTHONPATH=. python3 -m cim_resldpc.experiments.build_hf_cache --n_queries 200
PYTHONPATH=. python3 -m cim_resldpc.experiments.run_scaled_eval --encoder st --n_queries 200
PYTHONPATH=. python3 -m cim_resldpc.experiments.run_scaled_eval --encoder bge --corpora msmarco_hf --n_queries 200
PYTHONPATH=. python3 -m cim_resldpc.experiments.run_scaled_eval --rag_generator flan --n_queries 48 --corpora msmarco_hf
```

## Artifacts

- `cim_resldpc/results/sigir_eval_v14_minilm_n200.json`
- `cim_resldpc/results/sigir_eval_v14_bge_n200.json`
- `cim_resldpc/results/sigir_eval_v14_flan_rag_n48.json`
- caches: `data/cache/*_n200.json`

## Honest limits

- 200q still ≪ full MSMARCO Dev  
- Projection to 128-D loses some teacher fidelity  
- Flan-T5-small ≠ production edge LLM  
- NQ-open remains pseudo-passage based
