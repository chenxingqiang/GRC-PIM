# v1.3 Report — MiniLM + HuggingFace MSMARCO/NQ subsets

**Date:** 2026-07-10  
**Protocol:** teacher MiniLM-L6-v2 → project to `cim_dim=128` → base-3 GF(3) → LDPC/B′/BP/GEMV

## Why this step

v1.2 used TF-IDF on tiny fixtures. v1.3 answers the review question: *does CiM-ResLDPC still help with a real dense IR encoder on MSMARCO-style data?*

## Setup

```bash
pip install -r cim_resldpc/requirements-ir.txt
PYTHONPATH=. python3 -m cim_resldpc.experiments.build_hf_cache --n_queries 48
PYTHONPATH=. python3 -m cim_resldpc.experiments.run_hf_eval \
  --encoder st --nor nor_typical --n_queries 48 --cim_dim 128
```

Caches: `cim_resldpc/data/cache/msmarco_hf_n48.json`, `nq_hf_n48.json` (+ hard NQ variant).

## MSMARCO HF subset (48 queries, 403 passages) — MiniLM

| NOR | Method | nDCG@10 | MAP | vs noisy |
|-----|--------|---------|-----|----------|
| **typical** | dense clean | 0.703 | 0.605 | — |
| | CiM noisy | 0.567 | 0.441 | −19.3% vs clean |
| | **LDPC B′** | **0.609** | **0.488** | **+0.042** |
| | **LDPC BP** | **0.618** | **0.502** | **+0.051** |
| | GEMV+BP | 0.618 | 0.502 | +0.051 |
| stress | noisy | 0.334 | — | — |
| | B′ / BP / GEMV | 0.286 / 0.250 / 0.306 | — | GEMV least-bad |

**Encoder ablation (same MSMARCO cache, nor_typical):** MiniLM clean 0.703 ≫ TF-IDF clean 0.276 — confirms the strong-encoder step matters.

## NQ-open hard pseudo-corpus (default cache builder)

Easy NQ (question echoed in passage) saturates at 1.0 — not informative. Default builder now uses answer-only facts + distractors:

| Method | nDCG@10 | MAP | RAG F1 |
|--------|---------|-----|--------|
| BM25 | 0.069 | 0.058 | — |
| dense clean | 0.197 | 0.163 | 0.074 |
| CiM noisy | 0.111 | 0.088 | 0.014 |
| B′ | 0.107 | 0.076 | 0.010 |
| **BP / GEMV+BP** | **0.131** | **0.100** | **0.029** |

BP recovers ranking (+0.020 nDCG vs noisy) and answerability under hard NQ.

## Fixes landed with v1.3

- Multi-block LDPC soft LLR unpack for wide embeddings
- Teacher→CiM projection (`cim_dim`) for edge-realistic storage width
- HF cache builder + MiniLM eval entrypoint

## Honest limits

- 48-query subset ≠ full MSMARCO leaderboard  
- NQ-open has no native passages (pseudo-docs)  
- Stress NOR can still make ECC hurt ranking — report operating points  
- Extractive RAG proxy, not a generative LLM
