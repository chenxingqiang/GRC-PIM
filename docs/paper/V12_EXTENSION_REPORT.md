# v1.2 Extension Report — Strong Encoder / IR / GEMV / NOR / RAG

**Phase:** Development → Testing (post Verification roadmap)  
**Date:** 2026-07-10

## Purpose of each extension (recap)

| Extension | Why |
|-----------|-----|
| Strong encoder + MSMARCO/NQ mini | Make IR gains credible beyond hash+toy topics |
| Optional GRC GEMV | Connect title “residual” to real gated ΔLLR hardware stack |
| NOR calibration presets | Defend NOR wording with named stress maps (still GRC-PIM channel) |
| RAG generation proxy | Show retrieval quality → answerability / hallucination support |

## What shipped

- `encoders.py` — `tfidf_svd` (TF-IDF + fixed random projection, dim-stable)
- `data/msmarco_mini.json`, `data/nq_mini.json` + `ir_benchmarks.py`
- `gemv_hook.py` + `protect_and_recover(..., use_gemv=True)`
- `nor_calibration.py` — fresh / typical / aged / stress_sweep
- `rag_eval.py` — extractive F1, EM, support, hallucination rate
- `experiments/run_extended_eval.py`

## Key numbers (MSMARCO mini, tfidf_svd dim=128)

| NOR preset | clean | noisy | B′ | BP | GEMV+BP | B′−noisy |
|------------|-------|-------|----|----|---------|----------|
| nor_fresh | 0.885 | 0.845 | 0.845 | 0.845 | 0.845 | ~0 |
| **nor_typical** | 0.885 | 0.779 | **0.848** | 0.845 | 0.842 | **+0.069** |
| nor_aged | 0.885 | 0.671 | 0.627 | 0.507 | 0.551 | −0.044 |
| **nor_stress_sweep** | 0.885 | 0.495 | 0.557 | **0.692** | **0.669** | +0.062 / GEMV **+0.174** |

**Takeaway:** Under **typical** and **stress** NOR maps, LDPC (and GEMV at stress) improve dense nDCG vs unprotected CiM noise. Aged mid-corner can still hurt ranking even when ECC helps SER — same lesson as v1.1: always report the operating point.

## RAG (NQ mini, extractive proxy)

Hallucination rate stays ~0 when answers are extractive from retrieved text (by construction). F1 tracks retrieval hit quality (clean > noisy). This is a **lower bound** RAG claim until a real on-device LLM is plugged in.

## Honest limits (unchanged)

- Mini fixtures ≠ full MSMARCO/NQ leaderboard  
- NOR presets ≠ TCAD floating-gate physics  
- Extractive RAG ≠ generative LLM factuality  
- Optional `sentence-transformers` path exists when installed (`--encoder st`)

## Tests

`cim_resldpc/tests`: **24 passed**
