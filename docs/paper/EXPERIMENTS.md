# Experiments Draft (SIGIR-AP Full Paper)

Numbers from repo artifacts (`sigir_eval_v14_*`, `sigir_eval_v15_ablation.json`).  
Protocol: asymmetric RAG (digital query); teacher → random-project to `cim_dim` → base-3 GF(3) → multi-cycle PIM/NOR-mapped channel → B′ / BP recovery; dense-only is the scientific core.

## 5.1 Datasets

| Corpus | Source | Scale used | Notes |
|--------|--------|------------|-------|
| MSMARCO passage | `microsoft/ms_marco` v1.1 validation | **200 queries / 1720 passages** (also 48q cache) | Selected passages as relevance |
| NQ-open (hard) | `google-research-datasets/nq_open` | **200 queries / 911 docs** | Answer-only pseudo-passages + distractors |
| Geometric probe | synthetic clusters | 100 docs / 20 queries | Validates LDPC when geometry is known |
| SemiCorpusMini | bundled | 8–12 docs | Semiconductor domain flavor |

## 5.2 Encoders & CiM storage

- Teachers: **MiniLM-L6-v2**, **BGE-small-en-v1.5** (primary), TF-IDF projection (ablation)
- Edge storage width: project to `cim_dim ∈ {64,128,256}` then **2 GF(3) symbols/dim** (9 levels)
- Channel: GRC-PIM multi-cycle noise with **NOR presets** (`nor_fresh/typical/aged/stress`)

## 5.3 Main IR result (BGE, MSMARCO 200q, `nor_typical`, `cim_dim=128`, seed=42)

| Method | nDCG@10 | MAP |
|--------|---------|-----|
| BM25 | 0.528 | 0.402 |
| Dense clean (projected) | 0.673 | — |
| CiM noisy (no LDPC) | 0.445 | — |
| CiM LDPC B′ | 0.491 | — |
| **CiM LDPC BP** | **0.531** | — |
| **Hybrid BM25+BP** | **0.572** | — |

**BP recovers +0.086 nDCG vs unprotected CiM noise.** Hybrid exceeds BM25 alone (+0.045).

### Multi-seed stability (`nor_typical`, dim=128, seeds 0/1/2)

| Metric | Mean ± Std |
|--------|------------|
| Noisy nDCG | 0.492 ± 0.017 |
| BP nDCG | 0.567 ± 0.022 |
| **BP gain vs noisy** | **0.075 ± 0.015** |

## 5.4 Encoder comparison (MSMARCO 200q, `nor_typical`)

| Teacher | Clean | Noisy | BP | BP gain |
|---------|-------|-------|----|---------|
| MiniLM | 0.667 | 0.570 | 0.605 | +0.034 |
| **BGE-small** | 0.673 | 0.407 | **0.573** | **+0.166** (v1.4 run) / +0.086 (seed=42 ablation) |

BGE shows larger absolute recovery under the same NOR map (noise hurts BGE geometry more; BP restores more).

## 5.5 NOR operating-point sweep (BGE, dim=128, seed=0)

| NOR preset | Noisy | BP | BP gain |
|------------|-------|----|---------|
| fresh | 0.552 | 0.552 | ≈0 |
| **typical** | 0.486 | **0.540** | **+0.055** |
| aged | 0.249 | 0.132 | −0.117 |
| stress | 0.075 | 0.033 | −0.042 |

**Claim discipline:** CiM-ResLDPC helps in the **typical** regime; under extreme aged/stress maps, ranking can still degrade—report the operating point (same lesson as ECC SER≠IR).

## 5.6 `cim_dim` ablation (BGE, `nor_typical`, seed=0)

| cim_dim | Clean | BP gain vs noisy |
|---------|-------|------------------|
| 64 | 0.646 | +0.051 |
| 128 | 0.653 | +0.055 |
| 256 | 0.665 | **+0.179** |

Wider CiM vectors preserve more teacher signal and increase BP recovery headroom (at higher storage/MAC cost).

## 5.7 RAG generation

**Extractive** (answer span from passages): hallucination≈0 by construction; F1 tracks retrieval hit rate.

**Flan-T5-small** (MSMARCO 48q, MiniLM, `nor_typical`):

| Retriever | RAG F1 | Hallucination rate |
|-----------|--------|--------------------|
| clean | 0.293 | 0.104 |
| noisy | 0.279 | 0.062 |
| B′ | 0.278 | 0.104 |
| BP | 0.250 | 0.083 |

Generative setting yields **non-zero unsupported answers**, enabling a true RAG hallucination metric.

## 5.8 Geometric probe (controlled)

At p=0.03: noisy nDCG 0.905 → B′ **0.952** (SER 0.086→0.038). Establishes that when embedding geometry exists, LDPC soft recovery improves ranking.

## 5.9 Takeaways for the paper narrative

1. Hardware noise on CiM-stored dense vectors **materially hurts** MSMARCO nDCG.  
2. GF(3) LDPC (esp. BP) **recovers** ranking under NOR-typical maps; multi-seed gain **0.075±0.015** (BGE, dim=128).  
3. Hybrid BM25+BP is a strong **system** result (0.572 > BM25 0.528).  
4. Gains are **operating-point dependent**—do not claim universal win under aged/stress without qualification.  
5. Generative Flan RAG shows retrieval→generation coupling with measurable hallucination.
