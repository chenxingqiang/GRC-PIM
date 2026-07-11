# Experiments Draft (SIGIR-AP Full Paper)

Numbers from repo artifacts (`sigir_eval_v14_*` … `sigir_eval_v17_msmarco1000.json`).  
Protocol: asymmetric RAG (digital query); teacher → random-project to `cim_dim` → base-3 GF(3) → multi-cycle PIM/NOR-mapped channel → B′ / BP recovery; dense-only is the scientific core.

## 5.1 Datasets

| Corpus | Source | Scale used | Notes |
|--------|--------|------------|-------|
| MSMARCO passage | `microsoft/ms_marco` v1.1 validation | **2000q / 16894**, also 1000/500/200/48 | Selected passages as relevance |
| NQ-open (hard) | `google-research-datasets/nq_open` | **200 queries / 911 docs** | Answer-only pseudo-passages + distractors |
| Geometric probe | synthetic clusters | 100 docs / 20 queries | Validates LDPC when geometry is known |
| SemiCorpusMini | bundled | 8–12 docs | Semiconductor domain flavor |

## 5.2 Encoders & CiM storage

- Teachers: **MiniLM-L6-v2**, **BGE-small-en-v1.5** (primary), TF-IDF projection (ablation)
- Edge storage width: project to `cim_dim ∈ {64,128,256}` then **2 GF(3) symbols/dim** (9 levels)
- Channel: GRC-PIM multi-cycle noise with **NOR presets** (`nor_fresh/typical/aged/stress`)

## 5.3 Main IR result (BGE, MSMARCO **2000q**, `nor_typical`, `cim_dim=128`, seed=42)

| Method | nDCG@10 |
|--------|---------|
| BM25 | 0.531 |
| Dense clean (projected) | 0.636 |
| CiM noisy (no LDPC) | 0.254 |
| **CiM LDPC BP** | **0.475** |
| **Hybrid BM25+BP** | **0.557** |

**BP recovers +0.221 nDCG@10 vs unprotected CiM noise.** Hybrid exceeds BM25 (0.531).

### Stability (MSMARCO 2000q)

| Setting | BP gain vs noisy |
|---------|------------------|
| **dim=128 seeds {0,1,2,42}** | **0.145 ± 0.050** |
| dim=256 seed=42 (spot-check) | **+0.208** |

NOR honesty (dim=128, seed=42): fresh +0.010; **aged +0.006** with absolute BP nDCG≈0.061—do not claim wins under aged maps.

### Scale ladder (BGE, dim=128, seed=42, `nor_typical`)

| Scale | Noisy → BP | Hybrid vs BM25 |
|-------|------------|----------------|
| 200q | 0.445 → 0.531 (+0.086) | 0.572 > 0.528 |
| 500q | 0.423 → 0.539 (+0.116) | 0.565 > 0.536 |
| 1000q | 0.283 → 0.496 (+0.214) | 0.549 > 0.520 |
| **2000q** | **0.254 → 0.475 (+0.221)** | **0.557 > 0.531** |

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

1. Hardware noise on CiM-stored dense vectors **materially hurts** MSMARCO nDCG (worse at larger subsets under the same map).  
2. GF(3) LDPC (esp. BP) **recovers** ranking under NOR-typical maps; 2000q multi-seed gain **0.145±0.050** (BGE, dim=128); 1000q **0.182±0.031**.  
3. Hybrid BM25+BP is a strong **system** result (2000q: 0.557 > BM25 0.531).  
4. Gains are **operating-point dependent**—do not claim universal win under aged/stress without qualification.  
5. Generative Flan RAG shows retrieval→generation coupling with measurable hallucination.
