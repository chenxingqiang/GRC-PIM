# Method Draft (condensed)

## 3.1 Problem

Edge RAG stores document embeddings in CiM. Multi-cycle analog readout (drift, disturb, quantization) corrupts vectors and collapses nDCG. Prior CiM-RAG accelerates MAC but does not protect ternary embeddings with NB-LDPC or feed residual soft evidence into retrieval.

## 3.2 CiM-ResLDPC pipeline

1. **Teacher encode** (BGE/MiniLM) → project to edge width `cim_dim`.  
2. **Base-3 GF(3) pack** (2 symbols/dim).  
3. **LDPC encode** with shipped GF(3) NB-LDPC (systematic G).  
4. **Multi-cycle CiM channel** (GRC-PIM noise; NOR presets map stress → `(p_indep,p_drift,T)`).  
5. **Recover:** B′ soft-LLR product and/or BP; optional GRC GEMV gated ΔLLR before BP.  
6. **Retrieve:** asymmetric cosine (clean digital query vs recovered doc vectors).  
7. **Optional hybrid:** BM25 cutoff + dense + residual-confidence fusion.  
8. **Generate:** extractive or Flan-T5 conditioned on top-k passages.

## 3.3 TONEL-Res (representation)

Train/adapt embeddings with residual alignment to **channel-calibrated** reconstructions (no clean oracle) plus soft ternary quantization penalty. In large-scale tables we evaluate frozen teachers + projection for IR fairness; TONEL remains the training-time regularizer for CiM-aware fine-tuning.

## 3.4 What we do *not* claim

- NOR presets ≠ full floating-gate TCAD.  
- 200q MSMARCO ≠ full leaderboard.  
- Flan-T5-small ≠ production edge LLM.  
- BP SER↓ does not always imply nDCG↑ at every noise corner.
