# Introduction Draft (SIGIR-AP 2026)

**Title:** CiM-ResLDPC: Residual Error Calibrated Robust Retrieval-Augmented Generation on NOR Flash Compute-in-Memory Arrays

## 1. Introduction

Retrieval-augmented generation (RAG) is becoming the default way to ground large language models (LLMs) in private or long-tail knowledge. On edge devices, however, **local-first** constraints forbid uploading raw documents to the cloud, while interactive assistants still demand low retrieval latency. Compute-in-memory (CiM) is an attractive substrate for this setting: dense embeddings can be stored and scored inside memory arrays, avoiding costly off-chip data movement. NOR Flash CiM is especially appealing for on-device knowledge bases because it offers higher density than SRAM CiM at acceptable cost.

The difficulty is that **CiM is not a noiseless RAM**. Floating-gate / analog arrays exhibit read disturb, conductance drift, quantization, and analog MAC residuals. When document embeddings live in such arrays, hardware noise is not a minor implementation detail—it **directly corrupts the vector geometry that dense retrieval depends on**. In our measurements on MSMARCO passage subsets with a BGE teacher projected to a 128-D edge width, unprotected CiM noise drops nDCG@10 sharply under a typical NOR-mapped stress point (e.g., **0.641→0.283** at 1000 queries; **0.673→0.445** at 200 queries). That is a first-order IR failure, not a second-order hardware footnote.

Existing CiM-RAG systems (e.g., RoCR- and CiMRAG-style designs) primarily treat the array as a **MAC accelerator** for similarity. Noise handling, when present, is often a shallow mask or averaging step. In parallel, classical storage coding (including non-binary LDPC) repairs raw Flash bitstreams but is rarely connected to **embedding reconstruction or ranking**. The result is a three-way gap:

1. **Hardware noise ↔ representation gap:** multi-cycle soft evidence and residual calibration are not embedded into retrieval learning or scoring.  
2. **Alphabet / bit-width gap:** float teachers must be mapped to GF(q) / limited-width CiM storage without total semantic collapse.  
3. **Privacy ↔ latency gap:** heavy denoisers that ignore gating cannot meet edge budgets, while unprotected CiM cannot meet ranking quality.

We propose **CiM-ResLDPC**, a hardware–IR co-design that reuses the GRC-PIM gated residual stack as a retrieval reliability layer. Document embeddings are packed into GF(3) symbols, protected by non-binary LDPC, exposed to a multi-cycle CiM channel (with NOR operating-point presets), and recovered via B′ soft LLRs and/or belief propagation—optionally with gated GEMV residual ΔLLR. Queries remain on the digital side (**asymmetric RAG**), matching realistic edge pipelines where only the corpus is CiM-resident. A hybrid BM25 + residual-calibrated dense scorer provides a practical system mode.

**Contributions.**

- **TONEL-Res / CiM-aware representation path:** residual soft-information regularization toward *channel-calibrated* reconstructions (no clean-embedding oracle), with GF(3)-compatible quantization.  
- **End-to-end CiM-ResLDPC retrieval:** LDPC-protected ternary embeddings + B′/BP recovery + optional GRC GEMV, evaluated under dense-only scientific metrics.  
- **Edge system optimizations:** decode gating / hybrid sparse–dense fusion, local-only privacy contract, and generative RAG probes (Flan-T5) with measurable hallucination.

**Empirical highlight.** On MSMARCO (**2000 queries / 16894 passages**, BGE-small, `nor_typical`, `cim_dim=128`, seed=42), LDPC BP recovers **+0.221 nDCG@10** versus unprotected CiM noise (0.254→0.475); hybrid BM25+BP reaches **0.557** (above BM25 0.531). Multi-seed BP gain is **0.145±0.050**. At 1000q: +0.214 with multi-seed **0.182±0.031** (dim=128) / **0.180±0.029** (dim=256). Gains remain **operating-point and seed dependent**—we report failures rather than hide them.

## 1.1 Positioning

CiM-ResLDPC is complementary to GRC-PIM: GRC-PIM is the reliability layer between multi-cycle PIM readout and GF(3) decoding; CiM-ResLDPC is the IR layer that consumes those signals for ranking and RAG. We target SIGIR-AP’s industry–academia intersection: edge retrieval, hardware-aware IR, and deployable privacy-preserving RAG.
