# CiM-ResLDPC — Abstract Draft (SIGIR-AP 2026)

**Title:** CiM-ResLDPC: Residual Error Calibrated Robust Retrieval-Augmented Generation on NOR Flash Compute-in-Memory Arrays

**Chinese:** CiM-ResLDPC：面向NOR浮栅存算阵列的残差纠错鲁棒边缘检索增强生成

## Abstract

Edge retrieval-augmented generation (RAG) increasingly relies on compute-in-memory (CiM) to keep private corpora on-device under millisecond latency budgets. NOR Flash CiM offers the density needed for large knowledge bases, but floating-gate drift, read disturb, quantization, and analog MAC residuals corrupt stored embeddings and collapse ranking quality. Prior CiM-RAG systems accelerate similarity with in-array MAC while treating noise as a shallow mask; they neither inject multi-cycle soft evidence into representation learning nor couple GF(3) LDPC residual calibration with retrieval scoring.

We present **CiM-ResLDPC**, a hardware–IR co-design that reuses gated residual calibration (GRC-PIM) as a first-class retrieval signal. (1) **TONEL-Res** distills dense embeddings with a residual soft-information regularizer and GF(3)-aware quantization so representations remain robust under ternary CiM storage. (2) An end-to-end **three-layer pipeline** links NOR multi-cycle readout, residual-calibrated vectors, and residual-aware hybrid reranking (BM25 + dense + residual confidence). (3) **Edge optimizations**—decode pruning under low drift and BM25 candidate cutoff—reduce CiM cycles while preserving local-only privacy. On offline synthetic IR and a semiconductor mini-corpus, CiM-ResLDPC improves nDCG/MAP under PIM noise relative to noisy CiM baselines and reports latency / decode-skip hardware proxies suitable for SIGIR-AP evaluation.

## CCS Concepts

- Information systems → Retrieval models and ranking; Retrieval efficiency
- Hardware → Memory and dense storage; Error detection and correction

## Keywords

Retrieval-augmented generation; compute-in-memory; NOR Flash; LDPC; residual calibration; edge IR; noise-robust embedding
