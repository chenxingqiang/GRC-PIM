# CiM-ResLDPC — Abstract Draft (SIGIR-AP 2026)

**Title:** CiM-ResLDPC: Residual Error Calibrated Robust Retrieval-Augmented Generation on NOR Flash Compute-in-Memory Arrays

**Chinese:** CiM-ResLDPC：面向NOR浮栅存算阵列的残差纠错鲁棒边缘检索增强生成

## Abstract (v1.1 — narrowed claims)

Edge retrieval-augmented generation (RAG) increasingly relies on compute-in-memory (CiM) to keep private corpora on-device. Dense CiM substrates (including NOR Flash targets) offer capacity for local knowledge bases, but multi-cycle readout noise, quantization to constrained alphabets, and analog residuals corrupt stored embeddings and collapse ranking quality. Prior CiM-RAG systems mainly accelerate similarity with in-array MAC and treat noise as a shallow mask; they rarely protect ternary embeddings with non-binary LDPC or feed soft residual evidence into representation learning and dense scoring.

We present **CiM-ResLDPC**, a hardware–IR co-design on the GRC-PIM multi-cycle PIM channel. (1) **TONEL-Res** trains dense embeddings with a residual soft-information regularizer toward **channel-calibrated** reconstructions (B′ soft expectation / GF(3) BP)—without clean-embedding oracles—and GF(3)-aware quantization. (2) Embeddings are **LDPC-protected** for CiM storage; recovery uses B′/BP soft–hard paths before cosine retrieval. (3) An optional **hybrid** BM25 cutoff reduces candidates for edge latency, reported separately from the dense-only scientific table. Offline synthetic IR and a semiconductor mini-corpus evaluate dense robustness under PIM noise; generation-side RAG metrics and device-physics NOR models remain future work.

## CCS Concepts

- Information systems → Retrieval models and ranking; Retrieval efficiency
- Hardware → Memory and dense storage; Error detection and correction

## Keywords

Retrieval-augmented generation; compute-in-memory; NOR Flash; LDPC; residual calibration; edge IR; noise-robust embedding
