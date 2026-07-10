# Related Work Draft (camera-ready tone)

## 2.1 LLM-oriented RAG and retrieval noise

Modern RAG pipelines retrieve passages to condition LLM generation [Lewis et al.; Guu et al.]. Most IR work models *textual* noise, spurious correlations, or retrieval–generation mismatch. Edge local RAG adds a different noise source: **analog CiM storage and MAC**. Text denoisers and LLM-side filters do not correct corrupted embedding coordinates. Our focus is this hardware-induced ranking degradation and its propagation into answerability / hallucination proxies.

## 2.2 Dense retrieval and edge constraints

Dense encoders (DPR, Contriever, BGE, etc.) dominate MSMARCO-style ranking. Edge deployment forces compression: product quantization, binary codes, or low-dimensional projections. We adopt a transparent protocol—**frozen teacher → fixed projection to `cim_dim` → GF(3) packing**—so that measured gains can be attributed to the CiM protection/recovery path rather than to opaque end-to-end finetuning. Hybrid BM25+dense remains a strong systems baseline for latency; we report it as a *system* result, separate from dense-only scientific tables.

## 2.3 CiM-accelerated retrieval

CiM / PIM accelerators reduce data movement for similarity search and neural kernels. Representative CiM-RAG efforts emphasize in-array MAC throughput. Their noise treatments are typically insufficient for NOR-like multi-cycle soft channels: they do not expose residual soft LLRs to ranking, nor protect ternary embedding codes with NB-LDPC. SRAM CiM is quieter but capacity-limited; NOR Flash CiM is denser but noise-rich. **CiM-ResLDPC targets the latter regime.**

## 2.4 LDPC and soft residual calibration in storage

Non-binary LDPC and soft-decision decoding are mature in Flash controllers. GRC-PIM adds a gated residual GEMV calibrator on multi-cycle soft LLRs with safe fallback to B′. Prior art stops at bit/symbol reliability. We instead ask whether the same soft residual evidence can **reconstruct embedding geometry** and improve nDCG/MAP under CiM noise. Empirically, BP recovery helps under typical NOR-mapped stress; SER improvements do not automatically transfer to ranking at every corner—hence our operating-point tables.

## 2.5 Hardware-aware IR at SIGIR

Energy-aware ranking, on-device search, and systems-for-IR are rising SIGIR themes, yet few papers jointly report IR metrics and CiM channel models. Microelectronics venues rarely evaluate nDCG. CiM-ResLDPC sits in that sparse intersection and is written for SIGIR-AP’s applied + industrial audience.

## 2.6 Summary of differences

| Line of work | Typical focus | Missing for edge CiM-RAG |
|--------------|---------------|---------------------------|
| Dense IR / RAG | Semantics, fusion, generation | Analog CiM embedding corruption |
| CiM-RAG accelerators | MAC speed / energy | LDPC-protected embeddings + ranking soft evidence |
| Flash LDPC / GRC-PIM | BER/SER, gated ΔLLR | nDCG/MAP/RAG hallucination |
| **CiM-ResLDPC (ours)** | Cross-layer residual retrieval | — |
