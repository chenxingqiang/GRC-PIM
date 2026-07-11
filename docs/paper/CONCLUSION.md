# Conclusion & Limitations Draft

## 6. Conclusion

We presented **CiM-ResLDPC**, a cross-layer framework that treats NOR/PIM compute-in-memory noise as a first-class retrieval problem rather than a MAC-acceleration footnote. By packing dense teacher embeddings into GF(3) symbols, protecting them with non-binary LDPC, and recovering vectors with B′ soft LLRs and belief propagation (optionally gated GEMV residual calibration from GRC-PIM), the method reconnects storage soft evidence to ranking.

On MSMARCO passage subsets with a BGE-small teacher projected to edge widths, unprotected CiM noise causes large nDCG drops, while LDPC BP recovers ranking under typical NOR-mapped operating points. At **2000 queries / 16894 passages** (`cim_dim=128`, seed=42), BP recovers **+0.221 nDCG@10** (0.254→0.475); multi-seed gain is **0.145±0.050**; hybrid BM25+BP reaches **0.557** above BM25 **0.531**. At 1000q the same protocol yields +0.214 with multi-seed **0.182±0.031**. Absolute BP nDCG remains near ~0.48 across 1k–2k scales while unprotected noise worsens. Controlled geometric probes confirm that when embedding geometry exists, soft recovery improves both SER and nDCG. Generative Flan-T5 probes show non-zero hallucination rates, enabling RAG-side evaluation beyond extractive proxies.

The broader message for SIGIR-AP is that **hardware–IR co-design is actionable**: residual soft information from CiM channels can be turned into measurable retrieval robustness for private, on-device RAG—provided claims are tied to explicit noise operating points.

## 6.1 Limitations

- **Scale:** Reported MSMARCO numbers use stratified validation subsets (48/200/500/1000/2000 queries), not the full leaderboard protocol.  
- **NOR model:** Presets map stress to the GRC-PIM multi-cycle bit-flip channel; they are engineering calibrations, not TCAD floating-gate physics or chip traces.  
- **Projection:** Teacher→`cim_dim` random projection is transparent but suboptimal versus learned CiM-aware quantization.  
- **Seed / width sensitivity:** At narrow widths, some seeds (historically dim=128 seed=0 on 500q) can make BP hurt ranking; we report failures and prefer multi-seed or wider CiM vectors for stability claims.  
- **NQ-open:** Lacks native passages; hard pseudo-docs are diagnostic only.  
- **Generation:** Flan-T5-small is a probe, not a production edge LLM.  
- **TONEL training:** Large tables use frozen teachers for IR fairness; full CiM-aware fine-tuning remains future work.

## 6.2 Future work

Full MSMARCO Dev/DL evaluation; learned GF(3)/PQ maps; measured NOR array noise; on-device LLMs (Phi/TinyLlama); tighter coupling of GRC GEMV gates into rerank features; and end-to-end latency/energy on a CiM prototype SoC.
