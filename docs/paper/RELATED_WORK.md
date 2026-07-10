# Related Work Draft (Four Modules)

## Module 1 — LLM-oriented RAG bottlenecks (2026)

Cloud RAG primarily models *textual* noise and retrieval–generation mismatch. Edge local RAG additionally stores embeddings in analog CiM, so **hardware noise propagates into ranking and amplifies LLM hallucination**. Compressing float embeddings to CiM bit-width / GF(q) alphabets induces **semantic collapse**, especially for long-tail industrial knowledge. Local-first privacy forbids uploading raw documents, yet heavy denoisers violate **sub-10 ms** retrieval budgets. Existing text-only denoisers do not model floating-gate temporal soft evidence.

## Module 2 — CiM-accelerated retrieval

Representative systems (e.g., RoCR, CiMRAG) focus on **in-memory similarity acceleration**. Noise handling is typically random masking or simple averaging, without NOR-specific multi-cycle soft LLRs or gated residual correction. SRAM CiM is quieter but capacity-limited; NOR Flash CiM is dense but noise-rich—yet **no retrieval stack is specialized to NOR + GF(3) LDPC residual gating**. Hardware, ranking models, and ECC are optimized in silos without end-to-end feedback.

## Module 3 — LDPC in storage vs retrieval

Classical LDPC protects raw Flash bitstreams. Soft residuals are used to repair bits, **not to recalibrate embedding geometry or rerank documents**. Multi-cycle MAC residual soft evidence is an untapped signal for relevance scoring under analog noise. CiM-ResLDPC closes this gap by feeding GRC-PIM residual features into TONEL-Res training and hybrid fusion.

## Module 4 — Hardware–IR crossover at SIGIR

Edge retrieval, energy-aware ranking, and hardware-aware IR are rising, but few IR authors model Flash analog channels, and few microelectronics authors report nDCG/MAP/RAG factuality. SIGIR-AP’s industry–academia emphasis makes **cross-layer CiM + RAG** a differentiated submission niche.

## Positioning vs GRC-PIM

GRC-PIM provides the **reliability layer** (B′, gated ΔLLR, GF(3) BP). CiM-ResLDPC is the **IR layer** that consumes those signals for representation learning and retrieval—complementary, not competing, contributions.
