# Paper Outline & Submission Checklist (SIGIR-AP Full Research)

**Working title:** CiM-ResLDPC: Residual Error Calibrated Robust Retrieval-Augmented Generation on NOR Flash Compute-in-Memory Arrays

**Track:** Full Research Paper (7–9 pages) · backup: Industry Paper

## Section map → repo docs

| Section | Draft file | Status |
|---------|------------|--------|
| Abstract | `ABSTRACT.md` | v1.1 narrowed claims |
| §1 Intro | `INTRODUCTION.md` | **v1.6** with 200q/500q numbers |
| §2 Related Work | `RELATED_WORK.md` | **v1.6** camera-ready tone |
| §3 Method | `METHOD.md` | condensed pipeline |
| §4 (optional) System | design doc + GRC-PIM README | architecture |
| §5 Experiments | `EXPERIMENTS.md` + V14/V15/V16 reports | measured |
| §6 Conclusion | *todo short* | — |
| Ethics / Limits | SCIENTIFIC_REFLECTION + V16 notes | operating-point honesty |

## Recommended main tables for the PDF

1. **MSMARCO 500q, BGE, nor_typical, dim=128, seed=42** (primary IR table)  
2. **Multi-seed dim=256** stability (0.157±0.013 BP gain)  
3. **NOR sweep** (fresh/typical/aged) with explicit failure cases  
4. **Encoder ablation** MiniLM vs BGE (200q)  
5. **Hybrid system** BM25 vs BM25+BP  
6. **Flan RAG** hallucination (48q) as generation probe  
7. **Geometric probe** as controlled positive control  

## Narrative (one paragraph for reviewers)

Edge RAG needs local CiM storage → NOR density is attractive → analog noise destroys dense nDCG → prior CiM-RAG accelerates MAC without LDPC residual retrieval → we protect GF(3) embeddings and recover with B′/BP (optional GEMV) → on MSMARCO subsets BP restores ranking under typical NOR maps; hybrid beats BM25; gains are seed/dim/operating-point sensitive and we report that.

## Checklist before submit

- [ ] Replace mini-fixture-only claims with 200q/500q tables  
- [ ] Never cite v0 oracle-blend numbers  
- [ ] State NOR presets ≠ TCAD  
- [ ] State NQ-open uses pseudo-passages  
- [ ] Report pathological seed (dim=128 seed=0 on 500q)  
- [ ] Prefer dim=256 for stability claims  
- [ ] Separate dense-only vs hybrid metrics  
- [ ] Cite GRC-PIM as reliability substrate  
- [ ] ACM format title + CCS + keywords from ABSTRACT.md  
- [ ] Reproducibility: `requirements-ir.txt` + cache builders + JSON artifacts  

## Next optional upgrades

- Full MSMARCO Dev (thousands of queries)  
- Contriever/BGE-base without random projection (product-quant CiM map)  
- On-device LLM (Phi/TinyLlama) instead of Flan-T5-small  
- Measured NOR chip noise traces replacing presets  
