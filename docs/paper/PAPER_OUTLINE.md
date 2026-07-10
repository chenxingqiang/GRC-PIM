# Paper Outline & Submission Checklist (SIGIR-AP Full Research)

**Working title:** CiM-ResLDPC: Residual Error Calibrated Robust Retrieval-Augmented Generation on NOR Flash Compute-in-Memory Arrays

**Track:** Full Research Paper (7–9 pages) · backup: Industry Paper

## Section map → repo docs

| Section | Draft file | Status |
|---------|------------|--------|
| Abstract | `ABSTRACT.md` | v1.1 narrowed claims |
| §1 Intro | `INTRODUCTION.md` | **v1.7** with 1000q highlight |
| §2 Related Work | `RELATED_WORK.md` | camera-ready tone |
| §3 Method | `METHOD.md` | condensed pipeline |
| §4 (optional) System | design doc + GRC-PIM README | architecture |
| §5 Experiments | `EXPERIMENTS.md` + V14–V17 reports | measured through 1000q |
| §6 Conclusion | `CONCLUSION.md` | **v1.7** |
| Ethics / Limits | SCIENTIFIC_REFLECTION + Conclusion §6.1 | operating-point honesty |
| ACM LaTeX | `latex/main.tex` | skeleton (compile offline) |

## Recommended main tables for the PDF

1. **MSMARCO 1000q, BGE, nor_typical, dim=128, seed=42** (primary IR table)  
2. **Multi-seed dim=128 / dim=256** stability (0.182±0.031 / 0.180±0.029)  
3. **Scale ladder** 200→500→1000q  
4. **NOR sweep** (fresh/typical/aged) with explicit failure cases  
5. **Encoder ablation** MiniLM vs BGE (200q)  
6. **Hybrid system** BM25 vs BM25+BP  
7. **Flan RAG** hallucination (48q) as generation probe  
8. **Geometric probe** as controlled positive control  

## Narrative (one paragraph for reviewers)

Edge RAG needs local CiM storage → NOR density is attractive → analog noise destroys dense nDCG → prior CiM-RAG accelerates MAC without LDPC residual retrieval → we protect GF(3) embeddings and recover with B′/BP (optional GEMV) → on MSMARCO subsets up to 1000q BP restores ranking under typical NOR maps; hybrid beats BM25; gains are seed/dim/operating-point sensitive and we report that.

## Checklist before submit

- [x] Replace mini-fixture-only claims with 200q/500q/1000q tables  
- [x] Never cite v0 oracle-blend numbers  
- [x] State NOR presets ≠ TCAD  
- [x] State NQ-open uses pseudo-passages  
- [x] Report pathological seed (dim=128 seed=0 on 500q)  
- [x] Prefer multi-seed / dim=256 for stability claims  
- [x] Separate dense-only vs hybrid metrics  
- [x] Cite GRC-PIM as reliability substrate  
- [ ] ACM format title + CCS + keywords from ABSTRACT.md (LaTeX skeleton ready)  
- [x] Reproducibility: `requirements-ir.txt` + cache builders + JSON artifacts  

## Next optional upgrades

- Full MSMARCO Dev (thousands of queries)  
- Contriever/BGE-base without random projection (product-quant CiM map)  
- On-device LLM (Phi/TinyLlama) instead of Flan-T5-small  
- Measured NOR chip noise traces replacing presets  
