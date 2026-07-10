# Paper Outline & Submission Checklist (SIGIR-AP Full Research)

**Working title:** CiM-ResLDPC: Residual Error Calibrated Robust Retrieval-Augmented Generation on NOR Flash Compute-in-Memory Arrays

**Track:** Full Research Paper (7–9 pages) · backup: Industry Paper

## Section map → repo docs

| Section | Draft file | Status |
|---------|------------|--------|
| Abstract | `ABSTRACT.md` | updated with 2k scale |
| §1 Intro | `INTRODUCTION.md` | **v1.8** 2000q highlight |
| §2 Related Work | `RELATED_WORK.md` | camera-ready tone |
| §3 Method | `METHOD.md` | condensed pipeline |
| §5 Experiments | `EXPERIMENTS.md` + V14–V18 | measured through **2000q** |
| §6 Conclusion | `CONCLUSION.md` | **v1.8** |
| ACM LaTeX | `latex/main.tex` + `references.bib` | draft with cites |

## Recommended main tables for the PDF

1. **MSMARCO 2000q (primary) + 1000q**, BGE, nor_typical, dim=128, seed=42  
2. **Scale ladder** 200→500→1000→2000q  
3. **Multi-seed** stability (2000q: 0.145±0.050)  
4. **NOR sweep** fresh/typical/aged with aged failure honesty  
5. **Encoder ablation** MiniLM vs BGE (200q)  
6. **Hybrid system** BM25 vs BM25+BP  
7. **Flan RAG** hallucination (48q)  
8. **Geometric probe** positive control  

## Checklist before submit

- [x] Tables through 2000q stratified validation  
- [x] Never cite v0 oracle-blend numbers  
- [x] State NOR presets ≠ TCAD; report aged failure  
- [x] Separate dense-only vs hybrid  
- [x] ACM skeleton + `.bib`  
- [ ] Full Dev (~10k q) optional stretch  
- [ ] Camera-ready author block / DOI  

## Next optional upgrades

- Full MSMARCO Dev (~10k queries)  
- Contriever / BGE-base without random projection  
- On-device LLM (Phi/TinyLlama)  
- Measured NOR chip noise traces  
