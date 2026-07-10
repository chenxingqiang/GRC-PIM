# Scientific Reflection — CiM-ResLDPC v0 → v1.1

**Phase:** Verification (fact-check before further claims)  
**Date:** 2026-07-10  
**Verdict:** The *research direction* is sound and publishable in principle, but the **v0 implementation overstated hardware–IR coupling** and contained **evaluation confounds** that would not survive SIGIR review. v1.1 applies minimal, evidence-based corrections.

---

## 1. What is scientifically solid (keep)

| Claim | Evidence |
|-------|----------|
| Edge RAG on CiM faces embedding corruption from multi-cycle PIM noise | Reproduced: clean dense nDCG 0.79 → noisy CiM 0.37 (−53%) on SyntheticIR |
| GRC-PIM noise3d + B′ soft-LLR is a reusable channel model | Direct import from validated `pim_predecoder` |
| GF(3) ternary quantization is a legitimate CiM storage constraint | Matches GRC-PIM GF(3) alphabet; matrices are systematic (`info ‖ parity`) |
| Hybrid sparse+dense is a valid edge latency design | Standard IR practice; BM25 cutoff reduces dense MAC candidates |
| Cross-layer residual soft evidence → ranking is a real gap vs RoCR/CiMRAG-style MAC-only papers | Conceptual novelty holds *if* residual path is real, not oracle |

---

## 2. Critical integrity failures in v0 (must fix)

### F1 — Oracle leakage (fatal for claims)

`pipeline.index` / query path used:

```text
cal = 0.5 * recovered_noisy + 0.5 * clean_embedding
```

At inference, **clean float embeddings are not available** after CiM readout. This invents accuracy and invalidates “residual calibration” as the cause of gains.

**Fix:** Reconstruct only from channel soft information (B′ LLR soft expectation and/or GF(3) BP hard decision). Never mix `clean_emb` into calibrated vectors at index/query time.

### F2 — “ResLDPC” without LDPC on embeddings

Design diagram shows encode → noise → GRC → BP. v0 only did per-dim GF(3) quantize + bit-flip noise + B′ argmax. **No parity symbols, no `bp_decode`.** The paper title’s “LDPC” was aspirational.

**Fix:** Pack embedding symbols into GF(3) info blocks, `encode()` with shipped G, apply noise to full codeword, recover via B′ and optional BP; take information-symbol prefix as embedding.

### F3 — Result confounding (BM25 ≈ “ours”)

| Method | nDCG@10 |
|--------|---------|
| BM25 | 0.936 |
| CiM-ResLDPC (hybrid) | 0.933 |

Toy SyntheticIR has **lexical topic tokens by construction**, so BM25 saturates. Hybrid scores cannot attribute gains to residual calibration. Metric `ours_degradation_vs_clean = −0.19` compared hybrid to clean dense — **category error**.

**Fix:** Report **dense-only under noise** ablations (no BM25): `cim_noisy`, `cim_bprime`, `cim_bp`, `cim_soft` (+ optional hybrid as system result). Robustness = drop within the same scorer family.

### F4 — Unstable encoder hash

`HashBagEncoder` used Python `hash()` (randomized per process unless `PYTHONHASHSEED` fixed). Breaks reproducibility.

**Fix:** `hashlib.md5` / blake2 modulo bucket.

### F5 — TONEL alignment target was raw noise

Aligning the student to **uncorrected** noisy reconstruction pushes embeddings toward corruption when \(w\) is large. Design intent is alignment to **ECC-calibrated** reconstruction \(\hat e_{\mathrm{cal}}\).

**Fix:** TONEL residual target = soft/BP reconstruction from the channel (no clean blend); weight \(w\) still from drift/gate proxies.

### F6 — Over-claim vs hardware reality (documentation)

| Paper wording | Actual v0/v1.1 |
|---------------|----------------|
| “NOR Flash floating-gate” | Channel is **generic PIM multi-cycle bit-flip** from GRC-PIM (RRAM-oriented paper stack), not a device-physics NOR model |
| “Gated GEMV ΔLLR” | v1.1 uses **B′ + optional BP**; full INT8 GEMV checkpoint path is optional/future |
| “RAG / LLM hallucination rate” | **Retrieval-only** offline metrics; no LLM generation loop yet |
| “<10 ms edge” | Python wall-clock on toy corpus ≠ SoC cycle-accurate latency |

**Fix:** Narrow claims in abstract/design to “PIM/CiM multi-cycle noise (GRC-PIM channel); NOR as target substrate”; label GEMV as future hook; mark latency as software proxy.

---

## 3. Is the design *achievable* and *rigorous*?

**Achievable: yes**, with a narrowed claim:

> Under a multi-cycle PIM noise channel with GF(3) LDPC-protected ternary embeddings, soft residual evidence (B′ / BP) improves dense retrieval robustness versus hard noisy readout; hybrid BM25 is an engineering accelerator, not the scientific core.

**Not yet rigorous enough for camera-ready SIGIR** until:

1. Oracle-free reconstruction (F1) — **blocking**  
2. Real LDPC protect/recover path (F2) — **blocking for the title**  
3. Dense-only ablations (F3) — **blocking for contribution narrative**  
4. Stable encoder + honest claim language (F4–F6) — **blocking for reproducibility / review ethics**

Larger (non-blocking for v1.1 code, blocking for full paper): MSMARCO/NQ, strong dense teachers (BGE/Contriever), true GEMV gate cosim, device-calibrated NOR noise, RAG generation metrics.

---

## 4. Minimal design amendments (approved for Development)

1. Replace oracle blend with **soft LLR expectation** and **BP-decoded** info-symbol reconstruction.  
2. Add `ldpc_embed.py`: pack → encode → noise → B′/BP → unpack.  
3. Eval protocol: mandatory dense-only noise table; hybrid reported separately.  
4. Stable hashing; reflection + claim errata in docs.  
5. Keep GRC GEMV MLP as optional enhancement behind a flag; default path = B′+BP (honest).

---

## 6. v1.1 empirical outcome (after fixes)

**Protocol:** asymmetric RAG (clean digital query); base-3 multi-symbol GF(3) packing; dense-only table; geometric probe.

| Setting | Result |
|---------|--------|
| SyntheticIR dense @ \(p=0.05\) | noisy **0.134** → B′ **0.308** (+0.175 nDCG); soft **0.323** |
| Geometric probe @ \(p=0.03\) | noisy **0.905** → B′ **0.952** (SER 0.086→0.038) |
| Geometric probe @ \(p=0.05\) | noisy **0.858** → B′ **0.947** |
| BP vs B′ on ranking | **Not always better** — BP can over-correct ranking even when SER drops; paper must report both |

**Interpretation:** The design is **achievable and empirically supported** once oracle leakage, centroid collapse, and BM25 confounding are removed. Remaining gaps for camera-ready: strong text encoders, MSMARCO/NQ, optional GEMV gate, NOR device calibration, RAG generation metrics.

---

## 7. Decision (updated)

v1.1 is **scientifically honest enough to continue**. Treat geometric probe + dense-only gains as the core evidence; do not cite v0 hybrid≈BM25 numbers.
