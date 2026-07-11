# CiM-ResLDPC Design Document (SIGIR-AP 2026)

**Phase:** Design (v1.1 — amended after scientific reflection)  
**Title (EN):** CiM-ResLDPC: Residual Error Calibrated Robust Retrieval-Augmented Generation on NOR Flash Compute-in-Memory Arrays  
**Title (CN):** CiM-ResLDPC：面向NOR浮栅存算阵列的残差纠错鲁棒边缘检索增强生成  
**Base stack:** GRC-PIM (`pim_predecoder`) — multi-cycle PIM noise + B′ soft-LLR + GF(3) NB-LDPC BP (+ optional gated GEMV)  
**Repo:** GRC-PIM-IR  
**Claim scope (v1.1):** Channel = GRC-PIM multi-cycle PIM noise (NOR is the *target substrate*, not a fitted device model). Default residual path = B′ soft / BP — not clean-embedding oracle. Full INT8 GEMV gate is optional future hook. See `SCIENTIFIC_REFLECTION.md`.

---

## 1. Problem Statement

Edge RAG on NOR Flash CiM arrays faces three coupled failures:

1. **Hardware noise ↔ retrieval representation gap** — conductance drift, read disturb, quantization, and analog MAC residual errors corrupt embedding vectors; classical denoisers ignore multi-cycle soft evidence and LDPC residual soft information.
2. **Semantic collapse under GF(3) / bit-width constraints** — dense float embeddings do not map cleanly to ternary storage; naive compression destroys long-tail technical recall.
3. **Privacy + latency conflict** — local-first retrieval must stay on-device (<10 ms) while residual decoding is expensive without gating.

**Goal:** Fuse GRC-PIM residual soft information into dense retrieval representation learning and the retrieval–rerank pipeline (hardware-noise-aware RAG), not merely accelerate similarity with CiM MAC.

---

## 2. System Overview

```
Documents / Query
        │
        ▼
┌───────────────────┐
│  TONEL-Res Encoder │  residual-regularized dense embedding
│  (GF(3)-aware)     │
└─────────┬─────────┘
          │ store / query vectors (quantized to GF(3) symbols)
          ▼
┌───────────────────┐
│  NOR CiM Array     │  multi-cycle MAC readout (noise3d channel)
│  + GRC-PIM ECC     │  B′ + gated ΔLLR → GF(3) BP
└─────────┬─────────┘
          │ residual soft evidence (gate, ΔLLR mag, drift score)
          ▼
┌───────────────────┐
│  Residual-aware    │  hybrid BM25 + dense score + residual confidence
│  Retrieve + Rerank │  gated decode prune for low-residual docs
└─────────┬─────────┘
          │ top-k denoised passages
          ▼
     Edge LLM generate (local, no raw doc upload)
```

### Three-layer co-design

| Layer | Responsibility | Reuses |
|-------|----------------|--------|
| Hardware | Multi-cycle NOR/PIM readout, gated residual LDPC | `core/noise3d.py`, `core/gemv_calibrator.py`, `core/codec.py` |
| Representation | TONEL-Res embedding + GF(3) quantization | new `cim_resldpc/tonel.py` |
| Retrieval | CiM similarity + residual-aware hybrid rerank | new `cim_resldpc/pipeline.py` |

---

## 3. Contribution Specs

### C1 — TONEL-Res (residual-aware embedding distillation)

**Inputs**

- Clean embedding target \(e \in \mathbb{R}^d\) (teacher / float encoder).
- Multi-cycle noisy CiM reads of quantized embedding symbols.
- GRC-PIM outputs: \(\mathrm{LLR}_{B'}\), \(\Delta\mathrm{LLR}\), gate \(g\), drift \(d_{\mathrm{global}}\).

**Outputs**

- Student embedding \(e_\theta\) trained with residual soft-info regularizer.
- GF(3)-quantized code symbols for CiM storage.

**Loss**

\[
\mathcal{L} = \mathcal{L}_{\mathrm{retr}}(e_\theta; q, d^+)
  + \lambda_{\mathrm{res}}\, w(d_{\mathrm{global}}, g)\, \mathcal{L}_{\mathrm{align}}(e_\theta, \hat e_{\mathrm{cal}})
  + \lambda_{\mathrm{gf3}}\, \mathcal{L}_{\mathrm{quant}}(e_\theta)
\]

- \(\mathcal{L}_{\mathrm{retr}}\): InfoNCE / contrastive retrieval loss (synthetic corpus in v1).
- \(\mathcal{L}_{\mathrm{align}}\): MSE / cosine between student and residual-calibrated reconstructed embedding.
- \(w(\cdot)\): dynamic gate — higher weight on high-drift / high-gate blocks.
- \(\mathcal{L}_{\mathrm{quant}}\): soft ternary quantization penalty toward GF(3) levels.

**API**

```python
class TONELResEncoder(nn.Module):
    def encode(texts: list[str]) -> Tensor          # (B, d)
    def quantize_gf3(emb: Tensor) -> np.ndarray     # (B, n_sym) in {0,1,2}
    def residual_regularizer(...) -> Tensor
    def train_step(batch) -> dict[str, float]
```

### C2 — End-to-end CiM-ResLDPC retrieval pipeline

**Stages**

1. Encode query / docs with TONEL-Res; store GF(3) symbols in simulated CiM.
2. Query–doc similarity via noisy CiM GEMV (dot-product under `apply_pim_noise_3d` on stored codes).
3. Optional full GRC-PIM decode when residual gate opens; else skip (latency prune).
4. Rerank: \(\mathrm{score} = \alpha\,\mathrm{BM25} + \beta\,\mathrm{dense}_{\mathrm{cal}} + \gamma\,\mathrm{residual\_conf}\).

**API**

```python
class CiMResLDPCPipeline:
    def index(docs: list[Document]) -> None
    def retrieve(query: str, k: int = 10) -> list[ScoredDoc]
    def retrieve_with_metrics(query: str, k: int) -> RetrieveResult
```

### C3 — Edge real-time optimizations

1. **Residual decode prune:** if `composite_drift < τ_skip` and mean gate low → skip GEMV path (B′ only or majority).
2. **Hybrid sparse+dense:** BM25 first-stage cut to `M` candidates, dense+residual only on `M`.
3. **Local-only flag:** pipeline never serializes raw docs off-device (enforced by API contract / unit test).

---

## 4. Module Map (new package `cim_resldpc/`)

```
cim_resldpc/
├── __init__.py
├── config.py              # defaults: dim, τ_d, τ_g, τ_skip, λ_res, hybrid weights
├── noise_bridge.py        # wrap GRC-PIM noise + drift scores for embeddings
├── gf3_quantize.py        # float embedding ↔ GF(3) symbol packing
├── tonel.py               # TONEL-Res encoder + losses
├── residual_features.py   # extract residual soft evidence for rerank
├── hybrid_score.py        # BM25 + dense + residual confidence fusion
├── pipeline.py            # end-to-end index / retrieve
├── metrics.py             # nDCG, MAP, Recall@k, latency, noise degradation
├── datasets.py            # synthetic IR + semiconductor mini-corpus
├── baselines.py           # NaiveRAG, noisy CiM without residual, BM25-only
├── experiments/
│   └── run_sigir_eval.py  # main evaluation entry
├── tests/                 # unit + integration tests
└── results/               # JSON experiment outputs
```

**Dependency rule:** `cim_resldpc` imports `pim_predecoder.core.*`; it must not modify GRC-PIM ECC math unless a bug is proven.

---

## 5. Data Interfaces

### Document

```python
@dataclass
class Document:
    doc_id: str
    text: str
    labels: list[str] = field(default_factory=list)  # topic tags for synthetic eval
```

### ScoredDoc / RetrieveResult

```python
@dataclass
class ScoredDoc:
    doc_id: str
    score: float
    dense_score: float
    bm25_score: float
    residual_conf: float
    used_gemv: bool
    latency_ms: float

@dataclass
class RetrieveResult:
    query: str
    hits: list[ScoredDoc]
    metrics: dict[str, float]  # optional per-query
```

### Hardware noise profile (from GRC-PIM)

| Field | Source |
|-------|--------|
| `p_indep`, `p_drift`, `p_row_burst`, `n_cycles` | `apply_pim_noise_3d` |
| `drift_score`, `composite` | `enhanced_drift_scores` |
| `use_gemv`, `gate_mean`, `delta_mag` | GEMV calibrator diagnose / soft_llr |

---

## 6. Evaluation Plan (SIGIR-style, v1 runnable)

### Datasets (v1)

| Name | Role |
|------|------|
| `SyntheticIR` | Controlled relevance labels; bag-of-topics embeddings |
| `SemiCorpusMini` | Semiconductor / chip handbook snippets (bundled text) |
| Noise sweep | E1–E5-like drift strengths via `noise3d` params |

Full MSMARCO/NQ downloads are optional later; v1 must run offline with bundled data.

### Baselines

| ID | Description |
|----|-------------|
| `bm25` | Sparse only |
| `naive_dense` | Float dense, no CiM noise |
| `cim_noisy` | Dense on CiM without residual calibration |
| `cim_bp` | CiM + B′ / majority only |
| `cim_resldpc` | Full proposed pipeline |

### Metrics

- IR: nDCG@5/10, MAP, Recall@k  
- Robustness: ΔnDCG under noise vs clean  
- Hardware: mean retrieve latency_ms, decode skip rate, estimated energy proxy (MAC cycles)  
- RAG proxy: answer F1 on synthetic QA pairs (optional lightweight)

---

## 7. Test Plan Draft (executable in Development)

| ID | Case | Expected |
|----|------|----------|
| T1 | GF(3) quantize/dequantize roundtrip | reconstruction error bounded; symbols ∈ {0,1,2} |
| T2 | Noise bridge applies 3D PIM noise to packed embeddings | output shape `(B,T,2n)`; drift score ∈ [0,1] |
| T3 | Residual regularizer weight increases with drift | \(w\) monotone in synthetic high vs low drift |
| T4 | TONEL train_step returns finite losses | all loss keys finite, non-NaN |
| T5 | Hybrid score fusion ranks relevant doc above random | on SyntheticIR toy set |
| T6 | Decode prune skips GEMV when drift < τ_skip | `used_gemv=False` |
| T7 | Pipeline local-only: no network / no upload API | unit assert on interface flags |
| T8 | Metrics: nDCG@10 in [0,1]; noise degrades naive more than ResLDPC | relative robustness |
| T9 | End-to-end `run_sigir_eval` writes JSON with required keys | schema check |
| T10 | Import path: `cim_resldpc` uses GRC-PIM without circular imports | import smoke |

---

## 8. Defaults (config)

| Key | Default | Notes |
|-----|---------|-------|
| `embed_dim` | 64 | toy / edge-friendly; paper can scale to 384 |
| `n_cycles` | 4 | match GRC-PIM T=4 |
| `tau_d` | 0.025 | drift gate |
| `tau_g` | 0.6 | model confidence |
| `tau_skip` | 0.015 | prune below this composite/global drift |
| `lambda_res` | 0.5 | residual align weight |
| `lambda_gf3` | 0.1 | quantization weight |
| `alpha,beta,gamma` | 0.3, 0.5, 0.2 | BM25 / dense / residual |
| `bm25_cutoff_m` | 32 | hybrid first stage |
| `p_indep` | 5e-3 | default noise |
| `p_drift` | 5e-3 | |
| `p_row_burst` | 5e-3 | |

---

## 9. Paper Narrative (for Related Work / Intro)

Edge local RAG is required for privacy → NOR Flash CiM is the capacity-efficient substrate → floating-gate noise destroys nDCG → prior CiM-RAG (RoCR, CiMRAG) only accelerate MAC with weak noise masks → we inject GF(3) LDPC residual soft evidence into embedding learning and gated retrieval (TONEL-Res + CiM-ResLDPC) → dual wins on IR metrics and hardware latency/energy.

---

## 10. Out of Scope (v1)

- Full MSMARCO/NQ training of large encoders  
- Real NOR silicon tape-out (reuse GRC-PIM RTL cosim as hardware evidence)  
- Production LLM serving (RAG generation stub / extractive answer proxy)  
- Changing GRC-PIM ECC equations

---

## 11. Approval Checkpoint

Design is complete and ready for Development: implement tests T1–T10 first, then modules in §4, then run evaluation JSON for Testing phase.
