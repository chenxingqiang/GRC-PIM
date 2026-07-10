# CiM-ResLDPC

Residual error–calibrated robust **edge RAG** on NOR Flash / PIM compute-in-memory, built on the GRC-PIM gated residual stack (`pim_predecoder`).

**Paper target:** SIGIR-AP 2026 Full Research Paper  
**English title:** *CiM-ResLDPC: Residual Error Calibrated Robust Retrieval-Augmented Generation on NOR Flash Compute-in-Memory Arrays*

## Quick start

```bash
# from repository root
export PATH="$HOME/.local/bin:$PATH"
pip install numpy torch pytest --index-url https://download.pytorch.org/whl/cpu  # or CPU/GPU torch of choice
pip install numpy pytest  # if using default PyPI for pytest

python -m pytest cim_resldpc/tests -v
python -m cim_resldpc.experiments.run_sigir_eval --out_json cim_resldpc/results/sigir_eval.json
```

## Layout

| Path | Role |
|------|------|
| `tonel.py` | TONEL-Res residual-aware embedding learning |
| `noise_bridge.py` | GRC-PIM multi-cycle noise + drift gating bridge |
| `gf3_quantize.py` | Float ↔ GF(3) ternary quantization |
| `pipeline.py` | End-to-end hybrid retrieve + residual rerank |
| `datasets.py` | Synthetic IR + semiconductor mini-corpus |
| `experiments/run_sigir_eval.py` | Offline nDCG/MAP/latency eval |
| `../docs/design/CIM_RESLDPC_DESIGN.md` | Design spec |
| `../docs/paper/` | Abstract, related work, experiment plan |

## Relation to GRC-PIM

CiM-ResLDPC **does not replace** ECC. It consumes multi-cycle soft evidence and gated residual signals from GRC-PIM and feeds them into retrieval representation learning and reranking—closing the hardware-noise ↔ IR representation gap.

## v1.2 extensions

| Piece | Module | Purpose |
|-------|--------|---------|
| Strong offline encoder | `encoders.py` (`tfidf_svd`) | IR-credible dense vectors without HF download |
| MSMARCO/NQ mini | `data/*.json`, `ir_benchmarks.py` | Standard-schema IR fixtures (+ optional HF loaders) |
| Optional GRC GEMV | `gemv_hook.py` → `ldpc_embed.use_gemv` | Real gated ΔLLR before BP |
| NOR calibration presets | `nor_calibration.py` | Named stress maps onto GRC-PIM channel |
| RAG generation proxy | `rag_eval.py` | Extractive F1 / support / hallucination rate |

```bash
PYTHONPATH=. python3 -m cim_resldpc.experiments.run_extended_eval \
  --encoder tfidf_svd --nor nor_aged --out_json cim_resldpc/results/sigir_eval_extended.json
```

## v1.3 HuggingFace + MiniLM

```bash
pip install -r cim_resldpc/requirements-ir.txt
PYTHONPATH=. python3 -m cim_resldpc.experiments.build_hf_cache --n_queries 48
PYTHONPATH=. python3 -m cim_resldpc.experiments.run_hf_eval --encoder st --nor nor_typical
```

Report: [`../docs/paper/V13_HF_MINILM_REPORT.md`](../docs/paper/V13_HF_MINILM_REPORT.md)

## v1.4 Scaled (200q) + BGE + Flan RAG

```bash
PYTHONPATH=. python3 -m cim_resldpc.experiments.build_hf_cache --n_queries 200
PYTHONPATH=. python3 -m cim_resldpc.experiments.run_scaled_eval --encoder bge --n_queries 200 --corpora msmarco_hf
PYTHONPATH=. python3 -m cim_resldpc.experiments.run_scaled_eval --rag_generator flan --n_queries 48 --corpora msmarco_hf
```

Report: [`../docs/paper/V14_SCALED_BGE_FLAN_REPORT.md`](../docs/paper/V14_SCALED_BGE_FLAN_REPORT.md)

## v1.5 Ablations + paper drafts

```bash
PYTHONPATH=. python3 -m cim_resldpc.experiments.run_ablation_v15 --encoder bge --n_queries 200
```

- [`../docs/paper/V15_ABLATION_REPORT.md`](../docs/paper/V15_ABLATION_REPORT.md)
- [`../docs/paper/EXPERIMENTS.md`](../docs/paper/EXPERIMENTS.md)
- [`../docs/paper/METHOD.md`](../docs/paper/METHOD.md)
