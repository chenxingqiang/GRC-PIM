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
