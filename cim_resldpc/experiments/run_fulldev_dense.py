"""Lean full-MSMARCO-Dev dense-only eval (scientific core only).

Encodes the full cached MSMARCO validation subset with BGE, projects to
cim_dim, and measures nDCG@10 / Recall@10 for clean / noisy / B' / BP under a
NOR preset with an asymmetric (digital) query. Retrieval is vectorized. BM25,
hybrid, and Flan RAG are intentionally skipped (pure-Python O(n_docs*n_q) BM25
and LLM generation are infeasible at ~82k docs on CPU and are not the
dense-only scientific core).

Reproduce (from repo root):
  python3 -m cim_resldpc.experiments.run_fulldev_dense \
      --n_queries 10000 --nor nor_typical --cim_dim 128 --seed 42 \
      --out_json cim_resldpc/results/sigir_eval_v26_bge_msmarco_fulldev.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cim_resldpc.gf3_quantize import quantize_base3, dequantize_base3
from cim_resldpc.ldpc_embed import protect_and_recover
from cim_resldpc.metrics import ndcg_at_k, average_precision, recall_at_k, mean_metric
from cim_resldpc.noise_bridge import apply_embedding_cim_noise
from cim_resldpc.nor_calibration import get_nor_profile


def _norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _eval(q_emb, d_emb, q_rel, doc_ids, k=10, batch=256):
    d = _norm(d_emb.astype(np.float32))
    q = _norm(q_emb.astype(np.float32))
    id2i = {did: i for i, did in enumerate(doc_ids)}
    rel_idx = [set(id2i[r] for r in rels if r in id2i) for rels in q_rel]
    ndcgs, aps, recs = [], [], []
    for s in range(0, len(q), batch):
        sims = q[s:s + batch] @ d.T
        top = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        for r, row in enumerate(top):
            order = row[np.argsort(-sims[r, row])]
            rels = [1.0 if j in rel_idx[s + r] else 0.0 for j in order]
            ndcgs.append(ndcg_at_k(rels, k))
            aps.append(average_precision([int(x) for x in rels]))
            recs.append(recall_at_k([int(x) for x in rels], n_relevant=len(rel_idx[s + r]), k=k))
    summary = {"ndcg@10": mean_metric(ndcgs), "map": mean_metric(aps),
               "recall@10": mean_metric(recs), "n_queries": float(len(q))}
    return summary, np.asarray(ndcgs, dtype=np.float64)


def _paired_sig(a, b, n_boot=10000, seed=0):
    """Paired significance of per-query nDCG improvement a-b (a=method, b=baseline)."""
    from scipy import stats

    diff = a - b
    t, p = stats.ttest_rel(a, b)
    wsp = stats.wilcoxon(a, b).pvalue if np.any(diff != 0) else 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_delta": float(diff.mean()), "t": float(t), "p_ttest": float(p),
            "p_wilcoxon": float(wsp), "ci95": [float(lo), float(hi)], "n": int(len(diff))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_queries", type=int, default=10000)
    ap.add_argument("--nor", default="nor_typical")
    ap.add_argument("--cim_dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--st_model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args(argv)

    cache = Path(__file__).resolve().parents[1] / "data" / "cache" / f"msmarco_hf_n{args.n_queries}.json"
    raw = json.loads(cache.read_text())
    docs, queries = raw["documents"], raw["queries"]
    doc_ids = [d["doc_id"] for d in docs]
    q_rel = [q["relevant_doc_ids"] for q in queries]
    print(f"loaded {len(docs)} docs / {len(queries)} queries", flush=True)

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(args.st_model)
    t = time.time()
    d_raw = m.encode([d["text"] for d in docs], batch_size=64, show_progress_bar=False).astype(np.float32)
    q_raw = m.encode([q["query"] for q in queries], batch_size=64, show_progress_bar=False).astype(np.float32)
    print(f"encoded in {time.time()-t:.0f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    P = rng.normal(size=(d_raw.shape[1], args.cim_dim)).astype(np.float32) / np.sqrt(d_raw.shape[1])
    d_clean = np.clip((d_raw @ P) / (np.abs(d_raw @ P).max() + 1e-8), -1, 1)
    q_emb = np.clip((q_raw @ P) / (np.abs(q_raw @ P).max() + 1e-8), -1, 1)

    prof = get_nor_profile(args.nor)
    stored = quantize_base3(d_clean, n_sym_per_dim=2)
    read = apply_embedding_cim_noise(stored, prof, rng=np.random.default_rng(1))
    d_noisy = dequantize_base3(read.recovered_symbols, n_sym_per_dim=2, dim=args.cim_dim)
    t = time.time()
    rec = protect_and_recover(d_clean, prof, rng=np.random.default_rng(1), use_bp=True,
                              n_sym_per_dim=2, quant_mode="base3")
    print(f"recovered in {time.time()-t:.0f}s", flush=True)

    res, pq = {}, {}
    for name, demb in [("dense_clean", d_clean), ("cim_noisy_no_ldpc", d_noisy),
                       ("cim_ldpc_bprime", rec.emb_bprime), ("cim_ldpc_bp", rec.emb_bp)]:
        t = time.time()
        res[name], pq[name] = _eval(q_emb, demb, q_rel, doc_ids)
        print(f"  {name:22s} nDCG@10={res[name]['ndcg@10']:.4f}  ({time.time()-t:.0f}s)", flush=True)

    sig = {
        "bp_vs_noisy": _paired_sig(pq["cim_ldpc_bp"], pq["cim_noisy_no_ldpc"]),
        "bprime_vs_noisy": _paired_sig(pq["cim_ldpc_bprime"], pq["cim_noisy_no_ldpc"]),
    }
    for k2, v in sig.items():
        print(f"  sig {k2}: delta={v['mean_delta']:.4f} p_t={v['p_ttest']:.2e} "
              f"CI95=[{v['ci95'][0]:.4f},{v['ci95'][1]:.4f}]", flush=True)

    out = {"protocol_version": "v2.6", "encoder": "bge", "st_model": args.st_model,
           "corpus": "msmarco_hf_fulldev", "n_docs": len(docs), "n_queries": len(queries),
           "nor": args.nor, "cim_dim": args.cim_dim, "seed": args.seed,
           "notes": "Full MSMARCO v1.1 validation (~10k q). Dense-only scientific core "
                    "(asymmetric digital query); BM25/hybrid/RAG skipped (infeasible at scale). "
                    "Per-query paired significance (t-test, Wilcoxon, bootstrap 95% CI) of the "
                    "nDCG@10 improvement over the noisy baseline.",
           "dense_only": res,
           "significance": sig,
           "robustness": {"bp_gain_vs_noisy": res["cim_ldpc_bp"]["ndcg@10"] - res["cim_noisy_no_ldpc"]["ndcg@10"]}}
    Path(args.out_json).write_text(json.dumps(out, indent=1))
    print("wrote", args.out_json, flush=True)


if __name__ == "__main__":
    main()
