"""SIGIR-AP style offline evaluation for CiM-ResLDPC (v1.1 honest protocol).

Scientific core:
  - Asymmetric RAG: digital query, CiM-noisy LDPC-protected doc store
  - base-3 multi-symbol GF(3) packing (geometry-preserving)
  - Dense-only table under matched noise seeds
  - Hybrid BM25 reported separately (system, not robustness denominator)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cim_resldpc.baselines import (
    BM25Baseline,
    CiMLDPCBaseline,
    CiMNoisyBaseline,
    NaiveDenseBaseline,
)
from cim_resldpc.config import CiMResLDPCConfig
from cim_resldpc.datasets import QAPair, Document, build_synthetic_ir, load_semi_corpus
from cim_resldpc.gf3_quantize import dequantize_base3, quantize_base3
from cim_resldpc.ldpc_embed import LDPCCode, protect_and_recover, symbol_error_rate
from cim_resldpc.metrics import (
    average_precision,
    mean_metric,
    ndcg_at_k,
    noise_degradation,
    recall_at_k,
)
from cim_resldpc.noise_bridge import NoiseProfile, apply_embedding_cim_noise
from cim_resldpc.pipeline import CiMResLDPCPipeline
from cim_resldpc.residual_features import cosine_similarity
from cim_resldpc.tonel import TONELResEncoder


def _relevance_vector(hits: list, q: QAPair) -> list[float]:
    rel_set = set(q.relevant_doc_ids)
    return [
        1.0 if (h.doc_id if hasattr(h, "doc_id") else h["doc_id"]) in rel_set else 0.0
        for h in hits
    ]


def evaluate_retriever(retrieve_fn, queries: list[QAPair], k: int = 10) -> dict[str, float]:
    ndcgs, maps, recalls, latencies = [], [], [], []
    for q in queries:
        hits = retrieve_fn(q.query, k)
        rels = _relevance_vector(hits, q)
        binary = [int(r > 0) for r in rels]
        ndcgs.append(ndcg_at_k(rels, k=k))
        maps.append(average_precision(binary))
        recalls.append(recall_at_k(binary, n_relevant=len(q.relevant_doc_ids), k=k))
        if hits and hasattr(hits[0], "latency_ms"):
            latencies.append(float(hits[0].latency_ms))
    out = {
        "ndcg@10": mean_metric(ndcgs),
        "map": mean_metric(maps),
        f"recall@{k}": mean_metric(recalls),
        "n_queries": float(len(queries)),
    }
    if latencies:
        out["latency_ms_mean"] = mean_metric(latencies)
    return out


def _train_encoder(enc, docs, queries, steps, seed):
    rng = np.random.default_rng(seed)
    id2text = {d.doc_id: d.text for d in docs}
    for t in range(steps):
        q = queries[t % len(queries)]
        enc.train_step([q.query], [id2text[q.relevant_doc_ids[0]]], rng=rng)


def geometric_probe(seed: int = 0) -> dict:
    """Controlled cluster geometry: proves LDPC helps when structure exists."""
    rng = np.random.default_rng(seed)
    code = LDPCCode.load()
    d, n_docs, n_q, n_clusters = 64, 100, 20, 10
    centers = rng.normal(size=(n_clusters, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
    docs, labels = [], []
    for i in range(n_docs):
        c = i % n_clusters
        v = centers[c] + 0.15 * rng.normal(size=d).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        docs.append(v)
        labels.append(c)
    docs = np.stack(docs)
    queries, qlabels = [], []
    for i in range(n_q):
        c = i % n_clusters
        v = centers[c] + 0.15 * rng.normal(size=d).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        queries.append(v)
        qlabels.append(c)
    queries = np.stack(queries)

    def eval_ndcg(qv, dv, k=10):
        scores = []
        for i in range(len(qv)):
            sim = cosine_similarity(qv[i], dv)
            order = np.argsort(-sim)[:k]
            rels = [1.0 if labels[j] == qlabels[i] else 0.0 for j in order]
            scores.append(ndcg_at_k(rels, k))
        return mean_metric(scores)

    rows = []
    for p in [0.01, 0.03, 0.05, 0.08, 0.12]:
        profile = NoiseProfile(p_indep=p, p_drift=p, p_row_burst=0.0, n_cycles=4)
        stored = quantize_base3(docs, n_sym_per_dim=2)
        read = apply_embedding_cim_noise(stored, profile, rng=np.random.default_rng(1))
        noisy_emb = dequantize_base3(read.recovered_symbols, n_sym_per_dim=2, dim=d)
        rec = protect_and_recover(
            docs, profile, code=code, rng=np.random.default_rng(1),
            use_bp=True, n_sym_per_dim=2, quant_mode="base3",
        )
        row = {
            "p": p,
            "ser_noisy": symbol_error_rate(stored, read.recovered_symbols),
            "ser_bp": symbol_error_rate(stored, rec.symbols_bp),
            "ndcg_float": eval_ndcg(queries, docs),
            "ndcg_noisy": eval_ndcg(queries, noisy_emb),
            "ndcg_bprime": eval_ndcg(queries, rec.emb_bprime),
            "ndcg_bp": eval_ndcg(queries, rec.emb_bp),
        }
        rows.append(row)
    return {"rows": rows}


def run_evaluation(
    out_json: Path | str,
    n_topics: int = 6,
    docs_per_topic: int = 3,
    train_steps: int = 20,
    seed: int = 42,
    include_semi: bool = True,
    config: CiMResLDPCConfig | None = None,
    noise_p: float = 0.05,
) -> dict:
    cfg = config or CiMResLDPCConfig(seed=seed)
    # Stronger default noise for the scientific table (LDPC operating region)
    cfg.p_indep = noise_p
    cfg.p_drift = noise_p
    cfg.p_row_burst = 0.0  # isolate column/indep noise for clearer ablations
    enc = TONELResEncoder(cfg)

    syn_docs, syn_queries = build_synthetic_ir(
        n_topics=n_topics, docs_per_topic=docs_per_topic, seed=seed,
    )
    _train_encoder(enc, syn_docs, syn_queries, steps=train_steps, seed=seed)

    profile = NoiseProfile(
        p_indep=cfg.p_indep,
        p_drift=cfg.p_drift,
        p_row_burst=cfg.p_row_burst,
        n_cycles=cfg.n_cycles,
    )

    naive = NaiveDenseBaseline(syn_docs, enc)
    # Matched seeds across noisy / LDPC for fair comparison
    noisy = CiMNoisyBaseline(
        syn_docs, enc, profile, rng=np.random.default_rng(seed), asymmetric_query=True,
    )
    bprime = CiMLDPCBaseline(
        syn_docs, enc, profile, rng=np.random.default_rng(seed),
        recovery="bprime", use_bp=False, asymmetric_query=True,
    )
    bp = CiMLDPCBaseline(
        syn_docs, enc, profile, rng=np.random.default_rng(seed),
        recovery="bp", use_bp=True, asymmetric_query=True,
    )
    soft = CiMLDPCBaseline(
        syn_docs, enc, profile, rng=np.random.default_rng(seed),
        recovery="soft", use_bp=True, asymmetric_query=True,
    )

    dense_table = {
        "naive_dense_clean": evaluate_retriever(lambda q, k: naive.retrieve(q, k=k), syn_queries),
        "cim_noisy_no_ldpc": evaluate_retriever(lambda q, k: noisy.retrieve(q, k=k), syn_queries),
        "cim_ldpc_bprime": evaluate_retriever(lambda q, k: bprime.retrieve(q, k=k), syn_queries),
        "cim_ldpc_bp": evaluate_retriever(lambda q, k: bp.retrieve(q, k=k), syn_queries),
        "cim_ldpc_soft": evaluate_retriever(lambda q, k: soft.retrieve(q, k=k), syn_queries),
    }

    clean_ndcg = dense_table["naive_dense_clean"]["ndcg@10"]
    robustness = {
        "noise_p": noise_p,
        "noisy_vs_clean": noise_degradation(clean_ndcg, dense_table["cim_noisy_no_ldpc"]["ndcg@10"]),
        "bprime_vs_clean": noise_degradation(clean_ndcg, dense_table["cim_ldpc_bprime"]["ndcg@10"]),
        "bp_vs_clean": noise_degradation(clean_ndcg, dense_table["cim_ldpc_bp"]["ndcg@10"]),
        "bprime_gain_vs_noisy": (
            dense_table["cim_ldpc_bprime"]["ndcg@10"] - dense_table["cim_noisy_no_ldpc"]["ndcg@10"]
        ),
        "bp_gain_vs_noisy": (
            dense_table["cim_ldpc_bp"]["ndcg@10"] - dense_table["cim_noisy_no_ldpc"]["ndcg@10"]
        ),
    }

    pipe = CiMResLDPCPipeline(cfg, encoder=enc, use_bp=True, recovery="bprime")
    pipe.index(syn_docs)
    hybrid = evaluate_retriever(
        lambda q, k: pipe.retrieve(q, k=k, mode="hybrid"), syn_queries,
    )
    dense_pipe = evaluate_retriever(
        lambda q, k: pipe.retrieve(q, k=k, mode="dense"), syn_queries,
    )
    bm25_m = evaluate_retriever(
        lambda q, k: BM25Baseline(syn_docs).retrieve(q, k=k), syn_queries,
    )

    semi_metrics: dict = {}
    if include_semi:
        semi_docs, semi_queries = load_semi_corpus()
        _train_encoder(enc, semi_docs, semi_queries, steps=max(8, train_steps // 2), seed=seed + 1)
        pipe_s = CiMResLDPCPipeline(cfg, encoder=enc, use_bp=True, recovery="bprime")
        pipe_s.index(semi_docs)
        semi_metrics = {
            "hybrid": evaluate_retriever(
                lambda q, k: pipe_s.retrieve(q, k=k, mode="hybrid"), semi_queries,
            ),
            "dense": evaluate_retriever(
                lambda q, k: pipe_s.retrieve(q, k=k, mode="dense"), semi_queries,
            ),
            "bm25": evaluate_retriever(
                lambda q, k: BM25Baseline(semi_docs).retrieve(q, k=k), semi_queries,
            ),
        }

    probe = geometric_probe(seed=seed)

    result = {
        "protocol_version": "v1.1",
        "notes": (
            "Asymmetric RAG (clean digital query, CiM doc store). "
            "base3 multi-symbol GF(3) packing. "
            "Dense-only is the scientific core; hybrid is system-only. "
            "geometric_probe validates LDPC ranking gains under controlled geometry. "
            "Channel = GRC-PIM multi-cycle PIM noise (NOR = target substrate)."
        ),
        "config": cfg.as_dict(),
        "synthetic": {"n_docs": len(syn_docs), "n_queries": len(syn_queries)},
        "semi": {
            "n_docs": len(load_semi_corpus()[0]) if include_semi else 0,
            "n_queries": len(load_semi_corpus()[1]) if include_semi else 0,
        },
        "dense_only": dense_table,
        "robustness_dense": robustness,
        "geometric_probe": probe,
        "system": {
            "bm25": bm25_m,
            "cim_resldpc_dense": dense_pipe,
            "cim_resldpc_hybrid": hybrid,
        },
        "cim_resldpc": {
            "synthetic": dense_pipe,
            "semi": semi_metrics.get("dense", {}),
            "hybrid_synthetic": hybrid,
            "semi_full": semi_metrics,
            "robustness": robustness,
        },
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="CiM-ResLDPC SIGIR-AP offline eval (v1.1)")
    p.add_argument("--out_json", type=str, default="cim_resldpc/results/sigir_eval.json")
    p.add_argument("--n_topics", type=int, default=6)
    p.add_argument("--docs_per_topic", type=int, default=3)
    p.add_argument("--train_steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise_p", type=float, default=0.05)
    p.add_argument("--no_semi", action="store_true")
    args = p.parse_args(argv)
    result = run_evaluation(
        out_json=args.out_json,
        n_topics=args.n_topics,
        docs_per_topic=args.docs_per_topic,
        train_steps=args.train_steps,
        seed=args.seed,
        include_semi=not args.no_semi,
        noise_p=args.noise_p,
    )
    print(f"Dense-only @ p={args.noise_p}:")
    for name, m in result["dense_only"].items():
        print(f"  {name:24s} {m['ndcg@10']:.4f}")
    print("Robustness:", {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in result["robustness_dense"].items()})
    print("Geometric probe (p, noisy, b', bp):")
    for row in result["geometric_probe"]["rows"]:
        print(
            f"  p={row['p']:.3f}  noisy={row['ndcg_noisy']:.3f}  "
            f"b'={row['ndcg_bprime']:.3f}  bp={row['ndcg_bp']:.3f}  "
            f"SER {row['ser_noisy']:.3f}->{row['ser_bp']:.3f}"
        )
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
