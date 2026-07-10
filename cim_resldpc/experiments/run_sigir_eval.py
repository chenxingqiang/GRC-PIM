"""SIGIR-AP style offline evaluation for CiM-ResLDPC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cim_resldpc.baselines import BM25Baseline, CiMNoisyBaseline, NaiveDenseBaseline
from cim_resldpc.config import CiMResLDPCConfig
from cim_resldpc.datasets import QAPair, Document, build_synthetic_ir, load_semi_corpus
from cim_resldpc.metrics import (
    average_precision,
    mean_metric,
    ndcg_at_k,
    noise_degradation,
    recall_at_k,
)
from cim_resldpc.noise_bridge import NoiseProfile
from cim_resldpc.pipeline import CiMResLDPCPipeline
from cim_resldpc.tonel import TONELResEncoder


def _relevance_vector(hits: list, q: QAPair, graded: bool = True) -> list[float]:
    rel_set = set(q.relevant_doc_ids)
    out = []
    for h in hits:
        doc_id = h.doc_id if hasattr(h, "doc_id") else h["doc_id"]
        if doc_id in rel_set:
            out.append(1.0 if graded else 1)
        else:
            out.append(0.0 if graded else 0)
    return out


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


def _train_encoder(
    enc: TONELResEncoder,
    docs: list[Document],
    queries: list[QAPair],
    steps: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    id2text = {d.doc_id: d.text for d in docs}
    for t in range(steps):
        q = queries[t % len(queries)]
        pos_id = q.relevant_doc_ids[0]
        enc.train_step([q.query], [id2text[pos_id]], rng=rng)


def run_evaluation(
    out_json: Path | str,
    n_topics: int = 6,
    docs_per_topic: int = 3,
    train_steps: int = 20,
    seed: int = 42,
    include_semi: bool = True,
    config: CiMResLDPCConfig | None = None,
) -> dict:
    cfg = config or CiMResLDPCConfig(seed=seed)
    enc = TONELResEncoder(cfg)

    syn_docs, syn_queries = build_synthetic_ir(
        n_topics=n_topics, docs_per_topic=docs_per_topic, seed=seed,
    )
    _train_encoder(enc, syn_docs, syn_queries, steps=train_steps, seed=seed)

    # Proposed pipeline
    pipe = CiMResLDPCPipeline(cfg, encoder=enc)
    pipe.index(syn_docs)
    ours_syn = evaluate_retriever(lambda q, k: pipe.retrieve(q, k=k), syn_queries)

    # Baselines on synthetic
    bm25 = BM25Baseline(syn_docs)
    naive = NaiveDenseBaseline(syn_docs, enc)
    profile = NoiseProfile(
        p_indep=cfg.p_indep,
        p_drift=cfg.p_drift,
        p_row_burst=cfg.p_row_burst,
        n_cycles=cfg.n_cycles,
    )
    noisy = CiMNoisyBaseline(syn_docs, enc, profile, rng=np.random.default_rng(seed))

    baselines = {
        "bm25": evaluate_retriever(lambda q, k: bm25.retrieve(q, k=k), syn_queries),
        "naive_dense": evaluate_retriever(lambda q, k: naive.retrieve(q, k=k), syn_queries),
        "cim_noisy": evaluate_retriever(lambda q, k: noisy.retrieve(q, k=k), syn_queries),
    }

    # Robustness: clean naive vs noisy vs ours
    clean_ndcg = baselines["naive_dense"]["ndcg@10"]
    noisy_ndcg = baselines["cim_noisy"]["ndcg@10"]
    ours_ndcg = ours_syn["ndcg@10"]
    robustness = {
        "naive_degradation": noise_degradation(clean_ndcg, noisy_ndcg),
        "ours_degradation_vs_clean": noise_degradation(clean_ndcg, ours_ndcg),
        "skip_rate": float(np.mean(~pipe.used_gemv_flags)) if pipe.used_gemv_flags is not None else 0.0,
    }

    semi_metrics = {}
    if include_semi:
        semi_docs, semi_queries = load_semi_corpus()
        # Continue light training on semi
        _train_encoder(enc, semi_docs, semi_queries, steps=max(8, train_steps // 2), seed=seed + 1)
        pipe_s = CiMResLDPCPipeline(cfg, encoder=enc)
        pipe_s.index(semi_docs)
        semi_metrics = evaluate_retriever(
            lambda q, k: pipe_s.retrieve(q, k=k), semi_queries,
        )
        bm25_s = BM25Baseline(semi_docs)
        baselines["bm25_semi"] = evaluate_retriever(
            lambda q, k: bm25_s.retrieve(q, k=k), semi_queries,
        )

    result = {
        "config": cfg.as_dict(),
        "synthetic": {
            "n_docs": len(syn_docs),
            "n_queries": len(syn_queries),
        },
        "semi": {
            "n_docs": len(load_semi_corpus()[0]) if include_semi else 0,
            "n_queries": len(load_semi_corpus()[1]) if include_semi else 0,
        },
        "baselines": baselines,
        "cim_resldpc": {
            "synthetic": ours_syn,
            "semi": semi_metrics,
            "robustness": robustness,
        },
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="CiM-ResLDPC SIGIR-AP offline eval")
    p.add_argument(
        "--out_json",
        type=str,
        default="cim_resldpc/results/sigir_eval.json",
    )
    p.add_argument("--n_topics", type=int, default=6)
    p.add_argument("--docs_per_topic", type=int, default=3)
    p.add_argument("--train_steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_semi", action="store_true")
    args = p.parse_args(argv)
    result = run_evaluation(
        out_json=args.out_json,
        n_topics=args.n_topics,
        docs_per_topic=args.docs_per_topic,
        train_steps=args.train_steps,
        seed=args.seed,
        include_semi=not args.no_semi,
    )
    ours = result["cim_resldpc"]["synthetic"]
    print(
        f"CiM-ResLDPC synthetic nDCG@10={ours['ndcg@10']:.4f} "
        f"MAP={ours['map']:.4f} "
        f"latency_ms={ours.get('latency_ms_mean', 0):.3f}"
    )
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
