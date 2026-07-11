"""v1.8 focused paper-main eval on larger MSMARCO subsets.

Runs a compact grid (not full NOR×dim×seed) suitable for 2k+ queries:
  - seed sweep @ nor_typical / cim_dim=128
  - dim=256 @ seed=42 (stability spot-check)
  - NOR fresh/aged @ dim=128 / seed=42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cim_resldpc.experiments.run_ablation_v15 import (
    cache_teacher_embeddings,
    load_emb_cache,
    run_condition,
)
from cim_resldpc.nor_calibration import list_nor_presets


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="msmarco_hf")
    p.add_argument("--n_queries", type=int, default=2000)
    p.add_argument("--encoder", default="bge", choices=["bge", "st", "tfidf_svd"])
    p.add_argument(
        "--emb_cache",
        default="cim_resldpc/data/cache/bge_msmarco_n2000_emb.npz",
    )
    p.add_argument(
        "--out_json",
        default="cim_resldpc/results/sigir_eval_v18_msmarco2000.json",
    )
    p.add_argument("--seeds", default="0,1,2,42")
    p.add_argument("--force_encode", action="store_true")
    args = p.parse_args(argv)

    cache_path = Path(args.emb_cache)
    if args.force_encode or not cache_path.exists():
        print(f"Encoding {args.n_queries}q with {args.encoder}...", flush=True)
        cache_teacher_embeddings(args.corpus, args.n_queries, args.encoder, cache_path)
    cache = load_emb_cache(cache_path)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    seed_rows = []
    for seed in seeds:
        print(f"seed={seed} dim=128 nor_typical...", flush=True)
        row = run_condition(cache, "nor_typical", cim_dim=128, seed=seed, include_hybrid=True)
        print(
            f"  noisy={row['dense_only']['cim_noisy_no_ldpc']['ndcg@10']:.4f} "
            f"bp={row['dense_only']['cim_ldpc_bp']['ndcg@10']:.4f} "
            f"gain={row['robustness']['bp_gain_vs_noisy']:.4f} "
            f"hybrid={row['dense_only']['hybrid_bm25_bp']['ndcg@10']:.4f}",
            flush=True,
        )
        seed_rows.append(row)

    print("dim=256 seed=42 nor_typical...", flush=True)
    dim256 = run_condition(cache, "nor_typical", cim_dim=256, seed=42, include_hybrid=True)
    print(
        f"  noisy={dim256['dense_only']['cim_noisy_no_ldpc']['ndcg@10']:.4f} "
        f"bp={dim256['dense_only']['cim_ldpc_bp']['ndcg@10']:.4f} "
        f"gain={dim256['robustness']['bp_gain_vs_noisy']:.4f}",
        flush=True,
    )

    nor_rows = []
    for nor in ("nor_fresh", "nor_aged"):
        print(f"{nor} dim=128 seed=42...", flush=True)
        row = run_condition(cache, nor, cim_dim=128, seed=42, include_hybrid=True)
        print(
            f"  noisy={row['dense_only']['cim_noisy_no_ldpc']['ndcg@10']:.4f} "
            f"bp={row['dense_only']['cim_ldpc_bp']['ndcg@10']:.4f} "
            f"gain={row['robustness']['bp_gain_vs_noisy']:.4f}",
            flush=True,
        )
        nor_rows.append(row)

    bp_gains = [r["robustness"]["bp_gain_vs_noisy"] for r in seed_rows]
    paper_main = []
    for r in seed_rows:
        if r["seed"] == 42:
            paper_main.append(r)
            break
    paper_main.append(dim256)
    paper_main.extend(nor_rows)
    paper_main.extend([r for r in seed_rows if r["seed"] != 42])

    result = {
        "protocol_version": "v1.8",
        "encoder": cache.get("encoder"),
        "model": cache.get("model"),
        "corpus": cache.get("corpus"),
        "n_docs": cache.get("n_docs"),
        "n_queries": cache.get("n_queries"),
        "seed_sweep": seed_rows,
        "dim256_spotcheck": dim256,
        "nor_sweep_seed42": nor_rows,
        "paper_main_configs": paper_main,
        "seed_summary": {
            "bp_gain_mean": float(np.mean(bp_gains)),
            "bp_gain_std": float(np.std(bp_gains)),
            "bp_ndcg_mean": float(np.mean([r["dense_only"]["cim_ldpc_bp"]["ndcg@10"] for r in seed_rows])),
            "bp_ndcg_std": float(np.std([r["dense_only"]["cim_ldpc_bp"]["ndcg@10"] for r in seed_rows])),
            "noisy_ndcg_mean": float(np.mean([r["dense_only"]["cim_noisy_no_ldpc"]["ndcg@10"] for r in seed_rows])),
            "noisy_ndcg_std": float(np.std([r["dense_only"]["cim_noisy_no_ldpc"]["ndcg@10"] for r in seed_rows])),
            "n_seeds": len(seeds),
        },
        "nor_presets": list_nor_presets(),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("Seed summary:", result["seed_summary"], flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
