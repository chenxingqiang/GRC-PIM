"""v1.5 ablations: cache BGE embeddings, sweep NOR / seed / cim_dim / hybrid.

Encodes documents+queries once, then reuses float teacher vectors for fast
channel/LDPC sweeps (projection + noise + recover).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cim_resldpc.datasets import Document, QAPair
from cim_resldpc.dense_index import DenseHit
from cim_resldpc.encoders import build_encoder
from cim_resldpc.experiments.build_hf_cache import load_cached_or_build
from cim_resldpc.gf3_quantize import dequantize_base3, quantize_base3
from cim_resldpc.hybrid_score import build_bm25, fuse_scores
from cim_resldpc.ldpc_embed import protect_and_recover
from cim_resldpc.metrics import (
    average_precision,
    mean_metric,
    ndcg_at_k,
    noise_degradation,
    recall_at_k,
)
from cim_resldpc.noise_bridge import apply_embedding_cim_noise
from cim_resldpc.nor_calibration import get_nor_profile, list_nor_presets
from cim_resldpc.rag_eval import evaluate_rag
from cim_resldpc.residual_features import cosine_similarity


def _eval_from_sims(
    doc_ids: list[str],
    queries: list[QAPair],
    q_emb: np.ndarray,
    d_emb: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    ndcgs, maps, recalls = [], [], []
    for i, q in enumerate(queries):
        sim = cosine_similarity(q_emb[i], d_emb)
        order = np.argsort(-sim)[:k]
        rel_set = set(q.relevant_doc_ids)
        rels = [1.0 if doc_ids[j] in rel_set else 0.0 for j in order]
        binary = [int(r > 0) for r in rels]
        ndcgs.append(ndcg_at_k(rels, k=k))
        maps.append(average_precision(binary))
        recalls.append(recall_at_k(binary, n_relevant=len(q.relevant_doc_ids), k=k))
    return {
        "ndcg@10": mean_metric(ndcgs),
        "map": mean_metric(maps),
        f"recall@{k}": mean_metric(recalls),
        "n_queries": float(len(queries)),
    }


def _project(emb: np.ndarray, cim_dim: int, seed: int) -> np.ndarray:
    if emb.shape[1] == cim_dim:
        out = emb
    else:
        rng = np.random.default_rng(seed)
        proj = rng.normal(size=(emb.shape[1], cim_dim)).astype(np.float32)
        proj /= np.sqrt(cim_dim)
        out = emb @ proj
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    safe = np.maximum(norms, 1e-8)
    out = out / safe
    return np.where(norms < 1e-8, 0.0, out).astype(np.float32)


def recover_docs(
    emb: np.ndarray,
    nor_name: str,
    seed: int,
    mode: str,
) -> np.ndarray:
    """mode: clean | noisy | bprime | bp"""
    if mode == "clean":
        return emb
    profile = get_nor_profile(nor_name)
    rng = np.random.default_rng(seed)
    if mode == "noisy":
        sym = quantize_base3(emb, n_sym_per_dim=2)
        read = apply_embedding_cim_noise(sym, profile, rng=rng)
        return dequantize_base3(read.recovered_symbols, n_sym_per_dim=2, dim=emb.shape[1])
    rec = protect_and_recover(
        emb,
        profile=profile,
        rng=rng,
        use_bp=(mode == "bp"),
        n_sym_per_dim=2,
        quant_mode="base3",
    )
    return rec.emb_bp if mode == "bp" else rec.emb_bprime


def cache_teacher_embeddings(
    corpus: str,
    n_queries: int,
    encoder_kind: str,
    out_path: Path,
    st_model: str | None = None,
) -> dict:
    docs, queries, _ = load_cached_or_build(corpus, n_queries=n_queries, force=False)
    texts = [d.text for d in docs]
    qtexts = [q.query for q in queries]
    if encoder_kind == "bge":
        enc = build_encoder("st", st_model="BAAI/bge-small-en-v1.5")
        model_name = "BAAI/bge-small-en-v1.5"
    elif encoder_kind == "st":
        model_name = st_model or "sentence-transformers/all-MiniLM-L6-v2"
        enc = build_encoder("st", st_model=model_name)
    else:
        enc = build_encoder("tfidf_svd", dim=384, fit_texts=texts + qtexts)
        model_name = "tfidf_svd"

    print(f"encoding {len(texts)} docs + {len(qtexts)} queries with {model_name}...")
    d_emb = enc.encode(texts)
    q_emb = enc.encode(qtexts)
    payload = {
        "corpus": corpus,
        "encoder": encoder_kind,
        "model": model_name,
        "n_docs": len(docs),
        "n_queries": len(queries),
        "dim": int(d_emb.shape[1]),
        "doc_ids": [d.doc_id for d in docs],
        "queries": [
            {
                "qid": q.qid,
                "query": q.query,
                "relevant_doc_ids": q.relevant_doc_ids,
                "answer": q.answer,
            }
            for q in queries
        ],
        "doc_texts": texts,
        "d_emb": d_emb,
        "q_emb": q_emb,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        d_emb=d_emb,
        q_emb=q_emb,
        meta=json.dumps({k: v for k, v in payload.items() if k not in {"d_emb", "q_emb"}}),
    )
    print(f"wrote {out_path}")
    return payload


def load_emb_cache(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    meta["d_emb"] = z["d_emb"].astype(np.float32)
    meta["q_emb"] = z["q_emb"].astype(np.float32)
    return meta


def run_condition(
    cache: dict,
    nor_name: str,
    cim_dim: int,
    seed: int,
    include_hybrid: bool = True,
) -> dict:
    d_teacher = cache["d_emb"]
    q_teacher = cache["q_emb"]
    doc_ids = cache["doc_ids"]
    queries = [QAPair(**q) for q in cache["queries"]]
    texts = cache["doc_texts"]

    # Projection seed fixed to cim_dim/seed so dims are comparable across NOR
    d_proj = _project(d_teacher, cim_dim, seed=seed)
    q_proj = _project(q_teacher, cim_dim, seed=seed)

    table = {}
    for mode in ("clean", "noisy", "bprime", "bp"):
        d_rec = recover_docs(d_proj, nor_name, seed=seed + 17, mode=mode)
        # asymmetric: query stays projected-clean
        key = {
            "clean": "dense_clean",
            "noisy": "cim_noisy_no_ldpc",
            "bprime": "cim_ldpc_bprime",
            "bp": "cim_ldpc_bp",
        }[mode]
        table[key] = _eval_from_sims(doc_ids, queries, q_proj, d_rec)

    # BM25 + hybrid with BP docs
    bm25_index = build_bm25(doc_ids, texts)
    bm25_scores = np.stack([bm25_index.score(q.query) for q in queries], axis=0)
    # dense scores from BP recovery
    d_bp = recover_docs(d_proj, nor_name, seed=seed + 17, mode="bp")
    dense_scores = q_proj @ d_bp.T  # (Q, D) since both L2-normalized ~ cosine

    # residual conf proxy: 1 - mean abs(clean_proj - bp) per doc, broadcast
    delta = np.mean(np.abs(d_proj - d_bp), axis=1)
    res_conf = np.clip(1.0 - delta / (delta.max() + 1e-8), 0.0, 1.0)
    res_mat = np.broadcast_to(res_conf[None, :], dense_scores.shape)

    hybrid_ndcgs = []
    for i, q in enumerate(queries):
        fused = fuse_scores(bm25_scores[i], dense_scores[i], res_mat[i], 0.3, 0.5, 0.2)
        order = np.argsort(-fused)[:10]
        rel_set = set(q.relevant_doc_ids)
        rels = [1.0 if doc_ids[j] in rel_set else 0.0 for j in order]
        hybrid_ndcgs.append(ndcg_at_k(rels, 10))
    table["bm25"] = _eval_from_sims(
        doc_ids,
        queries,
        # fake: rank by bm25 only via scores
        np.zeros((len(queries), 1), dtype=np.float32),
        np.zeros((len(doc_ids), 1), dtype=np.float32),
    )
    # proper BM25 eval
    ndcgs = []
    maps = []
    for i, q in enumerate(queries):
        order = np.argsort(-bm25_scores[i])[:10]
        rel_set = set(q.relevant_doc_ids)
        rels = [1.0 if doc_ids[j] in rel_set else 0.0 for j in order]
        binary = [int(r > 0) for r in rels]
        ndcgs.append(ndcg_at_k(rels, 10))
        maps.append(average_precision(binary))
    table["bm25"] = {
        "ndcg@10": mean_metric(ndcgs),
        "map": mean_metric(maps),
        "recall@10": mean_metric(
            [
                recall_at_k(
                    [1 if doc_ids[j] in set(q.relevant_doc_ids) else 0
                     for j in np.argsort(-bm25_scores[i])[:10]],
                    n_relevant=len(q.relevant_doc_ids),
                    k=10,
                )
                for i, q in enumerate(queries)
            ]
        ),
        "n_queries": float(len(queries)),
    }
    if include_hybrid:
        table["hybrid_bm25_bp"] = {
            "ndcg@10": mean_metric(hybrid_ndcgs),
            "n_queries": float(len(queries)),
        }

    clean = table["dense_clean"]["ndcg@10"]
    noisy = table["cim_noisy_no_ldpc"]["ndcg@10"]
    return {
        "nor": nor_name,
        "cim_dim": cim_dim,
        "seed": seed,
        "dense_only": table,
        "robustness": {
            "noisy_vs_clean": noise_degradation(clean, noisy),
            "bprime_gain_vs_noisy": table["cim_ldpc_bprime"]["ndcg@10"] - noisy,
            "bp_gain_vs_noisy": table["cim_ldpc_bp"]["ndcg@10"] - noisy,
            "hybrid_vs_bm25": (
                table.get("hybrid_bm25_bp", {}).get("ndcg@10", 0) - table["bm25"]["ndcg@10"]
            ),
        },
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="msmarco_hf")
    p.add_argument("--n_queries", type=int, default=200)
    p.add_argument("--encoder", default="bge", choices=["bge", "st", "tfidf_svd"])
    p.add_argument("--emb_cache", default="cim_resldpc/data/cache/bge_msmarco_n200_emb.npz")
    p.add_argument("--force_encode", action="store_true")
    p.add_argument("--out_json", default="cim_resldpc/results/sigir_eval_v15_ablation.json")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--cim_dims", default="64,128,256")
    p.add_argument("--nors", default="nor_fresh,nor_typical,nor_aged,nor_stress_sweep")
    args = p.parse_args(argv)

    cache_path = Path(args.emb_cache)
    if args.force_encode or not cache_path.exists():
        cache_teacher_embeddings(args.corpus, args.n_queries, args.encoder, cache_path)
    cache = load_emb_cache(cache_path)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    dims = [int(x) for x in args.cim_dims.split(",") if x.strip()]
    nors = [x.strip() for x in args.nors.split(",") if x.strip()]

    rows = []
    # Primary grid: all NOR @ cim_dim=128 seed=0; all seeds @ typical/128; all dims @ typical/seed0
    for nor in nors:
        rows.append(run_condition(cache, nor, cim_dim=128, seed=0))
        print(f"NOR {nor}: noisy={rows[-1]['dense_only']['cim_noisy_no_ldpc']['ndcg@10']:.4f} "
              f"bp={rows[-1]['dense_only']['cim_ldpc_bp']['ndcg@10']:.4f} "
              f"gain={rows[-1]['robustness']['bp_gain_vs_noisy']:.4f}")

    seed_rows = []
    for seed in seeds:
        seed_rows.append(run_condition(cache, "nor_typical", cim_dim=128, seed=seed))
        print(f"seed {seed}: bp_gain={seed_rows[-1]['robustness']['bp_gain_vs_noisy']:.4f}")

    dim_rows = []
    for dim in dims:
        dim_rows.append(run_condition(cache, "nor_typical", cim_dim=dim, seed=0))
        print(f"dim {dim}: clean={dim_rows[-1]['dense_only']['dense_clean']['ndcg@10']:.4f} "
              f"bp_gain={dim_rows[-1]['robustness']['bp_gain_vs_noisy']:.4f}")

    # Multi-seed summary at typical/128
    bp_gains = [r["robustness"]["bp_gain_vs_noisy"] for r in seed_rows]
    bprime_gains = [r["robustness"]["bprime_gain_vs_noisy"] for r in seed_rows]
    bp_ndcg = [r["dense_only"]["cim_ldpc_bp"]["ndcg@10"] for r in seed_rows]
    noisy_ndcg = [r["dense_only"]["cim_noisy_no_ldpc"]["ndcg@10"] for r in seed_rows]

    result = {
        "protocol_version": "v1.5",
        "encoder": cache.get("encoder"),
        "model": cache.get("model"),
        "corpus": cache.get("corpus"),
        "n_docs": cache.get("n_docs"),
        "n_queries": cache.get("n_queries"),
        "nor_sweep": rows,
        "seed_sweep": seed_rows,
        "dim_sweep": dim_rows,
        "seed_summary": {
            "bp_gain_mean": float(np.mean(bp_gains)),
            "bp_gain_std": float(np.std(bp_gains)),
            "bprime_gain_mean": float(np.mean(bprime_gains)),
            "bprime_gain_std": float(np.std(bprime_gains)),
            "bp_ndcg_mean": float(np.mean(bp_ndcg)),
            "bp_ndcg_std": float(np.std(bp_ndcg)),
            "noisy_ndcg_mean": float(np.mean(noisy_ndcg)),
            "noisy_ndcg_std": float(np.std(noisy_ndcg)),
            "n_seeds": len(seeds),
        },
        "nor_presets": list_nor_presets(),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("\nSeed summary:", result["seed_summary"])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
