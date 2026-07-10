"""Extended SIGIR-AP eval: strong encoder, MSMARCO/NQ mini, NOR, GEMV, RAG gen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cim_resldpc.dense_index import EncoderDenseIndex
from cim_resldpc.encoders import build_encoder
from cim_resldpc.gemv_hook import load_gemv_calibrator
from cim_resldpc.hybrid_score import build_bm25
from cim_resldpc.ir_benchmarks import load_msmarco_mini, load_nq_mini
from cim_resldpc.metrics import (
    average_precision,
    mean_metric,
    ndcg_at_k,
    noise_degradation,
    recall_at_k,
)
from cim_resldpc.nor_calibration import get_nor_profile, list_nor_presets
from cim_resldpc.rag_eval import evaluate_rag
from cim_resldpc.datasets import QAPair


def _eval_ir(retrieve_fn, queries: list[QAPair], k: int = 10) -> dict[str, float]:
    ndcgs, maps, recalls = [], [], []
    for q in queries:
        hits = retrieve_fn(q.query, k)
        rel_set = set(q.relevant_doc_ids)
        rels = []
        for h in hits:
            did = h.doc_id if hasattr(h, "doc_id") else h["doc_id"]
            rels.append(1.0 if did in rel_set else 0.0)
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


def _bm25_retrieve(docs, queries_unused=None):
    index = build_bm25([d.doc_id for d in docs], [d.text for d in docs])

    class _B:
        def retrieve(self, query: str, k: int = 10):
            from cim_resldpc.dense_index import DenseHit

            scores = index.score(query)
            order = np.argsort(-scores)[:k]
            return [DenseHit(docs[i].doc_id, float(scores[i])) for i in order]

    return _B()


def eval_corpus(
    name: str,
    docs,
    queries,
    encoder_kind: str,
    nor_name: str,
    use_gemv: bool,
    seed: int,
    emb_dim: int = 128,
) -> dict:
    texts = [d.text for d in docs]
    ids = [d.doc_id for d in docs]
    fit_texts = texts + [q.query for q in queries]
    enc = build_encoder(encoder_kind, dim=emb_dim, fit_texts=fit_texts)
    profile = get_nor_profile(nor_name)
    rng = np.random.default_rng(seed)
    gemv = load_gemv_calibrator() if use_gemv else None

    clean = EncoderDenseIndex(
        ids, texts, enc, apply_cim=False, rng=rng,
    )
    # Unprotected noisy path: quantize + channel noise without LDPC
    from cim_resldpc.gf3_quantize import quantize_base3, dequantize_base3
    from cim_resldpc.noise_bridge import apply_embedding_cim_noise

    clean_emb = enc.encode(texts)
    sym = quantize_base3(clean_emb, n_sym_per_dim=2)
    read = apply_embedding_cim_noise(sym, profile, rng=np.random.default_rng(seed))
    noisy_emb = dequantize_base3(read.recovered_symbols, n_sym_per_dim=2, dim=clean_emb.shape[1])

    class _Noisy:
        def retrieve(self, query: str, k: int = 10):
            from cim_resldpc.dense_index import DenseHit
            from cim_resldpc.residual_features import cosine_similarity

            q = enc.encode([query])[0]
            sim = cosine_similarity(q, noisy_emb)
            order = np.argsort(-sim)[:k]
            return [DenseHit(ids[int(i)], float(sim[i])) for i in order]

    bprime = EncoderDenseIndex(
        ids, texts, enc, profile=profile, recovery="bprime", use_bp=False,
        use_gemv=False, apply_cim=True, rng=np.random.default_rng(seed),
    )
    bp = EncoderDenseIndex(
        ids, texts, enc, profile=profile, recovery="bp", use_bp=True,
        use_gemv=False, apply_cim=True, rng=np.random.default_rng(seed),
    )
    gemv_idx = None
    if use_gemv:
        gemv_idx = EncoderDenseIndex(
            ids, texts, enc, profile=profile, recovery="bprime", use_bp=True,
            use_gemv=True, gemv_calibrator=gemv, apply_cim=True,
            rng=np.random.default_rng(seed),
        )

    bm25 = _bm25_retrieve(docs)
    table = {
        "bm25": _eval_ir(bm25.retrieve, queries),
        "dense_clean": _eval_ir(clean.retrieve, queries),
        "cim_noisy_no_ldpc": _eval_ir(_Noisy().retrieve, queries),
        "cim_ldpc_bprime": _eval_ir(bprime.retrieve, queries),
        "cim_ldpc_bp": _eval_ir(bp.retrieve, queries),
    }
    if gemv_idx is not None:
        table["cim_ldpc_gemv_bp"] = _eval_ir(gemv_idx.retrieve, queries)

    clean_ndcg = table["dense_clean"]["ndcg@10"]
    robustness = {
        "nor_preset": nor_name,
        "encoder": encoder_kind,
        "embed_dim": enc.dim,
        "noisy_vs_clean": noise_degradation(clean_ndcg, table["cim_noisy_no_ldpc"]["ndcg@10"]),
        "bprime_gain_vs_noisy": (
            table["cim_ldpc_bprime"]["ndcg@10"] - table["cim_noisy_no_ldpc"]["ndcg@10"]
        ),
        "bp_gain_vs_noisy": (
            table["cim_ldpc_bp"]["ndcg@10"] - table["cim_noisy_no_ldpc"]["ndcg@10"]
        ),
    }
    if gemv_idx is not None:
        robustness["gemv_gain_vs_noisy"] = (
            table["cim_ldpc_gemv_bp"]["ndcg@10"] - table["cim_noisy_no_ldpc"]["ndcg@10"]
        )

    doc_by_id = {d.doc_id: d for d in docs}
    rag = {
        "clean": evaluate_rag(queries, doc_by_id, clean.retrieve, k=3),
        "noisy": evaluate_rag(queries, doc_by_id, _Noisy().retrieve, k=3),
        "bprime": evaluate_rag(queries, doc_by_id, bprime.retrieve, k=3),
        "bp": evaluate_rag(queries, doc_by_id, bp.retrieve, k=3),
        "bm25": evaluate_rag(queries, doc_by_id, bm25.retrieve, k=3),
    }
    if gemv_idx is not None:
        rag["gemv_bp"] = evaluate_rag(queries, doc_by_id, gemv_idx.retrieve, k=3)

    return {
        "corpus": name,
        "n_docs": len(docs),
        "n_queries": len(queries),
        "dense_only": table,
        "robustness": robustness,
        "rag_generation": rag,
    }


def run_extended(
    out_json: Path | str,
    encoder: str = "tfidf_svd",
    nor_name: str = "nor_aged",
    use_gemv: bool = True,
    seed: int = 42,
    emb_dim: int = 128,
) -> dict:
    ms_docs, ms_q = load_msmarco_mini()
    nq_docs, nq_q = load_nq_mini()
    result = {
        "protocol_version": "v1.2",
        "notes": (
            "Strong offline encoder (tfidf_svd by default). "
            "Bundled MSMARCO/NQ mini fixtures (not full HF dumps). "
            "NOR presets are engineering maps onto GRC-PIM channel. "
            "Optional GRC GEMV residual LLR before BP. "
            "RAG metrics use extractive answer proxy (no external LLM)."
        ),
        "nor_presets": list_nor_presets(),
        "msmarco_mini": eval_corpus(
            "msmarco_mini", ms_docs, ms_q, encoder, nor_name, use_gemv, seed, emb_dim,
        ),
        "nq_mini": eval_corpus(
            "nq_mini", nq_docs, nq_q, encoder, nor_name, use_gemv, seed, emb_dim,
        ),
    }
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_json", default="cim_resldpc/results/sigir_eval_extended.json")
    p.add_argument("--encoder", default="tfidf_svd", choices=["tfidf_svd", "hash", "st"])
    p.add_argument("--nor", default="nor_typical", choices=sorted(["nor_fresh", "nor_typical", "nor_aged", "nor_stress_sweep"]))
    p.add_argument("--no_gemv", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--emb_dim", type=int, default=128)
    args = p.parse_args(argv)
    result = run_extended(
        out_json=args.out_json,
        encoder=args.encoder,
        nor_name=args.nor,
        use_gemv=not args.no_gemv,
        seed=args.seed,
        emb_dim=args.emb_dim,
    )
    for corpus in ("msmarco_mini", "nq_mini"):
        block = result[corpus]
        print(f"\n=== {corpus} / {block['robustness']['encoder']} / {args.nor} ===")
        for k, v in block["dense_only"].items():
            print(f"  {k:22s} nDCG@10={v['ndcg@10']:.4f}")
        print("  robustness:", {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in block["robustness"].items()})
        print("  RAG F1 clean/noisy/bprime:",
              round(block["rag_generation"]["clean"]["rag_f1"], 3),
              round(block["rag_generation"]["noisy"]["rag_f1"], 3),
              round(block["rag_generation"]["bprime"]["rag_f1"], 3))
        print("  RAG hallu clean/noisy/bprime:",
              round(block["rag_generation"]["clean"]["rag_hallucination_rate"], 3),
              round(block["rag_generation"]["noisy"]["rag_hallucination_rate"], 3),
              round(block["rag_generation"]["bprime"]["rag_hallucination_rate"], 3))
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
