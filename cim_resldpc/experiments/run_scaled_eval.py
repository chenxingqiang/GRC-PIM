"""v1.4 scaled evaluation: 200q MSMARCO/NQ, MiniLM/BGE, optional Flan-T5 RAG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cim_resldpc.datasets import Document, QAPair
from cim_resldpc.dense_index import DenseHit, EncoderDenseIndex
from cim_resldpc.encoders import build_encoder
from cim_resldpc.experiments.build_hf_cache import load_cached_or_build
from cim_resldpc.gemv_hook import load_gemv_calibrator
from cim_resldpc.gf3_quantize import dequantize_base3, quantize_base3
from cim_resldpc.hybrid_score import build_bm25
from cim_resldpc.metrics import (
    average_precision,
    mean_metric,
    ndcg_at_k,
    noise_degradation,
    recall_at_k,
)
from cim_resldpc.noise_bridge import apply_embedding_cim_noise
from cim_resldpc.nor_calibration import get_nor_profile
from cim_resldpc.rag_eval import evaluate_rag
from cim_resldpc.rag_generators import build_generator, evaluate_rag_with_generator


def _eval_ir(retrieve_fn, queries: list[QAPair], k: int = 10) -> dict[str, float]:
    ndcgs, maps, recalls = [], [], []
    for q in queries:
        hits = retrieve_fn(q.query, k)
        rel_set = set(q.relevant_doc_ids)
        rels = [
            1.0 if (h.doc_id if hasattr(h, "doc_id") else h["doc_id"]) in rel_set else 0.0
            for h in hits
        ]
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


def _bm25(docs: list[Document]):
    index = build_bm25([d.doc_id for d in docs], [d.text for d in docs])

    class _B:
        def retrieve(self, query: str, k: int = 10):
            scores = index.score(query)
            order = np.argsort(-scores)[:k]
            return [DenseHit(docs[i].doc_id, float(scores[i])) for i in order]

    return _B()


def _project_encoder(teacher, cim_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    teacher_dim = int(teacher.dim)
    proj = None
    if teacher_dim != cim_dim:
        proj = rng.normal(size=(teacher_dim, cim_dim)).astype(np.float32)
        proj /= np.sqrt(cim_dim)

    class _Projected:
        name = f"{teacher.name}->proj{cim_dim}"

        @property
        def dim(self) -> int:
            return cim_dim

        def encode(self, xs):
            emb = teacher.encode(xs)
            out = emb if proj is None else emb @ proj
            norms = np.linalg.norm(out, axis=-1, keepdims=True)
            safe = np.maximum(norms, 1e-8)
            out = out / safe
            return np.where(norms < 1e-8, 0.0, out).astype(np.float32)

    return _Projected(), teacher_dim


def make_teacher(encoder_kind: str, st_model: str, fit_texts: list[str], dim: int = 384):
    if encoder_kind in {"st", "minilm"}:
        return build_encoder("st", st_model=st_model)
    if encoder_kind in {"bge", "bge-small"}:
        return build_encoder("st", st_model="BAAI/bge-small-en-v1.5")
    if encoder_kind in {"tfidf_svd", "tfidf"}:
        return build_encoder("tfidf_svd", dim=dim, fit_texts=fit_texts)
    raise ValueError(encoder_kind)


def eval_corpus(
    name: str,
    docs: list[Document],
    queries: list[QAPair],
    encoder_kind: str,
    nor_name: str,
    use_gemv: bool,
    seed: int,
    st_model: str,
    cim_dim: int,
    rag_generator: str = "extractive",
) -> dict:
    texts = [d.text for d in docs]
    ids = [d.doc_id for d in docs]
    fit_texts = texts + [q.query for q in queries]
    teacher = make_teacher(encoder_kind, st_model, fit_texts, dim=cim_dim)
    enc, teacher_dim = _project_encoder(teacher, cim_dim, seed)
    profile = get_nor_profile(nor_name)
    gemv = load_gemv_calibrator() if use_gemv else None

    clean = EncoderDenseIndex(ids, texts, enc, apply_cim=False, rng=np.random.default_rng(seed))
    clean_emb = enc.encode(texts)
    sym = quantize_base3(clean_emb, n_sym_per_dim=2)
    read = apply_embedding_cim_noise(sym, profile, rng=np.random.default_rng(seed))
    noisy_emb = dequantize_base3(read.recovered_symbols, n_sym_per_dim=2, dim=clean_emb.shape[1])

    class _Noisy:
        def retrieve(self, query: str, k: int = 10):
            from cim_resldpc.residual_features import cosine_similarity

            q = enc.encode([query])[0]
            sim = cosine_similarity(q, noisy_emb)
            order = np.argsort(-sim)[:k]
            return [DenseHit(ids[int(i)], float(sim[i])) for i in order]

    bprime = EncoderDenseIndex(
        ids, texts, enc, profile=profile, recovery="bprime", use_bp=False,
        apply_cim=True, rng=np.random.default_rng(seed),
    )
    bp = EncoderDenseIndex(
        ids, texts, enc, profile=profile, recovery="bp", use_bp=True,
        apply_cim=True, rng=np.random.default_rng(seed),
    )
    table = {
        "bm25": _eval_ir(_bm25(docs).retrieve, queries),
        "dense_clean": _eval_ir(clean.retrieve, queries),
        "cim_noisy_no_ldpc": _eval_ir(_Noisy().retrieve, queries),
        "cim_ldpc_bprime": _eval_ir(bprime.retrieve, queries),
        "cim_ldpc_bp": _eval_ir(bp.retrieve, queries),
    }
    gemv_idx = None
    if use_gemv:
        gemv_idx = EncoderDenseIndex(
            ids, texts, enc, profile=profile, recovery="bp", use_bp=True,
            use_gemv=True, gemv_calibrator=gemv, apply_cim=True,
            rng=np.random.default_rng(seed),
        )
        table["cim_ldpc_gemv_bp"] = _eval_ir(gemv_idx.retrieve, queries)

    clean_ndcg = table["dense_clean"]["ndcg@10"]
    robustness = {
        "nor_preset": nor_name,
        "encoder": encoder_kind,
        "st_model": st_model if encoder_kind in {"st", "minilm"} else (
            "BAAI/bge-small-en-v1.5" if encoder_kind.startswith("bge") else encoder_kind
        ),
        "teacher_dim": teacher_dim,
        "cim_dim": cim_dim,
        "noisy_vs_clean": noise_degradation(clean_ndcg, table["cim_noisy_no_ldpc"]["ndcg@10"]),
        "bprime_gain_vs_noisy": table["cim_ldpc_bprime"]["ndcg@10"] - table["cim_noisy_no_ldpc"]["ndcg@10"],
        "bp_gain_vs_noisy": table["cim_ldpc_bp"]["ndcg@10"] - table["cim_noisy_no_ldpc"]["ndcg@10"],
    }
    if gemv_idx is not None:
        robustness["gemv_gain_vs_noisy"] = (
            table["cim_ldpc_gemv_bp"]["ndcg@10"] - table["cim_noisy_no_ldpc"]["ndcg@10"]
        )

    doc_by_id = {d.doc_id: d for d in docs}
    gen = build_generator(rag_generator)
    retrievers = {
        "clean": clean.retrieve,
        "noisy": _Noisy().retrieve,
        "bprime": bprime.retrieve,
        "bp": bp.retrieve,
        "bm25": _bm25(docs).retrieve,
    }
    if gemv_idx is not None:
        retrievers["gemv_bp"] = gemv_idx.retrieve

    rag = {}
    for key, fn in retrievers.items():
        if rag_generator == "extractive":
            rag[key] = evaluate_rag(queries, doc_by_id, fn, k=3)
            rag[key]["generator"] = "extractive"
        else:
            rag[key] = evaluate_rag_with_generator(queries, doc_by_id, fn, gen, k=3)

    return {
        "corpus": name,
        "n_docs": len(docs),
        "n_queries": len(queries),
        "dense_only": table,
        "robustness": robustness,
        "rag_generation": rag,
    }


def run_scaled(
    out_json: Path | str,
    encoder: str = "st",
    nor_name: str = "nor_typical",
    use_gemv: bool = True,
    n_queries: int = 200,
    cim_dim: int = 128,
    seed: int = 42,
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    corpora: list[str] | None = None,
    rag_generator: str = "extractive",
) -> dict:
    corpora = corpora or ["msmarco_hf", "nq_hf"]
    blocks = {}
    for name in corpora:
        docs, queries, path = load_cached_or_build(name, n_queries=n_queries, force=False)
        print(f"loaded {name} from {path} ({len(docs)} docs / {len(queries)} queries)")
        blocks[name] = eval_corpus(
            name, docs, queries, encoder, nor_name, use_gemv, seed, st_model,
            cim_dim=cim_dim, rag_generator=rag_generator,
        )
    result = {
        "protocol_version": "v1.4",
        "notes": (
            "Scaled HF subsets (default 200q). Encoders: MiniLM / BGE-small / tfidf. "
            "Optional Flan-T5-small generative RAG. Teacher→cim_dim projection before CiM."
        ),
        "config": {
            "encoder": encoder,
            "st_model": st_model,
            "nor": nor_name,
            "use_gemv": use_gemv,
            "n_queries": n_queries,
            "cim_dim": cim_dim,
            "rag_generator": rag_generator,
            "seed": seed,
        },
        **blocks,
    }
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_json", default="cim_resldpc/results/sigir_eval_v14_scaled.json")
    p.add_argument("--encoder", default="st", choices=["st", "bge", "tfidf_svd"])
    p.add_argument("--nor", default="nor_typical")
    p.add_argument("--no_gemv", action="store_true")
    p.add_argument("--n_queries", type=int, default=200)
    p.add_argument("--cim_dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--st_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--corpora", default="msmarco_hf,nq_hf")
    p.add_argument("--rag_generator", default="extractive", choices=["extractive", "flan"])
    args = p.parse_args(argv)
    result = run_scaled(
        out_json=args.out_json,
        encoder=args.encoder,
        nor_name=args.nor,
        use_gemv=not args.no_gemv,
        n_queries=args.n_queries,
        cim_dim=args.cim_dim,
        seed=args.seed,
        st_model=args.st_model,
        corpora=[c.strip() for c in args.corpora.split(",") if c.strip()],
        rag_generator=args.rag_generator,
    )
    for name, block in result.items():
        if not isinstance(block, dict) or "dense_only" not in block:
            continue
        print(f"\n=== {name} / {args.encoder} / {args.nor} ===")
        for k, v in block["dense_only"].items():
            print(f"  {k:22s} nDCG@10={v['ndcg@10']:.4f} MAP={v['map']:.4f}")
        print("  gains:",
              round(block["robustness"]["bprime_gain_vs_noisy"], 4),
              round(block["robustness"]["bp_gain_vs_noisy"], 4),
              round(block["robustness"].get("gemv_gain_vs_noisy", 0.0), 4))
        rg = block["rag_generation"]
        print("  RAG F1:", {k: round(rg[k]["rag_f1"], 3) for k in rg})
        print("  RAG hallu:", {k: round(rg[k]["rag_hallucination_rate"], 3) for k in rg})
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
