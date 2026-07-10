"""v1.3 HF + MiniLM evaluation for CiM-ResLDPC.

Uses sentence-transformers MiniLM (real dense IR encoder) and cached
MSMARCO / NQ-open subsets. Falls back to bundled mini fixtures if HF/cache
unavailable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cim_resldpc.dense_index import EncoderDenseIndex
from cim_resldpc.encoders import build_encoder
from cim_resldpc.experiments.build_hf_cache import load_cached_or_build
from cim_resldpc.gemv_hook import load_gemv_calibrator
from cim_resldpc.gf3_quantize import dequantize_base3, quantize_base3
from cim_resldpc.hybrid_score import build_bm25
from cim_resldpc.ir_benchmarks import load_msmarco_mini, load_nq_mini
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
from cim_resldpc.datasets import Document, QAPair
from cim_resldpc.dense_index import DenseHit


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


def eval_corpus(
    name: str,
    docs: list[Document],
    queries: list[QAPair],
    encoder_kind: str,
    nor_name: str,
    use_gemv: bool,
    seed: int,
    st_model: str,
    cim_dim: int = 128,
) -> dict:
    """Encode with a strong teacher, then project to `cim_dim` for CiM storage.

    Rationale: edge NOR/CiM arrays cannot cheaply store full 384-D MiniLM
    vectors; projecting to 64–128-D matches the paper's bit-width story while
    keeping the teacher encoder for IR-quality semantics.
    """
    texts = [d.text for d in docs]
    ids = [d.doc_id for d in docs]
    fit_texts = texts + [q.query for q in queries]

    if encoder_kind in {"st", "sentence_transformer", "minilm"}:
        teacher = build_encoder("st", st_model=st_model)
    else:
        teacher = build_encoder(encoder_kind, dim=max(cim_dim, 128), fit_texts=fit_texts)

    # Fit a fixed random projection teacher_dim → cim_dim (deterministic)
    rng = np.random.default_rng(seed)
    teacher_dim = int(teacher.dim)
    if teacher_dim == cim_dim:
        proj = None
    else:
        proj = rng.normal(size=(teacher_dim, cim_dim)).astype(np.float32)
        proj /= np.sqrt(cim_dim)

    class _Projected:
        name = f"{teacher.name}->proj{cim_dim}"

        @property
        def dim(self) -> int:
            return cim_dim

        def encode(self, xs):
            emb = teacher.encode(xs)
            if proj is None:
                out = emb
            else:
                out = emb @ proj
            norms = np.linalg.norm(out, axis=-1, keepdims=True)
            safe = np.maximum(norms, 1e-8)
            out = out / safe
            return np.where(norms < 1e-8, 0.0, out).astype(np.float32)

    enc = _Projected()
    profile = get_nor_profile(nor_name)
    gemv = load_gemv_calibrator() if use_gemv else None

    clean = EncoderDenseIndex(ids, texts, enc, apply_cim=False, rng=np.random.default_rng(seed))

    clean_emb = enc.encode(texts)
    sym = quantize_base3(clean_emb, n_sym_per_dim=2)
    read = apply_embedding_cim_noise(sym, profile, rng=np.random.default_rng(seed))
    noisy_emb = dequantize_base3(
        read.recovered_symbols, n_sym_per_dim=2, dim=clean_emb.shape[1],
    )

    class _Noisy:
        def retrieve(self, query: str, k: int = 10):
            q = enc.encode([query])[0]
            from cim_resldpc.residual_features import cosine_similarity

            sim = cosine_similarity(q, noisy_emb)
            order = np.argsort(-sim)[:k]
            return [DenseHit(ids[int(i)], float(sim[i])) for i in order]

    bprime = EncoderDenseIndex(
        ids, texts, enc, profile=profile, recovery="bprime", use_bp=False,
        use_gemv=False, apply_cim=True, rng=np.random.default_rng(seed),
        n_sym_per_dim=2, quant_mode="base3",
    )
    bp = EncoderDenseIndex(
        ids, texts, enc, profile=profile, recovery="bp", use_bp=True,
        use_gemv=False, apply_cim=True, rng=np.random.default_rng(seed),
        n_sym_per_dim=2, quant_mode="base3",
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
            rng=np.random.default_rng(seed), n_sym_per_dim=2, quant_mode="base3",
        )
        table["cim_ldpc_gemv_bp"] = _eval_ir(gemv_idx.retrieve, queries)

    clean_ndcg = table["dense_clean"]["ndcg@10"]
    robustness = {
        "nor_preset": nor_name,
        "encoder": encoder_kind if encoder_kind != "st" else f"st:{st_model}",
        "teacher_dim": teacher_dim,
        "cim_dim": cim_dim,
        "embed_dim": int(enc.dim),
        "noisy_vs_clean": noise_degradation(clean_ndcg, table["cim_noisy_no_ldpc"]["ndcg@10"]),
        "bprime_gain_vs_noisy": table["cim_ldpc_bprime"]["ndcg@10"] - table["cim_noisy_no_ldpc"]["ndcg@10"],
        "bp_gain_vs_noisy": table["cim_ldpc_bp"]["ndcg@10"] - table["cim_noisy_no_ldpc"]["ndcg@10"],
    }
    if "cim_ldpc_gemv_bp" in table:
        robustness["gemv_gain_vs_noisy"] = (
            table["cim_ldpc_gemv_bp"]["ndcg@10"] - table["cim_noisy_no_ldpc"]["ndcg@10"]
        )

    doc_by_id = {d.doc_id: d for d in docs}
    rag = {
        "clean": evaluate_rag(queries, doc_by_id, clean.retrieve, k=3),
        "noisy": evaluate_rag(queries, doc_by_id, _Noisy().retrieve, k=3),
        "bprime": evaluate_rag(queries, doc_by_id, bprime.retrieve, k=3),
        "bp": evaluate_rag(queries, doc_by_id, bp.retrieve, k=3),
        "bm25": evaluate_rag(queries, doc_by_id, _bm25(docs).retrieve, k=3),
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


def load_corpus(name: str, n_queries: int, allow_hf: bool) -> tuple[list[Document], list[QAPair]]:
    if name == "msmarco_mini":
        return load_msmarco_mini()
    if name == "nq_mini":
        return load_nq_mini()
    if not allow_hf:
        raise RuntimeError(f"HF corpus {name} requested but --no_hf set")
    docs, queries, path = load_cached_or_build(name, n_queries=n_queries, force=False)
    print(f"loaded {name} from {path} ({len(docs)} docs / {len(queries)} queries)")
    return docs, queries


def run_hf_eval(
    out_json: Path | str,
    encoder: str = "st",
    nor_name: str = "nor_typical",
    use_gemv: bool = True,
    n_queries: int = 64,
    seed: int = 42,
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    allow_hf: bool = True,
    corpora: list[str] | None = None,
    cim_dim: int = 128,
) -> dict:
    corpora = corpora or ["msmarco_hf", "nq_hf"]
    blocks = {}
    for name in corpora:
        try:
            docs, queries = load_corpus(name, n_queries=n_queries, allow_hf=allow_hf)
        except Exception as e:
            print(f"SKIP {name}: {e}")
            continue
        blocks[name] = eval_corpus(
            name, docs, queries, encoder, nor_name, use_gemv, seed, st_model,
            cim_dim=cim_dim,
        )

    result = {
        "protocol_version": "v1.3",
        "notes": (
            "MiniLM (or chosen encoder) on HF MSMARCO/NQ-open subsets with local JSON cache. "
            "Teacher embeddings are projected to cim_dim before GF(3)/LDPC CiM storage. "
            "NQ-open passages are answer-grounded pseudo-docs (dataset has no passages). "
            "NOR presets remain engineering maps onto GRC-PIM channel."
        ),
        "nor_presets": list_nor_presets(),
        "config": {
            "encoder": encoder,
            "st_model": st_model,
            "nor": nor_name,
            "use_gemv": use_gemv,
            "n_queries": n_queries,
            "cim_dim": cim_dim,
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
    p.add_argument("--out_json", default="cim_resldpc/results/sigir_eval_hf_minilm.json")
    p.add_argument("--encoder", default="st", choices=["st", "tfidf_svd", "hash"])
    p.add_argument("--nor", default="nor_typical")
    p.add_argument("--no_gemv", action="store_true")
    p.add_argument("--n_queries", type=int, default=48)
    p.add_argument("--cim_dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--st_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--no_hf", action="store_true")
    p.add_argument(
        "--corpora",
        default="msmarco_hf,nq_hf",
        help="comma-separated: msmarco_hf,nq_hf,msmarco_mini,nq_mini",
    )
    args = p.parse_args(argv)
    result = run_hf_eval(
        out_json=args.out_json,
        encoder=args.encoder,
        nor_name=args.nor,
        use_gemv=not args.no_gemv,
        n_queries=args.n_queries,
        seed=args.seed,
        st_model=args.st_model,
        allow_hf=not args.no_hf,
        corpora=[c.strip() for c in args.corpora.split(",") if c.strip()],
        cim_dim=args.cim_dim,
    )
    for name, block in result.items():
        if not isinstance(block, dict) or "dense_only" not in block:
            continue
        print(f"\n=== {name} ===")
        for k, v in block["dense_only"].items():
            print(f"  {k:22s} nDCG@10={v['ndcg@10']:.4f} MAP={v['map']:.4f}")
        print("  robustness:", {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in block["robustness"].items()})
        rg = block["rag_generation"]
        print(
            "  RAG F1 c/n/b'/bp:",
            round(rg["clean"]["rag_f1"], 3),
            round(rg["noisy"]["rag_f1"], 3),
            round(rg["bprime"]["rag_f1"], 3),
            round(rg["bp"]["rag_f1"], 3),
        )
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
