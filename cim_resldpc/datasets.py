"""Bundled synthetic IR + semiconductor mini-corpus (offline, no downloads)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: str
    text: str
    labels: list[str] = field(default_factory=list)


@dataclass
class QAPair:
    qid: str
    query: str
    relevant_doc_ids: list[str]
    answer: str = ""


# Semiconductor / chip engineering snippets (unique industrial flavor)
SEMI_DOCS: list[Document] = [
    Document(
        "semi_001",
        "NOR flash floating-gate cells store multi-level conductance for compute-in-memory "
        "MAC; read disturb and charge trapping cause temporal drift across repeated reads.",
        ["nor", "cim", "drift"],
    ),
    Document(
        "semi_002",
        "GF(3) non-binary LDPC soft decoding uses per-symbol LLR vectors; residual GEMV "
        "calibration corrects multi-cycle PIM readout before belief propagation.",
        ["ldpc", "gf3", "residual"],
    ),
    Document(
        "semi_003",
        "SRAM CiM offers low noise but limited density; NOR flash CiM provides large "
        "capacity for on-device knowledge bases at the cost of complex analog noise.",
        ["sram", "nor", "capacity"],
    ),
    Document(
        "semi_004",
        "Edge RAG must keep private semiconductor process documents on-device; cloud "
        "upload of fab recipes violates local-first privacy constraints.",
        ["privacy", "edge", "rag"],
    ),
    Document(
        "semi_005",
        "Gated residual calibration opens the GEMV path only when global drift exceeds "
        "tau_d and model confidence exceeds tau_g, otherwise falling back to B-prime LLR.",
        ["gate", "grc", "drift"],
    ),
    Document(
        "semi_006",
        "Quantizing dense embeddings to ternary GF(3) symbols matches NOR multi-level "
        "storage; aggressive float compression without residual alignment collapses long-tail recall.",
        ["quantization", "embedding", "gf3"],
    ),
    Document(
        "semi_007",
        "Hybrid BM25 and dense retrieval reduces CiM MAC cycles by cutting candidates "
        "before residual-aware reranking on the edge SoC.",
        ["hybrid", "bm25", "latency"],
    ),
    Document(
        "semi_008",
        "IR-drop and thermal gradients induce column-correlated conductance drift in "
        "crossbar arrays, motivating multi-cycle soft evidence rather than single hard decisions.",
        ["irdrop", "thermal", "crossbar"],
    ),
]


SEMI_QUERIES: list[QAPair] = [
    QAPair("q1", "How does NOR flash CiM suffer from read disturb drift?", ["semi_001", "semi_008"], "read disturb"),
    QAPair("q2", "What is GF(3) LDPC residual GEMV calibration for PIM?", ["semi_002", "semi_005"], "residual GEMV"),
    QAPair("q3", "Why prefer NOR over SRAM for large on-device knowledge bases?", ["semi_003"], "capacity"),
    QAPair("q4", "How to keep semiconductor documents private in edge RAG?", ["semi_004"], "local-first"),
    QAPair("q5", "When does gated residual calibration fall back to B-prime?", ["semi_005"], "dual gate"),
    QAPair("q6", "How does GF(3) quantization affect embedding semantic collapse?", ["semi_006"], "ternary"),
    QAPair("q7", "How can hybrid BM25 reduce edge CiM retrieval latency?", ["semi_007"], "candidate cut"),
    QAPair("q8", "What causes column-correlated drift in crossbar MAC arrays?", ["semi_008", "semi_001"], "IR-drop"),
]


def build_synthetic_ir(n_topics: int = 8, docs_per_topic: int = 4, seed: int = 0) -> tuple[list[Document], list[QAPair]]:
    """Topic-bag synthetic corpus with clear relevance structure."""
    topics = [f"topic_{i}" for i in range(n_topics)]
    docs: list[Document] = []
    queries: list[QAPair] = []
    for ti, topic in enumerate(topics):
        rel_ids = []
        for j in range(docs_per_topic):
            did = f"syn_{ti}_{j}"
            # Shared topic tokens + unique filler
            text = (
                f"{topic} semiconductor retrieval document {j} "
                f"keywords {topic}_alpha {topic}_beta chip process node"
            )
            docs.append(Document(did, text, labels=[topic]))
            rel_ids.append(did)
        queries.append(
            QAPair(
                qid=f"sq_{ti}",
                query=f"find {topic} chip process documents {topic}_alpha",
                relevant_doc_ids=rel_ids[:2],
                answer=topic,
            )
        )
    # Add distractors
    for k in range(n_topics):
        docs.append(
            Document(
                f"syn_noise_{k}",
                f"unrelated automotive warranty policy number {k} insurance claim",
                labels=["noise"],
            )
        )
    return docs, queries


def load_semi_corpus() -> tuple[list[Document], list[QAPair]]:
    return list(SEMI_DOCS), list(SEMI_QUERIES)
