"""Baseline retrievers for CiM-ResLDPC evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datasets import Document
from .gf3_quantize import dequantize_from_gf3, quantize_to_gf3
from .hybrid_score import build_bm25
from .noise_bridge import NoiseProfile, apply_embedding_cim_noise
from .residual_features import cosine_similarity
from .tonel import TONELResEncoder


@dataclass
class BaselineHit:
    doc_id: str
    score: float


class BM25Baseline:
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.index = build_bm25([d.doc_id for d in docs], [d.text for d in docs])

    def retrieve(self, query: str, k: int = 10) -> list[BaselineHit]:
        scores = self.index.score(query)
        order = np.argsort(-scores)[:k]
        return [BaselineHit(self.docs[i].doc_id, float(scores[i])) for i in order]


class NaiveDenseBaseline:
    """Float dense retrieval without CiM noise."""

    def __init__(self, docs: list[Document], encoder: TONELResEncoder):
        self.docs = docs
        self.encoder = encoder
        self.doc_emb = encoder.encode_numpy([d.text for d in docs])

    def retrieve(self, query: str, k: int = 10) -> list[BaselineHit]:
        q = self.encoder.encode_numpy([query])[0]
        sim = cosine_similarity(q, self.doc_emb)
        order = np.argsort(-sim)[:k]
        return [BaselineHit(self.docs[i].doc_id, float(sim[i])) for i in order]


class CiMNoisyBaseline:
    """Dense retrieval after CiM noise without residual calibration."""

    def __init__(
        self,
        docs: list[Document],
        encoder: TONELResEncoder,
        profile: NoiseProfile,
        rng: np.random.Generator,
    ):
        self.docs = docs
        self.encoder = encoder
        self.profile = profile
        self.rng = rng
        clean = encoder.encode_numpy([d.text for d in docs])
        symbols = quantize_to_gf3(clean, levels=encoder.config.gf3_levels)
        readout = apply_embedding_cim_noise(symbols, profile, rng=rng)
        self.doc_emb = dequantize_from_gf3(
            readout.recovered_symbols, levels=encoder.config.gf3_levels,
        )

    def retrieve(self, query: str, k: int = 10) -> list[BaselineHit]:
        q = self.encoder.encode_numpy([query])[0]
        # Query also passes through a light noise path for fairness
        q_sym = quantize_to_gf3(q[None, :], levels=self.encoder.config.gf3_levels)
        q_read = apply_embedding_cim_noise(q_sym, self.profile, rng=self.rng)
        q_rec = dequantize_from_gf3(
            q_read.recovered_symbols, levels=self.encoder.config.gf3_levels,
        )[0]
        sim = cosine_similarity(q_rec, self.doc_emb)
        order = np.argsort(-sim)[:k]
        return [BaselineHit(self.docs[i].doc_id, float(sim[i])) for i in order]
