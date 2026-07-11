"""Baseline retrievers for CiM-ResLDPC evaluation (oracle-free)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datasets import Document
from .gf3_quantize import (
    dequantize_base3,
    dequantize_from_gf3,
    quantize_base3,
    quantize_to_gf3,
)
from .hybrid_score import build_bm25
from .ldpc_embed import LDPCCode, protect_and_recover
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
    """Quantize + bit noise without LDPC (asymmetric: clean digital query)."""

    def __init__(
        self,
        docs: list[Document],
        encoder: TONELResEncoder,
        profile: NoiseProfile,
        rng: np.random.Generator,
        asymmetric_query: bool = True,
    ):
        self.docs = docs
        self.encoder = encoder
        self.profile = profile
        self.rng = rng
        self.asymmetric_query = asymmetric_query
        self.n_sym = encoder.config.n_sym_per_dim
        self.quant_mode = encoder.config.quant_mode
        clean = encoder.encode_numpy([d.text for d in docs])
        if self.quant_mode == "base3":
            symbols = quantize_base3(clean, n_sym_per_dim=self.n_sym)
            readout = apply_embedding_cim_noise(symbols, profile, rng=rng)
            self.doc_emb = dequantize_base3(
                readout.recovered_symbols, n_sym_per_dim=self.n_sym, dim=clean.shape[1],
            )
        else:
            symbols = quantize_to_gf3(clean, levels=encoder.config.gf3_levels)
            readout = apply_embedding_cim_noise(symbols, profile, rng=rng)
            self.doc_emb = dequantize_from_gf3(
                readout.recovered_symbols, levels=encoder.config.gf3_levels,
            )

    def retrieve(self, query: str, k: int = 10) -> list[BaselineHit]:
        q = self.encoder.encode_numpy([query])[0]
        if self.asymmetric_query:
            q_rec = q
        else:
            if self.quant_mode == "base3":
                q_sym = quantize_base3(q[None, :], n_sym_per_dim=self.n_sym)
                q_read = apply_embedding_cim_noise(q_sym, self.profile, rng=self.rng)
                q_rec = dequantize_base3(
                    q_read.recovered_symbols, n_sym_per_dim=self.n_sym, dim=q.shape[0],
                )[0]
            else:
                q_sym = quantize_to_gf3(q[None, :], levels=self.encoder.config.gf3_levels)
                q_read = apply_embedding_cim_noise(q_sym, self.profile, rng=self.rng)
                q_rec = dequantize_from_gf3(
                    q_read.recovered_symbols, levels=self.encoder.config.gf3_levels,
                )[0]
        sim = cosine_similarity(q_rec, self.doc_emb)
        order = np.argsort(-sim)[:k]
        return [BaselineHit(self.docs[i].doc_id, float(sim[i])) for i in order]


class CiMLDPCBaseline:
    """LDPC-protected dense retrieval (asymmetric clean query by default)."""

    def __init__(
        self,
        docs: list[Document],
        encoder: TONELResEncoder,
        profile: NoiseProfile,
        rng: np.random.Generator,
        recovery: str = "bprime",
        use_bp: bool = True,
        code: LDPCCode | None = None,
        asymmetric_query: bool = True,
    ):
        self.docs = docs
        self.encoder = encoder
        self.profile = profile
        self.rng = rng
        self.recovery = recovery
        self.use_bp = use_bp
        self.asymmetric_query = asymmetric_query
        self.code = code or LDPCCode.load()
        clean = encoder.encode_numpy([d.text for d in docs])
        self.doc_emb = self._recover(clean)

    def _recover(self, emb: np.ndarray) -> np.ndarray:
        rec = protect_and_recover(
            emb,
            profile=self.profile,
            code=self.code,
            levels=self.encoder.config.gf3_levels,
            rng=self.rng,
            use_bp=self.use_bp,
            n_sym_per_dim=self.encoder.config.n_sym_per_dim,
            quant_mode=self.encoder.config.quant_mode,
        )
        if self.recovery == "soft":
            return rec.emb_soft
        if self.recovery == "bp":
            return rec.emb_bp
        return rec.emb_bprime

    def retrieve(self, query: str, k: int = 10) -> list[BaselineHit]:
        q = self.encoder.encode_numpy([query])[0]
        q_rec = q if self.asymmetric_query else self._recover(q[None, :])[0]
        sim = cosine_similarity(q_rec, self.doc_emb)
        order = np.argsort(-sim)[:k]
        return [BaselineHit(self.docs[i].doc_id, float(sim[i])) for i in order]
