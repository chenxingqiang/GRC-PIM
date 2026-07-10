"""Dense retrieval helpers that plug arbitrary TextEncoder + CiM recovery."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoders import TextEncoder
from .ldpc_embed import LDPCCode, protect_and_recover
from .noise_bridge import NoiseProfile
from .residual_features import cosine_similarity


@dataclass
class DenseHit:
    doc_id: str
    score: float


class EncoderDenseIndex:
    """Asymmetric dense index: encode once, optionally CiM-protect documents."""

    def __init__(
        self,
        doc_ids: list[str],
        texts: list[str],
        encoder: TextEncoder,
        profile: NoiseProfile | None = None,
        recovery: str = "bprime",
        use_bp: bool = True,
        use_gemv: bool = False,
        gemv_calibrator=None,
        rng: np.random.Generator | None = None,
        n_sym_per_dim: int = 2,
        quant_mode: str = "base3",
        code: LDPCCode | None = None,
        apply_cim: bool = True,
    ):
        self.doc_ids = list(doc_ids)
        self.encoder = encoder
        self.profile = profile
        self.recovery = recovery
        self.use_bp = use_bp
        self.use_gemv = use_gemv
        self.gemv_calibrator = gemv_calibrator
        self.rng = rng or np.random.default_rng(0)
        self.n_sym_per_dim = n_sym_per_dim
        self.quant_mode = quant_mode
        self.code = code or LDPCCode.load()
        self.clean_emb = encoder.encode(texts)
        if apply_cim:
            if profile is None:
                raise ValueError("profile required when apply_cim=True")
            self.doc_emb = self._recover(self.clean_emb)
        else:
            self.doc_emb = self.clean_emb

    def _recover(self, emb: np.ndarray) -> np.ndarray:
        rec = protect_and_recover(
            emb,
            profile=self.profile,
            code=self.code,
            rng=self.rng,
            use_bp=self.use_bp,
            n_sym_per_dim=self.n_sym_per_dim,
            quant_mode=self.quant_mode,
            use_gemv=self.use_gemv,
            gemv_calibrator=self.gemv_calibrator,
        )
        if self.recovery == "soft":
            return rec.emb_soft
        if self.recovery == "bp":
            return rec.emb_bp
        return rec.emb_bprime

    def retrieve(self, query: str, k: int = 10) -> list[DenseHit]:
        q = self.encoder.encode([query])[0]
        sim = cosine_similarity(q, self.doc_emb)
        order = np.argsort(-sim)[:k]
        return [DenseHit(self.doc_ids[int(i)], float(sim[i])) for i in order]
