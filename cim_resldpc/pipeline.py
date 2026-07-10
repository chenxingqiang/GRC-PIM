"""End-to-end CiM-ResLDPC retrieval pipeline (Contributions 2–3)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .baselines import BM25Baseline
from .config import CiMResLDPCConfig
from .datasets import Document
from .gf3_quantize import dequantize_from_gf3, quantize_to_gf3
from .hybrid_score import build_bm25, fuse_scores
from .noise_bridge import NoiseProfile, apply_embedding_cim_noise, should_skip_gemv
from .residual_features import evidence_from_readout, cosine_similarity
from .tonel import TONELResEncoder


@dataclass
class ScoredDoc:
    doc_id: str
    score: float
    dense_score: float
    bm25_score: float
    residual_conf: float
    used_gemv: bool
    latency_ms: float


@dataclass
class RetrieveResult:
    query: str
    hits: list[ScoredDoc]
    metrics: dict[str, float] = field(default_factory=dict)


class CiMResLDPCPipeline:
    """Three-layer co-designed edge retrieval pipeline."""

    def __init__(
        self,
        config: CiMResLDPCConfig | None = None,
        encoder: TONELResEncoder | None = None,
    ):
        self.config = config or CiMResLDPCConfig()
        if not self.config.local_only:
            raise ValueError("CiM-ResLDPC requires local_only=True for edge privacy")
        self.encoder = encoder or TONELResEncoder(self.config)
        self.docs: list[Document] = []
        self.doc_ids: list[str] = []
        self.texts: list[str] = []
        self.clean_emb: np.ndarray | None = None
        self.stored_symbols: np.ndarray | None = None
        self.calibrated_emb: np.ndarray | None = None
        self.residual_conf: np.ndarray | None = None
        self.used_gemv_flags: np.ndarray | None = None
        self.bm25 = None
        self._rng = np.random.default_rng(self.config.seed)

    @property
    def uploads_raw_documents(self) -> bool:
        """Privacy contract: never upload raw documents."""
        return False

    def index(self, docs: list[Document]) -> None:
        self.docs = list(docs)
        self.doc_ids = [d.doc_id for d in docs]
        self.texts = [d.text for d in docs]
        self.bm25 = build_bm25(self.doc_ids, self.texts)
        self.clean_emb = self.encoder.encode_numpy(self.texts)
        self.stored_symbols = quantize_to_gf3(
            self.clean_emb, levels=self.config.gf3_levels,
        )
        profile = NoiseProfile(
            p_indep=self.config.p_indep,
            p_drift=self.config.p_drift,
            p_row_burst=self.config.p_row_burst,
            n_cycles=self.config.n_cycles,
            drift_persist=self.config.drift_persist,
        )
        readout = apply_embedding_cim_noise(
            self.stored_symbols, profile, rng=self._rng,
        )
        # Per-doc residual evidence (batch-level drift used as proxy; refine per row)
        n = len(docs)
        conf = np.zeros(n, dtype=np.float32)
        used = np.zeros(n, dtype=bool)
        # Approximate per-doc drift via per-row bit variance
        bits = readout.noisy_bits_3d.astype(np.float32)
        per_doc_drift = bits.var(axis=1).mean(axis=1) * 4.0  # (B,)
        # Soft recovery always available from B'
        recovered = readout.recovered_symbols
        # Optional "GEMV" path: when not skipped, blend clean+recovered (proxy for ΔLLR)
        cal = np.zeros_like(self.clean_emb)
        for i in range(n):
            dscore = float(per_doc_drift[i])
            skip = should_skip_gemv(dscore, tau_skip=self.config.tau_skip)
            # Proxy gate from per-doc LLR sharpness
            llr = readout.llr_bp[i]
            probs = np.exp(llr - llr.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True) + 1e-8
            ent = -(probs * np.log(probs + 1e-8)).sum(axis=-1).mean()
            gate_mean = float(np.clip(1.0 - ent / np.log(3.0), 0.0, 1.0))
            delta_mag = float(np.mean(np.abs(
                self.clean_emb[i] - dequantize_from_gf3(
                    recovered[i], levels=self.config.gf3_levels,
                )
            )))
            # Build a mini readout-like object for evidence helper
            from .noise_bridge import NoisyReadout

            mini = NoisyReadout(
                noisy_bits_3d=readout.noisy_bits_3d[i:i + 1],
                drift_score=dscore,
                scores={"global_drift": dscore},
                recovered_symbols=recovered[i:i + 1],
                llr_bp=readout.llr_bp[i:i + 1],
            )
            ev = evidence_from_readout(
                mini,
                gate_mean=gate_mean,
                delta_mag=delta_mag,
                tau_d=self.config.tau_d,
                tau_g=self.config.tau_g,
                tau_skip=self.config.tau_skip,
            )
            conf[i] = ev.residual_conf
            used[i] = ev.used_gemv
            rec_f = dequantize_from_gf3(recovered[i], levels=self.config.gf3_levels)
            if ev.used_gemv:
                # Residual-calibrated blend toward clean teacher (simulates ΔLLR correction)
                cal[i] = 0.5 * rec_f + 0.5 * self.clean_emb[i]
            else:
                cal[i] = rec_f
        self.calibrated_emb = cal.astype(np.float32)
        self.residual_conf = conf
        self.used_gemv_flags = used

    def retrieve(self, query: str, k: int | None = None) -> list[ScoredDoc]:
        return self.retrieve_with_metrics(query, k=k).hits

    def retrieve_with_metrics(self, query: str, k: int | None = None) -> RetrieveResult:
        if self.calibrated_emb is None or self.bm25 is None:
            raise RuntimeError("Call index() before retrieve()")
        k = k or self.config.top_k
        t0 = time.perf_counter()

        bm25_scores = self.bm25.score(query)
        # Hybrid first-stage cutoff
        m = min(self.config.bm25_cutoff_m, len(self.docs))
        cand = np.argsort(-bm25_scores)[:m]

        q_emb = self.encoder.encode_numpy([query])[0]
        # Query-side CiM noise + optional residual path
        profile = NoiseProfile(
            p_indep=self.config.p_indep,
            p_drift=self.config.p_drift,
            p_row_burst=self.config.p_row_burst,
            n_cycles=self.config.n_cycles,
            drift_persist=self.config.drift_persist,
        )
        q_sym = quantize_to_gf3(q_emb[None, :], levels=self.config.gf3_levels)
        q_read = apply_embedding_cim_noise(q_sym, profile, rng=self._rng)
        q_drift = q_read.drift_score
        q_skip = should_skip_gemv(q_drift, tau_skip=self.config.tau_skip)
        q_rec = dequantize_from_gf3(
            q_read.recovered_symbols, levels=self.config.gf3_levels,
        )[0]
        if q_skip:
            q_cal = q_rec
            q_used = False
        else:
            # Blend with clean query embedding as residual correction proxy
            q_cal = 0.5 * q_rec + 0.5 * q_emb
            q_used = q_drift >= self.config.tau_d

        dense_full = cosine_similarity(q_cal, self.calibrated_emb)
        # Only score candidates deeply; others get -inf dense for fusion mask
        dense_scores = np.full(len(self.docs), -1.0, dtype=np.float32)
        res_conf = np.zeros(len(self.docs), dtype=np.float32)
        used_flags = np.zeros(len(self.docs), dtype=bool)
        for i in cand:
            dense_scores[i] = dense_full[i]
            res_conf[i] = self.residual_conf[i]
            used_flags[i] = self.used_gemv_flags[i] or q_used

        fused = fuse_scores(
            bm25_scores,
            dense_scores,
            res_conf,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
        )
        # Restrict ranking to candidates
        fused_masked = np.full_like(fused, -1e9)
        fused_masked[cand] = fused[cand]
        order = np.argsort(-fused_masked)[:k]
        latency = (time.perf_counter() - t0) * 1000.0

        hits = []
        for i in order:
            hits.append(
                ScoredDoc(
                    doc_id=self.doc_ids[int(i)],
                    score=float(fused_masked[i]),
                    dense_score=float(dense_scores[i]),
                    bm25_score=float(bm25_scores[i]),
                    residual_conf=float(res_conf[i]),
                    used_gemv=bool(used_flags[i]),
                    latency_ms=float(latency),
                )
            )
        skip_rate = float(np.mean(~self.used_gemv_flags)) if len(self.docs) else 0.0
        return RetrieveResult(
            query=query,
            hits=hits,
            metrics={
                "latency_ms": float(latency),
                "decode_skip_rate": skip_rate,
                "bm25_cutoff_m": float(m),
                "query_used_gemv": float(q_used),
            },
        )


# Re-export Document for package root convenience
__all__ = [
    "CiMResLDPCPipeline",
    "Document",
    "ScoredDoc",
    "RetrieveResult",
    "BM25Baseline",
]
