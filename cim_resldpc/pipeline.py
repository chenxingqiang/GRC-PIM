"""End-to-end CiM-ResLDPC retrieval pipeline (Contributions 2–3).

v1.1 scientific constraints:
  - No clean-embedding oracle at index/query time.
  - Embeddings are LDPC-protected; recovery uses B' soft / BP only.
  - Hybrid BM25 is optional system mode; dense-only mode exposes the IR claim.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .baselines import BM25Baseline
from .config import CiMResLDPCConfig
from .datasets import Document
from .hybrid_score import build_bm25, fuse_scores
from .ldpc_embed import LDPCCode, protect_and_recover
from .noise_bridge import NoiseProfile, should_skip_gemv
from .residual_features import cosine_similarity, evidence_from_readout
from .noise_bridge import NoisyReadout
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
    """Three-layer co-designed edge retrieval pipeline (oracle-free)."""

    def __init__(
        self,
        config: CiMResLDPCConfig | None = None,
        encoder: TONELResEncoder | None = None,
        use_bp: bool = True,
        recovery: str = "soft",
    ):
        """
        Args:
          recovery: 'soft' (B' soft expectation), 'bp' (BP hard dequant),
                    or 'bprime' (B' hard dequant).
        """
        self.config = config or CiMResLDPCConfig()
        if not self.config.local_only:
            raise ValueError("CiM-ResLDPC requires local_only=True for edge privacy")
        if recovery not in {"soft", "bp", "bprime"}:
            raise ValueError(f"unknown recovery={recovery!r}")
        self.encoder = encoder or TONELResEncoder(self.config)
        self.use_bp = use_bp
        self.recovery = recovery
        self.code = LDPCCode.load()
        self.docs: list[Document] = []
        self.doc_ids: list[str] = []
        self.texts: list[str] = []
        self.clean_emb: np.ndarray | None = None  # training/analysis only; not used in scoring
        self.calibrated_emb: np.ndarray | None = None
        self.residual_conf: np.ndarray | None = None
        self.used_bp_flags: np.ndarray | None = None
        self.bm25 = None
        self._rng = np.random.default_rng(self.config.seed)

    @property
    def used_gemv_flags(self) -> np.ndarray | None:
        """Backward-compatible alias: GEMV path not default; BP/soft residual used."""
        return self.used_bp_flags

    @property
    def uploads_raw_documents(self) -> bool:
        return False

    def _profile(self) -> NoiseProfile:
        return NoiseProfile(
            p_indep=self.config.p_indep,
            p_drift=self.config.p_drift,
            p_row_burst=self.config.p_row_burst,
            n_cycles=self.config.n_cycles,
            drift_persist=self.config.drift_persist,
        )

    def _recover_batch(self, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns calibrated_emb, residual_conf, used_bp_flags — no clean mix."""
        rec = protect_and_recover(
            emb,
            profile=self._profile(),
            code=self.code,
            levels=self.config.gf3_levels,
            rng=self._rng,
            use_bp=self.use_bp,
            n_sym_per_dim=self.config.n_sym_per_dim,
            quant_mode=self.config.quant_mode,
        )
        if self.recovery == "soft":
            cal = rec.emb_soft
        elif self.recovery == "bp":
            cal = rec.emb_bp
        else:
            cal = rec.emb_bprime

        n = emb.shape[0]
        # Per-row drift proxy: use batch drift for all (codeword-shared channel stats)
        # Refine: variance of soft vs BP disagreement as confidence
        disagree = np.mean(np.abs(rec.emb_soft - rec.emb_bp), axis=1)
        conf = np.zeros(n, dtype=np.float32)
        used = np.zeros(n, dtype=bool)
        for i in range(n):
            dscore = float(rec.drift_score)
            skip = should_skip_gemv(dscore, tau_skip=self.config.tau_skip)
            mini = NoisyReadout(
                noisy_bits_3d=np.zeros((1, self.config.n_cycles, 2), dtype=np.int64),
                drift_score=dscore,
                scores={"global_drift": dscore},
                recovered_symbols=rec.symbols_bp[i:i + 1],
                llr_bp=np.zeros((1, 1, 3), dtype=np.float32),
            )
            ev = evidence_from_readout(
                mini,
                gate_mean=rec.gate_proxy,
                delta_mag=float(disagree[i]),
                tau_d=self.config.tau_d,
                tau_g=self.config.tau_g,
                tau_skip=self.config.tau_skip,
            )
            # Under skip: still use soft reconstruction but mark unused heavy decode
            conf[i] = ev.residual_conf
            used[i] = (not skip) and rec.used_bp
        return cal.astype(np.float32), conf, used

    def index(self, docs: list[Document]) -> None:
        self.docs = list(docs)
        self.doc_ids = [d.doc_id for d in docs]
        self.texts = [d.text for d in docs]
        self.bm25 = build_bm25(self.doc_ids, self.texts)
        self.clean_emb = self.encoder.encode_numpy(self.texts)
        cal, conf, used = self._recover_batch(self.clean_emb)
        self.calibrated_emb = cal
        self.residual_conf = conf
        self.used_bp_flags = used

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        mode: str = "hybrid",
    ) -> list[ScoredDoc]:
        return self.retrieve_with_metrics(query, k=k, mode=mode).hits

    def retrieve_with_metrics(
        self,
        query: str,
        k: int | None = None,
        mode: str = "hybrid",
    ) -> RetrieveResult:
        """
        mode:
          - 'hybrid': BM25 cutoff + dense calibrated + residual conf (system)
          - 'dense': dense-only on calibrated embeddings (scientific core)
        """
        if self.calibrated_emb is None or self.bm25 is None:
            raise RuntimeError("Call index() before retrieve()")
        if mode not in {"hybrid", "dense"}:
            raise ValueError(f"unknown mode={mode!r}")
        k = k or self.config.top_k
        t0 = time.perf_counter()

        q_emb = self.encoder.encode_numpy([query])[0]
        if self.config.asymmetric_query:
            # Digital-side query (standard local RAG): only doc store is CiM-noisy
            q_cal = q_emb
            q_used = np.array([False])
        else:
            q_cal, q_conf, q_used = self._recover_batch(q_emb[None, :])
            q_cal = q_cal[0]
        dense_full = cosine_similarity(q_cal, self.calibrated_emb)

        if mode == "dense":
            order = np.argsort(-dense_full)[:k]
            latency = (time.perf_counter() - t0) * 1000.0
            hits = [
                ScoredDoc(
                    doc_id=self.doc_ids[int(i)],
                    score=float(dense_full[i]),
                    dense_score=float(dense_full[i]),
                    bm25_score=0.0,
                    residual_conf=float(self.residual_conf[i]),
                    used_gemv=bool(self.used_bp_flags[i] or q_used[0]),
                    latency_ms=float(latency),
                )
                for i in order
            ]
            skip_rate = float(np.mean(~self.used_bp_flags)) if len(self.docs) else 0.0
            return RetrieveResult(
                query=query,
                hits=hits,
                metrics={
                    "latency_ms": float(latency),
                    "decode_skip_rate": skip_rate,
                    "mode": 0.0,  # dense
                    "query_used_bp": float(q_used[0]),
                },
            )

        # hybrid
        bm25_scores = self.bm25.score(query)
        m = min(self.config.bm25_cutoff_m, len(self.docs))
        cand = np.argsort(-bm25_scores)[:m]

        dense_scores = np.full(len(self.docs), -1.0, dtype=np.float32)
        res_conf = np.zeros(len(self.docs), dtype=np.float32)
        used_flags = np.zeros(len(self.docs), dtype=bool)
        for i in cand:
            dense_scores[i] = dense_full[i]
            res_conf[i] = self.residual_conf[i]
            used_flags[i] = self.used_bp_flags[i] or bool(q_used[0])

        fused = fuse_scores(
            bm25_scores,
            dense_scores,
            res_conf,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
        )
        fused_masked = np.full_like(fused, -1e9)
        fused_masked[cand] = fused[cand]
        order = np.argsort(-fused_masked)[:k]
        latency = (time.perf_counter() - t0) * 1000.0

        hits = [
            ScoredDoc(
                doc_id=self.doc_ids[int(i)],
                score=float(fused_masked[i]),
                dense_score=float(dense_scores[i]),
                bm25_score=float(bm25_scores[i]),
                residual_conf=float(res_conf[i]),
                used_gemv=bool(used_flags[i]),
                latency_ms=float(latency),
            )
            for i in order
        ]
        skip_rate = float(np.mean(~self.used_bp_flags)) if len(self.docs) else 0.0
        return RetrieveResult(
            query=query,
            hits=hits,
            metrics={
                "latency_ms": float(latency),
                "decode_skip_rate": skip_rate,
                "bm25_cutoff_m": float(m),
                "mode": 1.0,  # hybrid
                "query_used_bp": float(q_used[0]),
            },
        )


__all__ = [
    "CiMResLDPCPipeline",
    "Document",
    "ScoredDoc",
    "RetrieveResult",
    "BM25Baseline",
]
