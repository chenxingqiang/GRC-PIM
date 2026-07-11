"""TONEL-Res: residual-aware embedding learning for CiM-ResLDPC."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CiMResLDPCConfig
from .gf3_quantize import quantize_to_gf3
from .ldpc_embed import LDPCCode, protect_and_recover
from .noise_bridge import NoiseProfile, residual_gate_weight


def stable_token_id(token: str, n_buckets: int) -> int:
    """Process-stable hash (unlike Python's salted hash())."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % n_buckets


class HashBagEncoder(nn.Module):
    """Lightweight bag-of-hash embedding (no external tokenizer / downloads)."""

    def __init__(self, dim: int = 64, n_buckets: int = 4096):
        super().__init__()
        self.dim = dim
        self.n_buckets = n_buckets
        self.embed = nn.Embedding(n_buckets, dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.proj = nn.Linear(dim, dim)

    def _hash_tokens(self, texts: list[str]) -> torch.Tensor:
        batch = []
        for text in texts:
            toks = [t for t in text.lower().split() if t]
            if not toks:
                toks = ["<empty>"]
            ids = [stable_token_id(t, self.n_buckets) for t in toks]
            idx = torch.tensor(ids, dtype=torch.long)
            vec = self.embed(idx).mean(dim=0)
            batch.append(vec)
        return torch.stack(batch, dim=0)

    def forward(self, texts: list[str]) -> torch.Tensor:
        x = self._hash_tokens(texts)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


@dataclass
class TONELLossBreakdown:
    retrieval: float
    residual_align: float
    quant: float
    total: float
    residual_weight: float


class TONELResEncoder(nn.Module):
    """Residual soft-info regularized dense encoder (Contribution 1).

    Alignment target is ECC-calibrated reconstruction from the CiM channel
    (B' soft expectation / BP), never a clean-embedding oracle blend.
    """

    def __init__(self, config: CiMResLDPCConfig | None = None):
        super().__init__()
        self.config = config or CiMResLDPCConfig()
        self.encoder = HashBagEncoder(dim=self.config.embed_dim)
        self.opt = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        self._code: LDPCCode | None = None

    @property
    def code(self) -> LDPCCode:
        if self._code is None:
            self._code = LDPCCode.load()
        return self._code

    def encode(self, texts: list[str]) -> torch.Tensor:
        return self.encoder(texts)

    def encode_numpy(self, texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            return self.encode(texts).cpu().numpy().astype(np.float32)

    def quantize_gf3(self, emb: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        return quantize_to_gf3(emb, levels=self.config.gf3_levels)

    def residual_regularizer(
        self,
        emb: torch.Tensor,
        calibrated_emb: torch.Tensor,
        weight: float,
    ) -> torch.Tensor:
        """Cosine alignment to channel-calibrated reconstruction."""
        emb_n = F.normalize(emb, dim=-1)
        cal_n = F.normalize(calibrated_emb, dim=-1)
        align = 1.0 - (emb_n * cal_n).sum(dim=-1).mean()
        return weight * align

    def quant_loss(self, emb: torch.Tensor) -> torch.Tensor:
        """Soft ternary quantization penalty toward {-1,0,1}."""
        levels = torch.tensor(self.config.gf3_levels, dtype=emb.dtype, device=emb.device)
        dist = (emb.unsqueeze(-1) - levels).abs()
        min_dist = dist.min(dim=-1).values
        return min_dist.mean()

    def contrastive_loss(
        self,
        q: torch.Tensor,
        pos: torch.Tensor,
        negs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        temp = self.config.contrastive_temp
        q = F.normalize(q, dim=-1)
        pos = F.normalize(pos, dim=-1)
        if negs is None:
            logits = (q @ pos.T) / temp
            labels = torch.arange(q.size(0), device=q.device)
            return F.cross_entropy(logits, labels)
        negs = F.normalize(negs, dim=-1)
        pos_logit = (q * pos).sum(dim=-1, keepdim=True) / temp
        neg_logit = (q @ negs.T) / temp
        logits = torch.cat([pos_logit, neg_logit], dim=-1)
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        return F.cross_entropy(logits, labels)

    def _calibrated_embedding_from_channel(
        self,
        emb_np: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        """LDPC-protected CiM readout → soft reconstruction (no clean blend)."""
        profile = NoiseProfile(
            p_indep=self.config.p_indep,
            p_drift=self.config.p_drift,
            p_row_burst=self.config.p_row_burst,
            n_cycles=self.config.n_cycles,
            drift_persist=self.config.drift_persist,
        )
        rec = protect_and_recover(
            emb_np,
            profile=profile,
            code=self.code,
            levels=self.config.gf3_levels,
            rng=rng,
            use_bp=True,
            n_sym_per_dim=self.config.n_sym_per_dim,
            quant_mode=self.config.quant_mode,
        )
        # Prefer B' hard recon under base3 (stable geometry); soft as residual cue.
        # Aligning to BP/B' reconstruction teaches noise-robust codes without oracle.
        cal = rec.emb_bprime
        w = residual_gate_weight(
            rec.drift_score,
            rec.gate_proxy,
            tau_d=self.config.tau_d,
            tau_g=self.config.tau_g,
        )
        return cal.astype(np.float32), w

    def train_step(
        self,
        queries: list[str],
        positives: list[str],
        rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        if rng is None:
            rng = np.random.default_rng(self.config.seed)
        self.train()
        self.opt.zero_grad()
        q = self.encode(queries)
        p = self.encode(positives)
        l_retr = self.contrastive_loss(q, p)

        with torch.no_grad():
            p_np = p.detach().cpu().numpy().astype(np.float32)
            cal_np, w = self._calibrated_embedding_from_channel(p_np, rng)
        cal = torch.from_numpy(cal_np).to(p.device)
        l_res = self.residual_regularizer(p, cal, weight=w)
        l_q = self.quant_loss(p)

        total = (
            l_retr
            + self.config.lambda_res * l_res
            + self.config.lambda_gf3 * l_q
        )
        total.backward()
        self.opt.step()
        return {
            "retrieval": float(l_retr.detach()),
            "residual_align": float(l_res.detach()),
            "quant": float(l_q.detach()),
            "total": float(total.detach()),
            "residual_weight": float(w),
        }
