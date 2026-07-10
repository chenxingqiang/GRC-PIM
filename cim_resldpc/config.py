"""Default hyperparameters for CiM-ResLDPC (see docs/design/CIM_RESLDPC_DESIGN.md)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CiMResLDPCConfig:
    embed_dim: int = 64
    n_cycles: int = 4
    tau_d: float = 0.025
    tau_g: float = 0.6
    tau_skip: float = 0.015
    lambda_res: float = 0.5
    lambda_gf3: float = 0.1
    alpha: float = 0.3  # BM25
    beta: float = 0.5   # dense
    gamma: float = 0.2  # residual confidence
    bm25_cutoff_m: int = 32
    p_indep: float = 5e-3
    p_drift: float = 5e-3
    p_row_burst: float = 5e-3
    drift_persist: int = 3
    top_k: int = 10
    local_only: bool = True
    seed: int = 42
    # GF(3) packing: each float dim maps to one ternary symbol after quantization
    gf3_levels: tuple[float, float, float] = (-1.0, 0.0, 1.0)
    learning_rate: float = 1e-3
    contrastive_temp: float = 0.07

    def as_dict(self) -> dict:
        return {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in self.__dict__.items()
        }
