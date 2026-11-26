"""GEMV-friendly residual LLR calibrator for multi-cycle PIM reads.

This model keeps the classical T-cycle soft-LLR product as the anchor and
learns only a bounded residual correction. The architecture is a shared tiny
MLP applied independently to every code symbol, so inference is just batched
GEMV over compact per-symbol features.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import GF, encode, init_llr_from_bits
from .noise import bits_to_symbols
from .noise3d import apply_pim_noise_3d


GEMV_FEATURES = 13


def global_drift_score(noisy_bits_3d: np.ndarray) -> float:
    """Cheap global score for temporal correlation / drift strength.

    The score is the mean per-bit disagreement across repeated reads, scaled to
    [0, 1]. In the synthetic channels used here, iid is around 0.015, weak drift
    around 0.02, and strong drift around 0.04 at p_bit=5e-3.
    """
    return float(noisy_bits_3d.astype(np.float32).var(axis=1).mean() * 4.0)


DEFAULT_DRIFT_WEIGHTS: dict[str, float] = {
    "global_drift": 0.35,
    "column_persistence": 0.25,
    "row_burst": 0.20,
    "syndrome_instability": 0.20,
}


def column_persistence_score(noisy_bits_3d: np.ndarray) -> float:
    """Per-column temporal correlation score in [0, 1].

    Measures how many columns (bit positions) show elevated temporal
    variance, indicating column-correlated drift where specific bit
    positions flip consistently across cycles.
    """
    per_bit_var = noisy_bits_3d.astype(np.float32).var(axis=1)  # (B, 2n)
    per_col_var = per_bit_var.mean(axis=0)  # (2n,) mean var per column
    k = max(1, int(per_col_var.shape[0] * 0.2))
    top_k = np.sort(per_col_var)[-k:]
    return float(np.clip(top_k.mean() * 16.0, 0.0, 1.0))


def row_burst_score(noisy_bits_3d: np.ndarray) -> float:
    """Row-level burst error score in [0, 1].

    Measures whether high-disagreement row-cycles are concentrated in
    a few rows rather than spread evenly across the batch, indicating
    row-level burst errors.  Uses per-row maximum per-cycle disagreement
    spread (top-10% vs bottom-10% mean), normalised by the overall mean.
    """
    batch_size, n_cycles, two_n = noisy_bits_3d.shape
    n = two_n // 2
    bits = noisy_bits_3d.astype(np.float32).reshape(batch_size, n_cycles, n, 2)

    # Per-cycle majority vote (across the batch)
    cycle_majority = (bits.mean(axis=0, keepdims=True)
                      >= 0.5).astype(np.float32)

    # Per row-cycle: fraction of bits disagreeing with batch majority
    per_row_cycle_disagree = np.abs(
        bits - cycle_majority).mean(axis=(2, 3))          # (B, T)

    # Worst cycle per row
    per_row_max = per_row_cycle_disagree.max(axis=1)       # (B,)

    # Top-10% vs bottom-10% spread normalised by mean
    top_n = int(max(1, batch_size * 0.1))
    sorted_vals = np.sort(per_row_max)
    top_mean = float(sorted_vals[-top_n:].mean())
    btm_mean = float(sorted_vals[:top_n].mean())
    all_mean = float(max(per_row_max.mean(), 1e-12))
    spread = (top_mean - btm_mean) / all_mean
    return float(np.clip(spread, 0.0, 1.0))


def syndrome_instability(noisy_bits_3d: np.ndarray,
                          H: np.ndarray) -> float:
    """NB-LDPC syndrome instability score in [0, 1].

    Computes the syndrome for each read cycle using the current noisy
    symbols, then measures how often the syndrome changes between cycles
    (random flip noise produces unstable syndromes) and how active the
    syndromes are overall. Requires the parity-check matrix H.
    """
    batch_size, n_cycles, two_n = noisy_bits_3d.shape
    n = two_n // 2
    m = H.shape[0]
    all_syns = np.zeros((batch_size, n_cycles, m), dtype=np.int64)
    for t in range(n_cycles):
        syms = bits_to_symbols(noisy_bits_3d[:, t, :]) % GF
        all_syns[:, t, :] = (syms @ H.T) % GF
    if n_cycles >= 2:
        diffs = [(all_syns[:, t] != all_syns[:, t - 1]).mean()
                 for t in range(1, n_cycles)]
        change_rate = float(np.mean(diffs))
    else:
        change_rate = 0.0
    activity = float(all_syns.astype(bool).astype(np.float32).mean())
    score = (change_rate + activity) / 2.0
    return float(np.clip(score, 0.0, 1.0))


def composite_drift_score(
    scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted combination of drift sub-scores into [0, 1] scalar."""
    if weights is None:
        weights = DEFAULT_DRIFT_WEIGHTS
    total = sum(weights.get(k, 0.0) * scores.get(k, 0.0) for k in weights)
    return float(np.clip(total, 0.0, 1.0))


def enhanced_drift_scores(
    noisy_bits_3d: np.ndarray,
    H: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all drift detection signals.

    Returns a dict with keys: global_drift, column_persistence, row_burst,
    syndrome_instability, and composite. When H is None, syndrome_instability
    and composite are set to NaN.
    """
    scores: dict[str, float] = {
        "global_drift": global_drift_score(noisy_bits_3d),
        "column_persistence": column_persistence_score(noisy_bits_3d),
        "row_burst": row_burst_score(noisy_bits_3d),
    }
    if H is not None:
        scores["syndrome_instability"] = syndrome_instability(noisy_bits_3d, H)
        scores["composite"] = composite_drift_score(scores)
    else:
        scores["syndrome_instability"] = float("nan")
        scores["composite"] = float("nan")
    return scores


def soft_llr_product(noisy_bits_3d: np.ndarray, p_bit: float) -> np.ndarray:
    """T-cycle soft-LLR product baseline B'."""
    n_cycles = noisy_bits_3d.shape[1]
    llr = sum(init_llr_from_bits(noisy_bits_3d[:, t, :], p_bit=p_bit)
              for t in range(n_cycles))
    llr -= llr.max(axis=-1, keepdims=True)
    return llr.astype(np.float32)


def weighted_soft_llr_product(noisy_bits_3d: np.ndarray, p_bit: float,
                              rho: float) -> np.ndarray:
    """Geometrically weighted T-cycle soft-LLR product (weighted-BP baseline).

    Cycle ``t`` (0-based loop index) contributes with weight ``rho**t``,
    i.e. cycle ``t`` counting from 1 carries ``rho**(t-1)`` and the first
    cycle always has weight 1. ``rho=1`` reproduces the
    plain B' product exactly; ``rho < 1`` discounts the later cycles, which
    is the natural correction for the assumption B' violates: drift
    persistence keeps a drifted column corrupted for consecutive reads, so
    the T reads are not conditionally independent evidence. Feeding this
    prior to the same BP decoder isolates the effect of re-weighting the
    multi-cycle evidence from the learned residual correction.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho!r}")
    n_cycles = noisy_bits_3d.shape[1]
    llr = sum(rho ** t * init_llr_from_bits(noisy_bits_3d[:, t, :], p_bit=p_bit)
              for t in range(n_cycles))
    llr -= llr.max(axis=-1, keepdims=True)
    return llr.astype(np.float32)


def build_gemv_features(noisy_bits_3d: np.ndarray,
                        p_bit: float,
                        raw_bits: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Build compact per-symbol features for the residual calibrator.

    The 13 default features are all invariant to the order of the T reads,
    so they cannot express *when* a column disagreed -- which is where drift
    persistence lives. ``raw_bits=True`` appends the symbol's 2T reads in
    +-1 coding (column ``13 + 2t + b`` holds cycle ``t``, bit ``b``).

    Returns:
      features: (B, n, 13 [+ 2T]), for a shared per-symbol GEMV/MLP.
      anchor:   (B, n, 3), B' soft-LLR product.
    """
    batch_size, n_cycles, two_n = noisy_bits_3d.shape
    n_symbols = two_n // 2
    llr_bp = soft_llr_product(noisy_bits_3d, p_bit)

    sym_counts = np.zeros((batch_size, n_symbols, 4), dtype=np.float32)
    for t in range(n_cycles):
        sym_t = bits_to_symbols(noisy_bits_3d[:, t, :])
        for symbol in range(4):
            sym_counts[:, :, symbol] += (sym_t == symbol)
    sym_counts /= float(n_cycles)

    bits = noisy_bits_3d.reshape(batch_size, n_cycles, n_symbols, 2).astype(np.float32)
    bit_mean = bits.mean(axis=1)
    bit_disagree = bits.var(axis=1) * 4.0

    probs = np.exp(llr_bp - llr_bp.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=-1, keepdims=True)

    col = np.linspace(-1.0, 1.0, n_symbols, dtype=np.float32)
    col = col[None, :, None].repeat(batch_size, axis=0)

    llr_scale = llr_bp / 20.0
    parts = [llr_scale, sym_counts, bit_mean, bit_disagree,
             entropy.astype(np.float32), col]
    if raw_bits:
        seq = bits.transpose(0, 2, 1, 3).reshape(batch_size, n_symbols, 2 * n_cycles)
        parts.append(2.0 * seq - 1.0)
    features = np.concatenate(parts, axis=-1)
    return features.astype(np.float32), llr_bp.astype(np.float32)


class GEMVResidualCalibrator(nn.Module):
    """Shared per-symbol residual model: anchor + gate * clipped_delta.

    ``raw_bit_cycles = T > 0`` selects the order-aware variant whose input
    adds the 2T raw reads of each symbol (``build_gemv_features(raw_bits=True)``).
    """

    def __init__(self, in_features: int | None = None,
                 hidden: int = 24, clip: float = 2.0,
                 raw_bit_cycles: int = 0):
        super().__init__()
        if in_features is None:
            in_features = GEMV_FEATURES + 2 * int(raw_bit_cycles)
        self.raw_bits = raw_bit_cycles > 0
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, GF)
        self.gate = nn.Linear(in_features, 1)
        self.clip = clip

    def forward(self, features: torch.Tensor,
                llr_anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(features))
        delta = self.clip * torch.tanh(self.fc2(h) / self.clip)
        gate = torch.sigmoid(self.gate(features))
        return llr_anchor + gate * delta, gate

    @torch.no_grad()
    def soft_llr(self, noisy_bits_3d: np.ndarray, p_bit: float,
                 device: str = "cpu", chunk: int = 512) -> tuple[np.ndarray, float]:
        self.eval()
        out = []
        gates = []
        for start in range(0, noisy_bits_3d.shape[0], chunk):
            feats, llr_bp = build_gemv_features(noisy_bits_3d[start:start + chunk], p_bit,
                                                raw_bits=self.raw_bits)
            x = torch.from_numpy(feats).to(device)
            anchor = torch.from_numpy(llr_bp).to(device)
            llr, gate = self.forward(x, anchor)
            llr = llr - llr.max(dim=-1, keepdim=True).values
            out.append(llr.cpu().numpy().astype(np.float32))
            gates.append(float(gate.mean().cpu()))
        return np.concatenate(out, axis=0), float(np.mean(gates))

    @torch.no_grad()
    def diagnose(self, noisy_bits_3d: np.ndarray, p_bit: float,
                 clean_symbols: np.ndarray | None = None,
                 device: str = "cpu", chunk: int = 512) -> dict[str, np.ndarray]:
        """Extract internal model state for distribution mismatch analysis.

        Returns dict with keys:
          hidden_mean: (B,) hidden layer activation magnitude
          hidden_act: (B, hidden) raw hidden activations (after ReLU)
          delta:     (B, n, 3) residual correction per symbol per class
          delta_mag: (B,) mean absolute delta across all symbols/classes
          gate:      (B, n, 1) per-symbol gate sigmoid value
          gate_mean: (B,) mean gate per batch item
          llr_out:   (B, n, 3) final LLR output
          logits:    (B, n, 3) pre-softmax LLR before class-dim normalisation
          ce_loss_per_symbol: (B, n) per-symbol cross-entropy (if clean_symbols given)
          ce_loss_mean: (B,) mean CE per batch item (if clean_symbols given)
        """
        self.eval()
        result: dict[str, list[np.ndarray]] = {
            "hidden_act": [], "hidden_mean": [], "delta": [], "delta_mag": [],
            "gate": [], "gate_mean": [], "llr_out": [], "logits": [],
        }
        if clean_symbols is not None:
            result["ce_loss_per_symbol"] = []
            result["ce_loss_mean"] = []

        for start in range(0, noisy_bits_3d.shape[0], chunk):
            end = min(start + chunk, noisy_bits_3d.shape[0])
            feats, llr_bp = build_gemv_features(noisy_bits_3d[start:end], p_bit,
                                                raw_bits=self.raw_bits)
            x = torch.from_numpy(feats).to(device)
            anchor = torch.from_numpy(llr_bp).to(device)

            h = F.relu(self.fc1(x))  # (chunk, n, hidden)
            delta_raw = self.fc2(h)  # (chunk, n, 3)
            delta = self.clip * torch.tanh(delta_raw / self.clip)
            gate_t = torch.sigmoid(self.gate(x))
            llr_raw = anchor + gate_t * delta
            llr = llr_raw - llr_raw.max(dim=-1, keepdim=True).values

            result["hidden_act"].append(h.cpu().numpy().astype(np.float32))
            result["hidden_mean"].append(h.abs().mean(dim=(-1, -2)).cpu().numpy().astype(np.float32))
            result["delta"].append(delta.cpu().numpy().astype(np.float32))
            result["delta_mag"].append(delta.abs().mean(dim=(-1, -2)).cpu().numpy().astype(np.float32))
            result["gate"].append(gate_t.squeeze(-1).cpu().numpy().astype(np.float32))
            result["gate_mean"].append(gate_t.mean(dim=(-1, -2)).cpu().numpy().astype(np.float32))
            result["llr_out"].append(llr.cpu().numpy().astype(np.float32))
            result["logits"].append(llr_raw.cpu().numpy().astype(np.float32))

            if clean_symbols is not None:
                y_t = torch.from_numpy(clean_symbols[start:end]).long().to(device)
                n_sym = y_t.shape[1]
                ce = F.cross_entropy(
                    llr_raw.reshape(-1, GF), y_t.reshape(-1), reduction="none",
                ).cpu().numpy().astype(np.float32).reshape(end - start, n_sym)
                result["ce_loss_per_symbol"].append(ce)
                result["ce_loss_mean"].append(ce.mean(axis=-1))

        out: dict[str, np.ndarray] = {}
        for k, v in result.items():
            out[k] = np.concatenate(v, axis=0)
        return out


class LinearResidualCalibrator(nn.Module):
    """Hidden-layer-free variant of GEMVResidualCalibrator.

    Identical anchor/gate/clip semantics, but the correction head is a single
    affine map of the same 13 features (56 parameters vs the GEMV head's 425).
    It isolates the value of the non-linear hidden layer at matched features,
    training data, loss and gating.
    """

    def __init__(self, in_features: int = GEMV_FEATURES, clip: float = 2.0):
        super().__init__()
        self.fc = nn.Linear(in_features, GF)
        self.gate = nn.Linear(in_features, 1)
        self.clip = clip

    def forward(self, features: torch.Tensor,
                llr_anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.clip * torch.tanh(self.fc(features) / self.clip)
        gate = torch.sigmoid(self.gate(features))
        return llr_anchor + gate * delta, gate

    @torch.no_grad()
    def soft_llr(self, noisy_bits_3d: np.ndarray, p_bit: float,
                 device: str = "cpu", chunk: int = 512) -> tuple[np.ndarray, float]:
        self.eval()
        out = []
        gates = []
        for start in range(0, noisy_bits_3d.shape[0], chunk):
            feats, llr_bp = build_gemv_features(noisy_bits_3d[start:start + chunk], p_bit)
            x = torch.from_numpy(feats).to(device)
            anchor = torch.from_numpy(llr_bp).to(device)
            llr, gate = self.forward(x, anchor)
            llr = llr - llr.max(dim=-1, keepdim=True).values
            out.append(llr.cpu().numpy().astype(np.float32))
            gates.append(float(gate.mean().cpu()))
        return np.concatenate(out, axis=0), float(np.mean(gates))


def make_training_batch_gemv(
    G: np.ndarray,
    batch_size: int,
    n_cycles: int,
    p_indep: float,
    p_drift: float,
    p_row_burst: float,
    rng: np.random.Generator,
    drift_persist: int = 3,
    raw_bits: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build features, B' anchor, and clean GF(3) targets."""
    info = rng.integers(0, GF, size=(batch_size, G.shape[0]))
    clean = encode(info, G)
    noisy_bits_3d = apply_pim_noise_3d(
        clean, n_cycles=n_cycles, p_indep=p_indep,
        p_drift=p_drift, p_row_burst=p_row_burst, rng=rng,
        drift_persist=drift_persist,
    )
    features, llr_bp = build_gemv_features(noisy_bits_3d, p_indep, raw_bits=raw_bits)
    return features, llr_bp, clean


class PerColumnAffineCalibrator(nn.Module):
    """Per-column affine LLR calibration: LLR_out[c, s] = a_c * LLR_B'[c, s] + b_c.

    This is the simplest learned baseline: a per-column scale and bias applied
    to the B' soft-LLR product. It tests whether the GEMV model's additional
    features and non-linearities provide meaningful benefit over a trivial
    per-column affine transform.
    """

    def __init__(self, n_columns: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_columns))
        self.bias = nn.Parameter(torch.zeros(n_columns))

    def forward(self, llr_bp: torch.Tensor) -> torch.Tensor:
        """Apply per-column affine transform to B' LLR.

        Args:
          llr_bp: (B, n, 3) B' soft-LLR product anchor.
        Returns:
          (B, n, 3) calibrated LLR.
        """
        return llr_bp * self.scale[None, :, None] + self.bias[None, :, None]

    @torch.no_grad()
    def soft_llr(self, noisy_bits_3d: np.ndarray, p_bit: float,
                 device: str = "cpu", chunk: int = 512) -> np.ndarray:
        """Apply per-column affine calibration to batch data."""
        self.eval()
        out = []
        for start in range(0, noisy_bits_3d.shape[0], chunk):
            _, llr_bp = build_gemv_features(noisy_bits_3d[start:start + chunk], p_bit)
            anchor = torch.from_numpy(llr_bp).to(device)
            llr = self.forward(anchor)
            llr = llr - llr.max(dim=-1, keepdim=True).values
            out.append(llr.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)


def train_affine_calibrator(
    G: np.ndarray,
    n_cycles: int = 4,
    p_indep_range: tuple[float, float] = (1e-4, 5e-3),
    p_drift: float = 5e-3,
    p_row_burst: float = 5e-3,
    n_steps: int = 900,
    batch: int = 128,
    lr: float = 2e-3,
    residual_l2: float = 1e-2,
    device: str = "cpu",
    seed: int = 0,
    log_every: int = 100,
) -> PerColumnAffineCalibrator:
    """Train a per-column affine LLR calibrator.

    Shares the same training data (synthetic multi-cycle PIM reads under
    drift/burst noise) and loss function (cross-entropy + L2 residual penalty)
    as the GEMV residual calibrator, but the model is a trivial per-column
    affine transform.
    """
    n_symbols = G.shape[1]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = PerColumnAffineCalibrator(n_columns=n_symbols).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for step in range(n_steps):
        p_bit = float(np.exp(rng.uniform(np.log(p_indep_range[0]),
                                          np.log(p_indep_range[1]))))
        _, llr_bp, clean = make_training_batch_gemv(
            G, batch_size=batch, n_cycles=n_cycles, p_indep=p_bit,
            p_drift=p_drift, p_row_burst=p_row_burst, rng=rng,
        )
        anchor = torch.from_numpy(llr_bp).to(device)
        y = torch.from_numpy(clean).long().to(device)

        out = model(anchor)
        loss = F.cross_entropy(out.reshape(-1, GF), y.reshape(-1))
        loss = loss + residual_l2 * torch.mean((out - anchor) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            acc = (out.argmax(dim=-1) == y).float().mean().item()
            print(f"[train-affine] step {step:5d}  p_bit={p_bit:.1e}  "
                  f"loss={loss.item():.4f}  symbol-acc={acc:.4f}")
    return model


def train_gemv_calibrator(
    G: np.ndarray,
    n_cycles: int = 4,
    p_indep_range: tuple[float, float] = (1e-4, 5e-3),
    p_drift: float = 5e-3,
    p_row_burst: float = 5e-3,
    n_steps: int = 900,
    batch: int = 128,
    hidden: int = 24,
    lr: float = 2e-3,
    residual_l2: float = 1e-2,
    clip: float = 2.0,
    device: str = "cpu",
    seed: int = 0,
    log_every: int = 100,
    raw_bits: bool = False,
    drift_persist_choices: tuple[int, ...] = (3,),
) -> GEMVResidualCalibrator:
    """Train the residual calibrator.

    ``clip`` is the saturation level ``c`` of ``Delta = c tanh(./c)`` and
    ``residual_l2`` is the shrinkage weight ``lambda``. Together they bound
    how far the calibrator may move away from the B' anchor: the largest
    amount by which it can re-rank a symbol is ``2c``, so a confidently
    wrong B' prior whose log-margin exceeds ``2c`` is out of reach whatever
    the features say. Both are exposed so that the ceiling can be swept
    rather than assumed (Results, *Comparison with classical baselines*).

    ``raw_bits`` adds the order-carrying raw reads to the input, and each
    step draws the channel's drift persistence uniformly from
    ``drift_persist_choices``; the default ``(3,)`` is the original
    single-structure training.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = GEMVResidualCalibrator(
        hidden=hidden, clip=clip,
        raw_bit_cycles=n_cycles if raw_bits else 0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    persists = tuple(int(k) for k in drift_persist_choices)

    for step in range(n_steps):
        p_bit = float(np.exp(rng.uniform(np.log(p_indep_range[0]),
                                          np.log(p_indep_range[1]))))
        persist = (persists[0] if len(persists) == 1
                   else int(persists[rng.integers(len(persists))]))
        features, llr_bp, clean = make_training_batch_gemv(
            G, batch_size=batch, n_cycles=n_cycles, p_indep=p_bit,
            p_drift=p_drift, p_row_burst=p_row_burst, rng=rng,
            drift_persist=persist, raw_bits=raw_bits,
        )
        x = torch.from_numpy(features).to(device)
        anchor = torch.from_numpy(llr_bp).to(device)
        y = torch.from_numpy(clean).long().to(device)

        out, gate = model(x, anchor)
        loss = F.cross_entropy(out.reshape(-1, GF), y.reshape(-1))
        loss = loss + residual_l2 * torch.mean((out - anchor) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            acc = (out.argmax(dim=-1) == y).float().mean().item()
            print(f"[train-gemv] step {step:5d}  p_bit={p_bit:.1e}  "
                  f"loss={loss.item():.4f}  symbol-acc={acc:.4f}  "
                  f"gate={float(gate.mean().detach()):.3f}")
    return model


def train_linear_calibrator(
    G: np.ndarray,
    n_cycles: int = 4,
    p_indep_range: tuple[float, float] = (1e-4, 5e-3),
    p_drift: float = 5e-3,
    p_row_burst: float = 5e-3,
    n_steps: int = 900,
    batch: int = 128,
    lr: float = 2e-3,
    residual_l2: float = 1e-2,
    device: str = "cpu",
    seed: int = 0,
    log_every: int = 100,
) -> LinearResidualCalibrator:
    """Train the hidden-layer-free variant of the GEMV residual calibrator.

    Shares the training-data generator, loss, optimiser settings and RNG
    usage of train_gemv_calibrator, so that the comparison between the two
    models isolates the architecture and nothing else.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = LinearResidualCalibrator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for step in range(n_steps):
        p_bit = float(np.exp(rng.uniform(np.log(p_indep_range[0]),
                                          np.log(p_indep_range[1]))))
        features, llr_bp, clean = make_training_batch_gemv(
            G, batch_size=batch, n_cycles=n_cycles, p_indep=p_bit,
            p_drift=p_drift, p_row_burst=p_row_burst, rng=rng,
        )
        x = torch.from_numpy(features).to(device)
        anchor = torch.from_numpy(llr_bp).to(device)
        y = torch.from_numpy(clean).long().to(device)

        out, gate = model(x, anchor)
        loss = F.cross_entropy(out.reshape(-1, GF), y.reshape(-1))
        loss = loss + residual_l2 * torch.mean((out - anchor) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            acc = (out.argmax(dim=-1) == y).float().mean().item()
            print(f"[train-linear] step {step:5d}  p_bit={p_bit:.1e}  "
                  f"loss={loss.item():.4f}  symbol-acc={acc:.4f}  "
                  f"gate={float(gate.mean().detach()):.3f}")
    return model