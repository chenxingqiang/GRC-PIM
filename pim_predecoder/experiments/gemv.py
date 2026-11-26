"""Evaluate B' + GEMV-friendly residual calibrator + NB-LDPC.

This is the CIM/PIM-friendly alternative to the 3D CNN posterior model. The
learned model is a tiny shared MLP/GEMV residual head anchored on the classical
T-cycle soft-LLR product baseline B'.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pim_predecoder.core.codec import GF, bp_decode, encode, load_HG
from pim_predecoder.core.codec import init_llr_from_bits
from pim_predecoder.core.gemv_calibrator import (
    GEMVResidualCalibrator,
    LinearResidualCalibrator,
    PerColumnAffineCalibrator,
    composite_drift_score,
    enhanced_drift_scores,
    global_drift_score,
    soft_llr_product,
    train_affine_calibrator,
    train_gemv_calibrator,
    train_linear_calibrator,
    weighted_soft_llr_product,
)
from pim_predecoder.core.noise import bits_to_symbols, symbols_to_bits
from pim_predecoder.core.noise3d import (
    apply_pim_noise_3d,
    apply_pim_noise_aihwkit,
    apply_pim_noise_conductance,
    collapse_majority,
)
from pim_predecoder.paths import CACHE


CACHE_DIR = CACHE


def _ser(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.not_equal(x, ref).mean())


def _ber(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.not_equal(symbols_to_bits(x), symbols_to_bits(ref)).mean())


def _count_err(x: np.ndarray, ref: np.ndarray) -> tuple[int, int, int, int]:
    """Return (symbol_errors, symbol_total, bit_errors, bit_total).

    SER/BER are means over codewords, so these counts are exactly additive
    across evaluation chunks. Streaming a large codeword budget in chunks and
    summing these counts reproduces the monolithic rate bit-exactly.
    """
    sym_err = int(np.not_equal(x, ref).sum())
    x_bits = symbols_to_bits(x)
    ref_bits = symbols_to_bits(ref)
    bit_err = int(np.not_equal(x_bits, ref_bits).sum())
    return sym_err, int(x.size), bit_err, int(x_bits.size)


class _PathAcc:
    """Accumulates exact error counts for one decode path across chunks."""

    __slots__ = ("sym_err", "sym_tot", "bit_err", "bit_tot", "iters_w", "w")

    def __init__(self) -> None:
        self.sym_err = 0
        self.sym_tot = 0
        self.bit_err = 0
        self.bit_tot = 0
        self.iters_w = 0.0
        self.w = 0

    def add(self, x: np.ndarray, ref: np.ndarray, iters: float) -> None:
        sym_err, sym_tot, bit_err, bit_tot = _count_err(x, ref)
        self.sym_err += sym_err
        self.sym_tot += sym_tot
        self.bit_err += bit_err
        self.bit_tot += bit_tot
        self.iters_w += float(iters) * ref.shape[0]
        self.w += int(ref.shape[0])

    @property
    def ser(self) -> float:
        return self.sym_err / self.sym_tot

    @property
    def ber(self) -> float:
        return self.bit_err / self.bit_tot

    @property
    def iters(self) -> float:
        return self.iters_w / self.w


def _post_metrics(llr: np.ndarray, H: np.ndarray, ref: np.ndarray,
                  bp_iters: int) -> tuple[float, float, float]:
    x_hat, _, iters = bp_decode(llr, H, n_iter=bp_iters, early_stop=True)
    return _ser(x_hat, ref), _ber(x_hat, ref), float(iters)


def _decode(llr: np.ndarray, H: np.ndarray,
            bp_iters: int) -> tuple[np.ndarray, float]:
    """Run BP and return (hard decisions, iterations used)."""
    x_hat, _, iters = bp_decode(llr, H, n_iter=bp_iters, early_stop=True)
    return x_hat, float(iters)


def _train_or_load_gemv(
    G: np.ndarray,
    n_cycles: int,
    p_drift: float,
    p_row_burst: float,
    n_steps: int,
    hidden: int,
    train_batch: int,
    device: str,
    seed: int,
    matrix_tag: str,
    force_train: bool = False,
    clip: float = 2.0,
    residual_l2: float = 1e-2,
    raw_bits: bool = False,
    train_persist: tuple[int, ...] = (3,),
) -> GEMVResidualCalibrator:
    tag = (
        f"gemv_{matrix_tag}_T{n_cycles}_d{p_drift:.0e}_b{p_row_burst:.0e}_"
        f"s{n_steps}_bs{train_batch}_h{hidden}_seed{seed}"
    )
    # Only the non-default ceiling appears in the tag, so every cached model
    # from before this knob existed stays addressable.
    if (clip, residual_l2) != (2.0, 1e-2):
        tag += f"_c{clip:g}_l{residual_l2:g}"
    if raw_bits:
        tag += "_raw"
    if tuple(train_persist) != (3,):
        tag += "_p" + "-".join(str(k) for k in train_persist)
    path = CACHE_DIR / f"{tag}.pt"
    model = GEMVResidualCalibrator(
        hidden=hidden, clip=clip,
        raw_bit_cycles=n_cycles if raw_bits else 0).to(device)
    if path.exists() and not force_train:
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"[cache] loaded GEMV model from {path}")
        return model

    print(f"[train] GEMV model {tag}")
    t0 = time.time()
    model = train_gemv_calibrator(
        G, n_cycles=n_cycles,
        p_drift=p_drift, p_row_burst=p_row_burst,
        n_steps=n_steps, batch=train_batch, hidden=hidden,
        clip=clip, residual_l2=residual_l2,
        device=device, seed=seed,
        raw_bits=raw_bits, drift_persist_choices=tuple(train_persist),
    )
    torch.save(model.state_dict(), path)
    print(f"[train] saved {path}  ({time.time() - t0:.1f}s)")
    return model


def _train_or_load_linear(
    G: np.ndarray,
    n_cycles: int,
    p_drift: float,
    p_row_burst: float,
    n_steps: int,
    train_batch: int,
    device: str,
    seed: int,
    matrix_tag: str,
    force_train: bool = False,
) -> LinearResidualCalibrator:
    tag = (
        f"linear_{matrix_tag}_T{n_cycles}_d{p_drift:.0e}_b{p_row_burst:.0e}_"
        f"s{n_steps}_bs{train_batch}_seed{seed}"
    )
    path = CACHE_DIR / f"{tag}.pt"
    model = LinearResidualCalibrator().to(device)
    if path.exists() and not force_train:
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"[cache] loaded linear model from {path}")
        return model

    print(f"[train] linear model {tag}")
    t0 = time.time()
    model = train_linear_calibrator(
        G, n_cycles=n_cycles,
        p_drift=p_drift, p_row_burst=p_row_burst,
        n_steps=n_steps, batch=train_batch,
        device=device, seed=seed,
    )
    torch.save(model.state_dict(), path)
    print(f"[train] saved {path}  ({time.time() - t0:.1f}s)")
    return model


def _train_or_load_affine(
    G: np.ndarray,
    n_cycles: int,
    p_drift: float,
    p_row_burst: float,
    n_steps: int,
    train_batch: int,
    device: str,
    seed: int,
    matrix_tag: str,
    force_train: bool = False,
) -> PerColumnAffineCalibrator:
    tag = (
        f"affine_{matrix_tag}_T{n_cycles}_d{p_drift:.0e}_b{p_row_burst:.0e}_"
        f"s{n_steps}_bs{train_batch}_seed{seed}"
    )
    path = CACHE_DIR / f"{tag}.pt"
    model = PerColumnAffineCalibrator(n_columns=G.shape[1]).to(device)
    if path.exists() and not force_train:
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"[cache] loaded affine model from {path}")
        return model

    print(f"[train] affine model {tag}")
    t0 = time.time()
    model = train_affine_calibrator(
        G, n_cycles=n_cycles,
        p_drift=p_drift, p_row_burst=p_row_burst,
        n_steps=n_steps, batch=train_batch,
        device=device, seed=seed,
    )
    torch.save(model.state_dict(), path)
    print(f"[train] saved {path}  ({time.time() - t0:.1f}s)")
    return model


def evaluate(
    model: GEMVResidualCalibrator | LinearResidualCalibrator | None,
    H: np.ndarray,
    G: np.ndarray,
    p_bits: list[float],
    n_codewords: int,
    n_cycles: int,
    test_p_drift: float,
    test_p_row_burst: float,
    bp_iters: int,
    device: str,
    seed: int,
    chunk: int,
    drift_score_threshold: float | None,
    model_gate_threshold: float | None,
    composite_drift_threshold: float | None = None,
    affine_model: PerColumnAffineCalibrator | None = None,
    eval_chunk: int | None = None,
    channel: str = "synthetic",
    cond_params: dict | None = None,
    drift_score_max: float | None = None,
    wbp_rho: float | None = None,
    drift_persist: int = 3,
) -> list[dict]:
    """Evaluate every pipeline over ``n_codewords`` codewords per noise level.

    The codeword budget is streamed in blocks of ``eval_chunk`` so that large
    Monte Carlo budgets stay within a bounded memory footprint: a monolithic
    100,000-codeword batch at K=1024 would allocate tens of GB for the
    (B, T, 2n) read tensor and its float intermediates. SER/BER are means over
    codewords and are accumulated as exact integer error counts, so the
    reported rates are mathematically identical to a monolithic run.

    ``eval_chunk=None`` evaluates in a single block, which reproduces the
    previous monolithic behaviour (and its random-number stream) exactly.

    Drift scores are batch statistics: they are computed per block and
    reported as a codeword-weighted mean plus the across-block s.d., which
    additionally quantifies how stable the gating signal is over the budget.

    ``channel`` selects the read channel. ``"synthetic"`` is the Bernoulli
    channel the calibrator is trained on, driven by ``p_bits`` /
    ``test_p_drift`` / ``test_p_row_burst``. ``"conductance"`` instead drives
    ``apply_pim_noise_conductance`` with ``cond_params``, which makes bit
    errors emerge from ADC threshold crossings and is therefore structurally
    out of distribution. ``"aihwkit"`` drives ``apply_pim_noise_aihwkit``,
    which keeps that same level placement and ADC rule but takes its noise
    physics from a third-party device-calibrated model, so it cross-checks the
    gate against a channel we did not author. Neither conductance-domain
    channel has a bit-flip-rate parameter, so the nominal ``p`` handed to
    B'/majority/affine/GEMV is measured from the realised readout per block and
    reported as ``p_effective``; the entry from ``p_bits`` is kept only as the
    row label. Measuring ``p`` this way gives B' the correct marginal
    statistics and withholds only the correlation structure, which makes the
    comparison a strictly harder test of the calibrator than feeding it a
    mismatched nominal rate would.

    ``drift_score_threshold`` and ``drift_score_max`` form a *band*, not a
    floor. The lower bound asks whether multi-cycle evidence exists at all;
    the upper bound asks whether the impairment is still within the strength
    the calibrator was calibrated for. Without the upper bound an arbitrarily
    severe channel passes the drift gate, and the model-confidence gate does
    not catch it: on the conductance-domain channel the mean gate score stays
    near 0.9 even where the residual makes the error rate worse.
    """
    rng = np.random.default_rng(seed)
    block = int(eval_chunk) if eval_chunk else int(n_codewords)
    block = max(1, min(block, n_codewords))
    rows = []
    if channel not in ("synthetic", "conductance", "aihwkit"):
        raise ValueError(f"unknown channel {channel!r}")
    cond_params = dict(cond_params or {})

    for p in p_bits:
        bp_acc, maj_acc = _PathAcc(), _PathAcc()
        gemv_acc = _PathAcc() if model is not None else None
        raw_acc = _PathAcc() if model is not None else None
        affine_acc = _PathAcc() if affine_model is not None else None
        wbp_acc = _PathAcc() if wbp_rho is not None else None
        read_acc = _PathAcc()

        score_rows: list[dict[str, float]] = []
        gates: list[float] = []
        weights: list[int] = []
        p_effs: list[float] = []
        raw_read_bits: list[float] = []
        raw_weights: list[int] = []
        done = 0

        while done < n_codewords:
            size = min(block, n_codewords - done)
            info = rng.integers(0, GF, size=(size, G.shape[0]))
            clean = encode(info, G)
            if channel in ("conductance", "aihwkit"):
                sampler = (apply_pim_noise_conductance
                           if channel == "conductance"
                           else apply_pim_noise_aihwkit)
                noisy_3d = sampler(
                    clean, n_cycles=n_cycles, rng=rng, **cond_params,
                )
                # No bit-flip-rate parameter exists for this channel: measure
                # the realised readout error rate and use it as the nominal p.
                p_eff = float(np.mean(
                    noisy_3d != symbols_to_bits(clean)[:, None, :]))
                p_llr = min(max(p_eff, 1e-9), 0.5 - 1e-9)
            else:
                noisy_3d = apply_pim_noise_3d(
                    clean, n_cycles=n_cycles, p_indep=p,
                    p_drift=test_p_drift, drift_persist=drift_persist,
                    p_row_burst=test_p_row_burst, rng=rng,
                )
                p_eff = p_llr = p
            p_effs.append(p_eff)

            # --- uncorrected readout reference: the rate every corrected
            # path is measured against. ``read_acc`` scores cycle 0 alone,
            # i.e. what a single-read scheme with no ECC would deliver, and
            # ``raw_read_bits`` is the bit disagreement over all T reads.
            read_acc.add(bits_to_symbols(noisy_3d[:, 0, :]), clean, 0.0)
            raw_read_bits.append(float(np.mean(
                noisy_3d != symbols_to_bits(clean)[:, None, :])))
            raw_weights.append(size)

            # --- B' baseline ---
            llr_bp = soft_llr_product(noisy_3d, p_llr)
            x_bp, it_bp = _decode(llr_bp, H, bp_iters)
            bp_acc.add(x_bp, clean, it_bp)

            # --- Weighted-BP baseline (geometric per-cycle discount of B') ---
            if wbp_acc is not None:
                llr_wbp = weighted_soft_llr_product(noisy_3d, p_llr, rho=wbp_rho)
                x_wbp, it_wbp = _decode(llr_wbp, H, bp_iters)
                wbp_acc.add(x_wbp, clean, it_wbp)

            # --- Majority Vote baseline ---
            maj_bits = collapse_majority(noisy_3d)                  # (B, 2n)
            llr_maj = init_llr_from_bits(maj_bits, p_bit=p_llr)
            x_maj, it_maj = _decode(llr_maj, H, bp_iters)
            maj_acc.add(x_maj, clean, it_maj)

            # --- Per-column Affine baseline ---
            if affine_acc is not None:
                llr_affine = affine_model.soft_llr(
                    noisy_3d, p_llr, device=device, chunk=chunk)
                x_aff, it_aff = _decode(llr_affine, H, bp_iters)
                affine_acc.add(x_aff, clean, it_aff)

            # --- drift signals (batch statistics, per block) ---
            score_rows.append(enhanced_drift_scores(noisy_3d, H))
            weights.append(size)

            # --- GEMV residual path ---
            if gemv_acc is not None:
                llr_gemv, gate = model.soft_llr(
                    noisy_3d, p_llr, device=device, chunk=chunk)
                gates.append(float(gate))
                x_gemv, it_gemv = _decode(llr_gemv, H, bp_iters)
                gemv_acc.add(x_gemv, clean, it_gemv)
                raw_acc.add(llr_gemv.argmax(axis=-1), clean, 0.0)

            done += size

        w = np.asarray(weights, dtype=np.float64)
        all_scores = {
            k: float(np.average([s[k] for s in score_rows], weights=w))
            for k in score_rows[0]
        }
        score_sd = {
            k: float(np.std([s[k] for s in score_rows], ddof=1))
            if len(score_rows) > 1 else 0.0
            for k in score_rows[0]
        }
        # Fraction of individual blocks whose drift score falls outside the
        # band. The gate decision below uses the aggregate score, so this is
        # the rate at which a hardware detector that published its score every
        # ``eval_chunk`` codewords would disagree with it. The released RTL
        # publishes once per codeword, so running with ``--eval_chunk 1``
        # measures the spurious-rejection rate the deployed gate actually
        # incurs -- which is what the band bounds are calibrated against
        # (see sn-article.tex Sec. 4.4).
        block_scores = np.asarray([s["global_drift"] for s in score_rows])
        out_of_band = np.zeros(block_scores.shape, dtype=bool)
        if drift_score_threshold is not None:
            out_of_band |= block_scores < drift_score_threshold
        if drift_score_max is not None:
            out_of_band |= block_scores > drift_score_max
        drift_out_of_band_frac = float(np.mean(out_of_band))

        # --- gate decision on the aggregated signals ---
        if gemv_acc is not None:
            gate = float(np.average(gates, weights=w))
            use_gemv = True
            if composite_drift_threshold is not None:
                composite = all_scores.get("composite", float("nan"))
                if not np.isnan(composite) and composite < composite_drift_threshold:
                    use_gemv = False
            elif (drift_score_threshold is not None
                    and all_scores["global_drift"] < drift_score_threshold):
                use_gemv = False
            if (drift_score_max is not None
                    and all_scores["global_drift"] > drift_score_max):
                use_gemv = False
            if model_gate_threshold is not None and gate < model_gate_threshold:
                use_gemv = False

            # Closed gate falls back to B', exactly as the hybrid LLR would.
            hybrid = gemv_acc if use_gemv else bp_acc
            gemv_post, gemv_ber, gemv_iters = hybrid.ser, hybrid.ber, hybrid.iters
            raw_gemv, raw_gemv_ber = raw_acc.ser, raw_acc.ber
        else:
            gemv_post = gemv_ber = gemv_iters = None
            raw_gemv = raw_gemv_ber = gate = use_gemv = None

        floor = 3.0 / (n_codewords * H.shape[1])
        row = dict(
            p_bit=p,
            p_effective=float(np.average(p_effs, weights=w)),
            channel=channel,
            Bp_post=bp_acc.ser,
            Bp_post_BER=bp_acc.ber,
            Bp_iters=bp_acc.iters,
            Maj_post=maj_acc.ser,
            Maj_post_BER=maj_acc.ber,
            Maj_iters=maj_acc.iters,
            GEMV_post=gemv_post,
            GEMV_post_BER=gemv_ber,
            GEMV_iters=gemv_iters,
            GEMV_raw_symbol=raw_gemv,
            GEMV_raw_BER=raw_gemv_ber,
            GEMV_gate=gate,
            drift_score=all_scores["global_drift"],
            column_persistence=all_scores["column_persistence"],
            row_burst=all_scores["row_burst"],
            syndrome_instability=all_scores["syndrome_instability"],
            composite_drift=all_scores["composite"],
            use_gemv=use_gemv,
            drift_score_max=drift_score_max,
            resolution_floor=floor,
            n_codewords_eval=bp_acc.w,
            n_blocks=len(score_rows),
            eval_chunk=block,
            drift_score_sd=score_sd["global_drift"],
            drift_out_of_band_frac=drift_out_of_band_frac,
            # Full per-block series, so the spurious-rejection rate of any
            # candidate band can be recomputed without re-running.
            drift_score_blocks=[round(float(x), 6) for x in block_scores],
            composite_drift_sd=score_sd["composite"],
            Bp_symbol_errors=bp_acc.sym_err,
            Bp_bit_errors=bp_acc.bit_err,
            # Uncorrected readout reference (no fusion, no decoder).
            raw_read_SER=read_acc.ser,
            raw_read_BER=read_acc.ber,
            raw_read_bit_rate=float(np.average(raw_read_bits,
                                               weights=np.asarray(raw_weights,
                                                                  dtype=np.float64))),
        )
        if gemv_acc is not None:
            row["GEMV_symbol_errors"] = (
                gemv_acc.sym_err if use_gemv else bp_acc.sym_err)
            row["GEMV_bit_errors"] = (
                gemv_acc.bit_err if use_gemv else bp_acc.bit_err)
            # What the calibrator delivers when admitted, recorded whether
            # or not the gates admit it. GEMV_post* above carry the gated
            # fallback, so without these a closed point hides the correction
            # it rejected and its cost needs a separate forced-open re-run.
            row["GEMV_ungated_post"] = gemv_acc.ser
            row["GEMV_ungated_post_BER"] = gemv_acc.ber
            row["GEMV_ungated_iters"] = gemv_acc.iters
            row["GEMV_ungated_symbol_errors"] = gemv_acc.sym_err
            row["GEMV_ungated_bit_errors"] = gemv_acc.bit_err
        if affine_acc is not None:
            row["Affine_post"] = affine_acc.ser
            row["Affine_post_BER"] = affine_acc.ber
            row["Affine_iters"] = affine_acc.iters
        if wbp_acc is not None:
            row["Wbp_post"] = wbp_acc.ser
            row["Wbp_post_BER"] = wbp_acc.ber
            row["Wbp_iters"] = wbp_acc.iters
        rows.append(row)

        # Build summary line
        parts = [f"p={p:.1e} | read={read_acc.ser:.2e}"
                 f" | B'={bp_acc.ser:.2e}({bp_acc.iters:.1f})"]
        if wbp_acc is not None:
            parts.append(f"Wbp={wbp_acc.ser:.2e}({wbp_acc.iters:.1f})")
        parts.append(f"Maj={maj_acc.ser:.2e}({maj_acc.iters:.1f})")
        if affine_acc is not None:
            parts.append(f"Aff={affine_acc.ser:.2e}({affine_acc.iters:.1f})")
        if gemv_post is not None:
            ratio = bp_acc.ser / gemv_post if gemv_post > 0 else float("inf")
            parts.append(f"GEMV={gemv_post:.2e}({gemv_iters:.1f})")
            parts.append(f"BER {bp_acc.ber:.2e}->{gemv_ber:.2e}")
            parts.append(f"raw={raw_gemv:.2e} gate={gate:.3f}")
            parts.append(f"drift={all_scores['global_drift']:.3f}"
                         f"+-{score_sd['global_drift']:.3f}")
            parts.append(f"use={int(use_gemv)} gain={ratio:.2g}x")
        parts.append(f"n={bp_acc.w}x{len(score_rows)}blk floor~{floor:.1e}")
        print(" | ".join(parts), flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_p_drift", type=float, default=5e-3)
    ap.add_argument("--train_p_row_burst", type=float, default=5e-3)
    ap.add_argument("--test_p_drift", type=float, default=5e-3)
    ap.add_argument("--test_p_row_burst", type=float, default=5e-3)
    ap.add_argument("--p_bits", nargs="+", type=float,
                    default=[5e-3, 3e-3, 1e-3, 3e-4, 1e-4])
    ap.add_argument("--n_codewords", type=int, default=2000)
    ap.add_argument("--n_cycles", type=int, default=4)
    ap.add_argument("--matrix_info", type=int, default=256,
                    help="Information-symbol length for NBLDPC DAC_TEST matrices.")
    ap.add_argument("--matrix_rate", default="8_9",
                    help="Rate tag for NBLDPC DAC_TEST matrices, e.g. 8_9 or 16_17.")
    ap.add_argument("--train_steps", type=int, default=900)
    ap.add_argument("--train_batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--clip", type=float, default=2.0,
                    help="Saturation level c of the residual Delta = c tanh(./c). "
                         "The calibrator can re-rank a symbol by at most 2c, so "
                         "a B' prior that is confidently wrong by more than 2c "
                         "nats is outside its reach whatever the features say. "
                         "Sweep this to separate the architecture's ceiling from "
                         "what the feature set can express.")
    ap.add_argument("--residual_lambda", type=float, default=1e-2,
                    help="Shrinkage weight lambda on <(g*Delta)^2>, which pulls "
                         "the calibrator back towards the B' anchor. Set to 0 to "
                         "remove the pull.")
    ap.add_argument("--model", choices=["gemv", "linear"], default="gemv",
                    help="Residual head trained and evaluated. 'gemv' is the "
                         "hidden-layer calibrator (425 params); 'linear' is a "
                         "linear head on the same 13 features (56 params) that "
                         "isolates the value of the hidden nonlinearity at an "
                         "identical training budget and data stream.")
    ap.add_argument("--bp_iters", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval_seed", type=int, default=None,
                    help="Random seed for the evaluation data stream. Defaults "
                         "to --seed. Set it to a held-out value to evaluate a "
                         "cached model on data disjoint from the test sets, "
                         "which is required when calibrating gate thresholds "
                         "without touching test data.")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--eval_chunk", type=int, default=None,
                    help="Stream the codeword budget in blocks of this size. "
                         "Required for large Monte Carlo budgets (e.g. 100000) "
                         "to bound peak memory; SER/BER are accumulated as "
                         "exact error counts and are unaffected. Omit to "
                         "evaluate in a single monolithic block.")
    ap.add_argument("--drift_score_threshold", type=float, default=None,
                    help="Fallback to B' when global drift score is below this value.")
    ap.add_argument("--drift_score_max", type=float, default=None,
                    help="Fallback to B' when the global drift score is ABOVE "
                         "this value. Together with --drift_score_threshold "
                         "this makes the drift gate a band: the lower bound "
                         "requires that multi-cycle evidence exists, the upper "
                         "bound requires that the impairment is no stronger "
                         "than the calibrator was calibrated for. Calibrate it "
                         "on held-out data as max(in-distribution drift score) "
                         "plus a margin on the across-block s.d.")
    ap.add_argument("--composite_drift_threshold", type=float, default=None,
                    help="Fallback to B' when composite drift score is below this value "
                         "(takes precedence over drift_score_threshold when set).")
    ap.add_argument("--model_gate_threshold", type=float, default=None,
                    help="Fallback to B' when the model's mean gate is below this value.")
    ap.add_argument("--wbp_rho", type=float, default=None,
                    help="Weighted-BP baseline: geometric per-cycle weight "
                         "applied to the B' accumulation, cycle t (counting "
                         "from 1) carrying rho**(t-1) so the first cycle "
                         "always has weight 1. Omit to skip the baseline (the "
                         "row schema is unchanged); 1.0 reproduces B' "
                         "exactly; <1 discounts the later cycles whose "
                         "evidence is correlated under drift persistence.")
    ap.add_argument("--raw_bit_features", action="store_true",
                    help="Order-aware calibrator: append each symbol's 2T raw "
                         "reads to the 13 order-invariant features.")
    ap.add_argument("--train_drift_persist", type=int, nargs="+", default=[3],
                    help="Drift persistences the calibrator is trained on; "
                         "each step draws one uniformly.")
    ap.add_argument("--drift_persist", type=int, default=3,
                    help="Cycles one drift event holds its column in the "
                         "synthetic channel.")
    ap.add_argument("--force_train", action="store_true")
    ap.add_argument("--channel", choices=["synthetic", "conductance", "aihwkit"],
                    default="synthetic",
                    help="Read channel for evaluation. 'synthetic' is the "
                         "Bernoulli channel the calibrator is trained on. "
                         "'conductance' drives the conductance-domain channel "
                         "(errors emerge from ADC threshold crossings), which "
                         "is structurally out of distribution; --p_bits then "
                         "only labels the row and the nominal bit-flip rate is "
                         "measured from the readout (reported as p_effective). "
                         "'aihwkit' keeps that ADC rule but takes the noise "
                         "physics from aihwkit's measured CMO/HfOx ReRAM model, "
                         "which cross-checks the gate against a channel we did "
                         "not author (requires the aihwkit package).")
    ap.add_argument("--cond_sigma_ln", type=float, default=0.0,
                    help="Conductance channel: sigma of the static log-normal "
                         "cell-to-cell conductance spread.")
    ap.add_argument("--cond_drift_nu", type=float, default=0.0,
                    help="Conductance channel: retention-drift exponent nu in "
                         "G(t) = G_0 (t/t_0)^-nu.")
    ap.add_argument("--cond_rtn_traps", type=int, default=0,
                    help="Conductance channel: number of two-state RTN traps "
                         "per cell (>1 gives multi-level RTN).")
    ap.add_argument("--cond_rtn_amp", type=float, default=0.0,
                    help="Conductance channel: fractional conductance step of "
                         "one active RTN trap.")
    ap.add_argument("--cond_rtn_switch", type=float, default=0.5,
                    help="Conductance channel: per-cycle trap switching "
                         "probability.")
    ap.add_argument("--aihw_t_inference", type=float, default=86400.0,
                    help="aihwkit channel: retention time in seconds since "
                         "programming; drives conductance decay and 1/f read "
                         "noise.")
    ap.add_argument("--aihw_level_pitch_us", type=float, default=8.0,
                    help="aihwkit channel: conductance spacing between adjacent "
                         "GF(3) levels in microsiemens, i.e. the ADC margin the "
                         "device noise must cross.")
    ap.add_argument("--aihw_acceptance_range", type=float, default=2.0,
                    help="aihwkit channel: program-and-verify acceptance range "
                         "in percent (the device model fits 0.2 and 2).")
    ap.add_argument("--aihw_prog_noise_scale", type=float, default=1.0,
                    help="aihwkit channel: scale on the programming noise.")
    ap.add_argument("--aihw_read_noise_scale", type=float, default=1.0,
                    help="aihwkit channel: scale on the 1/f read noise; 1 is "
                         "the measured device, 0 gives a static-only readout.")
    ap.add_argument("--aihw_drift_scale", type=float, default=1.0,
                    help="aihwkit channel: scale on the conductance decay and "
                         "relaxation.")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--label", default="gemv")
    args = ap.parse_args()

    H, G = load_HG(
        repo_root=str(Path(__file__).resolve().parent.parent),
        info_length=args.matrix_info,
        rate=args.matrix_rate,
    )
    matrix_tag = f"k{G.shape[0]}_n{G.shape[1]}_r{args.matrix_rate}"
    print(f"H {H.shape}  G {G.shape}  T={args.n_cycles}  matrix={matrix_tag}  label={args.label}")

    if args.model == "linear":
        model = _train_or_load_linear(
            G, n_cycles=args.n_cycles,
            p_drift=args.train_p_drift, p_row_burst=args.train_p_row_burst,
            n_steps=args.train_steps,
            train_batch=args.train_batch,
            device=args.device, seed=args.seed,
            matrix_tag=matrix_tag,
            force_train=args.force_train,
        )
    else:
        model = _train_or_load_gemv(
            G, n_cycles=args.n_cycles,
            p_drift=args.train_p_drift, p_row_burst=args.train_p_row_burst,
            n_steps=args.train_steps, hidden=args.hidden,
            train_batch=args.train_batch,
            device=args.device, seed=args.seed,
            matrix_tag=matrix_tag,
            force_train=args.force_train,
            clip=args.clip, residual_l2=args.residual_lambda,
            raw_bits=args.raw_bit_features,
            train_persist=tuple(args.train_drift_persist),
        )

    affine_model = _train_or_load_affine(
        G, n_cycles=args.n_cycles,
        p_drift=args.train_p_drift, p_row_burst=args.train_p_row_burst,
        n_steps=args.train_steps, train_batch=args.train_batch,
        device=args.device, seed=args.seed,
        matrix_tag=matrix_tag,
        force_train=args.force_train,
    )

    if args.channel == "aihwkit":
        cond_params = dict(
            t_inference=args.aihw_t_inference,
            level_pitch_us=args.aihw_level_pitch_us,
            acceptance_range=args.aihw_acceptance_range,
            prog_noise_scale=args.aihw_prog_noise_scale,
            read_noise_scale=args.aihw_read_noise_scale,
            drift_scale=args.aihw_drift_scale,
        )
    else:
        cond_params = dict(
            sigma_ln=args.cond_sigma_ln,
            drift_nu=args.cond_drift_nu,
            rtn_traps=args.cond_rtn_traps,
            rtn_amp=args.cond_rtn_amp,
            rtn_switch=args.cond_rtn_switch,
        )
    if args.channel in ("conductance", "aihwkit"):
        print(f"\n[eval] channel={args.channel} {cond_params} "
              f"n_cw={args.n_codewords}")
    else:
        print(f"\n[eval] test_drift={args.test_p_drift} "
              f"burst={args.test_p_row_burst} n_cw={args.n_codewords}")
    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed
    rows = evaluate(
        model, H, G, args.p_bits,
        n_codewords=args.n_codewords, n_cycles=args.n_cycles,
        test_p_drift=args.test_p_drift,
        test_p_row_burst=args.test_p_row_burst,
        bp_iters=args.bp_iters, device=args.device,
        seed=eval_seed, chunk=args.chunk,
        drift_score_threshold=args.drift_score_threshold,
        drift_score_max=args.drift_score_max,
        model_gate_threshold=args.model_gate_threshold,
        composite_drift_threshold=args.composite_drift_threshold,
        wbp_rho=args.wbp_rho,
        drift_persist=args.drift_persist,
        affine_model=affine_model,
        eval_chunk=args.eval_chunk,
        channel=args.channel,
        cond_params=cond_params,
    )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(dict(
            label=args.label,
            train_p_drift=args.train_p_drift,
            train_p_row_burst=args.train_p_row_burst,
            test_p_drift=args.test_p_drift,
            test_p_row_burst=args.test_p_row_burst,
            n_codewords=args.n_codewords,
            n_cycles=args.n_cycles,
            eval_chunk=args.eval_chunk,
            seed=args.seed,
            eval_seed=eval_seed,
            matrix_info=args.matrix_info,
            matrix_rate=args.matrix_rate,
            matrix_shape=dict(H=list(H.shape), G=list(G.shape)),
            model=type(model).__name__,
            clip=args.clip,
            residual_lambda=args.residual_lambda,
            raw_bit_features=args.raw_bit_features,
            train_drift_persist=args.train_drift_persist,
            baselines="majority_vote + per_column_affine",
            channel=args.channel,
            cond_params=(cond_params
                         if args.channel in ("conductance", "aihwkit")
                         else None),
            drift_score_threshold=args.drift_score_threshold,
            drift_score_max=args.drift_score_max,
            composite_drift_threshold=args.composite_drift_threshold,
            model_gate_threshold=args.model_gate_threshold,
            wbp_rho=args.wbp_rho,
            drift_persist=args.drift_persist,
            rows=rows,
        ), f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()