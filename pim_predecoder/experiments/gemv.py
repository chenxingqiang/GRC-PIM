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
    PerColumnAffineCalibrator,
    composite_drift_score,
    enhanced_drift_scores,
    global_drift_score,
    soft_llr_product,
    train_affine_calibrator,
    train_gemv_calibrator,
)
from pim_predecoder.core.noise import bits_to_symbols, symbols_to_bits
from pim_predecoder.core.noise3d import apply_pim_noise_3d, collapse_majority
from pim_predecoder.paths import CACHE


CACHE_DIR = CACHE


def _ser(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.not_equal(x, ref).mean())


def _ber(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.not_equal(symbols_to_bits(x), symbols_to_bits(ref)).mean())


def _post_metrics(llr: np.ndarray, H: np.ndarray, ref: np.ndarray,
                  bp_iters: int) -> tuple[float, float, float]:
    x_hat, _, iters = bp_decode(llr, H, n_iter=bp_iters, early_stop=True)
    return _ser(x_hat, ref), _ber(x_hat, ref), float(iters)


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
) -> GEMVResidualCalibrator:
    tag = (
        f"gemv_{matrix_tag}_T{n_cycles}_d{p_drift:.0e}_b{p_row_burst:.0e}_"
        f"s{n_steps}_bs{train_batch}_h{hidden}_seed{seed}"
    )
    path = CACHE_DIR / f"{tag}.pt"
    model = GEMVResidualCalibrator(hidden=hidden).to(device)
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
    model: GEMVResidualCalibrator | None,
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
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for p in p_bits:
        info = rng.integers(0, GF, size=(n_codewords, G.shape[0]))
        clean = encode(info, G)
        noisy_3d = apply_pim_noise_3d(
            clean, n_cycles=n_cycles, p_indep=p,
            p_drift=test_p_drift, p_row_burst=test_p_row_burst, rng=rng,
        )

        # --- B' baseline ---
        llr_bp = soft_llr_product(noisy_3d, p)
        bp_post, bp_ber, bp_iters_used = _post_metrics(llr_bp, H, clean, bp_iters)

        # --- Majority Vote baseline ---
        maj_bits = collapse_majority(noisy_3d)  # (B, 2n)
        llr_maj = init_llr_from_bits(maj_bits, p_bit=p)
        maj_post, maj_ber, maj_iters = _post_metrics(llr_maj, H, clean, bp_iters)

        # --- Per-column Affine baseline ---
        if affine_model is not None:
            llr_affine = affine_model.soft_llr(noisy_3d, p, device=device, chunk=chunk)
            affine_post, affine_ber, affine_iters = _post_metrics(
                llr_affine, H, clean, bp_iters)
        else:
            affine_post = affine_ber = affine_iters = None

        # --- GEMV residual (optional: no model = GRC-PIM not evaluated) ---
        if model is not None:
            # Compute all drift sub-scores
            all_scores = enhanced_drift_scores(noisy_3d, H)

            llr_gemv, gate = model.soft_llr(noisy_3d, p, device=device, chunk=chunk)
            use_gemv = True

            # Composite threshold: richer multi-signal gating, takes precedence
            if composite_drift_threshold is not None:
                composite = all_scores.get("composite", float("nan"))
                if not np.isnan(composite) and composite < composite_drift_threshold:
                    use_gemv = False
            elif drift_score_threshold is not None and all_scores["global_drift"] < drift_score_threshold:
                use_gemv = False
            if model_gate_threshold is not None and gate < model_gate_threshold:
                use_gemv = False

            llr_hybrid = llr_gemv if use_gemv else llr_bp
            gemv_post, gemv_ber, gemv_iters = _post_metrics(llr_hybrid, H, clean, bp_iters)
            raw_gemv = _ser(llr_gemv.argmax(axis=-1), clean)
            raw_gemv_ber = _ber(llr_gemv.argmax(axis=-1), clean)
        else:
            all_scores = enhanced_drift_scores(noisy_3d, H)
            gemv_post = gemv_ber = gemv_iters = raw_gemv = raw_gemv_ber = gate = use_gemv = None

        floor = 3.0 / (n_codewords * H.shape[1])
        row = dict(
            p_bit=p,
            Bp_post=bp_post,
            Bp_post_BER=bp_ber,
            Bp_iters=bp_iters_used,
            Maj_post=maj_post,
            Maj_post_BER=maj_ber,
            Maj_iters=maj_iters,
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
            resolution_floor=floor,
        )
        if affine_post is not None:
            row["Affine_post"] = affine_post
            row["Affine_post_BER"] = affine_ber
            row["Affine_iters"] = affine_iters
        rows.append(row)

        # Build summary line
        parts = [f"p={p:.1e} | B'={bp_post:.2e}({bp_iters_used:.1f})"]
        parts.append(f"Maj={maj_post:.2e}({maj_iters:.1f})")
        if affine_post is not None:
            parts.append(f"Aff={affine_post:.2e}({affine_iters:.1f})")
        if gemv_post is not None:
            ratio = bp_post / gemv_post if gemv_post > 0 else float("inf")
            parts.append(f"GEMV={gemv_post:.2e}({gemv_iters:.1f})")
            parts.append(f"BER {bp_ber:.2e}->{gemv_ber:.2e}")
            parts.append(f"raw={raw_gemv:.2e} gate={gate:.3f}")
            parts.append(f"drift={all_scores['global_drift']:.3f}")
            parts.append(f"use={int(use_gemv)} gain={ratio:.2g}x")
        parts.append(f"floor~{floor:.1e}")
        print(" | ".join(parts))
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
    ap.add_argument("--bp_iters", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--drift_score_threshold", type=float, default=None,
                    help="Fallback to B' when global drift score is below this value.")
    ap.add_argument("--composite_drift_threshold", type=float, default=None,
                    help="Fallback to B' when composite drift score is below this value "
                         "(takes precedence over drift_score_threshold when set).")
    ap.add_argument("--model_gate_threshold", type=float, default=None,
                    help="Fallback to B' when the model's mean gate is below this value.")
    ap.add_argument("--force_train", action="store_true")
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

    model = _train_or_load_gemv(
        G, n_cycles=args.n_cycles,
        p_drift=args.train_p_drift, p_row_burst=args.train_p_row_burst,
        n_steps=args.train_steps, hidden=args.hidden,
        train_batch=args.train_batch,
        device=args.device, seed=args.seed,
        matrix_tag=matrix_tag,
        force_train=args.force_train,
    )

    affine_model = _train_or_load_affine(
        G, n_cycles=args.n_cycles,
        p_drift=args.train_p_drift, p_row_burst=args.train_p_row_burst,
        n_steps=args.train_steps, train_batch=args.train_batch,
        device=args.device, seed=args.seed,
        matrix_tag=matrix_tag,
        force_train=args.force_train,
    )

    print(f"\n[eval] test_drift={args.test_p_drift} "
          f"burst={args.test_p_row_burst} n_cw={args.n_codewords}")
    rows = evaluate(
        model, H, G, args.p_bits,
        n_codewords=args.n_codewords, n_cycles=args.n_cycles,
        test_p_drift=args.test_p_drift,
        test_p_row_burst=args.test_p_row_burst,
        bp_iters=args.bp_iters, device=args.device,
        seed=args.seed, chunk=args.chunk,
        drift_score_threshold=args.drift_score_threshold,
        model_gate_threshold=args.model_gate_threshold,
        composite_drift_threshold=args.composite_drift_threshold,
        affine_model=affine_model,
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
            matrix_info=args.matrix_info,
            matrix_rate=args.matrix_rate,
            matrix_shape=dict(H=list(H.shape), G=list(G.shape)),
            model="GEMVResidualCalibrator",
            baselines="majority_vote + per_column_affine",
            drift_score_threshold=args.drift_score_threshold,
            composite_drift_threshold=args.composite_drift_threshold,
            model_gate_threshold=args.model_gate_threshold,
            rows=rows,
        ), f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()