"""Multi-seed + gate threshold sweep + BP iteration ablation.

Runs the GEMV residual calibrator experiment across:
  - Multiple RNG seeds for statistical significance
  - Gate threshold combinations (composite_drift, model_gate)
  - BP iteration counts (1, 2, 3)

Phases:
  1. Ungated baselines: all (seed, bp_iters) combos without gates
  2. Gate threshold sweep: single seed × bp_iters=3, grid of thresholds
  3. Gated validation: best thresholds from phase 2 run across all
     (seed, bp_iters) combos

Output: aggregated JSON with per-config mean ± std across seeds.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from pim_predecoder.paths import RESULTS

RESULTS_DIR = RESULTS / "ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

UNITS_PER_BIT = 2  # GF(3) → 2 bits per symbol


def _ber_to_ser(ber: float) -> float:
    """Approximate: SER from BER for GF(3)."""
    return float(np.clip(1.0 - (1.0 - ber) ** UNITS_PER_BIT, 0.0, 1.0))


def _run_one(
    matrix_info: int,
    matrix_rate: str,
    train_p_drift: float,
    train_p_row_burst: float,
    test_p_drift: float,
    test_p_row_burst: float,
    p_bits: list[float],
    n_codewords: int,
    n_cycles: int,
    train_steps: int,
    train_batch: int,
    hidden: int,
    seed: int,
    bp_iters: int,
    chunk: int,
    device: str,
    force_train: bool,
    composite_drift_threshold: float | None,
    model_gate_threshold: float | None,
    label: str,
    out_json: Path,
) -> dict:
    """Run a single experiment via subprocess and return the result dict."""
    cmd = [
        sys.executable, "-m", "pim_predecoder.experiments.gemv",
        "--matrix_info", str(matrix_info),
        "--matrix_rate", matrix_rate,
        "--train_p_drift", _fmt(train_p_drift),
        "--train_p_row_burst", _fmt(train_p_row_burst),
        "--test_p_drift", _fmt(test_p_drift),
        "--test_p_row_burst", _fmt(test_p_row_burst),
        "--p_bits", *[_fmt(p) for p in p_bits],
        "--n_codewords", str(n_codewords),
        "--n_cycles", str(n_cycles),
        "--train_steps", str(train_steps),
        "--train_batch", str(train_batch),
        "--hidden", str(hidden),
        "--bp_iters", str(bp_iters),
        "--seed", str(seed),
        "--chunk", str(chunk),
        "--device", device,
        "--label", label,
        "--out_json", str(out_json),
    ]
    if composite_drift_threshold is not None:
        cmd.extend(["--composite_drift_threshold", _fmt(composite_drift_threshold)])
    if model_gate_threshold is not None:
        cmd.extend(["--model_gate_threshold", _fmt(model_gate_threshold)])
    if force_train:
        cmd.append("--force_train")

    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0

    with open(out_json) as f:
        result = json.load(f)
    print(f"[ablation] {label}  {elapsed:.0f}s")
    return result


def _fmt(v: float) -> str:
    """Format float to avoid scientific notation for small values."""
    return f"{v:.10f}".rstrip("0").rstrip(".")


def _aggregate_across_seeds(results: list[dict]) -> list[dict]:
    """Aggregate per-p_bit rows across multiple seeds (mean ± std)."""
    by_pbit: dict[float, list[dict]] = {}
    for r in results:
        for row in r["rows"]:
            p = row["p_bit"]
            if p not in by_pbit:
                by_pbit[p] = []
            by_pbit[p].append(row)

    metrics = [
        "Bp_post", "Bp_post_BER", "Bp_iters",
        "GEMV_post", "GEMV_post_BER", "GEMV_iters",
        "GEMV_raw_symbol", "GEMV_raw_BER",
        "GEMV_gate", "drift_score",
        "column_persistence", "row_burst",
        "syndrome_instability", "composite_drift",
        "resolution_floor",
    ]
    bool_metrics = ["use_gemv"]

    aggregated = []
    for p_bit in sorted(by_pbit):
        rows = by_pbit[p_bit]
        agg = {"p_bit": p_bit, "n_seeds": len(rows)}
        for m in metrics:
            values = [r.get(m, 0.0) for r in rows]
            agg[f"{m}_mean"] = float(np.mean(values))
            agg[f"{m}_std"] = float(np.std(values, ddof=1) if len(values) > 1 else 0.0)
        for m in bool_metrics:
            values = [int(r.get(m, 0)) for r in rows]
            agg[f"{m}_fraction"] = float(np.mean(values))
        aggregated.append(agg)
    return aggregated


def _sweep_threshold_grid(
    composite_thresholds: list[float],
    model_gate_thresholds: list[float],
) -> list[tuple[float | None, float | None]]:
    """Build threshold grid including (None, None) and single-axis sweeps."""
    grid: list[tuple[float | None, float | None]] = [(None, None)]
    for c in composite_thresholds:
        grid.append((c, None))
        for m in model_gate_thresholds:
            grid.append((c, m))
    for m in model_gate_thresholds:
        if (None, m) not in grid:
            grid.append((None, m))
    return grid


def _select_best_thresholds(
    threshold_results: list[dict],
    p_bit: float,
    metric: str = "GEMV_post_BER",
) -> tuple[float | None, float | None]:
    """Find the (composite, model_gate) pair that minimizes `metric` at `p_bit`."""
    best_pair = (None, None)
    best_val = float("inf")
    for r in threshold_results:
        comp = r["_composite_drift_threshold"]
        gate = r["_model_gate_threshold"]
        for row in r["rows"]:
            if abs(row["p_bit"] - p_bit) > 1e-10:
                continue
            val = row.get(metric, float("inf"))
            if val < best_val:
                best_val = val
                best_pair = (comp, gate)
    return best_pair


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-seed + gate threshold sweep + BP iteration ablation")
    ap.add_argument("--matrix_info", type=int, default=1024)
    ap.add_argument("--matrix_rate", default="16_17")
    ap.add_argument("--train_p_drift", type=float, default=5e-3)
    ap.add_argument("--train_p_row_burst", type=float, default=5e-3)
    ap.add_argument("--test_p_drift", type=float, default=5e-3)
    ap.add_argument("--test_p_row_burst", type=float, default=5e-3)
    ap.add_argument("--p_bits", nargs="+", type=float,
                    default=[5e-3, 3e-3, 1e-3, 3e-4, 1e-4])
    ap.add_argument("--n_codewords", type=int, default=800)
    ap.add_argument("--n_cycles", type=int, default=4)
    ap.add_argument("--train_steps", type=int, default=600)
    ap.add_argument("--train_batch", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--bp_iters_list", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--composite_thresholds", nargs="+", type=float,
                    default=[0.005, 0.01, 0.015, 0.02, 0.03])
    ap.add_argument("--model_gate_thresholds", nargs="+", type=float,
                    default=[0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force_train", action="store_true")
    ap.add_argument("--out_json", default=str(
        RESULTS_DIR / "ablation_gemv.json"))
    ap.add_argument("--phases", nargs="+",
                    default=["ungated", "threshold-sweep", "gated"],
                    choices=["ungated", "threshold-sweep", "gated", "component-ablation"],
                    help="Which phases to run (default: all three).")
    ap.add_argument("--opt_p_bit", type=float, default=5e-3,
                    help="p_bit used to select best thresholds in phase 2.")
    args = ap.parse_args()

    matrix_tag = f"k{args.matrix_info}_n{args.matrix_info * 17 // 16}_r{args.matrix_rate}"
    out_json = Path(args.out_json)
    output: dict = {
        "meta": {
            "matrix_info": args.matrix_info,
            "matrix_rate": args.matrix_rate,
            "matrix_tag": matrix_tag,
            "train_p_drift": args.train_p_drift,
            "train_p_row_burst": args.train_p_row_burst,
            "test_p_drift": args.test_p_drift,
            "test_p_row_burst": args.test_p_row_burst,
            "n_codewords": args.n_codewords,
            "n_cycles": args.n_cycles,
            "train_steps": args.train_steps,
            "train_batch": args.train_batch,
            "hidden": args.hidden,
            "p_bits": args.p_bits,
            "seeds": args.seeds,
            "bp_iters_list": args.bp_iters_list,
            "composite_thresholds": args.composite_thresholds,
            "model_gate_thresholds": args.model_gate_thresholds,
            "device": args.device,
        },
        "phases": {},
    }

    # ------------------------------------------------------------------
    # Phase 1: Ungated baselines (all seeds × bp_iters)
    # ------------------------------------------------------------------
    if "ungated" in args.phases:
        print("\n" + "=" * 60)
        print("[phase 1] Ungated baselines: "
              f"{len(args.seeds)} seeds × {len(args.bp_iters_list)} bp_iters "
              f"= {len(args.seeds) * len(args.bp_iters_list)} runs")
        print("=" * 60)

        ungated_by_config: dict[tuple[int, int], list[dict]] = {}
        for seed, bp_iters in itertools.product(args.seeds, args.bp_iters_list):
            label = f"abl_u_s{seed}_b{bp_iters}"
            tmp_json = RESULTS_DIR / f"_tmp_{label}.json"
            try:
                result = _run_one(
                    matrix_info=args.matrix_info,
                    matrix_rate=args.matrix_rate,
                    train_p_drift=args.train_p_drift,
                    train_p_row_burst=args.train_p_row_burst,
                    test_p_drift=args.test_p_drift,
                    test_p_row_burst=args.test_p_row_burst,
                    p_bits=args.p_bits,
                    n_codewords=args.n_codewords,
                    n_cycles=args.n_cycles,
                    train_steps=args.train_steps,
                    train_batch=args.train_batch,
                    hidden=args.hidden,
                    seed=seed,
                    bp_iters=bp_iters,
                    chunk=args.chunk,
                    device=args.device,
                    force_train=args.force_train,
                    composite_drift_threshold=None,
                    model_gate_threshold=None,
                    label=label,
                    out_json=tmp_json,
                )
            except subprocess.CalledProcessError as e:
                print(f"[ablation] FAILED: {label}  exit={e.returncode}")
                continue

            key = (bp_iters, 0)  # 0 = no gate config index
            if key not in ungated_by_config:
                ungated_by_config[key] = []
            ungated_by_config[key].append(result)

        ungated_aggregated = {}
        for (bp_iters, _), results in sorted(ungated_by_config.items()):
            ungated_aggregated[f"bp{bp_iters}"] = _aggregate_across_seeds(results)

        output["phases"]["ungated"] = {
            "description": "No gates, multi-seed × bp_iters",
            "n_runs": sum(len(v) for v in ungated_by_config.values()),
            "aggregated": ungated_aggregated,
        }
        print(f"\n[phase 1] done: {sum(len(v) for v in ungated_by_config.values())} "
              f"runs aggregated across {len(args.bp_iters_list)} bp_iters configs")

    # ------------------------------------------------------------------
    # Phase 2: Gate threshold sweep (seed[0] × bp_iters[-1])
    # ------------------------------------------------------------------
    best_composite: float | None = None
    best_model_gate: float | None = None

    if "threshold-sweep" in args.phases:
        threshold_grid = _sweep_threshold_grid(
            args.composite_thresholds, args.model_gate_thresholds)
        n_sweep = len(threshold_grid)
        print(f"\n{'=' * 60}")
        print(f"[phase 2] Threshold sweep: {n_sweep} threshold combos "
              f"(seed={args.seeds[0]}, bp_iters={args.bp_iters_list[-1]})")
        print("=" * 60)

        sweep_seed = args.seeds[0]
        sweep_bp = args.bp_iters_list[-1]
        sweep_results: list[dict] = []

        for comp, gate in threshold_grid:
            comp_str = f"{comp:.3f}" if comp is not None else "none"
            gate_str = f"{gate:.2f}" if gate is not None else "none"
            label = f"abl_t_c{comp_str}_g{gate_str}"
            tmp_json = RESULTS_DIR / f"_tmp_{label}.json"

            try:
                result = _run_one(
                    matrix_info=args.matrix_info,
                    matrix_rate=args.matrix_rate,
                    train_p_drift=args.train_p_drift,
                    train_p_row_burst=args.train_p_row_burst,
                    test_p_drift=args.test_p_drift,
                    test_p_row_burst=args.test_p_row_burst,
                    p_bits=args.p_bits,
                    n_codewords=args.n_codewords,
                    n_cycles=args.n_cycles,
                    train_steps=args.train_steps,
                    train_batch=args.train_batch,
                    hidden=args.hidden,
                    seed=sweep_seed,
                    bp_iters=sweep_bp,
                    chunk=args.chunk,
                    device=args.device,
                    force_train=args.force_train,
                    composite_drift_threshold=comp,
                    model_gate_threshold=gate,
                    label=label,
                    out_json=tmp_json,
                )
            except subprocess.CalledProcessError as e:
                print(f"[ablation] FAILED: {label}  exit={e.returncode}")
                continue

            result["_composite_drift_threshold"] = comp
            result["_model_gate_threshold"] = gate
            sweep_results.append(result)

        # Find best thresholds
        best_composite, best_model_gate = _select_best_thresholds(
            sweep_results, p_bit=args.opt_p_bit, metric="GEMV_post_BER")
        print(f"\n[phase 2] best thresholds at p_bit={args.opt_p_bit:.1e}: "
              f"composite={best_composite}, model_gate={best_model_gate}")

        # Build sweep summary table
        sweep_table = []
        for r in sweep_results:
            for row in r["rows"]:
                if abs(row["p_bit"] - args.opt_p_bit) > 1e-10:
                    continue
                sweep_table.append({
                    "composite_threshold": r["_composite_drift_threshold"],
                    "model_gate_threshold": r["_model_gate_threshold"],
                    "Bp_post": row["Bp_post"],
                    "Bp_post_BER": row["Bp_post_BER"],
                    "GEMV_post": row["GEMV_post"],
                    "GEMV_post_BER": row["GEMV_post_BER"],
                    "GEMV_gate": row["GEMV_gate"],
                    "drift_score": row["drift_score"],
                    "composite_drift": row.get("composite_drift"),
                    "use_gemv": row["use_gemv"],
                })

        output["phases"]["threshold_sweep"] = {
            "description": f"Threshold sweep (seed={sweep_seed}, bp_iters={sweep_bp})",
            "n_combos": n_sweep,
            "n_completed": len(sweep_results),
            "best_composite_threshold": best_composite,
            "best_model_gate_threshold": best_model_gate,
            "sweep_table": sweep_table,
        }
        print(f"[phase 2] done: {len(sweep_results)}/{n_sweep} completed")

    # ------------------------------------------------------------------
    # Phase 3: Gated validation (all seeds × bp_iters, best thresholds)
    # ------------------------------------------------------------------
    if "gated" in args.phases:
        if best_composite is None and best_model_gate is None:
            best_composite = args.composite_thresholds[len(args.composite_thresholds) // 2]
            best_model_gate = args.model_gate_thresholds[len(args.model_gate_thresholds) // 2]
            print(f"\n[phase 3] no sweep results; using middle thresholds: "
                  f"composite={best_composite}, model_gate={best_model_gate}")

        print(f"\n{'=' * 60}")
        print(f"[phase 3] Gated validation: "
              f"{len(args.seeds)} seeds × {len(args.bp_iters_list)} bp_iters "
              f"= {len(args.seeds) * len(args.bp_iters_list)} runs")
        print(f"[phase 3] thresholds: composite={best_composite}, model_gate={best_model_gate}")
        print("=" * 60)

        gated_by_config: dict[tuple[int, int], list[dict]] = {}
        for seed, bp_iters in itertools.product(args.seeds, args.bp_iters_list):
            label = f"abl_g_s{seed}_b{bp_iters}"
            tmp_json = RESULTS_DIR / f"_tmp_{label}.json"

            try:
                result = _run_one(
                    matrix_info=args.matrix_info,
                    matrix_rate=args.matrix_rate,
                    train_p_drift=args.train_p_drift,
                    train_p_row_burst=args.train_p_row_burst,
                    test_p_drift=args.test_p_drift,
                    test_p_row_burst=args.test_p_row_burst,
                    p_bits=args.p_bits,
                    n_codewords=args.n_codewords,
                    n_cycles=args.n_cycles,
                    train_steps=args.train_steps,
                    train_batch=args.train_batch,
                    hidden=args.hidden,
                    seed=seed,
                    bp_iters=bp_iters,
                    chunk=args.chunk,
                    device=args.device,
                    force_train=args.force_train,
                    composite_drift_threshold=best_composite,
                    model_gate_threshold=best_model_gate,
                    label=label,
                    out_json=tmp_json,
                )
            except subprocess.CalledProcessError as e:
                print(f"[ablation] FAILED: {label}  exit={e.returncode}")
                continue

            key = (bp_iters, 0)
            if key not in gated_by_config:
                gated_by_config[key] = []
            gated_by_config[key].append(result)

        gated_aggregated = {}
        for (bp_iters, _), results in sorted(gated_by_config.items()):
            gated_aggregated[f"bp{bp_iters}"] = _aggregate_across_seeds(results)

        output["phases"]["gated"] = {
            "description": "Gated with best thresholds, multi-seed × bp_iters",
            "composite_drift_threshold": best_composite,
            "model_gate_threshold": best_model_gate,
            "n_runs": sum(len(v) for v in gated_by_config.values()),
            "aggregated": gated_aggregated,
        }
        print(f"\n[phase 3] done: {sum(len(v) for v in gated_by_config.values())} "
              f"runs aggregated")

    # ------------------------------------------------------------------
    # Phase 4: Component-level ablation
    #   Tests each gating mechanism in isolation to attribute contribution.
    #   Conditions (all evaluated at bp_iters=3, all seeds):
    #     1. MLP-only:       no gates, always-on GEMV residual
    #     2. Model-gate only: model_confidence gate, no drift gating
    #     3. Drift-gate only: composite_drift gate, no model-gate
    #     4. Both gates:      full proposed system (cached from Phase 3)
    # ------------------------------------------------------------------
    if "component-ablation" in args.phases:
        if best_composite is None and best_model_gate is None:
            best_composite = args.composite_thresholds[len(args.composite_thresholds) // 2]
            best_model_gate = args.model_gate_thresholds[len(args.model_gate_thresholds) // 2]
            print(f"\n[phase 4] no sweep results; using middle thresholds: "
                  f"composite={best_composite}, model_gate={best_model_gate}")

        bp_iters = args.bp_iters_list[-1]
        n_seeds = len(args.seeds)
        n_cond = 3  # MLP-only, model-gate-only, drift-gate-only (both-gates from phase 3)
        print(f"\n{'=' * 60}")
        print(f"[phase 4] Component-level ablation: "
              f"{n_seeds} seeds × {n_cond} conditions × bp={bp_iters} "
              f"= {n_seeds * n_cond} runs")
        print(f"[phase 4] thresholds: composite={best_composite}, model_gate={best_model_gate}")
        print("=" * 60)

        component_configs = {
            "mlp_only": (None, None),
            "model_gate_only": (None, best_model_gate),
            "drift_gate_only": (best_composite, None),
        }

        comp_by_config: dict[str, list[dict]] = {}
        for cond_name, (comp_th, gate_th) in component_configs.items():
            comp_by_config[cond_name] = []
            for seed in args.seeds:
                comp_str = f"{comp_th:.3f}" if comp_th is not None else "none"
                gate_str = f"{gate_th:.2f}" if gate_th is not None else "none"
                label = f"abl_c4_{cond_name}_s{seed}_b{bp_iters}"
                tmp_json = RESULTS_DIR / f"_tmp_{label}.json"

                try:
                    result = _run_one(
                        matrix_info=args.matrix_info,
                        matrix_rate=args.matrix_rate,
                        train_p_drift=args.train_p_drift,
                        train_p_row_burst=args.train_p_row_burst,
                        test_p_drift=args.test_p_drift,
                        test_p_row_burst=args.test_p_row_burst,
                        p_bits=args.p_bits,
                        n_codewords=args.n_codewords,
                        n_cycles=args.n_cycles,
                        train_steps=args.train_steps,
                        train_batch=args.train_batch,
                        hidden=args.hidden,
                        seed=seed,
                        bp_iters=bp_iters,
                        chunk=args.chunk,
                        device=args.device,
                        force_train=args.force_train,
                        composite_drift_threshold=comp_th,
                        model_gate_threshold=gate_th,
                        label=label,
                        out_json=tmp_json,
                    )
                except subprocess.CalledProcessError as e:
                    print(f"[ablation] FAILED: {label}  exit={e.returncode}")
                    continue

                comp_by_config[cond_name].append(result)

        # Also retrieve both-gates results from Phase 3 if available
        both_gates_results: list[dict] = []
        if "gated" in args.phases or "gated" in output["phases"]:
            # Re-read from Phase 3 aggregated section
            both_gates_bp_label = f"bp{bp_iters}"
            both_gates = output.get("phases", {}).get("gated", {}).get("aggregated", {}).get(both_gates_bp_label)
            if both_gates:
                comp_by_config["both_gates"] = []
                # Phase 3 results are already aggregated — we include them for comparison
                # But we need the raw per-seed results, so re-run the gated config if not done
                for seed in args.seeds:
                    label = f"abl_c4_both_s{seed}_b{bp_iters}"
                    tmp_json = RESULTS_DIR / f"_tmp_{label}.json"
                    try:
                        result = _run_one(
                            matrix_info=args.matrix_info,
                            matrix_rate=args.matrix_rate,
                            train_p_drift=args.train_p_drift,
                            train_p_row_burst=args.train_p_row_burst,
                            test_p_drift=args.test_p_drift,
                            test_p_row_burst=args.test_p_row_burst,
                            p_bits=args.p_bits,
                            n_codewords=args.n_codewords,
                            n_cycles=args.n_cycles,
                            train_steps=args.train_steps,
                            train_batch=args.train_batch,
                            hidden=args.hidden,
                            seed=seed,
                            bp_iters=bp_iters,
                            chunk=args.chunk,
                            device=args.device,
                            force_train=args.force_train,
                            composite_drift_threshold=best_composite,
                            model_gate_threshold=best_model_gate,
                            label=label,
                            out_json=tmp_json,
                        )
                        comp_by_config["both_gates"].append(result)
                    except subprocess.CalledProcessError as e:
                        print(f"[ablation] FAILED: {label}  exit={e.returncode}")

        # Aggregate each condition across seeds
        component_aggregated = {}
        for cond_name, results in comp_by_config.items():
            if results:
                component_aggregated[cond_name] = {
                    "composite_drift_threshold": (
                        best_composite if "drift" in cond_name or "both" in cond_name else None),
                    "model_gate_threshold": (
                        best_model_gate if "gate" in cond_name or "both" in cond_name else None),
                    "n_seeds": len(results),
                    "n_runs": len(results),
                    "rows": _aggregate_across_seeds(results),
                }

        output["phases"]["component_ablation"] = {
            "description": (
                "Isolated gating mechanism contributions. "
                "mlp_only=always-on GEMV, model_gate_only=confidence gate only, "
                "drift_gate_only=composite drift gate only, both_gates=full system."
            ),
            "composite_drift_threshold": best_composite,
            "model_gate_threshold": best_model_gate,
            "bp_iters": bp_iters,
            "aggregated": component_aggregated,
        }
        total_runs = sum(len(v) for v in comp_by_config.values())
        print(f"\n[phase 4] done: {total_runs} runs across "
              f"{len(comp_by_config)} conditions × ~{n_seeds} seeds")
        print(f"[phase 4] conditions: {list(comp_by_config.keys())}")

    # ------------------------------------------------------------------
    # Write output and cleanup
    # ------------------------------------------------------------------
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[ablation] wrote {out_json}")

    # Clean up temp files
    for tmp in RESULTS_DIR.glob("_tmp_*.json"):
        tmp.unlink()
    print("[ablation] cleaned up temp files")


if __name__ == "__main__":
    main()
