"""Read the released ECC result files, gate thresholds and hardware inputs.

Paths resolve relative to this module, including when the bundle is used
standalone. No manuscript, figure generator or plotting libraries are needed.
"""
from __future__ import annotations

import json
import statistics as stats
from dataclasses import dataclass
from pathlib import Path

__all__ = (
    "BASELINES", "E2_SEEDS", "GATE", "LINEAR_FORMAL",
    "N100K", "RECAL", "RESULTS", "SYNTH",
    "WBP_CALIB", "WBP_FORMAL", "AggregatedRow", "GateScenarioRow",
    "WbpCalibPoint", "aggregate_1024_e2", "aggregate_256_e2", "e5_admitted",
    "e5_gate_points", "gate_composition", "gate_scenarios", "inference_accuracy",
    "linear_formal", "row_at_p", "wbp_calibration", "wbp_formal",
)

RESULTS = Path(__file__).resolve().parent / "results"
BASELINES = RESULTS / "baselines"
N100K = RESULTS / "revision_n100k"
RECAL = N100K / "recal"
WBP_CALIB = N100K / "wbp" / "calib"
WBP_FORMAL = N100K / "wbp"
LINEAR_FORMAL = N100K / "linear"

# Gate thresholds in force for the reported results. Keep in sync with
# sn-article.tex (Table tab:design, Sec. 2.3, Sec. 4.3).
GATE = {"tau_d_1024": 0.011, "tau_d_256": 0.005,
        "tau_d_max_1024": 0.038, "tau_d_max_256": 0.045,
        "tau_g": 0.6}

# Synthesis and simulation inputs for the released hardware comparison.
SYNTH = {
    "top_area_um2_ng45": 24798.914,   # yosys, GRC_PIM_TOP flattened
    "mac_area_um2_ng45": 791.616,     # yosys, one INT8_MAC instance
    "nand2_um2_ng45": 0.798,          # Nangate45 NAND2_X1
    "nand2_um2_28nm": 0.32,           # typical 28 nm foundry NAND2_X1
    "pr_utilisation": 0.60,           # placement utilisation for a small macro
    "weight_sram_um2": 3000.0,        # 425-byte INT8 weight bank
    "cycles_per_symbol_1mac": 401,    # iverilog-measured, as delivered
    "macs_per_symbol": 397,           # 13x24 + 24x3 + 13
    "freq_hz": 200e6,
    "n_symbols_1024": 1088,
    "n_symbols_256": 288,
}

@dataclass(frozen=True)
class AggregatedRow:
    p_bit: float
    bp_mean: float
    bp_std: float
    gemv_mean: float
    gemv_std: float
    use_gemv_fraction: float
    resolution_floor: float
    # Per-seed B'/GRC-PIM ratios, in seed order. The manuscript's gain cells
    # are the mean and sample s.d. of these, not the ratio of the two means.
    gains: tuple[float, ...] = ()
    drift_scores: tuple[float, ...] = ()
    gate_scores: tuple[float, ...] = ()
    # The same quantities with both gates forced open: what the calibrator
    # delivers when admitted. Equal to the deployed values wherever the gates
    # are open; where one is closed the deployed value is B' and these are not.
    ungated_mean: float = float("nan")
    ungated_std: float = float("nan")
    ungated_gains: tuple[float, ...] = ()
    open_flags: tuple[bool, ...] = ()

    @property
    def n_seeds(self) -> int:
        return len(self.gains)

    @property
    def n_open(self) -> int:
        return round(self.use_gemv_fraction * self.n_seeds)

    @property
    def gain_seed_mean(self) -> float:
        return stats.mean(self.gains)

    @property
    def gain_seed_std(self) -> float:
        return stats.stdev(self.gains) if len(self.gains) > 1 else 0.0

    @property
    def ungated_gain_mean(self) -> float:
        return stats.mean(self.ungated_gains)

    @property
    def ungated_gain_std(self) -> float:
        return (stats.stdev(self.ungated_gains)
                if len(self.ungated_gains) > 1 else 0.0)

    @property
    def gate_active(self) -> bool:
        return self.use_gemv_fraction > 0

    @property
    def gain_mean(self) -> float:
        if self.gemv_mean <= 0:
            return 1.0
        return self.bp_mean / self.gemv_mean


@dataclass(frozen=True)
class GateScenarioRow:
    scenario: str
    drift_score: float
    use_gemv: bool
    bp_ber: float
    gemv_ber: float
    ber_floor: float
    acceptance_score: float


@dataclass(frozen=True)
class WbpCalibPoint:
    """Held-out calibration point of the weighted-BP baseline.

    ``mean_ratio``/``std_ratio`` aggregate, across the 3 held-out seeds, the
    mean over operating points of Wbp/B$'$ on the scale's headline metric
    (SER at the 256-symbol scale, BER at the 1024-symbol scale).
    """
    scale: int
    rho: float
    mean_ratio: float
    std_ratio: float
    per_seed: tuple[float, ...]
    per_seed_gains: tuple[float, ...]

    @property
    def excess(self) -> float:
        """Mean degradation over B$'$ in percent (0 at rho=1)."""
        return 100.0 * (self.mean_ratio - 1.0)

    @property
    def mean_gain(self) -> float:
        """Mean across seeds of their arithmetic means of pointwise B'/Wbp."""
        return stats.mean(self.per_seed_gains)

    @property
    def std_gain(self) -> float:
        """Across-seed s.d. of the per-seed gain."""
        return stats.stdev(self.per_seed_gains)


# Seeds behind the headline E2 tables and the formal baseline rows.
E2_SEEDS = tuple(range(1, 11))
# Seeds of the held-out weighted-BP calibration grid and of the secondary
# analyses that were not extended (gated linear diagnostic, crossed-seed,
# E5-without-gates, drift sweep, device-calibrated cross-check).
_CALIB_SEEDS = (1, 2, 3)


def wbp_calibration(directory: Path = WBP_CALIB) -> list[WbpCalibPoint]:
    """Aggregate the held-out weighted-BP calibration runs (Fig. 3).

    Each JSON file is one (scale, seed, rho) run on the E2 matched channel
    with a held-out evaluation seed; the grid is {0.9, 0.75, 0.5, 0.25} per
    scale because rho=1 is B$'$ itself. Returns one point per (scale, rho).
    """
    groups: dict[tuple[int, float], dict[int, dict]] = {}
    for path in sorted(directory.glob("wbp_calib_*.json")):
        stem_parts = path.stem.split("_")          # wbp calib {scale} s{seed} rho{tag}
        scale, seed = int(stem_parts[2]), int(stem_parts[3][1:])
        j = _load(path)
        groups.setdefault((scale, float(j["wbp_rho"])), {})[seed] = j

    points: list[WbpCalibPoint] = []
    for (scale, rho), by_seed in sorted(groups.items()):
        missing = [s for s in _CALIB_SEEDS if s not in by_seed]
        if missing:
            raise RuntimeError(
                f"calibration grid incomplete: {scale}-symbol scale, rho={rho} "
                f"missing seed(s) {missing}")
        bp_field, wbp_field = (
            ("Bp_post", "Wbp_post") if scale == 256
            else ("Bp_post_BER", "Wbp_post_BER")
        )
        per_seed: list[float] = []
        per_seed_gains: list[float] = []
        for seed in _CALIB_SEEDS:
            ratios = [row[wbp_field] / row[bp_field]
                      for row in by_seed[seed]["rows"] if row[bp_field] > 0]
            if not ratios:
                raise RuntimeError(
                    f"calibration run {scale}-symbol seed {seed} rho={rho}: "
                    f"no usable operating point (B$'$ decoded all-zero everywhere)")
            if 0 in ratios:
                raise ValueError(f"undefined B'/Wbp gain: {scale}-symbol seed {seed} "
                                 f"rho={rho} has a zero Wbp error rate")
            per_seed.append(stats.mean(ratios))
            per_seed_gains.append(stats.mean(1.0 / ratio for ratio in ratios))
        points.append(WbpCalibPoint(
            scale=scale,
            rho=rho,
            mean_ratio=stats.mean(per_seed),
            std_ratio=stats.stdev(per_seed),
            per_seed=tuple(per_seed),
            per_seed_gains=tuple(per_seed_gains),
        ))
    return points


def _wbp_run_path(directory: Path, scale: int, seed: int) -> Path:
    """Result file carrying the weighted-BP columns of one (scale, seed).

    Seeds 1-3 (and 256-symbol seeds 4-5) have a dedicated run under ``wbp/``.
    For the remaining added seeds the weighted-BP path was evaluated in the
    same run as the ungated linear head: the fusion rule has no learned
    part, so its columns do not depend on which model the run loads (checked
    bit-identical across the two models), and one run yields both baselines.
    The dedicated file wins when both exist; the combined file is used only
    from the released results tree and only if it records ``wbp_rho``.
    """
    primary = directory / f"E2_{scale}_s{seed}_n100k_rho09.json"
    if primary.exists() or directory != WBP_FORMAL:
        return primary
    combined = LINEAR_FORMAL / f"E2_{scale}_s{seed}_n100k_linear_ungated.json"
    if combined.exists() and _load(combined).get("wbp_rho") is not None:
        return combined
    return primary


def wbp_formal(directory: Path = WBP_FORMAL,
               seeds: tuple[int, ...] = E2_SEEDS) -> dict[int, dict]:
    """Aggregate the formal 10^5 weighted-BP (rho=0.9) runs (Table 2 row).

    Each JSON is one (scale, seed) run on the E2 matched channel with the
    test seeds and the full 100,000-codeword budget. Returns, per scale,
    the Table tab:baseline_comparison cells---(mean, std) across
    ``seeds`` of the 256-symbol SER at p_bit = 5e-3 and 3e-3, and of the
    1024-symbol BER at 5e-3---plus the error ratios against B$'$ on the
    scale's headline metric: per operating point shown and per seed over
    all operating points.
    """
    out: dict[int, dict] = {}
    for scale in (256, 1024):
        wbp_field = "Wbp_post" if scale == 256 else "Wbp_post_BER"
        bp_field = "Bp_post" if scale == 256 else "Bp_post_BER"
        cell_idx = (0, 1) if scale == 256 else (0,)
        rows: dict[int, list[dict]] = {}
        for seed in seeds:
            j = _load(_wbp_run_path(directory, scale, seed))
            if j.get("wbp_rho") != 0.9:
                raise RuntimeError(
                    f"formal WBP run {scale}-symbol seed {seed}: wbp_rho="
                    f"{j.get('wbp_rho')} != 0.9")
            rows[seed] = j["rows"]
        cells = [
            (stats.mean([rows[s][i][wbp_field] for s in seeds]),
             stats.stdev([rows[s][i][wbp_field] for s in seeds]))
            for i in cell_idx
        ]
        ratio_pt = tuple(
            stats.mean(
                rows[s][i][wbp_field] / rows[s][i][bp_field]
                for s in seeds if rows[s][i][bp_field] > 0)
            for i in cell_idx
        )
        gain_pt = tuple(
            stats.mean(
                rows[s][i][bp_field] / rows[s][i][wbp_field]
                for s in seeds if rows[s][i][wbp_field] > 0)
            for i in cell_idx
        )
        per_seed = []
        per_seed_gain = []
        for seed in seeds:
            usable = [r[wbp_field] / r[bp_field]
                      for r in rows[seed] if r[bp_field] > 0]
            if not usable:
                raise RuntimeError(
                    f"formal WBP run {scale}-symbol seed {seed}: no usable "
                    "operating point (B$'$ decoded all-zero everywhere)")
            per_seed.append(stats.mean(usable))
            per_seed_gain.append(stats.mean([1.0 / r for r in usable]))
        out[scale] = {
            "cells": cells,
            "ratio_pt": ratio_pt,
            "ratio_mean": stats.mean(per_seed),
            "ratio_std": stats.stdev(per_seed),
            # Mean over seeds of the per-seed gain B'/method, the convention
            # of every gain in the manuscript (not the reciprocal of the mean
            # ratio, which differs at the second decimal).
            "gain_pt": gain_pt,
            "gain_mean": stats.mean(per_seed_gain),
            "gain_std": stats.stdev(per_seed_gain),
        }
    return out


def linear_formal(directory: Path = LINEAR_FORMAL,
                  ungated: bool = False,
                  seeds: tuple[int, ...] | None = None) -> dict[int, dict]:
    """Aggregate the formal 10^5 linear-head runs (Table 2 row).

    The linear head consumes the same 13 features through the same gates
    (tau_d, tau_g) and the same training budget and data stream as the GEMV
    calibrator, so its error ratio against B$'$ isolates the value of the
    hidden nonlinearity. The row schema reuses the GEMV fields, and
    GEMV_post / GEMV_post_BER already carry the gated fallback to B$'$.
    With ungated=True the *_linear_ungated.json diagnostic runs are read
    instead (the same models with tau_g disabled, isolating what the linear
    head actually learns). Returns, per scale, the Table
    tab:baseline_comparison cells---(mean, std) across ``seeds`` of the
    256-symbol SER at p_bit = 5e-3 and 3e-3, and of the 1024-symbol BER at
    5e-3---plus the error ratios against B$'$: per operating point shown
    and per seed over all operating points.
    """
    suffix = "_ungated" if ungated else ""
    if seeds is None:
        seeds = E2_SEEDS if ungated else _CALIB_SEEDS
    out: dict[int, dict] = {}
    for scale in (256, 1024):
        lin_field = "GEMV_post" if scale == 256 else "GEMV_post_BER"
        bp_field = "Bp_post" if scale == 256 else "Bp_post_BER"
        cell_idx = (0, 1) if scale == 256 else (0,)
        rows: dict[int, list[dict]] = {}
        for seed in seeds:
            j = _load(directory /
                      f"E2_{scale}_s{seed}_n100k_linear{suffix}.json")
            if j.get("model") != "LinearResidualCalibrator":
                raise RuntimeError(
                    f"formal linear run {scale}-symbol seed {seed}: model="
                    f"{j.get('model')} != LinearResidualCalibrator")
            rows[seed] = j["rows"]
        cells = [
            (stats.mean([rows[s][i][lin_field] for s in seeds]),
             stats.stdev([rows[s][i][lin_field] for s in seeds]))
            for i in cell_idx
        ]
        ratio_pt = tuple(
            stats.mean(
                rows[s][i][lin_field] / rows[s][i][bp_field]
                for s in seeds if rows[s][i][bp_field] > 0)
            for i in cell_idx
        )
        gain_pt = tuple(
            stats.mean(
                rows[s][i][bp_field] / rows[s][i][lin_field]
                for s in seeds if rows[s][i][lin_field] > 0)
            for i in cell_idx
        )
        per_seed = []
        per_seed_gain = []
        for seed in seeds:
            usable = [r[lin_field] / r[bp_field]
                      for r in rows[seed] if r[bp_field] > 0]
            if not usable:
                raise RuntimeError(
                    f"formal linear run {scale}-symbol seed {seed}: no usable "
                    "operating point (B$'$ decoded all-zero everywhere)")
            per_seed.append(stats.mean(usable))
            per_seed_gain.append(stats.mean([1.0 / r for r in usable]))
        out[scale] = {
            "cells": cells,
            "ratio_pt": ratio_pt,
            "ratio_mean": stats.mean(per_seed),
            "ratio_std": stats.stdev(per_seed),
            # Mean over seeds of the per-seed gain B'/method, the convention
            # of every gain in the manuscript (not the reciprocal of the mean
            # ratio, which differs at the second decimal).
            "gain_pt": gain_pt,
            "gain_mean": stats.mean(per_seed_gain),
            "gain_std": stats.stdev(per_seed_gain),
        }
    return out


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _row_at(j: dict, p_bit: float) -> dict:
    for row in j["rows"]:
        if abs(row["p_bit"] - p_bit) < 1e-12:
            return row
    raise KeyError(f"p_bit={p_bit} not in {path_label(j)}")


def path_label(j: dict) -> str:
    return j.get("tag", j.get("experiment", "json"))


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return stats.mean(values), stats.stdev(values)


def _ungated_value(directory: Path, scale: int, seed: int, tag: str,
                   row: dict, gemv_field: str, p_bit: float) -> float:
    """Post-decoding error rate of the calibrator with both gates open.

    Runs made after the ``GEMV_ungated_*`` fields were added carry it in the
    row. Earlier runs (seeds 1-3) do not: there an open gate means the stored
    value already is the ungated one, and a closed gate needs the forced-open
    companion run of the same seed and data stream.
    """
    field = gemv_field.replace("GEMV_post", "GEMV_ungated_post")
    if field in row:
        return row[field]
    if row.get("use_gemv", False):
        return row[gemv_field]
    companion = directory / f"E2_{scale}_s{seed}_n100k_{tag}_ungated_full.json"
    if not companion.exists():
        raise FileNotFoundError(
            f"{scale}-symbol seed {seed} closes the gate at p_bit={p_bit:g} "
            f"and records no ungated outcome; expected {companion.name}")
    bp_field = gemv_field.replace("GEMV_post", "Bp_post")
    forced = _row_at(_load(companion), p_bit)
    if forced[bp_field] != row[bp_field]:
        raise RuntimeError(
            f"{companion.name}: B' differs from the gated run at "
            f"p_bit={p_bit:g}, so it is not the same data stream")
    return forced[gemv_field]


def _aggregate_e2(scale: int, tag: str, bp_field: str, gemv_field: str,
                  default_floor: float,
                  seeds: tuple[int, ...] = E2_SEEDS,
                  directory: Path | None = None) -> list[AggregatedRow]:
    """Aggregate the E2 runs of one code scale over ``seeds``.

    A missing seed file raises: silently averaging fewer seeds than the
    manuscript states would make every quoted spread wrong.
    """
    directory = directory or RECAL
    runs = [_load(directory / f"E2_{scale}_s{s}_n100k_{tag}.json")
            for s in seeds]
    p_bits = [r["p_bit"] for r in runs[0]["rows"]]
    rows: list[AggregatedRow] = []
    for p in p_bits:
        bp_vals: list[float] = []
        gemv_vals: list[float] = []
        ungated_vals: list[float] = []
        use_flags: list[bool] = []
        floors: list[float] = []
        drifts: list[float] = []
        gates: list[float] = []
        for seed, j in zip(seeds, runs):
            row = _row_at(j, p)
            use = bool(row.get("use_gemv", False))
            bp_vals.append(row[bp_field])
            # A closed gate falls back to B', exactly as the hybrid LLR does.
            gemv_vals.append(row[gemv_field] if use else row[bp_field])
            ungated_vals.append(_ungated_value(
                directory, scale, seed, tag, row, gemv_field, p))
            use_flags.append(use)
            floors.append(row.get("resolution_floor", default_floor))
            drifts.append(row.get("drift_score", float("nan")))
            gates.append(row.get("GEMV_gate", float("nan")))
        bp_m, bp_s = _mean_std(bp_vals)
        gemv_m, gemv_s = _mean_std(gemv_vals)
        ung_m, ung_s = _mean_std(ungated_vals)
        rows.append(
            AggregatedRow(
                p_bit=p,
                bp_mean=bp_m,
                bp_std=bp_s,
                gemv_mean=gemv_m,
                gemv_std=gemv_s,
                use_gemv_fraction=sum(use_flags) / len(use_flags),
                resolution_floor=stats.mean(floors),
                gains=tuple(b / g if g > 0 else 1.0
                            for b, g in zip(bp_vals, gemv_vals)),
                drift_scores=tuple(drifts),
                gate_scores=tuple(gates),
                ungated_mean=ung_m,
                ungated_std=ung_s,
                ungated_gains=tuple(b / g if g > 0 else 1.0
                                    for b, g in zip(bp_vals, ungated_vals)),
                open_flags=tuple(use_flags),
            )
        )
    return rows


def aggregate_1024_e2(seeds: tuple[int, ...] = E2_SEEDS,
                      directory: Path | None = None) -> list[AggregatedRow]:
    """Multi-seed 1024-symbol E2 rows aligned with Table tab:ablation_1024."""
    return _aggregate_e2(1024, "td011", "Bp_post_BER", "GEMV_post_BER",
                         2.8e-8, seeds, directory)


def aggregate_256_e2(seeds: tuple[int, ...] = E2_SEEDS,
                     directory: Path | None = None) -> list[AggregatedRow]:
    """Multi-seed 256-symbol E2 rows aligned with Table tab:ablation_256."""
    return _aggregate_e2(256, "td005", "Bp_post", "GEMV_post",
                         1.0e-7, seeds, directory)


def gate_composition(scale: int, tau_d: float | None = None,
                     tau_g: float | None = None,
                     tau_d_max: float | None = None,
                     seeds: tuple[int, ...] = E2_SEEDS,
                     directory: Path | None = None) -> tuple[float, int, int]:
    """Exact E2 gain under an arbitrary threshold triple, by composition.

    The gates enter the result only through the indicators
    ``tau_d <= d <= tau_d_max`` and ``g-bar >= tau_g``. Once the per-point
    scores and the calibrator's ungated outcome are recorded, the gain any
    threshold triple would have produced follows without re-simulation: an
    admitted point contributes its ungated gain and a rejected point exactly
    1. Thresholds left as ``None`` take their deployed values. Returns
    ``(mean gain over all point x seed combinations, points admitted,
    points)``.

    The ungated gain must be used for admitted points. The stored
    ``GEMV_post*`` of a point the deployed thresholds close is the B'
    fallback, so composing from it would credit a re-admitted point with a
    gain of 1.
    """
    tau_d = GATE[f"tau_d_{scale}"] if tau_d is None else tau_d
    tau_d_max = GATE[f"tau_d_max_{scale}"] if tau_d_max is None else tau_d_max
    tau_g = GATE["tau_g"] if tau_g is None else tau_g
    aggregate = aggregate_1024_e2 if scale == 1024 else aggregate_256_e2
    gains: list[float] = []
    opened = 0
    for row in aggregate(seeds, directory):
        for d, g, ungated in zip(row.drift_scores, row.gate_scores,
                                 row.ungated_gains):
            use = tau_d <= d <= tau_d_max and g >= tau_g
            gains.append(ungated if use else 1.0)
            opened += int(use)
    return stats.mean(gains), opened, len(gains)


def e5_gate_points(scale: int) -> list[tuple[float, float]]:
    """(acceptance score, drift score) of every E5 mismatch operating point.

    E5 is the i.i.d.-trained model tested on strong drift
    (``results/revision_n100k/nogate``, seeds 1-3 x 5 p_bit, 10^5 codewords).
    The scores do not depend on the thresholds, so these points give the
    number of mismatched evaluations any (tau_d, tau_g) pair would admit.
    """
    out: list[tuple[float, float]] = []
    for seed in _CALIB_SEEDS:
        j = _load(N100K / "nogate" / f"nogate_{scale}_s{seed}.json")
        out += [(r["GEMV_gate"], r["drift_score"]) for r in j["rows"]]
    return out


def e5_admitted(scale: int, tau_g: float) -> tuple[int, int]:
    """(E5 points admitted at ``tau_g`` with the deployed band, E5 points)."""
    lo, hi = GATE[f"tau_d_{scale}"], GATE[f"tau_d_max_{scale}"]
    pts = e5_gate_points(scale)
    return sum(1 for g, d in pts if lo <= d <= hi and g >= tau_g), len(pts)


def gate_scenarios(p_bit: float = 5e-3) -> list[GateScenarioRow]:
    """E1–E5 gate-case rows (800 codewords, seed 1) for Table tab:gate_cases / Fig. 4."""
    out: list[GateScenarioRow] = []
    for scenario in ("E1", "E2", "E3", "E4", "E5"):
        j = _load(RESULTS / f"paper1024_{scenario}_gemv_gated.json")
        row = _row_at(j, p_bit)
        # resolution_floor in the legacy JSON counts symbols, not bits.
        ber_floor = 3.0 / (j["n_codewords"] * j["matrix_shape"]["G"][1] * 2)
        out.append(
            GateScenarioRow(
                scenario=scenario,
                drift_score=row.get("drift_score", 0.0),
                use_gemv=bool(row.get("use_gemv", False)),
                bp_ber=row["Bp_post_BER"],
                gemv_ber=row["GEMV_post_BER"],
                ber_floor=ber_floor,
                acceptance_score=row["GEMV_gate"],
            )
        )
    return out


def inference_accuracy() -> dict:
    return _load(RESULTS / "inference_accuracy.json")


def row_at_p(rows: list[AggregatedRow], p_bit: float) -> AggregatedRow:
    for row in rows:
        if abs(row.p_bit - p_bit) < 1e-12:
            return row
    raise KeyError(p_bit)
