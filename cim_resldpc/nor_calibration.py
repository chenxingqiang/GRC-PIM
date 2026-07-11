"""NOR Flash floating-gate oriented CiM noise profiles (calibrated proxies).

Honest scope: these are *engineering calibrations* mapped onto the GRC-PIM
multi-cycle bit-flip channel, not a full TCAD floating-gate device model.
Parameters are chosen to reflect literature-typical NOR read-disturb /
retention drift regimes relative to the generic PIM defaults.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .noise_bridge import NoiseProfile


@dataclass(frozen=True)
class NORCalibration:
    """Named NOR operating point → NoiseProfile + metadata."""

    name: str
    description: str
    profile: NoiseProfile
    # Physical knobs (documentation / paper tables)
    vread_v: float
    temperature_c: float
    read_cycles_equiv: int
    source: str


# Calibrated presets (proxy mapping → p_indep / p_drift / p_row_burst / T)
NOR_PRESETS: dict[str, NORCalibration] = {
    "nor_fresh": NORCalibration(
        name="nor_fresh",
        description="Fresh programmed NOR MLC, room temp, few reads",
        profile=NoiseProfile(
            p_indep=1e-3, p_drift=5e-4, p_row_burst=2e-4, n_cycles=4, drift_persist=2,
        ),
        vread_v=1.0,
        temperature_c=25.0,
        read_cycles_equiv=10,
        source="proxy: low read-disturb / retention (engineering map to GRC-PIM channel)",
    ),
    "nor_typical": NORCalibration(
        name="nor_typical",
        description="Typical edge RAG duty: moderate read disturb + weak drift",
        profile=NoiseProfile(
            p_indep=5e-3, p_drift=5e-3, p_row_burst=1e-3, n_cycles=4, drift_persist=3,
        ),
        vread_v=1.2,
        temperature_c=55.0,
        read_cycles_equiv=1_000,
        source="proxy: mid stress aligned with GRC-PIM E2-like defaults",
    ),
    "nor_aged": NORCalibration(
        name="nor_aged",
        description="Aged / high read-count NOR: strong column drift + disturb",
        profile=NoiseProfile(
            p_indep=2e-2, p_drift=3e-2, p_row_burst=5e-3, n_cycles=4, drift_persist=4,
        ),
        vread_v=1.5,
        temperature_c=85.0,
        read_cycles_equiv=100_000,
        source="proxy: high-temp retention + cumulative read disturb",
    ),
    "nor_stress_sweep": NORCalibration(
        name="nor_stress_sweep",
        description="Aggressive stress corner for ablation (not nominal)",
        profile=NoiseProfile(
            p_indep=5e-2, p_drift=5e-2, p_row_burst=1e-2, n_cycles=6, drift_persist=4,
        ),
        vread_v=1.8,
        temperature_c=105.0,
        read_cycles_equiv=1_000_000,
        source="proxy: worst-case corner for robustness curves",
    ),
}


def get_nor_profile(name: str = "nor_typical") -> NoiseProfile:
    if name not in NOR_PRESETS:
        raise KeyError(f"unknown NOR preset {name!r}; choose from {sorted(NOR_PRESETS)}")
    return NOR_PRESETS[name].profile


def list_nor_presets() -> list[dict]:
    out = []
    for cal in NOR_PRESETS.values():
        d = {
            "name": cal.name,
            "description": cal.description,
            "vread_v": cal.vread_v,
            "temperature_c": cal.temperature_c,
            "read_cycles_equiv": cal.read_cycles_equiv,
            "source": cal.source,
            "profile": {
                "p_indep": cal.profile.p_indep,
                "p_drift": cal.profile.p_drift,
                "p_row_burst": cal.profile.p_row_burst,
                "n_cycles": cal.profile.n_cycles,
                "drift_persist": cal.profile.drift_persist,
            },
        }
        out.append(d)
    return out
