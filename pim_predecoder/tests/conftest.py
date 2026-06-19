"""Shared fixtures for pim_predecoder tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pim_predecoder.paths import CACHE, REPO, RTL


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def hg_default(repo_root: Path):
    from pim_predecoder.core.codec import load_default_HG

    return load_default_HG(repo_root)


@pytest.fixture(scope="session")
def hg_1024(repo_root: Path):
    from pim_predecoder.core.codec import load_HG

    return load_HG(repo_root, info_length=1024, rate="16_17")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def has_iverilog() -> bool:
    import os
    path = os.environ.get("PATH", "")
    extra = "/opt/homebrew/bin:/usr/local/bin"
    for directory in (path + ":" + extra).split(":"):
        if not directory:
            continue
        if (Path(directory) / "iverilog").is_file() and (Path(directory) / "vvp").is_file():
            return True
    return False


@pytest.fixture
def require_iverilog(has_iverilog: bool):
    if not has_iverilog:
        pytest.skip("iverilog/vvp not installed")


@pytest.fixture(scope="session")
def gemv_checkpoint() -> Path:
    path = CACHE / "gemv_k1024_n1088_r16_17_T4_d5e-03_b5e-03_s600_bs48_h24_seed1.pt"
    if not path.exists():
        pytest.skip(f"missing checkpoint: {path}")
    return path


@pytest.fixture(scope="session")
def rtl_weights_dir() -> Path:
    d = RTL / "Weights"
    if not (d / "quant_meta.json").exists():
        pytest.skip(f"missing RTL weights: {d}")
    return d
