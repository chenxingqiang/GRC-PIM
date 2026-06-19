"""Tests for pim_predecoder.paths."""
from __future__ import annotations

from pim_predecoder import paths


def test_paths_layout():
    assert paths.PKG.name == "pim_predecoder"
    assert paths.REPO == paths.PKG.parent
    assert paths.RTL == paths.PKG / "rtl"
    assert paths.RTL_WEIGHTS == paths.RTL / "Weights"
    assert paths.CACHE == paths.PKG / "_cache"
    assert paths.RESULTS == paths.PKG / "results"


def test_rtl_weights_exist():
    assert (paths.RTL_WEIGHTS / "fc1_w.hex").is_file()
    assert (paths.RTL_WEIGHTS / "quant_meta.json").is_file()
