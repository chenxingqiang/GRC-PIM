"""Shared paths for the pim_predecoder package."""
from __future__ import annotations

from pathlib import Path

PKG = Path(__file__).resolve().parent
REPO = PKG.parent
RTL = PKG / "rtl"
RTL_WEIGHTS = RTL / "Weights"
CACHE = PKG / "_cache"
RESULTS = PKG / "results"
