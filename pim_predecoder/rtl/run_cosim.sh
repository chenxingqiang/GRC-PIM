#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
VDIR="$ROOT/pim_predecoder/rtl"
OUT="${1:-/tmp/grc_pim_cosim.o}"

cd "$ROOT"
iverilog -g2012 -o "$OUT" \
  "$VDIR/GRC_LLR_LUT.v" \
  "$VDIR/DRIFT_DETECTOR_CODEWORD.v" \
  "$VDIR/INT8_GEMV_MLP.v" \
  "$VDIR/GRC_PIM_TOP.v" \
  "$VDIR/TestBench/GRC_PIM_TOP_COSIM_TB.v"

vvp "$OUT"
