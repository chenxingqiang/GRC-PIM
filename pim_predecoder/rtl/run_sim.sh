#!/usr/bin/env bash
# Run GRC-PIM Verilog testbench on Mac (requires: brew install icarus-verilog)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
VDIR="$ROOT/pim_predecoder/rtl"
TB="$VDIR/TestBench/GRC_PIM_TOP_TB.v"
OUT="${1:-/tmp/grc_pim_tb.o}"

if ! command -v iverilog >/dev/null 2>&1; then
  echo "iverilog not found. Install: brew install icarus-verilog" >&2
  exit 1
fi

cd "$ROOT"
iverilog -g2012 -o "$OUT" \
  -I "$VDIR" \
  "$VDIR/GRC_LLR_LUT.v" \
  "$VDIR/DRIFT_DETECTOR.v" \
  "$VDIR/INT8_GEMV_MLP.v" \
  "$VDIR/GRC_PIM_TOP.v" \
  "$TB"

vvp "$OUT"

LUT_OUT="${OUT%.o}_lut.o"
iverilog -g2012 -o "$LUT_OUT" -I "$VDIR" "$VDIR/GRC_LLR_LUT.v" "$VDIR/TestBench/GRC_LLR_LUT_TB.v"
vvp "$LUT_OUT"

echo "Verilog simulation finished OK"
