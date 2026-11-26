#!/usr/bin/env bash
# Post-synthesis area report for GRC_PIM_TOP (requires: brew install yosys).
#
# Produces a real gate-level area number for the calibrator instead of an
# analytical estimate, and projects it to a target technology plus a
# place-and-route utilisation factor. Answers the "area excludes P&R" critique
# with synthesis evidence.
#
# Usage:
#   run_synth.sh [-l liberty] [-n nand2_area_target] [-u utilisation] [-o out.json]
#
# The default liberty is the open Nangate45 library; download once with:
#   curl -sSL -o /tmp/lib/ng45.lib \
#     https://cdn.jsdelivr.net/gh/The-OpenROAD-Project/OpenROAD-flow-scripts@master/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
VDIR="$ROOT/pim_predecoder/rtl"

LIBERTY="/tmp/lib/ng45.lib"
NAND2_TARGET="0.32"     # 28 nm foundry NAND2_X1 cell area (um^2)
UTIL="0.60"             # P&R placement utilisation for a small digital macro
SRAM_UM2="3000"         # 425-byte INT8 weight bank (um^2)
OUT="$VDIR/synth_area.json"

# The NB-LDPC decoder this block is compared against is a 40 nm prototype chip
# (Shi et al.), so a ratio taken against the 28 nm projection above would be a
# cross-node comparison. We therefore also project to 40 nm and report the
# matched-node ratio. 0.65 um^2 is the 28 nm cell scaled by (40/28)^2, and the
# weight bank is scaled by the same factor.
NAND2_40NM="0.65"
SRAM_40NM="6000"
DECODER_MM2="0.8"       # Shi et al. 40 nm NB-LDPC decoder area

while getopts "l:n:u:s:o:" opt; do
  case "$opt" in
    l) LIBERTY="$OPTARG" ;;
    n) NAND2_TARGET="$OPTARG" ;;
    u) UTIL="$OPTARG" ;;
    s) SRAM_UM2="$OPTARG" ;;
    o) OUT="$OPTARG" ;;
    *) echo "usage: $0 [-l liberty] [-n nand2_area] [-u util] [-s sram_um2] [-o out.json]" >&2; exit 2 ;;
  esac
done

command -v yosys >/dev/null 2>&1 || { echo "yosys not found. Install: brew install yosys" >&2; exit 1; }
[ -r "$LIBERTY" ] || { echo "liberty not readable: $LIBERTY (see header for download)" >&2; exit 1; }

LOG="${OUT%.json}.log"
cd "$ROOT"

# Weight ROMs are loaded with $readmemh relative to the repo root.
yosys -l "$LOG" -p "
read_liberty -lib $LIBERTY;
read_verilog $VDIR/INT8_GEMV_MLP.v $VDIR/DRIFT_DETECTOR.v \
             $VDIR/DRIFT_DETECTOR_CODEWORD.v $VDIR/GRC_LLR_LUT.v \
             $VDIR/LLR_INT8_TO_ECC3.v $VDIR/GRC_PIM_TOP.v;
synth -top GRC_PIM_TOP -flatten;
dfflibmap -liberty $LIBERTY;
abc -liberty $LIBERTY;
opt_clean;
stat -liberty $LIBERTY;
" >/dev/null

python3 - "$LOG" "$LIBERTY" "$NAND2_TARGET" "$UTIL" "$SRAM_UM2" "$OUT" \
         "$NAND2_40NM" "$SRAM_40NM" "$DECODER_MM2" <<'PY'
import json, re, sys

log, liberty, nand2_target, util, sram_um2, out = sys.argv[1:7]
nand2_40, sram_40, decoder_mm2 = (float(v) for v in sys.argv[7:10])
nand2_target, util, sram_um2 = float(nand2_target), float(util), float(sram_um2)

text = open(log).read()
area = float(re.findall(r"Chip area for module \S*GRC_PIM_TOP\S*: ([0-9.]+)", text)[-1])
seq = re.findall(r"sequential elements: ([0-9.]+)", text)
cells = int(re.findall(r"^\s+(\d+) cells$", text, re.M)[-1])

lib = open(liberty).read()
m = re.search(r"cell \(NAND2_X1\)\s*\{(.{0,400})", lib, re.S)
nand2_ref = float(re.search(r"area\s*:\s*([0-9.]+)", m.group(1)).group(1))

ge = area / nand2_ref                 # technology-independent gate equivalents
core = ge * nand2_target              # cell area in the target technology
placed = core / util                  # after P&R placement utilisation
total = placed + sram_um2

# matched-node projection against the 40 nm decoder baseline
total_40 = ge * nand2_40 / util + sram_40

report = {
    "liberty": liberty,
    "cells": cells,
    "area_um2_liberty": area,
    "sequential_um2": float(seq[-1]) if seq else None,
    "nand2_um2_liberty": nand2_ref,
    "gate_equivalents": round(ge, 1),
    "nand2_um2_target": nand2_target,
    "core_cell_area_um2_target": round(core, 1),
    "pr_utilisation": util,
    "placed_area_um2": round(placed, 1),
    "weight_sram_um2": sram_um2,
    "total_area_mm2_with_pr": round(total / 1e6, 5),
    "nand2_um2_40nm": nand2_40,
    "weight_sram_um2_40nm": sram_40,
    "total_area_mm2_40nm": round(total_40 / 1e6, 5),
    "decoder_area_mm2_40nm": decoder_mm2,
    "ratio_to_decoder_matched_40nm": round(total_40 / 1e6 / decoder_mm2, 4),
}
with open(out, "w") as f:
    json.dump(report, f, indent=2)

print(f"cells                     {cells:,}")
print(f"area @ liberty            {area:,.0f} um2  ({liberty})")
print(f"gate equivalents          {ge:,.0f} GE")
print(f"core cells @ target tech  {core:,.0f} um2  (NAND2={nand2_target} um2)")
print(f"after P&R (util {util:.0%})      {placed:,.0f} um2")
print(f"+ weight SRAM             {sram_um2:,.0f} um2")
print(f"TOTAL with P&R @ 28 nm    {total/1e6:.4f} mm2")
print(f"TOTAL with P&R @ 40 nm    {total_40/1e6:.4f} mm2  (matched to the decoder node)")
print(f"ratio to decoder @ 40 nm  {total_40/1e6/decoder_mm2:.1%}  (vs {decoder_mm2} mm2)")
print(f"wrote {out}")
PY
