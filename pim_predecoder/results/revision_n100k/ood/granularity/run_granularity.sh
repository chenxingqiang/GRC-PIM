#!/usr/bin/env bash
# Gate-granularity measurement: how the drift score behaves when the detector
# publishes it every L codewords instead of once per evaluation block.
#
# The reported results gate on a block statistic (20,000 codewords), but the
# RTL detector decides far more often. This sweep measures the actual
# per-granularity spread so the band can be placed at the granularity it is
# executed at, and so the spurious-rejection rate is a measured quantity
# rather than an extrapolation.
#
# Two findings drive the design (see sn-article.tex Sec. 4.4):
#   1. The spread does NOT fall as 1/sqrt(L). Column-drift events in
#      core/noise3d.py are shared by every codeword of a block (as they are in
#      hardware, where the drifted columns are a property of the crossbar and
#      persist for ~1e4 s against a 21.8 us codeword), so averaging cannot
#      remove the dominant variance component.
#   2. At per-codeword granularity the in-distribution and OOD drift-score
#      distributions overlap, so no upper bound separates them perfectly and
#      its placement is an explicit trade-off.
#
# win_L*.json      in-distribution p=5e-3 at L = 1, 5, 10, 20
# id_L1.json       in-distribution p=5e-3 and 1e-3 at L = 1
# C3_L1.json       conductance RTN-dominated channel at L = 1
# C4_L1.json       conductance strong-drift channel at L = 1
#
# Each row carries drift_score_blocks (the full per-block series), so the
# rejection and leak rates of any candidate band can be recomputed from the
# stored JSON without re-running.
#
# Usage (from repo root):
#   bash pim_predecoder/results/revision_n100k/ood/granularity/run_granularity.sh
set -euo pipefail

OUT="pim_predecoder/results/revision_n100k/ood/granularity"
mkdir -p "$OUT"

N_CW=2000
TAU_D=0.011
TAU_D_MAX=0.038

COMMON=(--matrix_info 1024 --matrix_rate 16_17
        --train_p_drift 5e-3 --train_p_row_burst 5e-3
        --n_codewords "$N_CW" --n_cycles 4
        --train_steps 0 --hidden 24 --bp_iters 3 --seed 1
        --drift_score_threshold "$TAU_D" --drift_score_max "$TAU_D_MAX")

# Window sweep on the worst in-distribution point.
for L in 1 5 10 20; do
  out="$OUT/win_L${L}.json"
  [[ -f "$out" ]] && { echo "skip win_L$L (exists)"; continue; }
  python3 -m pim_predecoder.experiments.gemv "${COMMON[@]}" \
    --test_p_drift 5e-3 --test_p_row_burst 5e-3 --p_bits 5e-3 \
    --eval_chunk "$L" --out_json "$out" --label "window_L${L}"
done

# Per-codeword granularity: in-distribution at two noise levels.
out="$OUT/id_L1.json"
if [[ -f "$out" ]]; then
  echo "skip id_L1 (exists)"
else
  python3 -m pim_predecoder.experiments.gemv "${COMMON[@]}" \
    --test_p_drift 5e-3 --test_p_row_burst 5e-3 --p_bits 5e-3 1e-3 \
    --eval_chunk 1 --out_json "$out" --label "gran_id_L1"
fi

# Per-codeword granularity: the two conductance channels the band must reject.
for spec in "C3 0.00 3 0.05" "C4 0.15 0 0.00"; do
  read -r cid nu tr amp <<< "$spec"
  out="$OUT/${cid}_L1.json"
  [[ -f "$out" ]] && { echo "skip ${cid}_L1 (exists)"; continue; }
  python3 -m pim_predecoder.experiments.gemv "${COMMON[@]}" \
    --channel conductance \
    --cond_sigma_ln 0.05 --cond_drift_nu "$nu" \
    --cond_rtn_traps "$tr" --cond_rtn_amp "$amp" --cond_rtn_switch 0.5 \
    --p_bits 5e-3 --eval_chunk 1 \
    --out_json "$out" --label "gran_${cid}_L1"
done

echo "granularity sweep done; results in $OUT"
