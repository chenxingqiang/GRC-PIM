#!/usr/bin/env bash
# Drift-strength sweep: the one axis along which this channel actually changes.
#
# The p_bit sweep reported in the main tables holds p_drift and p_row_burst at
# 5e-3 regardless of p_bit, so the B' error rate is a plateau and the five
# "operating points" are close to a single point repeated. The impairment that
# the calibrator corrects is the drift, so this sweep varies p_drift over two
# and a half decades at a fixed p_bit = 1e-3 and a fixed p_row_burst = 5e-3,
# which is what produces a waterfall rather than a plateau and what places the
# drift band in terms of a physical channel parameter instead of a score.
#
# The model is the deployed E2 checkpoint (trained at p_drift = p_row_burst =
# 5e-3) and is loaded from _cache/; this script never retrains it. No gate
# thresholds are passed, so GEMV_post is the ungated output and use_gemv /
# drift_score record where the band would have opened.
#
# Run from the repo root:
#   bash pim_predecoder/results/revision_n100k/drift_sweep/run_drift_sweep.sh
set -euo pipefail

OUT="pim_predecoder/results/revision_n100k/drift_sweep"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

N_CW="${N_CW:-100000}"
CHUNK=2000
JOBS="${JOBS:-6}"
P_BIT="1e-3"
P_BURST="5e-3"

# label  test_p_drift
POINTS=(
  "d0     0"
  "d1e-4  1e-4"
  "d3e-4  3e-4"
  "d1e-3  1e-3"
  "d3e-3  3e-3"
  "d5e-3  5e-3"
  "d1e-2  1e-2"
  "d2e-2  2e-2"
)

run_one() {
  local name="$1" pdrift="$2" seed="$3"
  local out="$OUT/sweep_1024_${name}_s${seed}.json"
  local log="$LOGS/${name}_s${seed}.log"
  if [[ -f "$out" ]] && [[ "${FORCE:-0}" != 1 ]]; then
    echo "skip $name s$seed (exists; FORCE=1 to overwrite)"
    return 0
  fi
  python3 -m pim_predecoder.experiments.gemv \
    --matrix_info 1024 --matrix_rate 16_17 \
    --train_p_drift 5e-3 --train_p_row_burst 5e-3 \
    --test_p_drift "$pdrift" --test_p_row_burst "$P_BURST" \
    --p_bits "$P_BIT" \
    --n_codewords "$N_CW" --eval_chunk "$CHUNK" --n_cycles 4 \
    --train_steps 600 --train_batch 48 --hidden 24 --bp_iters 3 \
    --seed "$seed" --eval_seed "$seed" \
    --out_json "$out" --label "sweep_1024_${name}_s${seed}" \
    > "$log" 2>&1
  echo "done $name s$seed"
}
export -f run_one
export OUT LOGS N_CW CHUNK P_BIT P_BURST FORCE

for spec in "${POINTS[@]}"; do
  read -r name pdrift <<< "$spec"
  for seed in 1 2 3; do
    printf '%s\t%s\t%s\n' "$name" "$pdrift" "$seed"
  done
done | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _

echo "all drift-sweep points done; results in $OUT"
