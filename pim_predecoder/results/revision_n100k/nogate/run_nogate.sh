#!/usr/bin/env bash
# E5 without gates: an i.i.d.-trained calibrator forced onto strong-drift data.
#
# This is the experiment that establishes what the gates are worth, so it has to
# be run at the same budget as every other reported operating point. The earlier
# version of this table was a single seed at 2,000 codewords, and at that budget
# the headline "3% degradation" rested on 46 excess symbol errors out of 1,734 --
# about 1.1 sigma, i.e. not resolved. It also ran at the 256-symbol scale while
# being reported as 1024-symbol, which inflated the quoted error counts by N
# 1088/288 = 3.78x. Both scales are therefore run here, at 100,000 codewords per
# seed over three seeds.
#
# No gate thresholds are passed, so GEMV_post is the ungated calibrator output
# and use_gemv / GEMV_gate record what the gate would have decided. That is what
# makes a single run report both the un-gated damage and the gate's verdict.
#
# Run from the repo root:
#   bash pim_predecoder/results/revision_n100k/nogate/run_nogate.sh
set -euo pipefail

OUT="pim_predecoder/results/revision_n100k/nogate"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

N_CW="${N_CW:-100000}"
CHUNK=2000
JOBS="${JOBS:-6}"

# tag  matrix_info  rate   train_steps  train_batch
SCALES=(
  "1024 1024 16_17 600 48"
  "256  256  8_9   900 128"
)

run_one() {
  local tag="$1" info="$2" rate="$3" steps="$4" batch="$5" seed="$6"
  local out="$OUT/nogate_${tag}_s${seed}.json"
  local log="$LOGS/${tag}_s${seed}.log"
  if [[ -f "$out" ]] && [[ "${FORCE:-0}" != 1 ]]; then
    echo "skip $tag s$seed (exists; FORCE=1 to overwrite)"
    return 0
  fi
  # train on pure i.i.d. noise, test on strong matched drift
  python3 -m pim_predecoder.experiments.gemv \
    --matrix_info "$info" --matrix_rate "$rate" \
    --train_p_drift 0 --train_p_row_burst 0 \
    --test_p_drift 5e-3 --test_p_row_burst 5e-3 \
    --p_bits 5e-3 3e-3 1e-3 3e-4 1e-4 \
    --n_codewords "$N_CW" --eval_chunk "$CHUNK" --n_cycles 4 \
    --train_steps "$steps" --train_batch "$batch" --hidden 24 --bp_iters 3 \
    --seed "$seed" --eval_seed "$seed" \
    --out_json "$out" --label "nogate_${tag}_s${seed}" \
    > "$log" 2>&1
  echo "done $tag s$seed"
}
export -f run_one
export OUT LOGS N_CW CHUNK FORCE

for spec in "${SCALES[@]}"; do
  read -r tag info rate steps batch <<< "$spec"
  for seed in 1 2 3; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$info" "$rate" "$steps" "$batch" "$seed"
  done
done | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _

echo "all nogate points done; results in $OUT"
