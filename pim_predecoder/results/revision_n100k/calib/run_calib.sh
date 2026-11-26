#!/usr/bin/env bash
# Held-out validation sweep used to calibrate the drift-gate lower bound tau_d.
#
# This is the sweep Sec. 3.5 of the manuscript describes: three channels
# (matched strong drift, weak drift, pure i.i.d.) x five p_bit levels x three
# model seeds = 45 operating points per code scale. The gate thresholds are
# deliberately NOT passed, so every point reports what the calibrator would do
# if admitted; the thresholds are then placed from the resulting scores rather
# than assumed. Evaluation seeds are 101-103, disjoint from the 1-3 used by
# every reported test evaluation, which is what makes the set held out.
#
# Budget: 100,000 codewords per point, streamed as 50 blocks of 2,000. The
# across-block spread of the drift score is what the 2-s.d. placement rule in
# Sec. 3.5 consumes, so the block count matters as much as the total: the
# earlier 20,000-codeword version of this sweep gave that s.d. from only 10
# blocks, and its 1.4e-7 resolution floor sat above the ~1e-7 error rates that
# fix the lower bound, leaving the binding constraint unresolved. At 100,000 the
# floor drops to 2.8e-8 (1024) / 1.0e-7 (256) and the constraint resolves the
# other way: no point at either scale degrades B', the three weak-drift points
# that looked 0.80-0.92x worse at 20,000 return 1.34-2.01x, and tau_d is left
# bound only by the 2-s.d. margin below the matched band (Sec. 3.5, Sec. 4.3).
#
# The model checkpoints are the ones trained for the main results and are
# loaded from _cache/; this script never retrains them. Run it from the repo
# root so that _cache/ and results/ resolve to the dev tree:
#   bash pim_predecoder/results/revision_n100k/calib/run_calib.sh
set -euo pipefail

OUT="pim_predecoder/results/revision_n100k/calib"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

N_CW="${N_CW:-100000}"
CHUNK=2000
JOBS="${JOBS:-6}"

# scale  matrix_info  rate   train_steps  train_batch
SCALES=(
  "1024 1024 16_17 600 48"
  "256  256  8_9   900 128"
)

# channel  test_p_drift  test_p_row_burst
CHANNELS=(
  "matched 5e-3  5e-3"
  "weak    5e-4  1e-3"
  "iid     0     0"
)

run_one() {
  local tag="$1" info="$2" rate="$3" steps="$4" batch="$5" \
        ch="$6" tdrift="$7" tburst="$8" seed="$9"
  local out log
  if [[ "$tag" == 1024 ]]; then
    out="$OUT/val_1024_${ch}_s${seed}.json"; log="$LOGS/${ch}_s${seed}.log"
  else
    out="$OUT/val_256_${ch}_s${seed}.json"; log="$LOGS/256_${ch}_s${seed}.log"
  fi
  if [[ -f "$out" ]] && [[ "${FORCE:-0}" != 1 ]]; then
    echo "skip $tag $ch s$seed (exists; FORCE=1 to overwrite)"
    return 0
  fi
  python3 -m pim_predecoder.experiments.gemv \
    --matrix_info "$info" --matrix_rate "$rate" \
    --train_p_drift 5e-3 --train_p_row_burst 5e-3 \
    --test_p_drift "$tdrift" --test_p_row_burst "$tburst" \
    --p_bits 5e-3 3e-3 1e-3 3e-4 1e-4 \
    --n_codewords "$N_CW" --eval_chunk "$CHUNK" --n_cycles 4 \
    --train_steps "$steps" --train_batch "$batch" --hidden 24 --bp_iters 3 \
    --seed "$seed" --eval_seed "$((100 + seed))" \
    --out_json "$out" --label "val_${tag}_${ch}_s${seed}" \
    > "$log" 2>&1
  echo "done $tag $ch s$seed"
}
export -f run_one
export OUT LOGS N_CW CHUNK

for spec in "${SCALES[@]}"; do
  read -r tag info rate steps batch <<< "$spec"
  for chspec in "${CHANNELS[@]}"; do
    read -r ch tdrift tburst <<< "$chspec"
    for seed in 1 2 3; do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$tag" "$info" "$rate" "$steps" "$batch" "$ch" "$tdrift" "$tburst" "$seed"
    done
  done
done | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _

echo "all points done; results in $OUT"
