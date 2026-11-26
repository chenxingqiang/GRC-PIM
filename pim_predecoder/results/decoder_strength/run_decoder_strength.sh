#!/usr/bin/env bash
# Decoder-strength ablation: does the GEMV residual gain survive a stronger BP
# decoder?
#
# The deployed decoder runs 3 iterations. This sweep re-decodes the SAME
# channel realisations at 8 and 15 iterations, holding the calibrator, the
# gates and the codeword stream fixed, so the comparison against the deployed
# 3-iteration budget is paired and differs only in decoder strength. The
# deployed reference at each scale is bp_iters=3, run here at the same budget.
#
# Budget: 10^4 codewords per seed (the budget used for threshold calibration),
# not the 10^5 headline budget. At 15 iterations a 1024-symbol run costs
# ~0.25 s per codeword, so 10^5 would be ~7 h per seed. Results are reported
# as a robustness check, not as a replacement for the 10^5 headline numbers.
#
# The decoder's hardware budget is NOT rescaled here: the paper's 15 us
# reference stays tied to the 3-iteration deployment. Iteration counts are
# recorded per path so the actual (early-stopped) work is visible.
#
# Run from the repo root:
#   bash pim_predecoder/results/decoder_strength/run_decoder_strength.sh
set -euo pipefail

OUT="pim_predecoder/results/decoder_strength"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

N_CW="${N_CW:-10000}"
CHUNK=2000
JOBS="${JOBS:-10}"
FORCE="${FORCE:-0}"

run_one() {
  local scale="$1" iters="$2" seed="$3"
  local out="$OUT/E2_${scale}_s${seed}_b${iters}.json"
  local log="$LOGS/${scale}_s${seed}_b${iters}.log"
  if [[ -f "$out" && "$FORCE" != 1 ]]; then
    echo "skip ${scale} s${seed} b${iters} (exists)"
    return 0
  fi
  if [[ "$scale" == 1024 ]]; then
    python3 -m pim_predecoder.experiments.gemv \
      --matrix_info 1024 --matrix_rate 16_17 \
      --train_p_drift 5e-3 --train_p_row_burst 5e-3 \
      --test_p_drift 5e-3 --test_p_row_burst 5e-3 \
      --p_bits 5e-3 3e-3 1e-3 3e-4 1e-4 \
      --n_codewords "$N_CW" --eval_chunk "$CHUNK" --n_cycles 4 \
      --train_steps 600 --train_batch 48 --hidden 24 \
      --bp_iters "$iters" \
      --seed "$seed" --eval_seed "$seed" \
      --drift_score_threshold 0.011 --drift_score_max 0.038 \
      --model_gate_threshold 0.6 \
      --out_json "$out" --label "E2_1024_s${seed}_b${iters}" \
      > "$log" 2>&1
  else
    python3 -m pim_predecoder.experiments.gemv \
      --matrix_info 256 --matrix_rate 8_9 \
      --train_p_drift 5e-3 --train_p_row_burst 5e-3 \
      --test_p_drift 5e-3 --test_p_row_burst 5e-3 \
      --p_bits 5e-3 3e-3 1e-3 3e-4 1e-4 \
      --n_codewords "$N_CW" --eval_chunk "$CHUNK" --n_cycles 4 \
      --train_steps 900 --train_batch 128 --hidden 24 \
      --bp_iters "$iters" \
      --seed "$seed" --eval_seed "$seed" \
      --drift_score_threshold 0.005 --drift_score_max 0.045 \
      --model_gate_threshold 0.6 \
      --out_json "$out" --label "E2_256_s${seed}_b${iters}" \
      > "$log" 2>&1
  fi
  echo "done ${scale} s${seed} b${iters}"
}
export -f run_one
export OUT LOGS N_CW CHUNK FORCE

for scale in 1024 256; do
  for iters in 3 8 15; do
    for seed in 1 2 3; do
      printf '%s\t%s\t%s\n' "$scale" "$iters" "$seed"
    done
  done
done | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _

echo "all decoder-strength points done; results in $OUT"
