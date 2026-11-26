#!/usr/bin/env bash
# Cross evaluation of model seed and data-stream seed (Supplementary Table
# tab:crossed_seed). For each code scale, every cached model (trained with
# seed m, the runs behind Tables 1 and 2 of the main text) is re-evaluated on
# the evaluation data streams of the other two seeds (--eval_seed e), at the
# same 1e5-codeword budget and recalibrated thresholds as the matched-seed
# runs. The diagonal (m == e) runs are the matched-seed results themselves
# (results/revision_n100k/recal/) and are not re-run here.
#
# Because the evaluation stream is seeded independently of the model, the B'
# columns of run (m, e) are bit-identical to the matched-seed run of stream e,
# so each column shares one baseline and the table separates training
# variance (rows) from channel-stream variance (columns).
#
# Usage (from repo root):
#   bash pim_predecoder/results/revision_n100k/crossed/run_crossed.sh
set -euo pipefail

OUT="pim_predecoder/results/revision_n100k/crossed"
mkdir -p "$OUT"

run_scale () {
  local matrix_info=$1 matrix_rate=$2 tau_d=$3 steps=$4 batch=$5
  for m in 1 2 3; do
    for e in 1 2 3; do
      [[ "$m" == "$e" ]] && continue
      local out="$OUT/E2_${matrix_info}_m${m}_e${e}_n100k_crossed.json"
      if [[ -f "$out" ]]; then
        echo "skip ${matrix_info} m$m e$e (exists)"
        continue
      fi
      python3 -m pim_predecoder.experiments.gemv \
        --matrix_info "$matrix_info" --matrix_rate "$matrix_rate" \
        --train_p_drift 5e-3 --train_p_row_burst 5e-3 \
        --test_p_drift 5e-3 --test_p_row_burst 5e-3 \
        --p_bits 5e-3 3e-3 1e-3 3e-4 1e-4 \
        --n_codewords 100000 --eval_chunk 2000 --n_cycles 4 \
        --train_steps "$steps" --train_batch "$batch" --hidden 24 --bp_iters 3 \
        --seed "$m" --eval_seed "$e" \
        --drift_score_threshold "$tau_d" --model_gate_threshold 0.6 \
        --out_json "$out" --label "crossed_${matrix_info}_m${m}_e${e}" \
        > "$OUT/run_${matrix_info}_m${m}_e${e}.log" 2>&1 &
    done
  done
  wait
}

run_scale 1024 16_17 0.011 600 48
run_scale 256 8_9 0.005 900 128
echo "done; results in $OUT"
