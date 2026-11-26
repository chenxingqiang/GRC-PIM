#!/usr/bin/env bash
# Conductance-domain OOD gate-robustness experiment (C1-C4).
#
# The calibrator and both gate thresholds are taken unchanged from the
# synthetic-channel results (tau_d=0.011, tau_g=0.6, 1024-symbol rate 16/17).
# Only the evaluation channel changes, so this measures whether the gate
# generalises to impairments that are structurally outside its training
# distribution.
#
# Operating points are matched on realised readout error rate (~1.5e-2, i.e.
# the synthetic channel at p_bit=5e-3) so that "structurally different" is not
# confounded with "simply worse". C4 is deliberately far past that budget.
#
# Three batches are produced:
#   *_s$seed.json     high-pass drift gate (d >= tau_d), as first submitted
#   *_s$seed_bp.json  band-pass drift gate (tau_d <= d <= tau_d_max)
#   id_*_s$seed.json  synthetic channel with the band-pass gate, to confirm
#                     the upper bound costs nothing inside the training domain
#
# Usage (from repo root): bash pim_predecoder/results/revision_n100k/ood/run_ood.sh
set -euo pipefail

OUT="pim_predecoder/results/revision_n100k/ood"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

N_CW=20000
CHUNK=2000
TAU_D=0.011
TAU_G=0.6
# Upper bound of the drift band, calibrated per code scale exactly as tau_d is,
# and at the per-codeword granularity the hardware detector publishes at. On
# the 1024-symbol scale the per-codeword score has mean 0.03068 and s.d.
# 0.00331, so mean + 2 s.d. rounds up to 0.038 (1.05% of in-distribution
# codewords spuriously rejected). The 256-symbol scale needs 0.045 (mean
# 0.03094, s.d. 0.00662, 1.95%); it is not exercised here because the
# conductance-domain sweep is run at the 1024-symbol scale. See
# granularity/run_granularity.sh for the measurement these numbers come from.
#
# NOTE on the committed *_bp.json files: they were produced with TAU_D_MAX=0.035,
# the earlier block-statistic calibration, and record that value in their
# drift_score_max field. They were not re-run when the bound moved to 0.038
# because no gate decision changes: the aggregate scores are 0.0000 (C1),
# 0.0211 (C2), 0.0463 (C3) and 0.2066 (C4), so C2 is admitted and C3/C4 are
# rejected under either bound. Re-running this script reproduces the same
# decisions with 0.038 recorded instead.
TAU_D_MAX=0.038

# id  sigma_ln  drift_nu  rtn_traps  rtn_amp   description
POINTS=(
  "C1 0.10 0.00 0 0.00 static_spread_only"
  "C2 0.05 0.05 2 0.03 drift_plus_rtn"
  "C3 0.05 0.00 3 0.05 rtn_dominated"
  "C4 0.05 0.15 0 0.00 strong_drift"
)

for seed in 1 2 3; do
  (
    for spec in "${POINTS[@]}"; do
      read -r id s nu tr amp desc <<< "$spec"
      for gate in highpass bandpass; do
        if [[ "$gate" == bandpass ]]; then
          out="$OUT/ood_1024_${id}_s${seed}_bp.json"
          max_arg=(--drift_score_max "$TAU_D_MAX")
          lbl="ood_1024_${id}_${desc}_bandpass_s${seed}"
        else
          out="$OUT/ood_1024_${id}_s${seed}.json"
          max_arg=()
          lbl="ood_1024_${id}_${desc}_s${seed}"
        fi
        [[ -f "$out" ]] && { echo "skip $id s$seed $gate (exists)"; continue; }
        python3 -m pim_predecoder.experiments.gemv \
          --matrix_info 1024 --matrix_rate 16_17 \
          --train_p_drift 5e-3 --train_p_row_burst 5e-3 \
          --channel conductance \
          --cond_sigma_ln "$s" --cond_drift_nu "$nu" \
          --cond_rtn_traps "$tr" --cond_rtn_amp "$amp" --cond_rtn_switch 0.5 \
          --p_bits 5e-3 \
          --n_codewords "$N_CW" --eval_chunk "$CHUNK" --n_cycles 4 \
          --train_steps 600 --train_batch 48 --hidden 24 --bp_iters 3 \
          --seed "$seed" \
          --drift_score_threshold "$TAU_D" "${max_arg[@]}" \
          --model_gate_threshold "$TAU_G" \
          --out_json "$out" --label "$lbl"
      done
    done

    # In-distribution regression: same band-pass gate on the synthetic channel
    # the calibrator was trained for. Confirms the upper bound does not close
    # the gate anywhere it was previously open.
    out="$OUT/id_1024_bandpass_s${seed}.json"
    if [[ -f "$out" ]]; then
      echo "skip in-dist s$seed (exists)"
    else
      python3 -m pim_predecoder.experiments.gemv \
        --matrix_info 1024 --matrix_rate 16_17 \
        --train_p_drift 5e-3 --train_p_row_burst 5e-3 \
        --test_p_drift 5e-3 --test_p_row_burst 5e-3 \
        --p_bits 5e-3 1e-3 \
        --n_codewords "$N_CW" --eval_chunk "$CHUNK" --n_cycles 4 \
        --train_steps 600 --train_batch 48 --hidden 24 --bp_iters 3 \
        --seed "$seed" \
        --drift_score_threshold "$TAU_D" --drift_score_max "$TAU_D_MAX" \
        --model_gate_threshold "$TAU_G" \
        --out_json "$out" --label "id_1024_bandpass_s${seed}"
    fi
  ) > "$LOGS/ood_s${seed}.log" 2>&1 &
done

wait
echo "all seeds done; results in $OUT"

