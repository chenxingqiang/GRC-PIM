#!/usr/bin/env bash
# Cross-validation of the band-pass reliability gate against a third-party,
# device-calibrated read channel.
#
# The C1-C4 sweep in ../run_ood.sh uses our own physics-inspired conductance
# channel, so it cannot by itself rule out that the gate thresholds are an
# artefact of that implementation. This script repeats the experiment with the
# noise physics replaced by aihwkit's ReRamCMONoiseModel, whose coefficients
# are fitted to measured analog filamentary CMO/HfOx ReRAM from IBM Research
# Europe - Zurich. The GF(3) level placement, the ADC decision rule, the
# multi-cycle readout, the calibrator checkpoint and both gate thresholds are
# unchanged, so the only factor that varies is the authorship of the channel.
#
# Operating points are a regular grid in the two quantities that set the
# signal-to-noise ratio of a read:
#   * level_pitch_us - the conductance spacing between adjacent GF(3) levels,
#     i.e. the ADC margin the device noise has to cross. This is a macro design
#     choice, not a device property.
#   * read_noise_scale - a scale on the device model's own 1/f read-noise term.
#     1 is the measured device; larger values model a cell with stronger read
#     noise and are what moves the drift score across the gate band at a fixed
#     ADC margin.
# The grid is regular and reported in full, so no operating point is selected
# after the fact. Two extra points cover the regimes the grid cannot reach:
# a static-only readout (read noise switched off, the analogue of C1) and a
# heavily aged array (t = 1e8 s, the analogue of C4).
#
# Each point is run twice:
#   agrid_*_s$seed_ug.json  ungated, so the intrinsic gain of the calibrator is
#                           visible whether or not the gate would admit it
#   agrid_*_s$seed_bp.json  band-pass gate (tau_d <= d <= tau_d_max, g >= tau_g)
# Comparing the two shows whether each gate decision was the correct one.
#
# Requires the aihwkit package, which has no wheel for the repository's Python
# 3.14; set AIHW_PY to an interpreter that has it installed (see README).
#
# Usage (from repo root):
#   bash pim_predecoder/results/revision_n100k/ood/aihwkit/run_aihwkit.sh
set -euo pipefail

OUT="pim_predecoder/results/revision_n100k/ood/aihwkit"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

PY="${AIHW_PY:-.venv-aihwkit/bin/python}"
"$PY" -c "import aihwkit; print('aihwkit', aihwkit.__version__)" >/dev/null

N_CW=20000
CHUNK=2000
TAU_D=0.011
TAU_D_MAX=0.038
TAU_G=0.6
JOBS="${JOBS:-6}"

# id  t_inference  level_pitch_us  read_noise_scale
POINTS=(
  "p5r1 86400 5.0 1.0"
  "p5r2 86400 5.0 2.0"
  "p5r4 86400 5.0 4.0"
  "p6r1 86400 6.0 1.0"
  "p6r2 86400 6.0 2.0"
  "p6r4 86400 6.0 4.0"
  "p7r1 86400 7.0 1.0"
  "p7r2 86400 7.0 2.0"
  "p7r4 86400 7.0 4.0"
  "p8r1 86400 8.0 1.0"
  "p8r2 86400 8.0 2.0"
  "p8r4 86400 8.0 4.0"
  "static 1 3.0 0.0"
  "aged 100000000 4.0 1.0"
)

run_one() {
  local id="$1" t="$2" pitch="$3" rns="$4" seed="$5" gate="$6"
  local out lbl
  if [[ "$gate" == bandpass ]]; then
    out="$OUT/agrid_1024_${id}_s${seed}_bp.json"
    lbl="agrid_1024_${id}_bandpass_s${seed}"
    gate_arg=(--drift_score_threshold "$TAU_D"
              --drift_score_max "$TAU_D_MAX"
              --model_gate_threshold "$TAU_G")
  else
    out="$OUT/agrid_1024_${id}_s${seed}_ug.json"
    lbl="agrid_1024_${id}_ungated_s${seed}"
    gate_arg=()
  fi
  if [[ -f "$out" ]]; then
    echo "skip $id s$seed $gate (exists)"
    return 0
  fi
  "$PY" -m pim_predecoder.experiments.gemv \
    --matrix_info 1024 --matrix_rate 16_17 \
    --train_p_drift 5e-3 --train_p_row_burst 5e-3 \
    --channel aihwkit \
    --aihw_t_inference "$t" --aihw_level_pitch_us "$pitch" \
    --aihw_read_noise_scale "$rns" --aihw_acceptance_range 2.0 \
    --p_bits 5e-3 \
    --n_codewords "$N_CW" --eval_chunk "$CHUNK" --n_cycles 4 \
    --train_steps 600 --train_batch 48 --hidden 24 --bp_iters 3 \
    --seed "$seed" "${gate_arg[@]}" \
    --out_json "$out" --label "$lbl" \
    > "$LOGS/${id}_s${seed}_${gate}.log" 2>&1
  echo "done $id s$seed $gate"
}
export -f run_one
export OUT LOGS PY N_CW CHUNK TAU_D TAU_D_MAX TAU_G

for seed in 1 2 3; do
  for spec in "${POINTS[@]}"; do
    read -r id t pitch rns <<< "$spec"
    for gate in ungated bandpass; do
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$t" "$pitch" "$rns" "$seed" "$gate"
    done
  done
done | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _

echo "all points done; results in $OUT"
