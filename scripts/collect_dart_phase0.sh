#!/usr/bin/env bash
# Phase 0 DART collection matrix.  Defaults to the conservative pilot sigma;
# override DART_LEVELS/RUNS_PER_CELL/RUN for a denser recovery dataset.
set -o pipefail

ROOT=/home/windylab/code/isaac_arm_rl
DART_LEVELS="${DART_LEVELS:-0.001 0.002}"
RUNS_PER_CELL="${RUNS_PER_CELL:-1}"
RUN="${RUN:-38}"
seed="${DART_SEED_BASE:-1000}"

run_cell() {
    local radius="$1"
    local period="$2"
    local z_amp="$3"
    local sigma="$4"
    local repeat
    for repeat in $(seq 1 "$RUNS_PER_CELL"); do
        echo "[phase0] R=$radius T=$period Z=$z_amp sigma=$sigma repeat=$repeat seed=$seed"
        MB_R="$radius" MB_T="$period" MB_Z="$z_amp" \
            DART_SIGMA="$sigma" DART_SEED="$seed" RUN="$RUN" \
            bash "$ROOT/scripts/collect_mpc_v3.sh"
        seed=$((seed + 1))
    done
}

for sigma in $DART_LEVELS; do
    run_cell 0.02 1.0 0.0 "$sigma"
    run_cell 0.02 1.0 0.02 "$sigma"
    run_cell 0.01 0.5 0.01 "$sigma"
done

echo "[phase0] DONE"
