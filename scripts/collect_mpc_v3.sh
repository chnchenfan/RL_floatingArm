#!/usr/bin/env bash
# Collect one clean or DART-perturbed MPC rollout with auditable action logs.
set -o pipefail

WS=/home/windylab/code/windylab_ws
DEMO=$WS/src/arm-platform/demo
export MB_BASE_RADIUS="${MB_R:-0.02}"
export MB_BASE_PERIOD="${MB_T:-1.0}"
export MB_BASE_Z_AMP="${MB_Z:-0.0}"
export MB_DART_NOISE="${DART_SIGMA:-${MB_DART_NOISE:-0}}"
export MB_DART_CLIP_SIGMA="${DART_CLIP_SIGMA:-${MB_DART_CLIP_SIGMA:-2.5}}"
export MB_DART_SEED="${DART_SEED:-${MB_DART_SEED:-0}}"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
source "$WS/scripts/setup_acados_env.sh" 2>/dev/null \
    || source "$WS/src/arm-platform/scripts/setup_acados_env.sh" 2>/dev/null \
    || true
cd "$DEMO"

START_MARK=$(mktemp)
SIM=""
VIZ=""
MPC=""

stop_group() {
    local pid="$1"
    [ -z "$pid" ] && return
    kill -INT -- "-$pid" 2>/dev/null || true
}

cleanup() {
    stop_group "$MPC"
    stop_group "$VIZ"
    stop_group "$SIM"
    sleep 2
    for pid in "$MPC" "$VIZ" "$SIM"; do
        [ -z "$pid" ] && continue
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in "$MPC" "$VIZ" "$SIM"; do
        [ -z "$pid" ] && continue
        kill -KILL -- "-$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

echo "[c] config R=$MB_BASE_RADIUS T=$MB_BASE_PERIOD Z=$MB_BASE_Z_AMP DART_sigma=$MB_DART_NOISE seed=$MB_DART_SEED"
setsid ros2 launch manipulator student_arm.launch.py \
    use_rviz:=false velocity_feedforward_gain:=1.0 >/tmp/arm.log 2>&1 &
SIM=$!
sleep 15
setsid ros2 run manipulator trajectory_visualizer_demo.py --ros-args \
    -p trajectory_mode:=moving_base_circle \
    -p frame_id:=world \
    -p record_control_signal:=true \
    -p ignore_initial_sec:=3.0 >/tmp/viz.log 2>&1 &
VIZ=$!
sleep 3
setsid python3 move_base_circle_mpc_ik_demo.py --ros-args \
    -p record_loop_csv:=true \
    -p dart_noise_std:="$MB_DART_NOISE" \
    -p dart_noise_clip_sigma:="$MB_DART_CLIP_SIGMA" \
    -p dart_noise_seed:="$MB_DART_SEED" >/tmp/mpc.log 2>&1 &
MPC=$!

echo "[c] tracking ${RUN:-40}s..."
sleep "${RUN:-40}"
cleanup
trap - EXIT INT TERM

LC=$(find data -maxdepth 1 -type f -name 'moving_base_mpc_loop_*.csv' \
    -newer "$START_MARK" -print | sort | tail -1)
rm -f "$START_MARK"
echo "[c] loop CSV: $LC"
if [ -z "$LC" ]; then
    echo "[c] ERROR: no fresh loop CSV produced"
    exit 1
fi

python3 - "$LC" <<'PY'
import csv
import sys
import numpy as np

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
if not rows:
    raise SystemExit("[c] ERROR: fresh loop CSV is empty")

vec = lambda prefix: np.array([
    [float(row[f"{prefix}_{i}"]) for i in range(1, 8)] for row in rows
])
q = vec("q_meas")
q_next = vec("q_next")
q_expert = vec("q_expert")
q_pub = vec("q_pub")
u0 = vec("u0")
u_expert = vec("u_expert")
u_exec = vec("u_exec")
noise = vec("dart_noise_applied")
err = np.array([
    float(row["first_error_m"]) * 1000 for row in rows
    if row["first_error_m"] not in ("", "nan")
])
ptt = np.array([
    float(row["path_target_time"]) for row in rows
    if row.get("path_target_time", "") not in ("", "nan")
])
clipped = np.array([int(row["dart_noise_clipped"]) for row in rows])

assert np.max(np.abs((q_next - q) - u0)) < 1e-9
assert np.max(np.abs((q_expert - q) - u_expert)) < 1e-9
assert np.max(np.abs((q_pub - q) - u_exec)) < 1e-9
assert np.max(np.abs((q_expert + noise) - q_pub)) < 1e-9
assert np.max(np.abs(u_exec)) <= 0.020000001

print("[c] rows=%d q_meas_span(rad) max=%.3f" %
      (len(rows), (q.max(0) - q.min(0)).max()))
print("[c] raw |u0| mean=%.4f safe expert |u_expert| mean=%.4f actual |u_exec| mean=%.4f" %
      (np.abs(u0).mean(), np.abs(u_expert).mean(), np.abs(u_exec).mean()))
print("[c] DART applied std=%.6frad nonzero=%.1f%% clipped=%.1f%%" %
      (noise.std(), 100.0 * np.mean(np.abs(noise) > 1e-12),
       100.0 * clipped.mean()))
print("[c] path_target_time[%.1f,%.1f] track_err mean=%.2fmm p95=%.2fmm max=%.2fmm" %
      (ptt.min(), ptt.max(), err.mean(), np.percentile(err, 95), err.max()))
PY
