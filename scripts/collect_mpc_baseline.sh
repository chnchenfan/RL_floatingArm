#!/usr/bin/env bash
# Collect CLEAN baseline MPC data: launch the sim arm + run the real acados NMPC
# (move_base_circle) with per-tick logging, baseline disturbance (xy circle
# r=0.02, 1 Hz, no z), for RUN_SEC seconds. Output: fresh moving_base_mpc_loop CSV.
set -o pipefail   # NOT -u: sourcing ROS setup.bash references unbound vars
WS=/home/windylab/code/windylab_ws
DEMO=$WS/src/arm-platform/demo
RUN_SEC="${1:-45}"

# baseline disturbance via env-var overrides (no config edit)
export MB_BASE_RADIUS=0.02 MB_BASE_PERIOD=1.0 MB_BASE_Z_AMP=0.0

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
source "$WS/src/arm-platform/scripts/setup_acados_env.sh" 2>/dev/null || true

cd "$DEMO"
echo "[collect] baseline r=0.02 T=1.0s(1Hz) z=0 ; run=${RUN_SEC}s"

# 1) sim arm (no rviz)
setsid ros2 launch manipulator student_arm.launch.py \
    use_rviz:=false velocity_feedforward_gain:=1.0 > /tmp/sim_arm.log 2>&1 &
SIM_PID=$!
echo "[collect] sim arm launching (pid $SIM_PID), waiting 12s..."
sleep 12

# 2) MPC demo with loop recording
before=$(ls -1 data/moving_base_mpc_loop_*.csv 2>/dev/null | wc -l)
setsid python3 move_base_circle_mpc_ik_demo.py --ros-args \
    -p record_loop_csv:=true > /tmp/mpc_demo.log 2>&1 &
MPC_PID=$!
echo "[collect] MPC running (pid $MPC_PID) for ${RUN_SEC}s..."
sleep "$RUN_SEC"

# 3) stop
kill -INT $MPC_PID 2>/dev/null; sleep 3
kill -9 $MPC_PID 2>/dev/null
kill -9 -- -$SIM_PID 2>/dev/null; pkill -9 -f student_arm 2>/dev/null
pkill -9 -f move_base_circle_mpc 2>/dev/null
sleep 2

newf=$(ls -1t data/moving_base_mpc_loop_*.csv 2>/dev/null | head -1)
echo "[collect] newest loop CSV: $newf"
if [ -n "$newf" ]; then
    echo "[collect] rows: $(($(wc -l < "$newf")-1))"
    python3 -c "
import csv
r=list(csv.DictReader(open('$newf')))
if r:
    pt=[float(x['path_time']) for x in r]
    er=[float(x['first_error_m'])*1000 for x in r if x['first_error_m']]
    st=[float(x['solve_time_ms']) for x in r if x['solve_time_ms']]
    import numpy as np
    print(f'[collect] path_time range [{min(pt):.2f},{max(pt):.2f}] (should sweep, not stuck at 0)')
    print(f'[collect] MPC track err mean={np.mean(er):.2f}mm max={np.max(er):.2f}mm')
    print(f'[collect] MPC solve_time mean={np.mean(st):.2f}ms p95={np.percentile(st,95):.2f}ms')
"
fi
echo "[collect] DONE"
