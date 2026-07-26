#!/bin/bash
# Collect whole-task DART expert data: plant (sim) + MPC demo, headless.
# Runs: (sigma, seed) pairs passed as "sigma:seed" args.
set -o pipefail
set +u
source /opt/ros/humble/setup.bash
source ~/code/windylab_ws/install/setup.bash
source ~/code/windylab_ws/src/arm-platform/scripts/setup_acados_env.sh

LOG_DIR=/tmp/claude-1000/-home-windylab-code-isaac-arm-rl/b4daf91e-c531-4209-9e66-49675a4c6aac/scratchpad/collect_logs
DEMO_DIR=~/code/windylab_ws/src/arm-platform/demo
mkdir -p "$LOG_DIR"

echo "[collect] cleaning stale FastDDS shm"
rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null

echo "[collect] starting plant (sim, headless)"
ros2 launch manipulator student_arm.launch.py \
  arm_type:=sim use_rviz:=False velocity_feedforward_gain:=1.0 \
  > "$LOG_DIR/plant.log" 2>&1 &
PLANT_PID=$!

for i in $(seq 1 40); do
  if timeout 3 ros2 topic echo /joint_states --once > /dev/null 2>&1; then
    echo "[collect] /joint_states up after ${i} checks"
    break
  fi
  sleep 1
done
if ! timeout 3 ros2 topic echo /joint_states --once > /dev/null 2>&1; then
  echo "[collect] FATAL: plant did not come up"; kill -INT $PLANT_PID; exit 1
fi
sleep 2

for spec in "$@"; do
  SIGMA="${spec%%:*}"
  SEED="${spec##*:}"
  echo "[collect] === run sigma=$SIGMA seed=$SEED ==="
  (cd "$DEMO_DIR" && timeout --signal=INT --kill-after=15 118 \
    /usr/bin/python3 move_base_circle_mpc_ik_demo.py --ros-args \
      -p trajectory_shape:=whole \
      -p record_loop_csv:=true \
      -p dart_noise_std:="$SIGMA" \
      -p dart_noise_seed:="$SEED" \
      > "$LOG_DIR/mpc_s${SIGMA}_seed${SEED}.log" 2>&1)
  echo "[collect] run sigma=$SIGMA seed=$SEED exited rc=$?"
  sleep 4
done

echo "[collect] stopping plant"
kill -INT $PLANT_PID 2>/dev/null
sleep 3
kill -TERM $PLANT_PID 2>/dev/null
wait $PLANT_PID 2>/dev/null
pkill -INT -f smooth_position_controller 2>/dev/null
sleep 2
echo "[collect] done"
