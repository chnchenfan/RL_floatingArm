#!/usr/bin/env bash
# Run the whole-task RL controller in Isaac and show it live with the SAME
# MPC trajectory visualizer used for the NMPC baseline.
#
# Usage:
#   bash ros_bridge/view_whole_live_rviz.sh                 # Phase-1 only
#   bash ros_bridge/view_whole_live_rviz.sh <phase2_ckpt>   # Phase-1+Phase-2
#
# Start order mirrors the MPC workflow: bridge/visualizer first, Isaac last,
# because the visualizer anchors t=0 to the first /student/joint_command.
set -eo pipefail

CHECKPOINT="${1:-}"
ROOT=/home/windylab/code/isaac_arm_rl
WS=/home/windylab/code/windylab_ws
URDF="$WS/src/arm-platform/config/arm.urdf"
RVIZ="$ROOT/ros_bridge/arm_rl.rviz"
UDP_PORT=45831

set +u
source /opt/ros/humble/setup.bash
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
set -u

# stale FastDDS shm segments break /joint_states delivery (see MPC handoff)
if ! pgrep -f ros2 > /dev/null 2>&1; then
  rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null || true
fi

echo "[whole-live] checkpoint=${CHECKPOINT:-phase1_only}"
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

ros2 run robot_state_publisher robot_state_publisher "$URDF" &

/usr/bin/python3 "$ROOT/ros_bridge/live_stream_node.py" --ros-args \
    -p udp_host:=127.0.0.1 -p udp_port:="$UDP_PORT" &

# the unchanged MPC visualizer: expected/actual/command-FK topics + KPIs
ros2 run manipulator trajectory_visualizer_demo.py --ros-args \
    -p trajectory_mode:=moving_base_circle \
    -p trajectory_shape:=whole \
    -p frame_id:=world \
    -p record_control_signal:=true \
    -p ignore_initial_sec:=3.0 \
    -p show_matplotlib:=false &

sleep 3

EXTRA_ARGS=()
if [ -n "$CHECKPOINT" ]; then
  EXTRA_ARGS+=(--checkpoint "$CHECKPOINT")
fi
OMNI_KIT_ACCEPT_EULA=YES "$ROOT/env_isaaclab/bin/python" \
    "$ROOT/scripts/live_phase2_whole_stream.py" \
    --host 127.0.0.1 --port "$UDP_PORT" "${EXTRA_ARGS[@]}" &

rviz2 -d "$RVIZ"
