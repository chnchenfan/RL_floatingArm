#!/usr/bin/env bash
# Run the selected Phase-1 policy online in Isaac and display live state in RViz.
# Usage: bash ros_bridge/view_phase1_live_rviz.sh [easy|medium|hard] [rate_hz]
# ROS setup files read a few optional variables before defining them, so defer
# nounset until after both setup files have been sourced.
set -eo pipefail

CONDITION="${1:-medium}"
RATE="${2:-50.0}"
ROOT=/home/windylab/code/isaac_arm_rl
WS=/home/windylab/code/windylab_ws
URDF="$WS/src/arm-platform/config/arm.urdf"
RVIZ="$ROOT/ros_bridge/arm_rl.rviz"
UDP_PORT=45831

case "$CONDITION" in
  easy|medium|hard) ;;
  *)
    echo "condition must be easy, medium, or hard" >&2
    exit 2
    ;;
esac

source /opt/ros/humble/setup.bash
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
set -u

echo "[view-live] Phase-1 online inference, condition=$CONDITION rate=${RATE}Hz"
echo "[view-live] RViz may appear before Isaac finishes its initial GPU startup."
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

ros2 run robot_state_publisher robot_state_publisher "$URDF" &

python3 "$ROOT/ros_bridge/live_stream_node.py" --ros-args \
    -p udp_host:=127.0.0.1 -p udp_port:="$UDP_PORT" &

OMNI_KIT_ACCEPT_EULA=YES "$ROOT/env_isaaclab/bin/python" \
    "$ROOT/scripts/live_phase1_stream.py" \
    --condition "$CONDITION" --host 127.0.0.1 --port "$UDP_PORT" \
    --rate-hz "$RATE" --physics-substeps 2 --sim-device cuda:0 &

rviz2 -d "$RVIZ"
