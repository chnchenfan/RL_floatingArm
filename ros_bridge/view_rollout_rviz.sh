#!/usr/bin/env bash
# View a rollout in RViz: robot_state_publisher(arm.urdf) + rviz2 + replay node.
# Does NOT run the arm controller, so it won't fight the replay on /joint_states.
#
#   bash ros_bridge/view_rollout_rviz.sh [rollout.csv] [rate_hz]
#
# Optional 3rd terminal for the expected-vs-actual tracking plot:
#   ros2 run manipulator trajectory_visualizer_demo.py --ros-args \
#     -p trajectory_mode:=circle -p frame_id:=base_link
# ROS setup files are not nounset-safe; enable it after sourcing them.
set -o pipefail

ROLLOUT="${1:-/home/windylab/code/isaac_arm_rl/data/rollout_synth.csv}"
RATE="${2:-50.0}"
WS=/home/windylab/code/windylab_ws
URDF="$WS/src/arm-platform/config/arm.urdf"
BRIDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
RVIZ="$BRIDGE_DIR/arm_rl.rviz"

source /opt/ros/humble/setup.bash
# source the built workspace so package://dummy_description meshes resolve in RViz
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
set -u

echo "[view] rollout=$ROLLOUT rate=${RATE}Hz"
trap 'kill $(jobs -p) 2>/dev/null' EXIT

# Pass the URDF path through robot_state_publisher's file fallback. Inlining
# this multi-line XML as a ROS CLI parameter fails argument parsing.
ros2 run robot_state_publisher robot_state_publisher "$URDF" &

python3 "$BRIDGE_DIR/replay_rollout_node.py" --ros-args \
    -p rollout_file:="$ROLLOUT" -p publish_rate_hz:="$RATE" -p loop:=true &

rviz2 -d "$RVIZ"   # foreground; closing RViz stops everything via the trap
