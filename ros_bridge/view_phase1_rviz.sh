#!/usr/bin/env bash
# Replay one selected Phase-1 rollout in RViz.
# Usage: bash ros_bridge/view_phase1_rviz.sh [easy|medium|hard] [rate_hz]
set -euo pipefail

CONDITION="${1:-medium}"
RATE="${2:-50.0}"
ROOT=/home/windylab/code/isaac_arm_rl

case "$CONDITION" in
  easy|medium|hard) ;;
  *)
    echo "condition must be easy, medium, or hard" >&2
    exit 2
    ;;
esac

ROLLOUT="$ROOT/data/rollout_phase1_${CONDITION}.csv"
if [ ! -f "$ROLLOUT" ]; then
  echo "missing $ROLLOUT" >&2
  echo "generate it with: source env_isaaclab/bin/activate && python3 scripts/export_phase1_rviz_rollouts.py" >&2
  exit 1
fi

exec bash "$ROOT/ros_bridge/view_rollout_rviz.sh" "$ROLLOUT" "$RATE"
