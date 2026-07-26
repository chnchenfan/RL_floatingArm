#!/usr/bin/env bash
# Phase 0 installer: Isaac Lab 2.3.2 + Isaac Sim 5.0 + torch cu128 (Blackwell/sm_120)
# Runs inside the uv-created Python 3.11 venv. Logs everything to logs/install.log.
set -uo pipefail

PROJ=/home/windylab/code/isaac_arm_rl
VENV="$PROJ/env_isaaclab"
source "$VENV/bin/activate"

echo "=== python: $(python --version) at $(which python) ==="
echo "=== proxy: http=$http_proxy https=$https_proxy ==="

# Robust pip network settings for the Clash proxy path.
export PIP_DEFAULT_TIMEOUT=120
PIP="python -m pip"

echo "=== [1/2] installing isaaclab[isaacsim,all]==2.3.2.post1 (pulls Isaac Sim 5.0) ==="
$PIP install --retries 5 --timeout 120 \
    "isaaclab[isaacsim,all]==2.3.2.post1" \
    --extra-index-url https://pypi.nvidia.com
RC1=$?
echo "=== [1/2] exit code: $RC1 ==="

echo "=== [2/2] overriding torch -> 2.7.0 cu128 (Blackwell support) ==="
$PIP install --retries 5 --timeout 120 -U \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
RC2=$?
echo "=== [2/2] exit code: $RC2 ==="

echo "=== pip freeze snapshot (isaac/torch) ==="
python -m pip freeze | grep -iE "isaac|torch|omni|warp|newton" || true

echo "=== DONE rc1=$RC1 rc2=$RC2 ==="
