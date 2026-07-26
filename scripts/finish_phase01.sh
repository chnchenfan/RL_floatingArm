#!/usr/bin/env bash
# Run AFTER a reboot (fixes the 595.71.05 -> 595.84 driver/module mismatch).
# Completes Phase 0 verify + Phase 1 (URDF import, Isaac FK, cross-check).
set -uo pipefail
PROJ=/home/windylab/code/isaac_arm_rl
cd "$PROJ"

echo "=== [0] driver sanity: nvidia-smi must succeed ==="
if ! nvidia-smi >/dev/null 2>&1; then
    echo "!! nvidia-smi still failing -> module/library still mismatched."
    echo "!! Reboot again (or reload nvidia kmods) before continuing."
    nvidia-smi 2>&1 | head -3
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

source env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES

echo "=== [1] Phase 1: URDF -> USD ==="
# NOTE: SimulationApp forces exit 0 on teardown, so rc is unreliable; gate on the
# actual output file instead.
python scripts/import_urdf.py
[ -f usd/arm.usd ] || { echo "!! import failed: usd/arm.usd not produced"; exit 2; }

echo "=== [2] Phase 1: Isaac FK for reference configs ==="
python scripts/fk_isaac.py || { echo "isaac fk failed"; exit 3; }

echo "=== [3] Phase 1: cross-check Isaac vs Pinocchio (py3.10 pinocchio) ==="
deactivate 2>/dev/null || true
/usr/bin/python3 scripts/compare_fk.py
RC=$?
echo "=== finish_phase01 done, compare rc=$RC ==="
exit $RC
