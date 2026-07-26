#!/usr/bin/env python3
"""Generate a synthetic rollout (pinocchio CLIK tracking an EE circle) to smoke-
test the Isaac->RViz replay bridge before any RL policy exists.

Runs on the ROS-side interpreter (py3.10 + pinocchio):
    python3 scripts/make_synthetic_rollout.py

Writes data/rollout_synth.csv in the shared rollout format. Replay it with
ros_bridge/replay_rollout_node.py and view in RViz / trajectory_visualizer.
"""
import os
import sys

import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rollout_io import write_rollout  # noqa: E402

ARM_URDF = "/home/windylab/code/windylab_ws/src/arm-platform/config/arm.urdf"
EE_FRAME = "link7"
OUT = "/home/windylab/code/isaac_arm_rl/data/rollout_synth.csv"

# EE circle in base frame (y-z plane), reachable for this arm (~0.55m reach).
CENTER = np.array([0.35, 0.0, 0.30])
RADIUS = 0.08
PERIOD = 4.0
N_PERIODS = 2.0
RATE_HZ = 50.0


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    model = pin.buildModelFromUrdf(ARM_URDF)
    data = model.createData()
    fid = model.getFrameId(EE_FRAME)

    dt = 1.0 / RATE_HZ
    T = int(N_PERIODS * PERIOD * RATE_HZ)
    times = np.arange(T) * dt

    q = pin.neutral(model)
    lo = np.where(np.isfinite(model.lowerPositionLimit),
                  model.lowerPositionLimit, -np.pi)
    hi = np.where(np.isfinite(model.upperPositionLimit),
                  model.upperPositionLimit, np.pi)

    qs, ee_ach, ee_tgt = [], [], []
    for t in times:
        ang = 2.0 * np.pi * t / PERIOD
        target = CENTER + RADIUS * np.array([0.0, np.cos(ang), np.sin(ang)])
        # position-only CLIK (redundant 7DOF), a few Gauss-Newton steps/tick
        for _ in range(20):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacement(model, data, fid)
            oMf = data.oMf[fid]
            err = target - oMf.translation
            if np.linalg.norm(err) < 1e-5:
                break
            J = pin.computeFrameJacobian(model, data, q, fid,
                                         pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), err)
            q = pin.integrate(model, q, dq)
            q = np.clip(q, lo, hi)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, fid)
        qs.append(q.copy())
        ee_ach.append(data.oMf[fid].translation.copy())
        ee_tgt.append(target.copy())

    qs = np.array(qs)
    ee_ach = np.array(ee_ach)
    err_mm = np.linalg.norm(ee_ach - np.array(ee_tgt), axis=1) * 1000.0
    write_rollout(OUT, times, qs, ee=ee_ach, tgt=np.array(ee_tgt),
                  source="synthetic_clik",
                  note=f"EE circle r={RADIUS} T={PERIOD}s, tracking err "
                       f"max={err_mm.max():.2f}mm mean={err_mm.mean():.2f}mm")
    print(f"[ok] wrote {OUT}: {T} samples @ {RATE_HZ}Hz")
    print(f"[ok] CLIK tracking error: max={err_mm.max():.3f}mm "
          f"mean={err_mm.mean():.3f}mm (positive control for the bridge)")
    print(f"[ok] q range per joint (deg):")
    print("     lo=", np.round(np.rad2deg(qs.min(0)), 1))
    print("     hi=", np.round(np.rad2deg(qs.max(0)), 1))


if __name__ == "__main__":
    main()
