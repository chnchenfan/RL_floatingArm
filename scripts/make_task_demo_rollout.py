#!/usr/bin/env python3
"""IK demo of the FULL task: base shakes (facing yaw + 1s jitter) while link7
tracks the world-frame shapes (circle/sine/triangle/windylab) around the body.
This is the MPC-equivalent reference (pinocchio IK in the base frame, exactly
like move_base_circle_mpc_ik_demo.py). Produces a rollout with base pose so RViz
shows the base shaking + the arm drawing.

    python3 scripts/make_task_demo_rollout.py
"""
import os
import sys

import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/windylab/code/windylab_ws/src/arm-platform/demo")
from rollout_io import write_rollout                 # noqa: E402
from task_spec.task_model import TaskModel            # noqa: E402
from pinocchio_ik import PinocchioIK                  # noqa: E402

OUT = "/home/windylab/code/isaac_arm_rl/data/rollout_task.csv"
RATE_HZ = 50.0


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tm = TaskModel()
    ik = PinocchioIK()

    dt = 1.0 / RATE_HZ
    T = int(tm.period * RATE_HZ)
    times = np.arange(T) * dt

    q = None
    qs, base_pos, base_quat, ee_w, tgt_w, draw = [], [], [], [], [], []
    perr, oerr = [], []
    for t in times:
        p_b, R_b = tm.ee_target_base(float(t))          # base-frame target (arm sees this)
        out = ik.solve(p_b, target_rotation=R_b, q_init=q, rot_weight=0.2)
        q = out[0] if isinstance(out, (tuple, list)) else out

        pin.forwardKinematics(ik.model, ik.data, q)
        pin.updateFramePlacement(ik.model, ik.data, ik.frame_id)
        oMf = ik.data.oMf[ik.frame_id]                  # EE in base frame
        # to world:
        bp, bR = tm.base_pose_world(float(t))
        ee_world = bp + bR @ oMf.translation
        pw, _ = tm.ee_target_world(float(t))

        qs.append(q.copy())
        base_pos.append(bp)
        base_quat.append(_R_to_xyzw(bR))
        ee_w.append(ee_world)
        tgt_w.append(pw)
        draw.append(tm.is_drawing(float(t)))
        perr.append(np.linalg.norm(oMf.translation - p_b) * 1000.0)
        Re = oMf.rotation.T @ R_b
        oerr.append(np.degrees(abs(np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1)))))

    qs = np.array(qs); perr = np.array(perr); oerr = np.array(oerr)
    write_rollout(OUT, times, qs, ee=np.array(ee_w), tgt=np.array(tgt_w),
                  base_pos=np.array(base_pos), base_quat=np.array(base_quat),
                  drawing=np.array(draw), source="task_ik_6d",
                  note="draw/turn phases, base 1s jitter + smooth facing turns")
    print(f"[ok] wrote {OUT}: {T} samples, period={tm.period}s")
    print(f"[ok] IK tracking (base frame): pos max={perr.max():.2f}mm "
          f"mean={perr.mean():.2f}mm | orient max={oerr.max():.1f}deg "
          f"mean={oerr.mean():.1f}deg")


def _R_to_xyzw(R):
    q = pin.Quaternion(R)
    return np.array([q.x, q.y, q.z, q.w])


if __name__ == "__main__":
    main()
