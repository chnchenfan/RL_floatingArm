#!/usr/bin/env python3
"""Track the full shape schedule (circle/sine/triangle/windylab around the base)
with pinocchio 6D IK (position + 90-deg orientation lock) to (a) produce a
rollout for RViz and (b) report per-segment tracking error == REACHABILITY.

Base is fixed here (world==base) so residual error reveals what the fixed-base
arm can/can't reach. Large residual on a segment => out of workspace.

    python3 scripts/make_shape_rollout.py [--radius 0.30] [--height 0.30]
"""
import argparse
import os
import sys

import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rollout_io import write_rollout                      # noqa: E402
from task_spec.shapes import default_schedule, ORIENTATION_TARGET  # noqa: E402

ARM_URDF = "/home/windylab/code/windylab_ws/src/arm-platform/config/arm.urdf"
EE_FRAME = "link7"
OUT = "/home/windylab/code/isaac_arm_rl/data/rollout_shapes.csv"
RATE_HZ = 50.0


def ik_step(model, data, fid, q, oMdes, lo, hi, iters=60, damp=1e-4):
    for _ in range(iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, fid)
        iMd = data.oMf[fid].actInv(oMdes)
        err = pin.log6(iMd).vector            # 6D error in EE-local frame
        if np.linalg.norm(err) < 1e-6:
            break
        J = pin.computeFrameJacobian(model, data, q, fid)      # LOCAL
        J = -pin.Jlog6(iMd.inverse()) @ J
        v = -J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
        q = pin.integrate(model, q, v)
        q = np.clip(q, lo, hi)
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.30)
    ap.add_argument("--height", type=float, default=0.30)
    ap.add_argument("--seg-dur", type=float, default=5.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    model = pin.buildModelFromUrdf(ARM_URDF)
    data = model.createData()
    fid = model.getFrameId(EE_FRAME)
    lo = np.where(np.isfinite(model.lowerPositionLimit),
                  model.lowerPositionLimit, -np.pi)
    hi = np.where(np.isfinite(model.upperPositionLimit),
                  model.upperPositionLimit, np.pi)

    sch = default_schedule(radius=args.radius, height=args.height,
                           seg_dur=args.seg_dur)
    dt = 1.0 / RATE_HZ
    T = int(sch.period * RATE_HZ)
    times = np.arange(T) * dt

    q = pin.neutral(model)
    qs, ee_ach, ee_tgt = [], [], []
    perr, oerr, seg_of = [], [], []
    for t in times:
        p_tgt, R_tgt = sch.world_target(t)
        oMdes = pin.SE3(np.asarray(R_tgt), np.asarray(p_tgt))
        q = ik_step(model, data, fid, q, oMdes, lo, hi)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, fid)
        oMf = data.oMf[fid]
        qs.append(q.copy())
        ee_ach.append(oMf.translation.copy())
        ee_tgt.append(p_tgt)
        perr.append(np.linalg.norm(oMf.translation - p_tgt) * 1000.0)   # mm
        Rerr = oMf.rotation.T @ np.asarray(R_tgt)
        ang = np.degrees(abs(np.arccos(np.clip((np.trace(Rerr) - 1) / 2, -1, 1))))
        oerr.append(ang)
        # which segment
        tt = t % sch.period
        seg_of.append(int(np.searchsorted(sch.starts, tt, side="right") - 1))

    qs = np.array(qs); perr = np.array(perr); oerr = np.array(oerr)
    seg_of = np.array(seg_of)
    write_rollout(OUT, times, qs, ee=np.array(ee_ach), tgt=np.array(ee_tgt),
                  source="shape_ik_6d",
                  note=f"schedule r={args.radius} h={args.height}")
    print(f"[ok] wrote {OUT}: {T} samples, period={sch.period}s")
    print(f"\nREACHABILITY per segment (pos err mm / orient err deg):")
    print(f"{'segment':>18} {'pos_max':>9} {'pos_mean':>9} {'ori_max':>9} {'ori_mean':>9}  verdict")
    for i, seg in enumerate(sch.segments):
        m = seg_of == i
        pm, pa = perr[m].max(), perr[m].mean()
        om, oa = oerr[m].max(), oerr[m].mean()
        ok = pm < 5.0 and om < 5.0
        v = "REACHABLE" if ok else ("pos OOR" if pm >= 5 else "ori OOR")
        print(f"{seg['shape']+'@'+str(seg['azimuth']):>18} "
              f"{pm:9.2f} {pa:9.2f} {om:9.2f} {oa:9.2f}  {v}")
    print(f"\noverall pos: max={perr.max():.1f}mm mean={perr.mean():.1f}mm | "
          f"orient: max={oerr.max():.1f}deg mean={oerr.mean():.1f}deg")


if __name__ == "__main__":
    main()
