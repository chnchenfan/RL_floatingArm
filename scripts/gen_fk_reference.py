#!/usr/bin/env python3
"""Phase 1 validation, step A: generate FK reference with Pinocchio (system py3.10).

Produces a deterministic set of joint configurations and the corresponding
world/base-frame pose of the `link7` end-effector frame (the same frame the
existing pinocchio IK uses, EE_FRAME='link7'). Isaac Sim will be fed the SAME
configs and must reproduce these poses to mm/deg tolerance.

Run with the ROS-side interpreter that has pinocchio 3.x:
    python3 scripts/gen_fk_reference.py
"""
import json
import os

import numpy as np
import pinocchio as pin

ARM_URDF = "/home/windylab/code/windylab_ws/src/arm-platform/config/arm.urdf"
EE_FRAME = "link7"           # matches src/arm-platform/demo/pinocchio_ik.py
OUT_DIR = "/home/windylab/code/isaac_arm_rl/assets"
N_RANDOM = 20
SEED = 12345


def quat_xyzw(R: np.ndarray) -> list:
    q = pin.Quaternion(R)            # normalized
    return [float(q.x), float(q.y), float(q.z), float(q.w)]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    model = pin.buildModelFromUrdf(ARM_URDF)   # fixed base, nq=7
    data = model.createData()
    fid = model.getFrameId(EE_FRAME)
    assert model.existFrame(EE_FRAME), f"frame {EE_FRAME} missing"

    lo = model.lowerPositionLimit.copy()
    hi = model.upperPositionLimit.copy()
    # Guard against +/-inf limits (continuous joints) -> clamp to +/-pi.
    lo = np.where(np.isfinite(lo), lo, -np.pi)
    hi = np.where(np.isfinite(hi), hi, np.pi)

    rng = np.random.default_rng(SEED)
    configs = [np.zeros(model.nq)]                       # home config first
    # a few axis-isolating configs: rotate one joint at a time to +30 deg
    for j in range(model.nq):
        q = np.zeros(model.nq)
        q[j] = np.deg2rad(30.0)
        q[j] = float(np.clip(q[j], lo[j], hi[j]))
        configs.append(q)
    # random configs inside limits (with 5% margin off the hard stops)
    for _ in range(N_RANDOM):
        q = lo + (hi - lo) * (0.05 + 0.90 * rng.random(model.nq))
        configs.append(q)

    ref = []
    for i, q in enumerate(configs):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, fid)
        oMf = data.oMf[fid]
        ref.append({
            "index": i,
            "q": [float(v) for v in q],
            "pos": [float(v) for v in oMf.translation],   # meters, base frame
            "quat_xyzw": quat_xyzw(oMf.rotation),
        })

    meta = {
        "urdf": ARM_URDF,
        "ee_frame": EE_FRAME,
        "nq": int(model.nq),
        "joint_lower": [float(v) for v in lo],
        "joint_upper": [float(v) for v in hi],
        "pinocchio_version": pin.__version__,
        "n_configs": len(configs),
    }
    with open(os.path.join(OUT_DIR, "fk_configs.json"), "w") as f:
        json.dump({"meta": meta, "configs": [r["q"] for r in ref]}, f, indent=2)
    with open(os.path.join(OUT_DIR, "fk_reference_pinocchio.json"), "w") as f:
        json.dump({"meta": meta, "poses": ref}, f, indent=2)

    print(f"[ok] nq={model.nq} frame={EE_FRAME} configs={len(configs)}")
    print(f"[ok] joint limits lo={np.round(lo,3)} hi={np.round(hi,3)}")
    print(f"[ok] wrote fk_configs.json + fk_reference_pinocchio.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
