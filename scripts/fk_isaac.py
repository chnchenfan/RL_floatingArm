#!/usr/bin/env python3
"""Phase 1, step C: compute link7 FK in Isaac Sim for the reference configs.

Runs inside the Isaac Lab py3.11 venv:
    source env_isaaclab/bin/activate
    python scripts/fk_isaac.py

Uses Isaac Lab's Articulation API on CPU with rendering OFF (the raw RTX
renderer crashes on this 8GB Blackwell laptop and is unnecessary for FK). For
each config in assets/fk_configs.json it writes the joint state, steps physics
once without rendering, and reads body world poses; stores the RELATIVE pose
base_link->link7 (frame-independent), matching Pinocchio's oMf.
Output: assets/fk_isaac.json.
"""
import json
import os

import numpy as np

# Accept the Omniverse EULA non-interactively (headless first boot).
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJ = "/home/windylab/code/isaac_arm_rl"
USD_PATH = os.path.join(PROJ, "usd", "arm.usd")
CONFIGS = os.path.join(PROJ, "assets", "fk_configs.json")
OUT = os.path.join(PROJ, "assets", "fk_isaac.json")
BASE_LINK = "base_link"
EE_LINK = "link7"
JOINT_ORDER = [f"joint{i}" for i in range(1, 8)]   # urdf/pinocchio order


def R_from_quat_wxyz(w, x, y, z):
    return np.array([
        [1 - 2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1 - 2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x+y*y)],
    ])


def T_from(pos, quat_wxyz):
    T = np.eye(4)
    T[:3, :3] = R_from_quat_wxyz(*quat_wxyz)
    T[:3, 3] = pos
    return T


def R_to_quat_xyzw(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([x, y, z, w])
    return (q / np.linalg.norm(q)).tolist()


def main() -> None:
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    simulation_app = app_launcher.app

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext, SimulationCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg

    cfgs = json.load(open(CONFIGS))["configs"]

    # Gravity OFF: pure kinematic FK. With free joints (0 stiffness) gravity
    # would drift the set config over the physics step and pollute the FK read.
    sim = SimulationContext(
        SimulationCfg(dt=1.0 / 120.0, device="cpu", gravity=(0.0, 0.0, 0.0))
    )

    arm_cfg = ArticulationCfg(
        prim_path="/World/arm",
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=0.0, damping=0.0
            )
        },
    )
    arm = Articulation(arm_cfg)
    sim.reset()

    joint_names = list(arm.data.joint_names)
    body_names = list(arm.data.body_names)
    assert BASE_LINK in body_names and EE_LINK in body_names, \
        f"bodies missing: base={BASE_LINK in body_names} ee={EE_LINK in body_names} have={body_names}"
    bi_base = body_names.index(BASE_LINK)
    bi_ee = body_names.index(EE_LINK)
    perm = [joint_names.index(j) for j in JOINT_ORDER]   # urdf order -> sim dof idx

    poses = []
    for i, q_urdf in enumerate(cfgs):
        q_sim = torch.zeros((1, len(joint_names)), dtype=torch.float32)
        for k, ji in enumerate(perm):
            q_sim[0, ji] = float(q_urdf[k])
        dq = torch.zeros_like(q_sim)
        arm.write_joint_state_to_sim(q_sim, dq)
        arm.write_data_to_sim()
        sim.step(render=False)
        arm.update(1.0 / 120.0)

        bp = arm.data.body_pos_w[0, bi_base].cpu().numpy()
        bq = arm.data.body_quat_w[0, bi_base].cpu().numpy()   # wxyz
        ep = arm.data.body_pos_w[0, bi_ee].cpu().numpy()
        eq = arm.data.body_quat_w[0, bi_ee].cpu().numpy()
        T_wb = T_from(bp, bq)
        T_we = T_from(ep, eq)
        T_be = np.linalg.inv(T_wb) @ T_we
        poses.append({
            "index": i,
            "pos": T_be[:3, 3].tolist(),
            "quat_xyzw": R_to_quat_xyzw(T_be[:3, :3]),
        })

    meta = {
        "usd": USD_PATH, "ee_link": EE_LINK, "base_link": BASE_LINK,
        "joint_names": joint_names, "body_names": body_names,
        "joint_order": JOINT_ORDER, "n": len(poses),
    }
    with open(OUT, "w") as f:
        json.dump({"meta": meta, "poses": poses}, f, indent=2)
    print(f"[ok] joint_names={joint_names}")
    print(f"[ok] body_names={body_names}")
    print(f"[ok] wrote {OUT} ({len(poses)} poses)")

    # Kit's simulation_app.close() can hang and leave a core spinning at 100%
    # (that was the source of the laptop overheating). Force-exit after the
    # result is safely written; the OS reclaims the GPU/threads cleanly.
    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
