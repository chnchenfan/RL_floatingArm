#!/usr/bin/env python3
"""Sanity-check the vectorized IK expert: drive the single-circle env with
expert_action() and confirm tracking error is small (the expert must be good
before we imitate it)."""
import os
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def main():
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False).app
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from env.arm_track_env import ArmTrackEnv, ArmTrackEnvCfg

    cfg = ArmTrackEnvCfg()
    cfg.scene.num_envs = 64
    cfg.single_circle = True
    cfg.inertial_coupling = False
    cfg.w_ori_ik = 0.0
    env = ArmTrackEnv(cfg)
    obs, _ = env.reset()
    q0 = torch.tensor(
        [0.1065, -0.3383, -0.2484, 1.4405, -0.9845, 0.0786, -3.1447],
        device=env.device,
    ).repeat(env.num_envs, 1)
    env._start[:] = 0
    env._dist["phase"][:] = 0.0
    env._dist["amp_t"][:] = 0.02
    env._dist["amp_z"][:] = 0.0
    env._dist["amp_a"][:] = torch.deg2rad(torch.tensor(0.1, device=env.device))
    env._dist["freq_t"][:] = 1.0
    env._dist["freq_a"][:] = 1.0
    env.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    env.robot.set_joint_position_target(q0)
    env.step(torch.zeros_like(q0))
    errs = []
    for k in range(400):
        act = env.expert_action()
        env.step(act)
        errs.append(torch.linalg.norm(env._p_tgt - env._p_ee, dim=-1).mean().item() * 1000)
        if k % 80 == 0:
            print(f"[expert] step {k:3d} pos_err={errs[-1]:.2f}mm", flush=True)
    import numpy as np
    e = np.array(errs[100:])   # after transient
    print(f"[expert] steady pos_err mean={e.mean():.2f}mm max={e.max():.2f}mm "
          f"(good if < ~5mm)", flush=True)
    import sys as _s; _s.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
