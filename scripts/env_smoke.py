#!/usr/bin/env python3
"""Smoke-test ArmTrackEnv: build, reset, step with random actions, sanity-check
observations/rewards. No training. Confirms the env runs on GPU physics here.

    source env_isaaclab/bin/activate
    python scripts/env_smoke.py
"""
import os
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def main():
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False).app

    import time
    import torch
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.arm_track_env import ArmTrackEnv, ArmTrackEnvCfg

    cfg = ArmTrackEnvCfg()
    cfg.scene.num_envs = 64
    assert abs(cfg.max_joint_step - 0.02) < 1e-12
    t0 = time.time()
    env = ArmTrackEnv(cfg)
    print(f"[env] built in {time.time()-t0:.1f}s  num_envs={env.num_envs} "
          f"obs={env.cfg.observation_space} act={env.cfg.action_space}", flush=True)

    obs, _ = env.reset()
    print(f"[env] reset obs['policy'] shape={tuple(obs['policy'].shape)} "
          f"finite={torch.isfinite(obs['policy']).all().item()}", flush=True)

    t0 = time.time()
    N = 100
    rew_hist, err_hist = [], []
    for k in range(N):
        act = torch.rand(env.num_envs, 7, device=env.device) * 2 - 1
        obs, rew, term, trunc, info = env.step(act)
        if k == 0:
            # The next decision must see the action that was actually executed
            # on this step, never the yet-unknown next action.
            if not torch.allclose(obs["policy"][:, -7:], act, atol=1e-7, rtol=0.0):
                raise RuntimeError("last-executed-action observation contract failed")
            print("[env] action contract PASS: obs_(t+1)[-7:] == executed a_t",
                  flush=True)
        rew_hist.append(rew.mean().item())
        err_hist.append(torch.linalg.norm(
            env._p_tgt - env._p_ee, dim=-1).mean().item() * 1000)
        if k % 25 == 0:
            print(f"[env] step {k:3d} rew={rew.mean().item():+.3f} "
                  f"pos_err={err_hist[-1]:.1f}mm "
                  f"finite={torch.isfinite(obs['policy']).all().item()}", flush=True)
    dt = time.time() - t0
    print(f"[env] {N} steps x {env.num_envs} envs in {dt:.1f}s "
          f"= {N*env.num_envs/dt:.0f} env-steps/s", flush=True)
    print(f"[env] DONE rew[{min(rew_hist):.2f},{max(rew_hist):.2f}] "
          f"pos_err start={err_hist[0]:.0f}mm end={err_hist[-1]:.0f}mm "
          f"(random policy, no learning)", flush=True)
    import sys as _s; _s.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
