#!/usr/bin/env python3
"""Imitation learning (DAgger) of the MPC-equivalent IK expert on the single-
circle task. Distills the expert's precision into a fast MLP policy.

    TRAIN_ENVS=4096 IMIT_ITERS=300 python scripts/imitation_train.py

Saves policy -> logs/imitation/circle/policy.pt  (+ tracking-error log).
"""
import os
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ENVS = int(os.environ.get("TRAIN_ENVS", "4096"))
ITERS = int(os.environ.get("IMIT_ITERS", "300"))
ROLL = int(os.environ.get("IMIT_ROLL", "50"))      # steps collected per iter
OUT_DIR = "/home/windylab/code/isaac_arm_rl/logs/imitation/circle"


def main():
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False).app
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    import torch.nn as nn
    import numpy as np
    from env.arm_track_env import ArmTrackEnv, ArmTrackEnvCfg

    cfg = ArmTrackEnvCfg()
    cfg.scene.num_envs = ENVS
    cfg.single_circle = True
    env = ArmTrackEnv(cfg)
    dev = env.device
    obs_dim, act_dim = cfg.observation_space, cfg.action_space

    policy = nn.Sequential(
        nn.Linear(obs_dim, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 64), nn.ELU(),
        nn.Linear(64, act_dim), nn.Tanh(),
    ).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    os.makedirs(OUT_DIR, exist_ok=True)
    obs_d, _ = env.reset()
    obs = obs_d["policy"]
    print(f"[imit] envs={ENVS} iters={ITERS} roll={ROLL} obs={obs_dim}", flush=True)

    # DAgger replay buffer (aggregate all collected (obs, expert) -> no forgetting)
    CAP = 1_500_000
    buf_o = torch.zeros(CAP, obs_dim, device=dev)
    buf_a = torch.zeros(CAP, act_dim, device=dev)
    filled = 0; ptr = 0

    def save(tag):
        torch.save({"state_dict": policy.state_dict(), "obs_dim": obs_dim,
                    "act_dim": act_dim, "arch": [256, 128, 64]},
                   os.path.join(OUT_DIR, f"policy{tag}.pt"))

    for it in range(ITERS):
        beta = max(0.0, 1.0 - it / (0.2 * ITERS))     # anneal to pure policy by 20%
        errs = []
        for _ in range(ROLL):
            with torch.no_grad():
                exp = env.expert_action()
                pol = policy(obs)
            m = obs.shape[0]
            e = (ptr + m) % CAP
            if ptr + m <= CAP:
                buf_o[ptr:ptr + m] = obs; buf_a[ptr:ptr + m] = exp
            else:
                k = CAP - ptr
                buf_o[ptr:] = obs[:k]; buf_a[ptr:] = exp[:k]
                buf_o[:m - k] = obs[k:]; buf_a[:m - k] = exp[k:]
            ptr = e; filled = min(filled + m, CAP)
            step_act = beta * exp + (1.0 - beta) * pol
            obs_d, _, _, _, _ = env.step(step_act)
            obs = obs_d["policy"]
            errs.append(torch.linalg.norm(env._p_tgt - env._p_ee, dim=-1).mean().item())
        # train on random minibatches from the WHOLE buffer (DAgger aggregate)
        bs = 8192; mse_last = 0.0
        for _ in range(200):
            b = torch.randint(0, filled, (bs,), device=dev)
            loss = ((policy(buf_o[b]) - buf_a[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            mse_last = loss.item()
        if it % 10 == 0 or it == ITERS - 1:
            print(f"[imit] it {it:3d} beta={beta:.2f} mse={mse_last:.4f} "
                  f"track_err={np.mean(errs)*1000:.2f}mm buf={filled}", flush=True)
        if it % 50 == 0 and it > 0:
            save("")
    save("")
    print(f"[imit] DONE saved {OUT_DIR}/policy.pt", flush=True)
    import sys as _s; _s.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
