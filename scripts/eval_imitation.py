#!/usr/bin/env python3
"""Evaluate the imitation MLP policy on the single-circle task with the config
disturbance; report tracking error + latency, export RViz rollout + compare to
the IK expert.

    python scripts/eval_imitation.py --policy logs/imitation/circle/policy.pt
"""
import argparse
import math
import os
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
OUT = "/home/windylab/code/isaac_arm_rl/data/rollout_imit.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--expert", action="store_true", help="eval the IK expert instead")
    args = ap.parse_args()

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False).app
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import numpy as np
    import torch
    import torch.nn as nn
    from env.arm_track_env import ArmTrackEnv, ArmTrackEnvCfg
    from rollout_io import write_rollout

    cfg = ArmTrackEnvCfg()
    cfg.scene.num_envs = 1
    cfg.single_circle = True
    cfg.start_phase_random = False
    env = ArmTrackEnv(cfg)

    if not args.expert:
        ck = torch.load(args.policy, map_location=env.device)
        a = ck["arch"]
        policy = nn.Sequential(
            nn.Linear(ck["obs_dim"], a[0]), nn.ELU(),
            nn.Linear(a[0], a[1]), nn.ELU(),
            nn.Linear(a[1], a[2]), nn.ELU(),
            nn.Linear(a[2], ck["act_dim"]), nn.Tanh()).to(env.device)
        policy.load_state_dict(ck["state_dict"]); policy.eval()

    obs_d, _ = env.reset()
    # config-matched disturbance AFTER reset (reset resamples it), then refresh obs
    d = env._dist
    d["wave_t"][:] = 0; d["wave_a"][:] = 1
    d["amp_t"][:] = 0.02; d["amp_a"][:] = math.radians(0.1)
    d["amp_z"][:] = 0.0; d["z_phase"][:] = math.pi / 3.0
    d["freq_t"][:] = 1.0; d["freq_a"][:] = 1.0; d["phase"][:] = 0.0
    env._start[:] = 0
    obs = env._get_observations()["policy"]
    T = env.targets.T
    qs, bpos, bquat, eew, tgtw, draw = [], [], [], [], [], []
    perr, oerr, lat = [], [], []
    for k in range(T):
        t0 = time.perf_counter()
        with torch.no_grad():
            act = env.expert_action() if args.expert else policy(obs)
        lat.append((time.perf_counter() - t0) * 1e3)
        obs_d, _, _, _, _ = env.step(act)
        obs = obs_d["policy"]
        p_ee = env._p_ee[0].cpu().numpy(); p_tg = env._p_tgt[0].cpu().numpy()
        perr.append(np.linalg.norm(p_ee - p_tg) * 1000)
        Rerr = env._R_ee[0].T @ env._R_tgt[0]
        oerr.append(float(np.degrees(np.arccos(np.clip(
            (torch.trace(Rerr).item() - 1) / 2, -1, 1)))))
        bp, bR = env.targets.base_pose(env._phase() % env.targets.T, env._dist)
        bpn = bp[0].cpu().numpy(); bRn = bR[0].cpu().numpy()
        qs.append(env.robot.data.joint_pos[0].cpu().numpy())
        bpos.append(bpn); bquat.append(_xyzw(bRn))
        eew.append(bpn + bRn @ p_ee); tgtw.append(bpn + bRn @ p_tg)
        draw.append(env._draw[0].item() > 0.5)

    write_rollout(OUT, np.arange(T) / 50.0, np.array(qs), ee=np.array(eew),
                  tgt=np.array(tgtw), base_pos=np.array(bpos),
                  base_quat=np.array(bquat), drawing=np.array(draw),
                  source="imit" if not args.expert else "expert")
    perr = np.array(perr); oerr = np.array(oerr); lat = np.array(lat)
    dm = np.array(draw)
    tag = "EXPERT(IK)" if args.expert else "IMITATION policy"
    print(f"\n[eval] {tag}  ->  {OUT}")
    print(f"[eval] pos: mean={perr[dm].mean():.2f}mm rms={np.sqrt((perr[dm]**2).mean()):.2f}mm "
          f"max={perr[dm].max():.2f}mm | orient mean={oerr[dm].mean():.2f}deg")
    print(f"[eval] latency: mean={lat.mean():.3f}ms p95={np.percentile(lat,95):.3f}ms")
    import sys as _s; _s.stdout.flush()
    os._exit(0)


def _xyzw(R):
    import numpy as np
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1) * 2; w = .25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = .25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = .25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = .25 * s
    q = np.array([x, y, z, w]); return q / np.linalg.norm(q)


if __name__ == "__main__":
    main()
