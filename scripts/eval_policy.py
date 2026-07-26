#!/usr/bin/env python3
"""Evaluate a trained policy on ONE deterministic episode with the config-matched
base disturbance (1 Hz / 2 cm / 0.1 deg, same as the IK/MPC demo rollout_task.csv),
export a RViz-viewable rollout, and report tracking error + inference latency.

    source env_isaaclab/bin/activate
    python scripts/eval_policy.py --checkpoint logs/rsl_rl/arm_track/envs2048/model_XXXX.pt

Output: data/rollout_rl.csv  (+ printed tracking/latency stats for RL-vs-MPC).
"""
import argparse
import math
import os
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

OUT = "/home/windylab/code/isaac_arm_rl/data/rollout_rl.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False).app

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import numpy as np
    import torch
    from env.arm_track_env import ArmTrackEnv, ArmTrackEnvCfg
    from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg,
                                    RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)
    from rsl_rl.runners import OnPolicyRunner
    from rollout_io import write_rollout

    cfg = ArmTrackEnvCfg()
    cfg.scene.num_envs = 1
    cfg.start_phase_random = False
    env = ArmTrackEnv(cfg)

    # fix the disturbance to the config-matched values (same as IK/MPC demo)
    d = env._dist
    d["wave_t"][:] = 0            # circle translation
    d["wave_a"][:] = 1            # sine attitude
    d["amp_t"][:] = 0.02
    d["amp_z"][:] = 0.0
    d["z_phase"][:] = math.pi / 3.0
    d["amp_a"][:] = math.radians(0.1)
    d["freq_t"][:] = 1.0
    d["freq_a"][:] = 1.0
    d["phase"][:] = 0.0
    env._start[:] = 0

    wrapped = RslRlVecEnvWrapper(env)
    agent = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24, max_iterations=1, experiment_name="arm_track",
        empirical_normalization=False,
        policy=RslRlPpoActorCriticCfg(init_noise_std=1.0,
            actor_hidden_dims=[256, 128, 64], critic_hidden_dims=[256, 128, 64],
            activation="elu"),
        algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True,
            clip_param=0.2, entropy_coef=0.005, num_learning_epochs=5,
            num_mini_batches=4, learning_rate=1e-3, schedule="adaptive",
            gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
    )
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=env.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    T = env.targets.T
    obs, _ = wrapped.reset()
    qs, bpos, bquat, eew, tgtw, draw = [], [], [], [], [], []
    perr, oerr, lat = [], [], []
    for k in range(T):
        t0 = time.perf_counter()
        with torch.no_grad():
            act = policy(obs)
        lat.append((time.perf_counter() - t0) * 1e3)      # ms per inference
        obs, _, _, _ = wrapped.step(act)                   # rsl_rl: obs,rew,done,extras

        p_ee = env._p_ee[0].cpu().numpy()
        p_tg = env._p_tgt[0].cpu().numpy()
        perr.append(np.linalg.norm(p_ee - p_tg) * 1000)
        Rerr = env._R_ee[0].T @ env._R_tgt[0]
        oerr.append(float(np.degrees(np.arccos(np.clip(
            (torch.trace(Rerr).item() - 1) / 2, -1, 1)))))
        # world-frame base pose + ee/target for RViz
        bp, bR = env.targets.base_pose(env._phase() % env.targets.T, env._dist)
        qs.append(env.robot.data.joint_pos[0].cpu().numpy())
        bpos.append(bp[0].cpu().numpy())
        bquat.append(_R_xyzw(bR[0].cpu().numpy()))
        eew.append(bp[0].cpu().numpy() + bR[0].cpu().numpy() @ p_ee)
        tgtw.append(bp[0].cpu().numpy() + bR[0].cpu().numpy() @ p_tg)
        draw.append(env._draw[0].item() > 0.5)

    dt = 1.0 / 50.0
    write_rollout(OUT, np.arange(T) * dt, np.array(qs), ee=np.array(eew),
                  tgt=np.array(tgtw), base_pos=np.array(bpos),
                  base_quat=np.array(bquat), drawing=np.array(draw),
                  source="rl_policy", note=f"ckpt={os.path.basename(args.checkpoint)}")
    perr = np.array(perr); oerr = np.array(oerr); lat = np.array(lat)
    dm = np.array(draw)
    print(f"\n[eval] wrote {OUT}")
    print(f"[eval] RL tracking (drawing phases): "
          f"pos mean={perr[dm].mean():.1f}mm p95={np.percentile(perr[dm],95):.1f}mm "
          f"max={perr[dm].max():.1f}mm | orient mean={oerr[dm].mean():.1f}deg")
    print(f"[eval] RL inference latency: mean={lat.mean():.3f}ms "
          f"p95={np.percentile(lat,95):.3f}ms max={lat.max():.3f}ms")
    import sys as _s; _s.stdout.flush()
    os._exit(0)


def _R_xyzw(R):
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
