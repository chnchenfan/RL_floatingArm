#!/usr/bin/env python3
"""Closed-loop full-cycle evaluation of the whole-task Phase-1 diffusion policy.

Covers at least one complete 73.039-s cycle (3652 ticks) per environment,
with start phases randomized over the entire cycle and states seeded from the
clean-MPC state bank.  Reports overall / pen-down / per-task / transition
position error, orientation error, divergence, and inference latency.

    PHASE1_POLICY=... ./env_isaaclab/bin/python scripts/eval_phase1_whole.py
"""

from __future__ import annotations

import math
import os
import sys
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ENVS = int(os.environ.get("PHASE1_EVAL_ENVS", "1024"))
STEPS = int(os.environ.get("PHASE1_EVAL_STEPS", "3700"))
DDIM_STEPS = int(os.environ.get("PHASE1_DDIM_STEPS", "8"))
EXEC_HORIZON = int(os.environ.get("PHASE1_EXEC_HORIZON", "1"))
ACTION_GAIN = float(os.environ.get("PHASE1_ACTION_GAIN", "1.0"))
Q_SEED_NOISE = float(os.environ.get("PHASE1_Q_SEED_NOISE", "0.001"))
JITTER = float(os.environ.get("PHASE1_EVAL_JITTER", "0.0"))
SEED = int(os.environ.get("PHASE1_EVAL_SEED", "20260726"))
CHECKPOINT = os.environ.get(
    "PHASE1_POLICY",
    "/home/windylab/code/isaac_arm_rl/logs/phase1_whole_diffusion/"
    "policy_whole.pt")
STATE_BANK = os.environ.get(
    "PHASE1_STATE_BANK",
    "/home/windylab/code/isaac_arm_rl/data/whole_v2_state_bank.npz")
OUT_NPZ = os.environ.get("PHASE1_EVAL_OUT", "")


def main():
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    import numpy as np
    import torch

    root = "/home/windylab/code/isaac_arm_rl"
    sys.path.insert(0, root)
    from distill.diffusion_policy import make_policy_from_checkpoint
    from env.arm_track_env import ArmTrackEnv, ArmTrackEnvCfg
    from env.residual_arm_track_env import _load_state_bank, _draw_bank_states

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if int(len(checkpoint["obs_mean"])) != 47:
        raise ValueError("whole Phase-1 checkpoint must be 47-D v2 obs")

    cfg = ArmTrackEnvCfg()
    cfg.scene.num_envs = ENVS
    cfg.single_circle = False
    cfg.whole_task = True
    cfg.max_joint_step = 0.04
    cfg.start_phase_random = True
    cfg.extended_phase1_obs = True
    cfg.observation_space = 47
    cfg.episode_length_s = 1.0e6      # no timeout resets during eval
    cfg.inertial_coupling = False     # Phase-1 is MPC imitation
    # calibrated plant model (see WholeResidualArmTrackEnvCfg; delay must be 0)
    cfg.command_delay_ticks = 0
    cfg.plant_stiffness = 80000.0
    cfg.plant_damping = 4000.0
    env = ArmTrackEnv(cfg)
    device = env.device
    policy = make_policy_from_checkpoint(checkpoint, device)
    obs_mean = torch.as_tensor(
        checkpoint["obs_mean"], device=device, dtype=torch.float32)
    obs_std = torch.as_tensor(
        checkpoint["obs_std"], device=device, dtype=torch.float32)

    env.reset()
    # exact config-matched base motion for evaluation
    d = env._dist
    exact = env.targets.sample_whole_disturbance(ENVS, jitter=JITTER)
    for key, value in exact.items():
        d[key].copy_(value)

    T = env.targets.T
    start = torch.randint(
        0, T, (ENVS,), device=device,
        generator=torch.Generator(device=device).manual_seed(SEED))
    env._start.copy_(start)
    bank = _load_state_bank(STATE_BANK, device)
    q_seed, dq_seed, prev_action = _draw_bank_states(bank, start)
    q_seed += Q_SEED_NOISE * torch.randn_like(q_seed)
    env._last_executed_action.copy_(prev_action)
    env._prev_action.copy_(prev_action)
    env.episode_length_buf.zero_()
    env.robot.write_joint_state_to_sim(q_seed, dq_seed)
    env.robot.set_joint_position_target(q_seed)
    env.sync_command_state()

    obs = env._get_observations()["policy"]
    err0 = torch.linalg.norm(env._p_tgt - env._p_ee, dim=-1)
    print(f"[whole-eval] seeded mean={err0.mean()*1000:.2f}mm "
          f"p95={torch.quantile(err0, .95)*1000:.2f}mm envs={ENVS} "
          f"steps={STEPS} gain={ACTION_GAIN} jitter={JITTER}", flush=True)

    history = torch.stack([obs, obs], dim=1)
    queued, queued_index = None, 0
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)

    pos_err = torch.empty(STEPS, ENVS, device=device)
    ori_err = torch.empty(STEPS, ENVS, device=device)
    pen_mask = torch.empty(STEPS, ENVS, dtype=torch.bool, device=device)
    task_ids = torch.empty(STEPS, ENVS, dtype=torch.long, device=device)
    sat_frac = torch.zeros(STEPS, device=device)
    inference_ms = []

    for step in range(STEPS):
        if queued is None or queued_index >= min(
                EXEC_HORIZON, queued.shape[1]):
            cond = ((history - obs_mean) / obs_std).clamp(
                -10.0, 10.0).flatten(1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                queued = policy.sample(
                    cond, inference_steps=DDIM_STEPS,
                    generator=generator, deterministic=True)
            torch.cuda.synchronize()
            inference_ms.append((time.perf_counter() - t0) * 1000.0)
            queued_index = 0
        action = (ACTION_GAIN * queued[:, queued_index].float()).clamp(
            -1.0, 1.0)
        queued_index += 1
        idx = (env._start + env.episode_length_buf) % T
        obs = env.step(action)[0]["policy"]
        history = torch.stack([history[:, 1], obs], dim=1)
        pos_err[step] = torch.linalg.norm(env._p_tgt - env._p_ee, dim=-1)
        Rerr = torch.bmm(env._R_ee.transpose(1, 2), env._R_tgt)
        cos = ((Rerr[:, 0, 0] + Rerr[:, 1, 1] + Rerr[:, 2, 2]) - 1) * 0.5
        ori_err[step] = torch.arccos(torch.clamp(cos, -1.0, 1.0))
        pen_mask[step] = env.targets.pen[idx]
        task_ids[step] = env.targets.task_id[idx]
        sat_frac[step] = (action.abs() > 0.999).float().mean()
        if step % 500 == 0:
            print(f"[whole-eval] step={step:4d} "
                  f"mean={pos_err[step].mean()*1000:.2f}mm "
                  f"p95={torch.quantile(pos_err[step], .95)*1000:.2f}mm "
                  f"max={pos_err[step].max()*1000:.2f}mm", flush=True)

    def stats(mask=None, name="overall"):
        v = pos_err if mask is None else pos_err[mask]
        v = v.flatten() * 1000.0
        o = (ori_err if mask is None else ori_err[mask]).flatten()
        if v.numel() == 0:
            print(f"[whole-eval] {name}: EMPTY", flush=True)
            return None
        line = (f"[whole-eval] {name}: mean={v.mean():.3f} "
                f"rmse={v.square().mean().sqrt():.3f} "
                f"p95={torch.quantile(v, .95):.3f} "
                f"p99={torch.quantile(v, .99):.3f} max={v.max():.3f}mm "
                f"ori_mean={torch.rad2deg(o.mean()):.2f}deg "
                f"ori_p95={torch.rad2deg(torch.quantile(o, .95)):.2f}deg")
        print(line, flush=True)
        return {
            "name": name, "mean": v.mean().item(),
            "rmse": v.square().mean().sqrt().item(),
            "p95": torch.quantile(v, .95).item(),
            "p99": torch.quantile(v, .99).item(), "max": v.max().item(),
            "ori_mean_deg": torch.rad2deg(o.mean()).item(),
            "ori_p95_deg": torch.rad2deg(torch.quantile(o, .95)).item(),
            "n": v.numel(),
        }

    print("[whole-eval] RESULTS", flush=True)
    results = [stats(None, "overall"), stats(pen_mask, "pen_down")]
    task_names = ("circle", "triangle", "windylab", "sine", "square")
    for tid, tname in enumerate(task_names):
        results.append(stats(task_ids == tid, f"task_{tname}"))
    results.append(stats(task_ids < 0, "transition"))

    per_env_max = pos_err.max(dim=0).values
    diverged = (per_env_max > 0.05).float().mean().item()
    latency = np.asarray(inference_ms)
    print(f"[whole-eval] diverged(>50mm)={100*diverged:.2f}% "
          f"saturation={sat_frac.mean():.4f} "
          f"latency mean={latency.mean():.2f} "
          f"p95={np.percentile(latency, 95):.2f} "
          f"p99={np.percentile(latency, 99):.2f} "
          f"max={latency.max():.2f}ms", flush=True)
    if OUT_NPZ:
        np.savez_compressed(
            OUT_NPZ,
            results=np.array(
                [r for r in results if r is not None], dtype=object),
            diverged_fraction=diverged,
            saturation=sat_frac.mean().item(),
            latency_ms=latency,
            envs=ENVS, steps=STEPS, gain=ACTION_GAIN,
            checkpoint=CHECKPOINT, seed=SEED)
        print(f"[whole-eval] wrote {OUT_NPZ}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
