#!/usr/bin/env python3
"""Closed-loop Phase-1 diffusion stress test over 4096 Isaac environments."""

from __future__ import annotations

import math
import os
import sys
import time
import csv

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ENVS = int(os.environ.get("PHASE1_EVAL_ENVS", "4096"))
STEPS = int(os.environ.get("PHASE1_EVAL_STEPS", "400"))
DDIM_STEPS = int(os.environ.get("PHASE1_DDIM_STEPS", "8"))
EXEC_HORIZON = int(os.environ.get("PHASE1_EXEC_HORIZON", "1"))
DETERMINISTIC = os.environ.get("PHASE1_DETERMINISTIC", "1") != "0"
ACTION_GAIN = float(os.environ.get("PHASE1_ACTION_GAIN", "1.0"))
ACTION_GAINS_TEXT = os.environ.get("PHASE1_ACTION_GAINS", "")
Q_SEED_NOISE = float(os.environ.get("PHASE1_Q_SEED_NOISE", "0.001"))
ORACLE_CSV = os.environ.get("PHASE1_EVAL_ORACLE_CSV", "0") == "1"
CHECKPOINT = os.environ.get(
    "PHASE1_POLICY",
    "/home/windylab/code/isaac_arm_rl/logs/phase1_diffusion/policy.pt",
)


def main():
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    import numpy as np
    import torch

    root = "/home/windylab/code/isaac_arm_rl"
    sys.path.insert(0, root)
    from distill.diffusion_policy import make_policy_from_checkpoint
    from env.arm_track_env import ArmTrackEnv, ArmTrackEnvCfg

    torch.manual_seed(20260724)
    torch.cuda.manual_seed_all(20260724)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    observation_dim = int(len(checkpoint["obs_mean"]))
    if observation_dim not in (36, 47):
        raise ValueError(
            f"unsupported checkpoint observation dim: {observation_dim}")
    cfg = ArmTrackEnvCfg()
    cfg.scene.num_envs = ENVS
    cfg.single_circle = True
    cfg.start_phase_random = True
    cfg.extended_phase1_obs = observation_dim == 47
    cfg.observation_space = observation_dim
    # Phase 1 is MPC imitation. The fictitious inertial load belongs to the
    # Phase-2 residual-RL benchmark and is intentionally disabled here.
    cfg.inertial_coupling = False
    env = ArmTrackEnv(cfg)
    device = env.device
    policy = make_policy_from_checkpoint(checkpoint, device)
    obs_mean = torch.as_tensor(
        checkpoint["obs_mean"], device=device, dtype=torch.float32)
    obs_std = torch.as_tensor(
        checkpoint["obs_std"], device=device, dtype=torch.float32)
    obs_horizon = int(checkpoint["obs_horizon"])
    if obs_horizon != 2:
        raise ValueError(f"evaluator expects obs_horizon=2, got {obs_horizon}")

    obs_dict, _ = env.reset()
    # Three exact logged base-motion conditions, spread over 4096 envs.
    group = torch.arange(ENVS, device=device) % 3
    if ACTION_GAINS_TEXT:
        action_gains = [
            float(value) for value in ACTION_GAINS_TEXT.split(",")]
        if len(action_gains) != 3:
            raise ValueError("PHASE1_ACTION_GAINS requires three values")
        action_gain_env = torch.as_tensor(
            action_gains, device=device)[group, None]
    else:
        action_gains = [ACTION_GAIN] * 3
        action_gain_env = torch.full(
            (ENVS, 1), ACTION_GAIN, device=device)
    d = env._dist
    d["wave_t"][:] = 0
    d["wave_a"][:] = 1
    d["amp_t"][:] = torch.where(
        group == 2, torch.tensor(0.01, device=device),
        torch.tensor(0.02, device=device))
    d["amp_z"][:] = torch.where(
        group == 0, torch.tensor(0.0, device=device),
        torch.where(group == 1, torch.tensor(0.02, device=device),
                    torch.tensor(0.01, device=device)))
    d["z_phase"][:] = math.pi / 3.0
    d["amp_a"][:] = math.radians(0.1)
    d["freq_t"][:] = torch.where(
        group == 2, torch.tensor(2.0, device=device),
        torch.tensor(1.0, device=device))
    d["freq_a"].copy_(d["freq_t"])
    d["phase"][:] = 0.0

    # Seed each environment from a random phase of a real clean MPC rollout.
    # This mirrors the deployed reset/entry controller and avoids asking a
    # local IK controller to find the correct kinematic branch from all-zero q.
    seed_csvs = (
        "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
        "moving_base_mpc_loop_20260724_051521.csv",
        "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
        "moving_base_mpc_loop_20260724_051912.csv",
        "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
        "moving_base_mpc_loop_20260724_052123.csv",
    )
    q_seed = torch.empty(ENVS, 7, device=device)
    dq_seed = torch.empty(ENVS, 7, device=device)
    prev_action_seed = torch.empty(ENVS, 7, device=device)
    start_seed = torch.empty(ENVS, dtype=torch.long, device=device)
    row_seed = torch.empty(ENVS, dtype=torch.long, device=device)
    q_pub_tables = []
    rng = np.random.default_rng(20260724)
    for gid, path in enumerate(seed_csvs):
        rows = list(csv.DictReader(open(path)))
        ids = np.flatnonzero((torch.arange(ENVS).cpu().numpy() % 3) == gid)
        if ORACLE_CSV:
            if len(rows) <= STEPS + 1:
                raise ValueError(
                    f"oracle CSV {path} has {len(rows)} rows, needs >{STEPS + 1}")
            pick = rng.integers(0, len(rows) - STEPS - 1, size=len(ids))
        else:
            pick = rng.integers(0, len(rows), size=len(ids))
        q_np = np.array([
            [float(rows[j][f"q_meas_{k}"]) for k in range(1, 8)]
            for j in pick
        ], dtype=np.float32)
        dq_np = np.array([
            [float(rows[j][f"dq_meas_{k}"]) for k in range(1, 8)]
            for j in pick
        ], dtype=np.float32)
        previous = np.maximum(pick - 1, 0)
        prev_action_np = np.array([
            [
                (float(rows[j][f"q_pub_{k}"]) -
                 float(rows[j][f"q_meas_{k}"])) / cfg.max_joint_step
                for k in range(1, 8)
            ]
            for j in previous
        ], dtype=np.float32).clip(-1.0, 1.0)
        t_np = np.array([float(rows[j]["t_solve"]) for j in pick])
        ids_t = torch.as_tensor(ids, device=device)
        q_pub_table = torch.as_tensor(np.array([
            [float(row[f"q_pub_{k}"]) for k in range(1, 8)]
            for row in rows
        ], dtype=np.float32), device=device)
        q_pub_tables.append(q_pub_table)
        q_seed[ids_t] = torch.as_tensor(q_np, device=device)
        dq_seed[ids_t] = torch.as_tensor(dq_np, device=device)
        prev_action_seed[ids_t] = torch.as_tensor(
            prev_action_np, device=device)
        start_seed[ids_t] = torch.as_tensor(
            np.round(t_np * 50.0).astype(np.int64) % env.targets.T,
            device=device,
        )
        row_seed[ids_t] = torch.as_tensor(pick, device=device)
    q_seed += Q_SEED_NOISE * torch.randn_like(q_seed)
    env._start.copy_(start_seed)
    env._last_executed_action.copy_(prev_action_seed)
    env._prev_action.copy_(prev_action_seed)
    env.robot.write_joint_state_to_sim(q_seed, dq_seed)
    env.robot.set_joint_position_target(q_seed)
    # Deployment hand-off: the entry controller supplies one valid on-trajectory
    # observation. Duplicate it only for the first history window; subsequent
    # windows contain two actual consecutive policy states.
    obs = env._get_observations()["policy"]
    initial_error = torch.linalg.norm(env._p_tgt - env._p_ee, dim=-1)
    print(
        f"[eval4096] seeded mean={initial_error.mean().item()*1000:.2f}mm "
        f"p95={torch.quantile(initial_error, 0.95).item()*1000:.2f}mm "
        f"max={initial_error.max().item()*1000:.2f}mm "
        f"q_noise={Q_SEED_NOISE:.4f}rad checkpoint_step={checkpoint['step']}",
        flush=True,
    )
    history = torch.stack([obs, obs], dim=1)
    queued = None
    queued_index = 0
    all_errors = []
    inference_ms = []
    obs_abs_z = []
    generator = torch.Generator(device=device)
    generator.manual_seed(20260724)

    for step in range(STEPS):
        if ORACLE_CSV:
            action = torch.empty(ENVS, 7, device=device)
            current_q = env.robot.data.joint_pos
            for gid, table in enumerate(q_pub_tables):
                ids_t = torch.nonzero(group == gid, as_tuple=False)[:, 0]
                row = (row_seed[ids_t] + step) % len(table)
                desired_q = table[row]
                action[ids_t] = (
                    (desired_q - current_q[ids_t]) / cfg.max_joint_step
                ).clamp(-1.0, 1.0)
        elif queued is None or queued_index >= min(
                EXEC_HORIZON, queued.shape[1]):
            cond = ((history - obs_mean) / obs_std).clamp(-10.0, 10.0).flatten(1)
            obs_abs_z.append(
                ((history[:, -1] - obs_mean) / obs_std).abs().mean(dim=0))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                queued = policy.sample(
                    cond,
                    inference_steps=DDIM_STEPS,
                    generator=generator,
                    deterministic=DETERMINISTIC,
                )
            torch.cuda.synchronize()
            inference_ms.append((time.perf_counter() - t0) * 1000.0)
            queued_index = 0
        if not ORACLE_CSV:
            action = (
                action_gain_env * queued[:, queued_index].float()
            ).clamp(-1.0, 1.0)
            queued_index += 1
        obs = env.step(action)[0]["policy"]
        history = torch.stack([history[:, 1], obs], dim=1)
        error = torch.linalg.norm(env._p_tgt - env._p_ee, dim=-1)
        all_errors.append(error)
        if step % 100 == 0:
            print(
                f"[eval4096] step={step:3d} mean={error.mean().item()*1000:.2f}mm "
                f"p95={torch.quantile(error, 0.95).item()*1000:.2f}mm "
                f"max={error.max().item()*1000:.2f}mm",
                flush=True,
            )

    errors = torch.stack(all_errors)
    print("[eval4096] RESULTS", flush=True)
    names = ("R.02/T1/z0", "R.02/T1/z.02", "R.01/T.5/z.01")
    for gid, name in enumerate(names):
        values = errors[:, group == gid].flatten()
        per_env_max = errors[:, group == gid].max(dim=0).values
        print(
            f"[eval4096] {name}: mean={values.mean().item()*1000:.2f} "
            f"p95={torch.quantile(values, .95).item()*1000:.2f} "
            f"p99={torch.quantile(values, .99).item()*1000:.2f} "
            f"max={values.max().item()*1000:.2f}mm "
            f"env_max_p95={torch.quantile(per_env_max, .95).item()*1000:.2f}mm "
            f"diverged={100*(per_env_max > .05).float().mean().item():.2f}%",
            flush=True,
        )
    if not ORACLE_CSV:
        latency = np.asarray(inference_ms)
        feature_z = torch.stack(obs_abs_z).mean(dim=0)
        print(
            "[eval4096] obs mean|z| "
            f"q={feature_z[0:7].mean().item():.2f} "
            f"dq={feature_z[7:14].mean().item():.2f} "
            f"target={feature_z[14:23].mean().item():.2f} "
            f"ee/error={feature_z[23:29].mean().item():.2f} "
            f"prev_action={feature_z[29:36].mean().item():.2f}",
            flush=True,
        )
        if observation_dim == 47:
            print(
                "[eval4096] obs-v2 mean|z| "
                f"phase={feature_z[36:38].mean().item():.2f} "
                f"base_vel={feature_z[38:41].mean().item():.2f} "
                f"base_accel={feature_z[41:44].mean().item():.2f} "
                f"condition={feature_z[44:47].mean().item():.2f}",
                flush=True,
            )
        print(
            f"[eval4096] chunk inference batch={ENVS} DDIM={DDIM_STEPS}: "
            f"mean={latency.mean():.2f}ms "
            f"p95={np.percentile(latency,95):.2f}ms "
            f"amortized/tick={latency.mean()/EXEC_HORIZON:.2f}ms "
            f"deterministic={int(DETERMINISTIC)} "
            f"action_gains={action_gains}",
            flush=True,
        )
    else:
        print("[eval4096] controller=CSV_ABSOLUTE_Q_ORACLE", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
