#!/usr/bin/env python3
"""Export the selected Phase-1 diffusion policy as three RViz rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ROOT = "/home/windylab/code/isaac_arm_rl"
SELECTED_CONFIG = f"{ROOT}/config/phase1_selected_policy.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--q-noise", type=float, default=0.002)
    parser.add_argument("--output-dir", default=f"{ROOT}/data")
    args = parser.parse_args()

    selected = json.load(open(SELECTED_CONFIG))
    checkpoint_path = selected["checkpoint"]
    gains = [
        item["gain"]
        for item in selected["deployment"]["action_gain_by_condition"]
    ]

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    import numpy as np
    import torch

    sys.path.insert(0, ROOT)
    from distill.diffusion_policy import make_policy_from_checkpoint
    from env.arm_track_env import ArmTrackEnv, ArmTrackEnvCfg
    from rollout_io import write_rollout

    torch.manual_seed(20260724)
    torch.cuda.manual_seed_all(20260724)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    observation_dim = len(checkpoint["obs_mean"])

    cfg = ArmTrackEnvCfg()
    cfg.scene.num_envs = 3
    cfg.single_circle = True
    cfg.start_phase_random = False
    cfg.inertial_coupling = False
    cfg.extended_phase1_obs = observation_dim == 47
    cfg.observation_space = observation_dim
    env = ArmTrackEnv(cfg)
    device = env.device
    policy = make_policy_from_checkpoint(checkpoint, device)
    obs_mean = torch.as_tensor(
        checkpoint["obs_mean"], device=device, dtype=torch.float32)
    obs_std = torch.as_tensor(
        checkpoint["obs_std"], device=device, dtype=torch.float32)
    action_gain = torch.tensor(gains, device=device)[:, None]

    env.reset()
    d = env._dist
    d["wave_t"][:] = 0
    d["wave_a"][:] = 1
    d["amp_t"][:] = torch.tensor([0.02, 0.02, 0.01], device=device)
    d["amp_z"][:] = torch.tensor([0.0, 0.02, 0.01], device=device)
    d["z_phase"][:] = math.pi / 3.0
    d["amp_a"][:] = math.radians(0.1)
    d["freq_t"][:] = torch.tensor([1.0, 1.0, 2.0], device=device)
    d["freq_a"].copy_(d["freq_t"])
    d["phase"][:] = 0.0
    env._start[:] = 0

    seed_csvs = (
        "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
        "moving_base_mpc_loop_20260724_051521.csv",
        "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
        "moving_base_mpc_loop_20260724_051912.csv",
        "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
        "moving_base_mpc_loop_20260724_052123.csv",
    )
    q_seed = torch.empty(3, 7, device=device)
    dq_seed = torch.empty(3, 7, device=device)
    for env_id, path in enumerate(seed_csvs):
        with open(path, newline="") as stream:
            row = next(csv.DictReader(stream))
        q_seed[env_id] = torch.tensor(
            [float(row[f"q_meas_{joint}"]) for joint in range(1, 8)],
            device=device,
        )
        dq_seed[env_id] = torch.tensor(
            [float(row[f"dq_meas_{joint}"]) for joint in range(1, 8)],
            device=device,
        )
    q_seed += args.q_noise * torch.randn_like(q_seed)
    env._last_executed_action.zero_()
    env._prev_action.zero_()
    env.robot.write_joint_state_to_sim(q_seed, dq_seed)
    env.robot.set_joint_position_target(q_seed)
    obs = env._get_observations()["policy"]
    history = torch.stack([obs, obs], dim=1)

    q_log = [[] for _ in range(3)]
    base_pos_log = [[] for _ in range(3)]
    base_quat_log = [[] for _ in range(3)]
    ee_world_log = [[] for _ in range(3)]
    target_world_log = [[] for _ in range(3)]
    error_log = [[] for _ in range(3)]

    for _ in range(args.steps):
        condition = (
            (history - obs_mean) / obs_std
        ).clamp(-10.0, 10.0).flatten(1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            chunk = policy.sample(
                condition,
                inference_steps=selected["deployment"]["ddim_steps"],
                deterministic=True,
            )
        action = (action_gain * chunk[:, 0].float()).clamp(-1.0, 1.0)
        obs = env.step(action)[0]["policy"]
        history = torch.stack([history[:, 1], obs], dim=1)

        base_pos, base_rotation = env.targets.base_pose(
            env._phase() % env.targets.T, env._dist)
        ee_world = base_pos + torch.bmm(
            base_rotation, env._p_ee[:, :, None])[:, :, 0]
        target_world = base_pos + torch.bmm(
            base_rotation, env._p_tgt[:, :, None])[:, :, 0]
        error = torch.linalg.norm(env._p_tgt - env._p_ee, dim=-1)
        for env_id in range(3):
            q_log[env_id].append(
                env.robot.data.joint_pos[env_id].cpu().numpy().copy())
            base_pos_log[env_id].append(
                base_pos[env_id].cpu().numpy().copy())
            base_quat_log[env_id].append(
                _rotation_to_xyzw(
                    base_rotation[env_id].cpu().numpy()))
            ee_world_log[env_id].append(
                ee_world[env_id].cpu().numpy().copy())
            target_world_log[env_id].append(
                target_world[env_id].cpu().numpy().copy())
            error_log[env_id].append(error[env_id].item() * 1000.0)

    os.makedirs(args.output_dir, exist_ok=True)
    names = ("easy", "medium", "hard")
    times = np.arange(args.steps, dtype=np.float64) / 50.0
    for env_id, name in enumerate(names):
        output = os.path.join(
            args.output_dir, f"rollout_phase1_{name}.csv")
        write_rollout(
            output,
            times,
            np.asarray(q_log[env_id]),
            ee=np.asarray(ee_world_log[env_id]),
            tgt=np.asarray(target_world_log[env_id]),
            base_pos=np.asarray(base_pos_log[env_id]),
            base_quat=np.asarray(base_quat_log[env_id]),
            drawing=np.ones(args.steps, dtype=bool),
            source="phase1_diffusion_selected",
            note=(
                f"condition={name}; checkpoint={os.path.basename(checkpoint_path)};"
                f" gain={gains[env_id]}; q_noise={args.q_noise}"
            ),
        )
        errors = np.asarray(error_log[env_id])
        print(
            f"[rviz-export] {name}: {output} "
            f"mean={errors.mean():.2f}mm "
            f"p95={np.percentile(errors, 95):.2f}mm "
            f"max={errors.max():.2f}mm",
            flush=True,
        )
    sys.stdout.flush()
    os._exit(0)


def _rotation_to_xyzw(rotation):
    import numpy as np

    trace = np.trace(rotation)
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(
                1.0 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(
                1.0 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.array([x, y, z, w])
    return quaternion / np.linalg.norm(quaternion)


if __name__ == "__main__":
    main()
