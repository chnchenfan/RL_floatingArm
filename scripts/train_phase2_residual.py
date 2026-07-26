#!/usr/bin/env python3
"""Train a bounded PPO residual on top of the frozen Phase-1 policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ROOT = "/home/windylab/code/isaac_arm_rl"
LOG_ROOT = f"{ROOT}/logs/phase2_residual"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--tag", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="render the training environment (use few envs only)",
    )
    args = parser.parse_args()
    if min(
        args.num_envs,
        args.iterations,
        args.steps_per_env,
        args.save_interval,
    ) <= 0:
        parser.error("numeric training arguments must be positive")
    if args.gui and args.num_envs > 64:
        parser.error("--gui supports at most 64 envs; use headless for 4096")
    if args.resume and not os.path.isfile(args.resume):
        parser.error(f"resume checkpoint not found: {args.resume}")
    return args


def main():
    args = parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(
        headless=not args.gui,
        enable_cameras=False,
    ).app
    import torch

    sys.path.insert(0, ROOT)
    from env.residual_arm_track_env import (
        ResidualArmTrackEnv,
        ResidualArmTrackEnvCfg,
    )
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
        RslRlVecEnvWrapper,
    )
    from rsl_rl.runners import OnPolicyRunner

    torch.manual_seed(20260725)
    torch.cuda.manual_seed_all(20260725)

    env_cfg = ResidualArmTrackEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    environment = ResidualArmTrackEnv(env_cfg)
    wrapped = RslRlVecEnvWrapper(environment, clip_actions=1.0)

    agent_cfg = RslRlOnPolicyRunnerCfg(
        seed=20260725,
        device="cuda:0",
        num_steps_per_env=args.steps_per_env,
        max_iterations=args.iterations,
        save_interval=args.save_interval,
        experiment_name="phase2_residual",
        run_name=args.tag,
        logger="tensorboard",
        obs_groups={
            "policy": ["policy"],
            "critic": ["policy"],
        },
        clip_actions=1.0,
        empirical_normalization=False,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=0.15,
            noise_std_type="log",
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=5,
            num_mini_batches=8,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"envs{args.num_envs}"
    log_dir = os.path.join(LOG_ROOT, f"{timestamp}_{tag}")
    os.makedirs(log_dir, exist_ok=True)
    runner = OnPolicyRunner(
        wrapped,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device="cuda:0",
    )

    if args.resume:
        runner.load(args.resume)
        print(f"[phase2] resumed {args.resume}", flush=True)
    else:
        _zero_initialize_residual_mean(runner, torch)
        _save_initial_checkpoint(
            runner,
            torch,
            os.path.join(log_dir, "model_init.pt"),
            args,
            env_cfg,
        )

    manifest = {
        "phase": 2,
        "algorithm": "bounded_residual_ppo_orientation_safe",
        "base_checkpoint": env_cfg.phase1_checkpoint,
        "residual_scale": env_cfg.residual_scale,
        "inertial_force_frame": "world",
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "steps_per_env": args.steps_per_env,
        "seed": 20260725,
        "reward": {
            "precision_sigma_m": env_cfg.precision_sigma_m,
            "tracking_sigma_m": env_cfg.tracking_sigma_m,
            "orientation_sigma_rad": env_cfg.orientation_sigma_rad,
            "orientation_limit_rad": env_cfg.orientation_limit_rad,
            "w_precision": env_cfg.w_precision,
            "w_tracking": env_cfg.w_tracking,
            "w_orientation": env_cfg.w_orientation,
            "w_orientation_limit": env_cfg.w_orientation_limit,
            "w_residual": env_cfg.w_residual,
            "w_residual_rate": env_cfg.w_residual_rate,
            "w_saturation": env_cfg.w_saturation,
        },
    }
    with open(
        os.path.join(log_dir, "phase2_manifest.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    print(
        "[phase2] START "
        f"envs={args.num_envs} iterations={args.iterations} "
        f"residual_scale={env_cfg.residual_scale} log_dir={log_dir}",
        flush=True,
    )
    runner.learn(
        num_learning_iterations=args.iterations,
        init_at_random_ep_len=False,
    )
    print("[phase2] DONE", flush=True)
    sys.stdout.flush()
    del app
    os._exit(0)


def _zero_initialize_residual_mean(runner, torch):
    actor = runner.alg.policy.actor
    linear_layers = [
        module for module in actor.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    if not linear_layers:
        raise RuntimeError("residual actor has no linear output layer")
    output_layer = linear_layers[-1]
    torch.nn.init.zeros_(output_layer.weight)
    torch.nn.init.zeros_(output_layer.bias)
    with torch.no_grad():
        parameter_norm = (
            output_layer.weight.norm() + output_layer.bias.norm()).item()
    if parameter_norm != 0.0:
        raise RuntimeError("failed to zero-initialize residual actor mean")
    print("[phase2] residual actor mean initialized exactly to zero", flush=True)


def _save_initial_checkpoint(runner, torch, path, args, env_cfg):
    torch.save(
        {
            "model_state_dict": runner.alg.policy.state_dict(),
            "optimizer_state_dict": runner.alg.optimizer.state_dict(),
            "iter": 0,
            "infos": {
                "base_checkpoint": env_cfg.phase1_checkpoint,
                "residual_scale": env_cfg.residual_scale,
                "num_envs": args.num_envs,
            },
        },
        path,
    )
    print(f"[phase2] saved zero-residual baseline {path}", flush=True)


if __name__ == "__main__":
    main()
