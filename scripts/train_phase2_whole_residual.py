#!/usr/bin/env python3
"""Train the whole-task bounded PPO residual on the frozen whole Phase-1 policy.

Same PPO recipe as the validated single-circle Phase-2, but on
WholeResidualArmTrackEnvCfg: random-phase 12-s windows over the exact
73.039-s whole cycle, state-bank resets, config-matched base disturbance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ROOT = "/home/windylab/code/isaac_arm_rl"
LOG_ROOT = f"{ROOT}/logs/phase2_whole_residual"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--tag", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--phase1-checkpoint", default="")
    parser.add_argument("--episode-length-s", type=float, default=12.0)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.steps_per_env,
           args.save_interval) <= 0:
        parser.error("numeric training arguments must be positive")
    if args.resume and not os.path.isfile(args.resume):
        parser.error(f"resume checkpoint not found: {args.resume}")
    return args


def main():
    args = parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    import torch

    sys.path.insert(0, ROOT)
    from env.residual_arm_track_env import (
        ResidualArmTrackEnv,
        WholeResidualArmTrackEnvCfg,
    )
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
        RslRlVecEnvWrapper,
    )
    from rsl_rl.runners import OnPolicyRunner

    torch.manual_seed(20260726)
    torch.cuda.manual_seed_all(20260726)

    env_cfg = WholeResidualArmTrackEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.residual_scale = args.residual_scale
    if args.phase1_checkpoint:
        env_cfg.phase1_checkpoint = args.phase1_checkpoint
    environment = ResidualArmTrackEnv(env_cfg)
    wrapped = RslRlVecEnvWrapper(environment, clip_actions=1.0)

    agent_cfg = RslRlOnPolicyRunnerCfg(
        seed=20260726,
        device="cuda:0",
        num_steps_per_env=args.steps_per_env,
        max_iterations=args.iterations,
        save_interval=args.save_interval,
        experiment_name="phase2_whole_residual",
        run_name=args.tag,
        logger="tensorboard",
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
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
        wrapped, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

    if args.resume:
        runner.load(args.resume)
        print(f"[phase2-whole] resumed {args.resume}", flush=True)
    else:
        _zero_initialize_residual_mean(runner, torch)

    manifest = {
        "phase": 2,
        "task": "whole_v2",
        "schema_version": "whole_v2_oneshot_73.0394904589",
        "algorithm": "bounded_residual_ppo_orientation_safe",
        "base_checkpoint": env_cfg.phase1_checkpoint,
        "residual_scale": env_cfg.residual_scale,
        "episode_length_s": env_cfg.episode_length_s,
        "max_joint_step": env_cfg.max_joint_step,
        "state_bank": env_cfg.state_bank_path,
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "steps_per_env": args.steps_per_env,
        "seed": 20260726,
    }
    with open(os.path.join(log_dir, "phase2_whole_manifest.json"),
              "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    print(f"[phase2-whole] START envs={args.num_envs} "
          f"iterations={args.iterations} "
          f"residual_scale={env_cfg.residual_scale} log_dir={log_dir}",
          flush=True)
    runner.learn(
        num_learning_iterations=args.iterations,
        init_at_random_ep_len=False,
    )
    print("[phase2-whole] DONE", flush=True)
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
    print("[phase2-whole] residual actor mean initialized exactly to zero",
          flush=True)


if __name__ == "__main__":
    main()
