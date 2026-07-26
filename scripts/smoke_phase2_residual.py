#!/usr/bin/env python3
"""Small headless safety test for the Phase-2 residual environment."""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ROOT = "/home/windylab/code/isaac_arm_rl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.steps <= 0:
        parser.error("--num-envs and --steps must be positive")

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    import torch

    sys.path.insert(0, ROOT)
    from env.residual_arm_track_env import (
        ResidualArmTrackEnv,
        ResidualArmTrackEnvCfg,
    )

    torch.manual_seed(20260725)
    torch.cuda.manual_seed_all(20260725)
    cfg = ResidualArmTrackEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = ResidualArmTrackEnv(cfg)
    observations, _ = env.reset()
    policy_observation = observations["policy"]
    assert policy_observation.shape == (args.num_envs, 54)
    assert torch.isfinite(policy_observation).all()

    errors = []
    max_zero_residual_difference = 0.0
    zero_residual = torch.zeros(args.num_envs, 7, device=env.device)
    for _ in range(args.steps):
        expected_final = env._base_action.clone()
        observations, rewards, terminated, truncated, _ = env.step(
            zero_residual)
        difference = torch.max(
            torch.abs(env._final_action - expected_final)).item()
        max_zero_residual_difference = max(
            max_zero_residual_difference, difference)
        error = torch.linalg.norm(
            env._p_tgt - env._p_ee, dim=-1) * 1000.0
        errors.append(error)
        assert observations["policy"].shape == (args.num_envs, 54)
        assert torch.isfinite(observations["policy"]).all()
        assert torch.isfinite(rewards).all()
        assert not terminated.any()
        assert not truncated.any()

    errors = torch.stack(errors)
    condition_counts = torch.bincount(
        env._condition_id, minlength=3).cpu().tolist()
    print(
        "[phase2-smoke] PASS "
        f"envs={args.num_envs} steps={args.steps} "
        f"conditions={condition_counts} "
        f"zero_residual_action_diff={max_zero_residual_difference:.3e} "
        f"error_mean={errors.mean().item():.2f}mm "
        f"error_max={errors.max().item():.2f}mm",
        flush=True,
    )
    sys.stdout.flush()
    del app
    os._exit(0)


if __name__ == "__main__":
    main()
