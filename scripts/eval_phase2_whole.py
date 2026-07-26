#!/usr/bin/env python3
"""Paired full-cycle evaluation of whole-task Phase-2 residual checkpoints.

Every checkpoint sees exactly the same start phases, state-bank hand-off
states, joint noise, and base-motion condition; ``model_init.pt`` (zero
residual) is the Phase-1 baseline under the identical inertial plant.  The
horizon covers a complete 73.039-s cycle regardless of start phase.

    ./env_isaaclab/bin/python scripts/eval_phase2_whole.py \
        --run-dir logs/phase2_whole_residual/<run> \
        --checkpoints model_init.pt,model_950.pt --steps 3700
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ROOT = Path("/home/windylab/code/isaac_arm_rl")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=3700)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--q-noise", type=float, default=0.001)
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", default="model_init.pt")
    parser.add_argument("--residual-scale", type=float, default=-1.0)
    parser.add_argument("--phase1-checkpoint", default="")
    parser.add_argument("--no-inertia", action="store_true")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()
    paths = []
    for value in args.checkpoints.split(","):
        path = Path(value.strip())
        if not path.is_absolute():
            path = args.run_dir / path
        if not path.is_file():
            parser.error(f"checkpoint not found: {path}")
        paths.append(path)
    args.checkpoint_paths = paths
    return args


def main():
    args = parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    import numpy as np
    import torch
    from rsl_rl.modules import ActorCritic

    sys.path.insert(0, str(ROOT))
    from env.residual_arm_track_env import (  # noqa: E402
        ResidualArmTrackEnv,
        WholeResidualArmTrackEnvCfg,
        _load_state_bank,
        _draw_bank_states,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = WholeResidualArmTrackEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.inertial_coupling = not args.no_inertia
    cfg.divergence_threshold_m = 1.0e6      # never hide tail errors via reset
    cfg.episode_length_s = (args.steps + 10) / 50.0
    if args.residual_scale > 0.0:
        cfg.residual_scale = args.residual_scale
    if args.phase1_checkpoint:
        cfg.phase1_checkpoint = args.phase1_checkpoint
    env = ResidualArmTrackEnv(cfg)
    device = env.device
    T = env.targets.T

    # ---- frozen scenario shared by all checkpoints
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    start = torch.randint(0, T, (args.num_envs,), device=device,
                          generator=generator)
    bank = _load_state_bank(cfg.state_bank_path, device)
    q0, dq0, prev0 = _draw_bank_states(bank, start)
    q0 = q0 + args.q_noise * torch.randn(
        q0.shape, device=device, generator=generator)
    dist = env.targets.sample_whole_disturbance(
        args.num_envs, jitter=args.jitter)

    actor_critic = ActorCritic(
        {"policy": torch.zeros(
            args.num_envs, cfg.observation_space, device=device)},
        {"policy": ["policy"], "critic": ["policy"]},
        cfg.action_space,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        init_noise_std=0.15,
        noise_std_type="log",
    ).to(device)
    actor_critic.eval()

    def reset_scenario():
        env.reset()
        for key, value in dist.items():
            env._dist[key].copy_(value)
        env._start.copy_(start)
        env._condition_id.zero_()
        env.episode_length_buf.zero_()
        env.robot.write_joint_state_to_sim(q0.clone(), dq0.clone())
        env.robot.set_joint_position_target(q0.clone())
        env.sync_command_state()
        env._last_executed_action.copy_(prev0)
        env._prev_action.copy_(prev0)
        env._history_needs_reset[:] = True
        env._base_action.zero_()
        env._previous_residual_action.zero_()
        return env._get_observations()["policy"]

    task_names = ("circle", "triangle", "windylab", "sine", "square")
    all_results = {}
    baseline_env_mean = None
    for path in args.checkpoint_paths:
        state = torch.load(path, map_location=device, weights_only=False)
        actor_critic.load_state_dict(state["model_state_dict"])
        actor_critic.eval()
        obs = reset_scenario()
        pos_err = torch.empty(args.steps, args.num_envs, device=device)
        ori_err = torch.empty(args.steps, args.num_envs, device=device)
        pen_mask = torch.empty(
            args.steps, args.num_envs, dtype=torch.bool, device=device)
        task_ids = torch.empty(
            args.steps, args.num_envs, dtype=torch.long, device=device)
        residual_sq = 0.0
        saturation = 0.0
        t_start = time.time()
        for step in range(args.steps):
            with torch.no_grad():
                residual = actor_critic.act_inference({"policy": obs})
            idx = (env._start + env.episode_length_buf) % T
            obs = env.step(residual)[0]["policy"]
            pos_err[step] = torch.linalg.norm(
                env._p_tgt - env._p_ee, dim=-1)
            Rerr = torch.bmm(env._R_ee.transpose(1, 2), env._R_tgt)
            cos = ((Rerr[:, 0, 0] + Rerr[:, 1, 1] + Rerr[:, 2, 2]) - 1) * 0.5
            ori_err[step] = torch.arccos(torch.clamp(cos, -1.0, 1.0))
            pen_mask[step] = env.targets.pen[idx]
            task_ids[step] = env.targets.task_id[idx]
            residual_sq += env._residual_action.square().mean().item()
            saturation += (
                env._final_action.abs() > 0.999).float().mean().item()
            if step % 500 == 0:
                print(f"[p2-whole] {path.name} step={step:4d} "
                      f"mean={pos_err[step].mean()*1000:.3f}mm "
                      f"p95={torch.quantile(pos_err[step], .95)*1000:.3f}mm",
                      flush=True)

        def stats(mask, name):
            v = (pos_err if mask is None else pos_err[mask]).flatten() * 1000.0
            o = (ori_err if mask is None else ori_err[mask]).flatten()
            if v.numel() == 0:
                return None
            entry = {
                "name": name,
                "mean_mm": v.mean().item(),
                "rmse_mm": v.square().mean().sqrt().item(),
                "p95_mm": torch.quantile(v, .95).item(),
                "p99_mm": torch.quantile(v, .99).item(),
                "max_mm": v.max().item(),
                "ori_mean_deg": torch.rad2deg(o.mean()).item(),
                "ori_p95_deg": torch.rad2deg(
                    torch.quantile(o, .95)).item(),
                "ori_max_deg": torch.rad2deg(o.max()).item(),
                "n": int(v.numel()),
            }
            print(f"[p2-whole] {path.name} {name}: "
                  f"mean={entry['mean_mm']:.3f} rmse={entry['rmse_mm']:.3f} "
                  f"p95={entry['p95_mm']:.3f} p99={entry['p99_mm']:.3f} "
                  f"max={entry['max_mm']:.3f}mm "
                  f"ori={entry['ori_mean_deg']:.2f}/"
                  f"{entry['ori_p95_deg']:.2f}deg", flush=True)
            return entry

        breakdown = [stats(None, "overall"), stats(pen_mask, "pen_down")]
        for tid, tname in enumerate(task_names):
            breakdown.append(stats(task_ids == tid, f"task_{tname}"))
        breakdown.append(stats(task_ids < 0, "transition"))
        per_env_mean = pos_err.mean(dim=0)
        per_env_max = pos_err.max(dim=0).values
        diverged = (per_env_max > 0.05).float().mean().item()
        summary = {
            "checkpoint": str(path),
            "residual_scale": float(env.cfg.residual_scale),
            "breakdown": [b for b in breakdown if b is not None],
            "diverged_fraction": diverged,
            "residual_rms": (residual_sq / args.steps) ** 0.5,
            "saturation_fraction": saturation / args.steps,
            "wall_sec": time.time() - t_start,
        }
        if baseline_env_mean is None:
            baseline_env_mean = per_env_mean
            summary["paired_improvement_fraction"] = None
        else:
            improved = (per_env_mean < baseline_env_mean).float().mean()
            summary["paired_improvement_fraction"] = improved.item()
            print(f"[p2-whole] {path.name} paired: improves "
                  f"{100*improved:.1f}% of envs vs "
                  f"{args.checkpoint_paths[0].name}", flush=True)
        print(f"[p2-whole] {path.name} diverged={100*diverged:.2f}% "
              f"residual_rms={summary['residual_rms']:.4f} "
              f"saturation={summary['saturation_fraction']:.4f}", flush=True)
        all_results[path.name] = summary

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as stream:
            json.dump({
                "num_envs": args.num_envs, "steps": args.steps,
                "seed": args.seed, "jitter": args.jitter,
                "inertial": not args.no_inertia,
                "schema_version": "whole_v2_oneshot_73.0394904589",
                "results": all_results,
            }, stream, indent=2)
        print(f"[p2-whole] wrote {args.out_json}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
