#!/usr/bin/env python3
"""Paired closed-loop evaluation for Phase-2 residual checkpoints.

Every candidate sees exactly the same MPC hand-off states, joint perturbations,
base-motion conditions, and simulation horizon.  The zero-initialized
``model_init.pt`` is the frozen Phase-1 baseline under the same inertial load.

Example:
    python scripts/eval_phase2_checkpoints.py \
        --num-envs 1024 --steps 400 \
        --checkpoints model_init.pt,model_500.pt,model_750.pt,model_999.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ROOT = Path("/home/windylab/code/isaac_arm_rl")
DEFAULT_RUN_DIR = (
    ROOT / "logs/phase2_residual/20260724_231620_formal4096"
)
SEED_CSVS = (
    Path(
        "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
        "moving_base_mpc_loop_20260724_051521.csv"
    ),
    Path(
        "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
        "moving_base_mpc_loop_20260724_051912.csv"
    ),
    Path(
        "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
        "moving_base_mpc_loop_20260724_052123.csv"
    ),
)
CONDITION_NAMES = ("easy", "medium", "hard")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--q-noise", type=float, default=0.002)
    parser.add_argument(
        "--checkpoints",
        default=(
            "model_init.pt,model_500.pt,model_750.pt,"
            "model_775.pt,model_900.pt,model_999.pt"
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--residual-scales",
        default="",
        help=(
            "optional comma-separated residual scales, one per checkpoint; "
            "a single value is broadcast to every checkpoint"
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=("random_handoff", "phase_zero"),
        default="random_handoff",
        help=(
            "random_handoff samples matched phases from clean MPC CSVs; "
            "phase_zero uses each condition's first logged state"
        ),
    )
    parser.add_argument(
        "--no-inertia",
        action="store_true",
        help="disable the Phase-2 fictitious inertial forces",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "logs/phase2_residual/eval",
    )
    args = parser.parse_args()
    if args.num_envs < 3:
        parser.error("--num-envs must be at least 3")
    if args.steps <= 0 or args.q_noise < 0.0:
        parser.error("--steps must be positive and --q-noise non-negative")
    checkpoints = []
    for value in args.checkpoints.split(","):
        path = Path(value.strip())
        if not path.is_absolute():
            path = args.run_dir / path
        if not path.is_file():
            parser.error(f"checkpoint not found: {path}")
        checkpoints.append(path)
    args.checkpoint_paths = checkpoints
    if args.residual_scales:
        try:
            residual_scales = [
                float(value.strip())
                for value in args.residual_scales.split(",")
            ]
        except ValueError as error:
            parser.error(f"invalid --residual-scales: {error}")
        if len(residual_scales) == 1:
            residual_scales *= len(checkpoints)
        if len(residual_scales) != len(checkpoints):
            parser.error(
                "--residual-scales requires one value or one value per "
                "checkpoint"
            )
        if any(value <= 0.0 for value in residual_scales):
            parser.error("--residual-scales values must be positive")
        args.residual_scale_values = residual_scales
    else:
        args.residual_scale_values = [None] * len(checkpoints)
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
        ResidualArmTrackEnvCfg,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    cfg = ResidualArmTrackEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.inertial_coupling = not args.no_inertia
    # Do not auto-reset a failed controller: resetting would hide its tail error.
    cfg.divergence_threshold_m = 1.0e6
    cfg.episode_length_s = max(
        cfg.episode_length_s,
        (args.steps + 10) / 50.0,
    )
    env = ResidualArmTrackEnv(cfg)
    device = env.device

    group = torch.arange(args.num_envs, device=device) % 3
    scenario = _build_scenario(
        args,
        group,
        env,
        torch,
        np,
    )

    dummy_observations = {
        "policy": torch.zeros(
            args.num_envs, cfg.observation_space, device=device
        )
    }
    actor_critic = ActorCritic(
        dummy_observations,
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

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode = "noinertia" if args.no_inertia else "inertia"
    output_dir = args.output_dir / (
        f"{timestamp}_{args.protocol}_{mode}_envs{args.num_envs}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    per_env_outputs = {}
    baseline_per_env_mean = None
    baseline_name = None
    for index, checkpoint_path in enumerate(args.checkpoint_paths):
        residual_scale = args.residual_scale_values[index]
        if residual_scale is None:
            residual_scale = float(cfg.residual_scale)
        env.cfg.residual_scale = residual_scale
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        actor_critic.load_state_dict(checkpoint["model_state_dict"])
        actor_critic.eval()
        name = checkpoint_path.stem
        if args.residual_scales:
            scale_suffix = f"{residual_scale:.3f}".replace(".", "p")
            name = f"{name}_s{scale_suffix}"
        observations, initial_error = _reset_to_scenario(
            env, scenario, group, args, torch
        )
        result = _rollout(
            env,
            observations,
            actor_critic,
            group,
            args,
            torch,
        )
        result["name"] = name
        result["checkpoint"] = str(checkpoint_path)
        result["checkpoint_iter"] = int(checkpoint.get("iter", -1))
        result["residual_scale"] = residual_scale
        result["initial_error_mm"] = _distribution_metrics(
            initial_error * 1000.0,
            torch,
        )
        per_env_mean = result.pop("_per_env_mean_mm")
        per_env_max = result.pop("_per_env_max_mm")
        per_env_outputs[f"{name}_mean_mm"] = per_env_mean.cpu().numpy()
        per_env_outputs[f"{name}_max_mm"] = per_env_max.cpu().numpy()
        if index == 0:
            baseline_per_env_mean = per_env_mean
            baseline_name = name
            result["paired_vs_baseline"] = {
                "baseline": name,
                "mean_delta_mm": 0.0,
                "median_delta_mm": 0.0,
                "win_fraction": 0.0,
                "worse_by_1mm_fraction": 0.0,
            }
        else:
            delta = per_env_mean - baseline_per_env_mean
            result["paired_vs_baseline"] = {
                "baseline": baseline_name,
                "mean_delta_mm": float(delta.mean().item()),
                "median_delta_mm": float(
                    torch.quantile(delta, 0.5).item()
                ),
                "win_fraction": float((delta < 0.0).float().mean().item()),
                "worse_by_1mm_fraction": float(
                    (delta > 1.0).float().mean().item()
                ),
            }
        summaries.append(result)
        _print_result(result)

    document = {
        "schema_version": 1,
        "protocol": args.protocol,
        "inertial_coupling": not args.no_inertia,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seed": args.seed,
        "q_noise_rad": args.q_noise,
        "conditions": list(CONDITION_NAMES),
        "condition_counts": [
            int((group == gid).sum().item()) for gid in range(3)
        ],
        "baseline": baseline_name,
        "results": summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "per_env_metrics.npz",
        condition_id=group.cpu().numpy(),
        **per_env_outputs,
    )
    print(f"[phase2-eval] SUMMARY {summary_path}", flush=True)
    env.close()
    sys.stdout.flush()
    del app
    os._exit(0)


def _build_scenario(args, group, env, torch, np):
    """Build one immutable, paired set of initial conditions."""
    num_envs = args.num_envs
    device = env.device
    q = torch.empty(num_envs, 7, device=device)
    dq = torch.empty_like(q)
    previous_action = torch.zeros_like(q)
    start = torch.zeros(num_envs, dtype=torch.long, device=device)
    rng = np.random.default_rng(args.seed)
    group_cpu = group.cpu().numpy()

    for condition_id, path in enumerate(SEED_CSVS):
        rows = list(csv.DictReader(path.open()))
        ids = np.flatnonzero(group_cpu == condition_id)
        if args.protocol == "random_handoff":
            picks = rng.integers(0, len(rows), size=len(ids))
        else:
            picks = np.zeros(len(ids), dtype=np.int64)
        q_np = np.asarray(
            [
                [
                    float(rows[row][f"q_meas_{joint}"])
                    for joint in range(1, 8)
                ]
                for row in picks
            ],
            dtype=np.float32,
        )
        dq_np = np.asarray(
            [
                [
                    float(rows[row][f"dq_meas_{joint}"])
                    for joint in range(1, 8)
                ]
                for row in picks
            ],
            dtype=np.float32,
        )
        previous_rows = np.maximum(picks - 1, 0)
        if "u_exec_1" in rows[0]:
            previous_np = np.asarray(
                [
                    [
                        float(rows[row][f"u_exec_{joint}"]) / 0.02
                        for joint in range(1, 8)
                    ]
                    for row in previous_rows
                ],
                dtype=np.float32,
            )
        else:
            previous_np = np.asarray(
                [
                    [
                        (
                            float(rows[row][f"q_pub_{joint}"])
                            - float(rows[row][f"q_meas_{joint}"])
                        )
                        / 0.02
                        for joint in range(1, 8)
                    ]
                    for row in previous_rows
                ],
                dtype=np.float32,
            )
        if args.protocol == "random_handoff":
            start_np = np.asarray(
                [
                    round(float(rows[row]["t_solve"]) * 50.0)
                    % env.targets.T
                    for row in picks
                ],
                dtype=np.int64,
            )
        else:
            start_np = np.zeros(len(ids), dtype=np.int64)
        ids_t = torch.as_tensor(ids, device=device)
        q[ids_t] = torch.as_tensor(q_np, device=device)
        dq[ids_t] = torch.as_tensor(dq_np, device=device)
        previous_action[ids_t] = torch.as_tensor(
            previous_np.clip(-1.0, 1.0),
            device=device,
        )
        start[ids_t] = torch.as_tensor(start_np, device=device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1)
    noise = torch.randn(
        num_envs,
        7,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    q += args.q_noise * noise
    return {
        "q": q,
        "dq": dq,
        "previous_action": previous_action,
        "start": start,
    }


def _reset_to_scenario(env, scenario, group, args, torch):
    env.reset(seed=args.seed)
    d = env._dist
    d["wave_t"][:] = 0
    d["wave_a"][:] = 1
    d["amp_t"][:] = torch.where(
        group == 2,
        torch.tensor(0.01, device=env.device),
        torch.tensor(0.02, device=env.device),
    )
    d["amp_z"][:] = torch.where(
        group == 0,
        torch.tensor(0.0, device=env.device),
        torch.where(
            group == 1,
            torch.tensor(0.02, device=env.device),
            torch.tensor(0.01, device=env.device),
        ),
    )
    d["z_phase"][:] = math.pi / 3.0
    d["amp_a"][:] = math.radians(0.1)
    d["freq_t"][:] = torch.where(
        group == 2,
        torch.tensor(2.0, device=env.device),
        torch.tensor(1.0, device=env.device),
    )
    d["freq_a"].copy_(d["freq_t"])
    d["phase"][:] = 0.0

    env._condition_id.copy_(group)
    env._start.copy_(scenario["start"])
    env.episode_length_buf[:] = 0
    env._last_executed_action.copy_(scenario["previous_action"])
    env._prev_action.copy_(scenario["previous_action"])
    env.actions.zero_()
    env._history_needs_reset[:] = True
    env._base_action.zero_()
    env._residual_action.zero_()
    env._previous_residual_action.zero_()
    env._residual_delta.zero_()
    env._final_action.zero_()
    env.robot.write_joint_state_to_sim(scenario["q"], scenario["dq"])
    env.robot.set_joint_position_target(scenario["q"])
    env.scene.write_data_to_sim()
    env.sim.forward()
    observations = env._get_observations()
    initial_error = torch.linalg.norm(
        env._p_tgt - env._p_ee,
        dim=-1,
    )
    return observations, initial_error


def _rollout(env, observations, actor_critic, group, args, torch):
    errors = []
    orientation_errors = []
    residuals = []
    saturations = []
    raw_clipped = []
    inference_ms = []
    for step in range(args.steps):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.inference_mode():
            raw_residual = actor_critic.act_inference(observations)
        torch.cuda.synchronize()
        inference_ms.append((time.perf_counter() - start_time) * 1000.0)
        observations, _, _, _, _ = env.step(raw_residual)
        error = torch.linalg.norm(env._p_tgt - env._p_ee, dim=-1)
        rotation_error = torch.bmm(
            env._R_ee.transpose(1, 2),
            env._R_tgt,
        )
        cosine = (
            rotation_error[:, 0, 0]
            + rotation_error[:, 1, 1]
            + rotation_error[:, 2, 2]
            - 1.0
        ) * 0.5
        orientation_error = torch.arccos(cosine.clamp(-1.0, 1.0))
        errors.append(error)
        orientation_errors.append(orientation_error)
        residuals.append(env._residual_action.clone())
        saturations.append(
            (env._final_action.abs() > 0.999).float().mean(dim=-1)
        )
        raw_clipped.append(
            (raw_residual.abs() > 1.0).float().mean(dim=-1)
        )
        if step % 100 == 0:
            print(
                f"[phase2-eval] step={step:3d}/{args.steps} "
                f"mean={error.mean().item()*1000.0:.3f}mm "
                f"max={error.max().item()*1000.0:.3f}mm",
                flush=True,
            )

    errors = torch.stack(errors) * 1000.0
    orientation_errors = torch.rad2deg(torch.stack(orientation_errors))
    residuals = torch.stack(residuals)
    saturations = torch.stack(saturations)
    raw_clipped = torch.stack(raw_clipped)
    per_env_mean = errors.mean(dim=0)
    per_env_max = errors.max(dim=0).values

    condition_results = {}
    for condition_id, name in enumerate(CONDITION_NAMES):
        mask = group == condition_id
        values = errors[:, mask]
        orientation_values = orientation_errors[:, mask]
        condition_results[name] = {
            **_distribution_metrics(values.flatten(), torch),
            "env_max_p95_mm": float(
                torch.quantile(values.max(dim=0).values, 0.95).item()
            ),
            "diverged_fraction": float(
                (values.max(dim=0).values > 50.0).float().mean().item()
            ),
            "orientation_mean_deg": float(
                orientation_values.mean().item()
            ),
            "orientation_p95_deg": float(
                torch.quantile(
                    orientation_values.flatten(), 0.95
                ).item()
            ),
        }

    latency = torch.as_tensor(inference_ms)
    return {
        "overall": {
            **_distribution_metrics(errors.flatten(), torch),
            "env_max_p95_mm": float(
                torch.quantile(per_env_max, 0.95).item()
            ),
            "diverged_fraction": float(
                (per_env_max > 50.0).float().mean().item()
            ),
            "orientation_mean_deg": float(
                orientation_errors.mean().item()
            ),
            "orientation_p95_deg": float(
                torch.quantile(
                    orientation_errors.flatten(), 0.95
                ).item()
            ),
        },
        "conditions": condition_results,
        "residual_rms": float(
            torch.sqrt(residuals.square().mean()).item()
        ),
        "residual_abs_p95": float(
            torch.quantile(residuals.abs().flatten(), 0.95).item()
        ),
        "final_action_saturation_fraction": float(
            saturations.mean().item()
        ),
        "raw_residual_clip_fraction": float(raw_clipped.mean().item()),
        "batch_inference_ms_mean": float(latency.mean().item()),
        "batch_inference_ms_p95": float(
            torch.quantile(latency, 0.95).item()
        ),
        "_per_env_mean_mm": per_env_mean,
        "_per_env_max_mm": per_env_max,
    }


def _distribution_metrics(values, torch):
    return {
        "mean_mm": float(values.mean().item()),
        "rms_mm": float(torch.sqrt(values.square().mean()).item()),
        "p95_mm": float(torch.quantile(values, 0.95).item()),
        "p99_mm": float(torch.quantile(values, 0.99).item()),
        "max_mm": float(values.max().item()),
    }


def _print_result(result):
    overall = result["overall"]
    paired = result["paired_vs_baseline"]
    print(
        "[phase2-eval] RESULT "
        f"name={result['name']} "
        f"mean={overall['mean_mm']:.3f}mm "
        f"rms={overall['rms_mm']:.3f}mm "
        f"p95={overall['p95_mm']:.3f}mm "
        f"p99={overall['p99_mm']:.3f}mm "
        f"max={overall['max_mm']:.3f}mm "
        f"ori_mean={overall['orientation_mean_deg']:.3f}deg "
        f"ori_p95={overall['orientation_p95_deg']:.3f}deg "
        f"diverged={100.0*overall['diverged_fraction']:.3f}% "
        f"residual_rms={result['residual_rms']:.4f} "
        f"saturation={100.0*result['final_action_saturation_fraction']:.3f}% "
        f"delta={paired['mean_delta_mm']:+.3f}mm "
        f"wins={100.0*paired['win_fraction']:.1f}%",
        flush=True,
    )
    for name in CONDITION_NAMES:
        metrics = result["conditions"][name]
        print(
            "[phase2-eval] CONDITION "
            f"model={result['name']} condition={name} "
            f"mean={metrics['mean_mm']:.3f}mm "
            f"p95={metrics['p95_mm']:.3f}mm "
            f"p99={metrics['p99_mm']:.3f}mm "
            f"max={metrics['max_mm']:.3f}mm "
            f"diverged={100.0*metrics['diverged_fraction']:.3f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
