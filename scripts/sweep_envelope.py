#!/usr/bin/env python3
"""Sweep the selected Phase-2 controller over a 3-D base-motion (R, f) grid.

Each parallel Isaac environment owns one grid cell.  The policy receives the
continuous radius/frequency features, but a fixed Phase-1 action gain of 0.90
is used throughout so the heatmap has no hidden discrete gain boundaries.
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
DEFAULT_CONFIG = ROOT / "config/phase2_selected_policy.json"
DEFAULT_ACTOR = ROOT / "exports/phase2_robust/residual_actor.ts"
DEFAULT_OUTPUT = ROOT / "reports/phase3"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--rmax", type=float, default=0.05)
    parser.add_argument("--fmin", type=float, default=0.25)
    parser.add_argument("--fmax", type=float, default=5.0)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="settling ticks excluded from metrics",
    )
    parser.add_argument("--measure-steps", type=int, default=200)
    parser.add_argument("--phase1-gain", type=float, default=0.90)
    parser.add_argument("--rms-limit-mm", type=float, default=2.0)
    parser.add_argument("--max-limit-mm", type=float, default=10.0)
    parser.add_argument("--orientation-limit-deg", type=float, default=7.0)
    parser.add_argument("--tag", default="phase2_robust_sphere")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.rmax <= 0.0:
        parser.error("--rmax must be positive")
    if args.fmin <= 0.0 or args.fmax <= args.fmin:
        parser.error("require 0 < --fmin < --fmax")
    if args.grid < 4:
        parser.error("--grid must be at least 4")
    if args.warmup_steps < 0 or args.measure_steps <= 0:
        parser.error("warmup must be non-negative and measurement positive")
    if not 0.0 < args.phase1_gain <= 1.0:
        parser.error("--phase1-gain must be in (0, 1]")
    if min(
        args.rms_limit_mm,
        args.max_limit_mm,
        args.orientation_limit_deg,
    ) <= 0.0:
        parser.error("feasibility limits must be positive")
    return args


def _load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _step_plant(env, residual_action):
    env._pre_physics_step(residual_action)
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    env.episode_length_buf += 1
    env.common_step_counter += 1
    return env._get_observations()["policy"]


def main():
    args = parse_args()
    selection = _load_json(args.config)
    phase1_selection = _load_json(selection["base_policy_config"])

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT))
    from env.residual_arm_track_env import (
        ResidualArmTrackEnv,
        ResidualArmTrackEnvCfg,
    )

    torch.manual_seed(20260725)
    torch.cuda.manual_seed_all(20260725)
    grid = args.grid
    num_envs = grid * grid
    cfg = ResidualArmTrackEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.phase1_checkpoint = phase1_selection["checkpoint"]
    cfg.phase1_ddim_steps = int(
        phase1_selection["deployment"]["ddim_steps"]
    )
    cfg.residual_scale = float(selection["deployment"]["residual_scale"])
    cfg.inertial_coupling = True
    cfg.condition_jitter_fraction = 0.0
    cfg.episode_length_s = (
        args.warmup_steps + args.measure_steps + 10
    ) / 50.0
    env = ResidualArmTrackEnv(cfg)
    device = env.device
    actor = torch.jit.load(str(args.actor), map_location=device).eval()

    radius = torch.linspace(0.0, args.rmax, grid, device=device)
    frequency = torch.linspace(
        args.fmin, args.fmax, grid, device=device
    )
    radius_grid, frequency_grid = torch.meshgrid(
        radius, frequency, indexing="ij"
    )
    radius_flat = radius_grid.reshape(-1)
    frequency_flat = frequency_grid.reshape(-1)

    env.reset()
    sweep = env.targets.set_sphere_sweep(
        radius_flat, frequency_flat, att_deg=0.0
    )
    for key, value in sweep.items():
        env._dist[key].copy_(value)
    env._start.zero_()
    # The condition id only selects a Phase-1 gain.  Force every entry to the
    # medium slot and overwrite that slot with one documented continuous-sweep
    # gain, avoiding an artificial boundary in the map.
    env._condition_id.fill_(1)
    env._phase1_gain[1] = args.phase1_gain
    q_seed = env._seed_q[1].expand(num_envs, -1).clone()
    dq_seed = env._seed_dq[1].expand(num_envs, -1).clone()
    env._history_needs_reset.fill_(True)
    env._base_action.zero_()
    env._residual_action.zero_()
    env._previous_residual_action.zero_()
    env._residual_delta.zero_()
    env._final_action.zero_()
    env._last_executed_action.zero_()
    env._prev_action.zero_()
    env.robot.write_joint_state_to_sim(q_seed, dq_seed)
    env.robot.set_joint_position_target(q_seed)
    env.scene.write_data_to_sim()
    env.sim.forward()
    observation = env._get_observations()["policy"]

    sum_squared_position = torch.zeros(num_envs, device=device)
    max_position = torch.zeros(num_envs, device=device)
    sum_orientation = torch.zeros(num_envs, device=device)
    max_orientation = torch.zeros(num_envs, device=device)
    saturation_sum = torch.zeros(num_envs, device=device)
    total_steps = args.warmup_steps + args.measure_steps
    started = time.monotonic()
    for step in range(total_steps):
        with torch.inference_mode():
            residual_action = actor(observation)
        observation = _step_plant(env, residual_action)
        if step >= args.warmup_steps:
            position_error = torch.linalg.norm(
                env._p_tgt - env._p_ee, dim=-1
            )
            rotation_error = torch.bmm(
                env._R_ee.transpose(1, 2), env._R_tgt
            )
            cosine = (
                rotation_error[:, 0, 0]
                + rotation_error[:, 1, 1]
                + rotation_error[:, 2, 2]
                - 1.0
            ) * 0.5
            orientation_error = torch.rad2deg(
                torch.arccos(torch.clamp(cosine, -1.0, 1.0))
            )
            sum_squared_position += position_error.square()
            max_position = torch.maximum(max_position, position_error)
            sum_orientation += orientation_error
            max_orientation = torch.maximum(
                max_orientation, orientation_error
            )
            saturation_sum += (
                env._final_action.abs() > 0.999
            ).float().mean(dim=-1)
        if step % 50 == 0 or step + 1 == total_steps:
            print(
                f"[envelope] step={step + 1}/{total_steps} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    rms_position_mm = (
        torch.sqrt(sum_squared_position / args.measure_steps) * 1000.0
    )
    max_position_mm = max_position * 1000.0
    mean_orientation_deg = sum_orientation / args.measure_steps
    saturation_fraction = saturation_sum / args.measure_steps
    feasible = (
        (rms_position_mm <= args.rms_limit_mm)
        & (max_position_mm <= args.max_limit_mm)
        & (mean_orientation_deg <= args.orientation_limit_deg)
    )

    radius_np = radius.detach().cpu().numpy()
    frequency_np = frequency.detach().cpu().numpy()
    rms_np = rms_position_mm.reshape(grid, grid).detach().cpu().numpy()
    max_np = max_position_mm.reshape(grid, grid).detach().cpu().numpy()
    orientation_np = (
        mean_orientation_deg.reshape(grid, grid).detach().cpu().numpy()
    )
    orientation_max_np = (
        max_orientation.reshape(grid, grid).detach().cpu().numpy()
    )
    saturation_np = (
        saturation_fraction.reshape(grid, grid).detach().cpu().numpy()
    )
    feasible_np = feasible.reshape(grid, grid).detach().cpu().numpy()

    max_observed_feasible_radius = []
    contiguous_feasible_radius = []
    for frequency_index in range(grid):
        valid_indices = np.flatnonzero(feasible_np[:, frequency_index])
        max_observed_feasible_radius.append(
            float(radius_np[valid_indices[-1]])
            if valid_indices.size
            else None
        )
        column = feasible_np[:, frequency_index]
        if not column[0]:
            contiguous_feasible_radius.append(None)
        else:
            first_invalid = np.flatnonzero(~column)
            last_index = (
                int(first_invalid[0]) - 1
                if first_invalid.size
                else len(radius_np) - 1
            )
            contiguous_feasible_radius.append(
                float(radius_np[last_index])
            )
    elapsed = time.monotonic() - started
    summary = {
        "schema_version": 1,
        "controller": "phase1_diffusion_plus_phase2_residual",
        "checkpoint": selection["checkpoint"],
        "residual_scale": cfg.residual_scale,
        "disturbance": "3d_sphere_translation",
        "inertial_coupling": True,
        "grid": grid,
        "num_envs": num_envs,
        "radius_range_m": [0.0, args.rmax],
        "frequency_range_hz": [args.fmin, args.fmax],
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "phase1_gain": args.phase1_gain,
        "feasibility_rule": {
            "position_rms_mm_max": args.rms_limit_mm,
            "position_max_mm_max": args.max_limit_mm,
            "orientation_mean_deg_max": args.orientation_limit_deg,
        },
        "feasible_grid_fraction": float(feasible_np.mean()),
        "position_rms_mm": {
            "min": float(rms_np.min()),
            "median": float(np.median(rms_np)),
            "max": float(rms_np.max()),
        },
        "position_max_mm": {
            "min": float(max_np.min()),
            "median": float(np.median(max_np)),
            "max": float(max_np.max()),
        },
        "orientation_mean_deg": {
            "min": float(orientation_np.min()),
            "median": float(np.median(orientation_np)),
            "max": float(orientation_np.max()),
        },
        "final_action_saturation_fraction": {
            "mean": float(saturation_np.mean()),
            "max": float(saturation_np.max()),
        },
        "frequency_hz": frequency_np.tolist(),
        "max_observed_feasible_radius_m_by_frequency": (
            max_observed_feasible_radius
        ),
        "contiguous_feasible_radius_from_zero_m_by_frequency": (
            contiguous_feasible_radius
        ),
        "elapsed_seconds": elapsed,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / f"envelope_{args.tag}.npz"
    json_path = args.output_dir / f"envelope_{args.tag}.json"
    png_path = args.output_dir / f"envelope_{args.tag}.png"
    np.savez(
        npz_path,
        radius_m=radius_np,
        frequency_hz=frequency_np,
        position_rms_mm=rms_np,
        position_max_mm=max_np,
        orientation_mean_deg=orientation_np,
        orientation_max_deg=orientation_max_np,
        saturation_fraction=saturation_np,
        feasible=feasible_np,
    )
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    _plot(
        radius_np,
        frequency_np,
        rms_np,
        max_np,
        feasible_np,
        png_path,
    )
    print(json.dumps(summary, indent=2))
    print(f"[envelope] wrote {npz_path}")
    print(f"[envelope] wrote {json_path}")
    print(f"[envelope] wrote {png_path}")
    sys.stdout.flush()
    del app
    os._exit(0)


def _plot(radius, frequency, rms, maximum, feasible, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, values, title, levels in (
        (axes[0], rms, "Position RMS error (mm)", [1, 2, 5, 10]),
        (axes[1], maximum, "Position maximum error (mm)", [2, 5, 10, 20]),
    ):
        image = axis.pcolormesh(
            radius * 100.0,
            frequency,
            values.T,
            shading="auto",
            cmap="turbo",
        )
        valid_levels = [
            value
            for value in levels
            if values.min() < value < values.max()
        ]
        if valid_levels:
            contours = axis.contour(
                radius * 100.0,
                frequency,
                values.T,
                levels=valid_levels,
                colors="white",
                linewidths=1.0,
            )
            axis.clabel(contours, fmt="%g mm", fontsize=8)
        axis.contour(
            radius * 100.0,
            frequency,
            feasible.T.astype(float),
            levels=[0.5],
            colors="black",
            linewidths=2.0,
        )
        axis.set_xlabel("3-D base radius R (cm)")
        axis.set_ylabel("base frequency f (Hz)")
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
    figure.suptitle(
        "Phase-2 robust policy; black boundary = declared feasible envelope"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


if __name__ == "__main__":
    main()
