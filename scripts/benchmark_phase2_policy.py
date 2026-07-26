#!/usr/bin/env python3
"""Benchmark the exported Phase-1 + Phase-2 controller without Isaac physics."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/home/windylab/code/isaac_arm_rl")
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "config/phase2_selected_policy.json"
DEFAULT_ACTOR = ROOT / "exports/phase2_robust/residual_actor.ts"
DEFAULT_OUTPUT = ROOT / "reports/phase3/phase2_policy_latency.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 10:
        parser.error("--warmup must be >= 1 and --iterations must be >= 10")
    return args


def _load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
    }


def main():
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.actor.is_file():
        raise FileNotFoundError(args.actor)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the deployment benchmark")

    selection = _load_json(args.config)
    phase1_selection = _load_json(selection["base_policy_config"])
    checkpoint = torch.load(
        phase1_selection["checkpoint"], map_location="cpu", weights_only=False
    )

    from distill.diffusion_policy import make_policy_from_checkpoint

    phase1 = make_policy_from_checkpoint(checkpoint, device)
    residual_actor = torch.jit.load(
        str(args.actor), map_location=device
    ).eval()
    obs_mean = torch.as_tensor(
        checkpoint["obs_mean"], device=device, dtype=torch.float32
    )
    obs_std = torch.as_tensor(
        checkpoint["obs_std"], device=device, dtype=torch.float32
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    # A fixed, plausible normalized history makes repeated timing deterministic
    # while avoiding the all-zero special case in every network layer.
    normalized_history = 0.5 * torch.randn(
        1,
        int(phase1_selection["observation_horizon"]),
        int(phase1_selection["observation_dim"]),
        device=device,
        generator=generator,
    )
    raw_history = normalized_history * obs_std + obs_mean
    gain = float(
        phase1_selection["deployment"]["action_gain_by_condition"][1]["gain"]
    )
    residual_scale = float(selection["deployment"]["residual_scale"])
    ddim_steps = int(phase1_selection["deployment"]["ddim_steps"])

    phase1_times = []
    residual_times = []
    full_times = []
    total = args.warmup + args.iterations
    final_action = None
    for iteration in range(total):
        _synchronize(device)
        full_started = time.perf_counter()
        normalized = (
            (raw_history - obs_mean) / obs_std
        ).clamp(-10.0, 10.0)
        phase1_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            chunk = phase1.sample(
                normalized.flatten(1),
                inference_steps=ddim_steps,
                deterministic=True,
            )
        _synchronize(device)
        phase1_finished = time.perf_counter()
        base_action = (gain * chunk[:, 0].float()).clamp(-1.0, 1.0)
        with torch.inference_mode():
            residual_action = residual_actor(
                torch.cat([normalized[:, 1], base_action], dim=-1)
            )
            final_action = (
                base_action + residual_scale * residual_action
            ).clamp(-1.0, 1.0)
        _synchronize(device)
        finished = time.perf_counter()
        if iteration >= args.warmup:
            phase1_times.append(
                (phase1_finished - phase1_started) * 1000.0
            )
            residual_times.append(
                (finished - phase1_finished) * 1000.0
            )
            full_times.append((finished - full_started) * 1000.0)

    result = {
        "schema_version": 1,
        "controller": "phase1_diffusion_plus_phase2_residual",
        "batch_size": 1,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "deterministic_ddim_steps": ddim_steps,
        "residual_scale": residual_scale,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "phase1_diffusion": _distribution(phase1_times),
        "phase2_residual_and_composition": _distribution(residual_times),
        "full_controller": _distribution(full_times),
        "control_period_ms_at_50hz": 20.0,
        "full_controller_deadline_fraction": float(
            np.mean(np.asarray(full_times) <= 20.0)
        ),
        "sample_final_action": final_action[0].detach().cpu().tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    print(f"[benchmark] wrote {args.output}")


if __name__ == "__main__":
    main()
