#!/usr/bin/env python3
"""Export the selected Phase-2 residual actor as a deployment bundle.

The Phase-1 diffusion policy remains in its native checkpoint because its
deterministic DDIM sampler is part of the controller.  The small Phase-2 actor
is exported independently to TorchScript and ONNX.  Its output is clipped to
the same [-1, 1] interval used by the Isaac evaluation environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from torch import nn


ROOT = Path("/home/windylab/code/isaac_arm_rl")
DEFAULT_CONFIG = ROOT / "config/phase2_selected_policy.json"
DEFAULT_OUTPUT = ROOT / "exports/phase2_robust"


class SafeResidualActor(nn.Module):
    """The deterministic PPO mean actor with the deployment safety clip."""

    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.actor(observation), -1.0, 1.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _copy(source: Path, destination: Path):
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    selection = _load_json(args.config)
    phase1_config_path = Path(selection["base_policy_config"])
    phase1_selection = _load_json(phase1_config_path)
    phase1_checkpoint_path = Path(phase1_selection["checkpoint"])
    phase2_checkpoint_path = Path(selection["checkpoint"])
    for path in (
        args.config,
        phase1_config_path,
        phase1_checkpoint_path,
        phase2_checkpoint_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    observation_dim = int(selection["observation_dim"])
    action_dim = int(selection["action_dim"])
    checkpoint = torch.load(
        phase2_checkpoint_path, map_location="cpu", weights_only=False
    )
    actor_state = {
        key.removeprefix("actor."): value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("actor.")
    }
    actor = SafeResidualActor(observation_dim, action_dim).eval()
    actor.actor.load_state_dict(actor_state, strict=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torchscript_path = args.output_dir / "residual_actor.ts"
    onnx_path = args.output_dir / "residual_actor.onnx"
    phase1_checkpoint_copy = args.output_dir / "phase1_diffusion_policy.pt"
    phase2_checkpoint_copy = args.output_dir / "phase2_training_checkpoint.pt"
    phase1_config_copy = args.output_dir / "phase1_selected_policy.json"
    phase2_config_copy = args.output_dir / "phase2_selected_policy.json"

    example = torch.linspace(
        -2.0,
        2.0,
        steps=3 * observation_dim,
        dtype=torch.float32,
    ).reshape(3, observation_dim)
    with torch.inference_mode():
        expected = actor(example)

    scripted = torch.jit.script(actor)
    scripted.save(str(torchscript_path))
    loaded_script = torch.jit.load(str(torchscript_path), map_location="cpu")
    with torch.inference_mode():
        scripted_output = loaded_script(example)
    max_torchscript_error = float(
        torch.max(torch.abs(expected - scripted_output)).item()
    )
    if max_torchscript_error > 1.0e-6:
        raise RuntimeError(
            "TorchScript validation failed: "
            f"max_abs_error={max_torchscript_error}"
        )

    torch.onnx.export(
        actor,
        example,
        str(onnx_path),
        input_names=["residual_observation"],
        output_names=["residual_action"],
        dynamic_axes={
            "residual_observation": {0: "batch"},
            "residual_action": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    import onnx

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    onnx_input_shape = [
        dimension.dim_param or dimension.dim_value
        for dimension in onnx_model.graph.input[0].type.tensor_type.shape.dim
    ]
    onnx_output_shape = [
        dimension.dim_param or dimension.dim_value
        for dimension in onnx_model.graph.output[0].type.tensor_type.shape.dim
    ]
    if onnx_input_shape != ["batch", observation_dim]:
        raise RuntimeError(f"unexpected ONNX input shape: {onnx_input_shape}")
    if onnx_output_shape != ["batch", action_dim]:
        raise RuntimeError(f"unexpected ONNX output shape: {onnx_output_shape}")

    _copy(phase1_checkpoint_path, phase1_checkpoint_copy)
    _copy(phase2_checkpoint_path, phase2_checkpoint_copy)
    _copy(phase1_config_path, phase1_config_copy)
    _copy(args.config, phase2_config_copy)

    residual_scale = float(selection["deployment"]["residual_scale"])
    manifest = {
        "schema_version": 1,
        "phase": 2,
        "selection": selection["selection"],
        "controller": {
            "phase1": {
                "type": "deterministic_diffusion_ddim",
                "checkpoint": phase1_checkpoint_copy.name,
                "configuration": phase1_config_copy.name,
                "observation_dim": int(phase1_selection["observation_dim"]),
                "observation_horizon": int(
                    phase1_selection["observation_horizon"]
                ),
                "action_dim": action_dim,
                "ddim_steps": int(
                    phase1_selection["deployment"]["ddim_steps"]
                ),
            },
            "phase2": {
                "type": "deterministic_residual_ppo_mean_actor",
                "torchscript": torchscript_path.name,
                "onnx": onnx_path.name,
                "training_checkpoint": phase2_checkpoint_copy.name,
                "configuration": phase2_config_copy.name,
                "observation_dim": observation_dim,
                "action_dim": action_dim,
                "output_clip": [-1.0, 1.0],
            },
            "composition": {
                "residual_scale": residual_scale,
                "formula": (
                    "final_action = clip(phase1_action + "
                    f"{residual_scale:.2f} * residual_action, -1, 1)"
                ),
                "action_scale_rad": float(
                    phase1_selection["action_scale_rad"]
                ),
                "control_rate_hz": 50.0,
            },
        },
        "validation": {
            "torchscript_max_abs_error": max_torchscript_error,
            "onnx_checker": "passed",
            "onnx_input_shape": onnx_input_shape,
            "onnx_output_shape": onnx_output_shape,
            "phase2_summary": selection["validation"]["summary"],
        },
    }
    manifest_path = args.output_dir / "deployment_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    artifact_paths = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    checksums_path = args.output_dir / "SHA256SUMS"
    with checksums_path.open("w", encoding="utf-8") as stream:
        for path in artifact_paths:
            stream.write(f"{_sha256(path)}  {path.name}\n")

    print(f"[export] bundle={args.output_dir}")
    for path in artifact_paths:
        print(f"[export] {path.name} bytes={path.stat().st_size}")
    print(
        "[export] validation "
        f"torchscript_max_abs_error={max_torchscript_error:.3e} "
        "onnx_checker=passed"
    )


if __name__ == "__main__":
    main()
