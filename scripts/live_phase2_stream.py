#!/usr/bin/env python3
"""Run the selected Phase-1 + Phase-2 controller and stream live Isaac ticks.

The process is intentionally headless: Isaac supplies the live physics state
and policy inference, while ROS 2/RViz performs the visualization in a
different Python process.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ROOT = Path("/home/windylab/code/isaac_arm_rl")
SELECTED_CONFIG = ROOT / "config/phase2_selected_policy.json"
DEFAULT_ACTOR = ROOT / "exports/phase2_robust/residual_actor.ts"
CONTROL_HZ = 50.0

CONDITIONS = {
    "easy": {
        "index": 0,
        "amp_t": 0.02,
        "amp_z": 0.0,
        "freq": 1.0,
        "seed_csv": (
            "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
            "moving_base_mpc_loop_20260724_051521.csv"
        ),
    },
    "medium": {
        "index": 1,
        "amp_t": 0.02,
        "amp_z": 0.02,
        "freq": 1.0,
        "seed_csv": (
            "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
            "moving_base_mpc_loop_20260724_051912.csv"
        ),
    },
    "hard": {
        "index": 2,
        "amp_t": 0.01,
        "amp_z": 0.01,
        "freq": 2.0,
        "seed_csv": (
            "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
            "moving_base_mpc_loop_20260724_052123.csv"
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition", choices=tuple(CONDITIONS), default="medium"
    )
    parser.add_argument("--config", type=Path, default=SELECTED_CONFIG)
    parser.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=45831)
    parser.add_argument("--rate-hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--cycle-steps", type=int, default=400)
    parser.add_argument(
        "--physics-substeps",
        type=int,
        choices=(1, 2),
        default=2,
        help="PhysX steps per 50 Hz control tick (validation uses 2)",
    )
    parser.add_argument(
        "--sim-device",
        choices=("cpu", "cuda:0"),
        default="cuda:0",
        help="PhysX device; Phase-1 diffusion still uses CUDA when available",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Stop after this many cycles; 0 runs until interrupted",
    )
    parser.add_argument("--q-noise", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.rate_hz <= 0.0:
        parser.error("--rate-hz must be > 0")
    if args.cycle_steps <= 0:
        parser.error("--cycle-steps must be > 0")
    if args.max_cycles < 0:
        parser.error("--max-cycles must be >= 0")
    if args.q_noise < 0.0:
        parser.error("--q-noise must be >= 0")
    if not args.config.is_file():
        parser.error(f"selection config not found: {args.config}")
    if not args.actor.is_file():
        parser.error(
            f"exported residual actor not found: {args.actor}; "
            "run scripts/export_phase2_policy.py first"
        )
    return args


def main():
    args = parse_args()
    selected = _load_json(args.config)
    phase1_selected = _load_json(Path(selected["base_policy_config"]))
    condition = CONDITIONS[args.condition]

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT))
    from env.residual_arm_track_env import (
        ResidualArmTrackEnv,
        ResidualArmTrackEnvCfg,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = ResidualArmTrackEnvCfg()
    cfg.scene.num_envs = 1
    cfg.decimation = args.physics_substeps
    cfg.sim.dt = 1.0 / (CONTROL_HZ * args.physics_substeps)
    cfg.sim.device = args.sim_device
    cfg.phase1_checkpoint = phase1_selected["checkpoint"]
    cfg.phase1_ddim_steps = int(
        phase1_selected["deployment"]["ddim_steps"]
    )
    cfg.phase1_policy_device = (
        "cuda:0" if torch.cuda.is_available() else args.sim_device
    )
    cfg.residual_scale = float(selected["deployment"]["residual_scale"])
    cfg.inertial_coupling = True
    cfg.condition_jitter_fraction = 0.0
    env = ResidualArmTrackEnv(cfg)
    actor = torch.jit.load(str(args.actor), map_location=env.device).eval()

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    destination = (args.host, args.port)
    should_run = True

    def request_stop(_signum, _frame):
        nonlocal should_run
        should_run = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(
        "[phase2-live] loaded "
        f"phase1={Path(cfg.phase1_checkpoint).name} "
        f"residual={args.actor.name} scale={cfg.residual_scale:.2f} "
        f"condition={args.condition} inertia=on "
        f"physics={args.physics_substeps}x"
        f"{CONTROL_HZ * args.physics_substeps:.0f}Hz@{env.device} "
        f"phase1_policy={env._phase1_device} "
        f"stream=udp://{args.host}:{args.port} "
        f"target_rate={args.rate_hz:.1f}Hz",
        flush=True,
    )

    global_seq = 0
    cycle = 0
    period = 1.0 / args.rate_hz
    next_deadline = time.monotonic()
    phase1_window = []
    residual_window = []
    plant_window = []
    loop_window = []

    while should_run and (
        args.max_cycles == 0 or cycle < args.max_cycles
    ):
        observation, phase1_ms = _reset_condition(
            env=env,
            torch=torch,
            condition=condition,
            q_noise=args.q_noise,
            seed=args.seed + cycle,
        )
        # Warm the small TorchScript graph without altering controller state.
        with torch.inference_mode():
            for _ in range(4):
                actor(observation)
        cycle_errors = []
        cycle_orientation_errors = []
        cycle_started = time.monotonic()

        for step in range(args.cycle_steps):
            if not should_run:
                break
            loop_started = time.perf_counter()

            _synchronize(torch, env.device)
            residual_started = time.perf_counter()
            with torch.inference_mode():
                residual_action = actor(observation)
            _synchronize(torch, env.device)
            residual_ms = (
                time.perf_counter() - residual_started
            ) * 1000.0

            env._pre_physics_step(residual_action)
            plant_started = time.perf_counter()
            for _ in range(env.cfg.decimation):
                env._sim_step_counter += 1
                env._apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
            env.episode_length_buf += 1
            env.common_step_counter += 1
            plant_ms = (time.perf_counter() - plant_started) * 1000.0

            _synchronize(torch, env._phase1_device)
            phase1_started = time.perf_counter()
            next_observation = env._get_observations()["policy"]
            _synchronize(torch, env._phase1_device)
            next_phase1_ms = (
                time.perf_counter() - phase1_started
            ) * 1000.0

            base_position, base_rotation = env.targets.base_pose(
                env._phase() % env.targets.T, env._dist
            )
            ee_world = base_position + torch.bmm(
                base_rotation, env._p_ee[:, :, None]
            )[:, :, 0]
            target_world = base_position + torch.bmm(
                base_rotation, env._p_tgt[:, :, None]
            )[:, :, 0]
            error_mm = (
                torch.linalg.norm(
                    env._p_tgt - env._p_ee, dim=-1
                )[0].item()
                * 1000.0
            )
            orientation_error_deg = _orientation_error_deg(env, torch)
            policy_ms = phase1_ms + residual_ms
            loop_ms = (time.perf_counter() - loop_started) * 1000.0
            packet = {
                "version": 1,
                "source": "phase2_residual_live",
                "checkpoint": args.actor.name,
                "base_checkpoint": Path(cfg.phase1_checkpoint).name,
                "condition": args.condition,
                "residual_scale": cfg.residual_scale,
                "seq": global_seq,
                "cycle": cycle,
                "step": step,
                "sim_time": step / CONTROL_HZ,
                "wall_time_ns": time.time_ns(),
                "q": _tolist(env.robot.data.joint_pos[0]),
                "dq": _tolist(env.robot.data.joint_vel[0]),
                "base_pos": _tolist(base_position[0]),
                "base_quat": _rotation_to_xyzw(
                    base_rotation[0].detach().cpu().numpy()
                ).tolist(),
                "ee": _tolist(ee_world[0]),
                "target": _tolist(target_world[0]),
                "error_mm": error_mm,
                "orientation_error_deg": orientation_error_deg,
                "phase1_inference_ms": phase1_ms,
                "residual_inference_ms": residual_ms,
                "policy_inference_ms": policy_ms,
                "plant_ms": plant_ms,
                "loop_ms": loop_ms,
                "base_action": _tolist(env._base_action[0]),
                "residual_action": _tolist(env._residual_action[0]),
                "final_action": _tolist(env._final_action[0]),
            }
            payload = json.dumps(
                packet, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            udp.sendto(payload, destination)

            global_seq += 1
            phase1_window.append(phase1_ms)
            residual_window.append(residual_ms)
            plant_window.append(plant_ms)
            loop_window.append(loop_ms)
            cycle_errors.append(error_mm)
            cycle_orientation_errors.append(orientation_error_deg)
            if len(phase1_window) >= round(args.rate_hz):
                print(
                    "[phase2-live] "
                    f"cycle={cycle} step={step:03d} "
                    f"error={error_mm:.2f}mm "
                    f"orientation={orientation_error_deg:.2f}deg "
                    f"phase1={np.mean(phase1_window):.2f}ms "
                    f"residual={np.mean(residual_window):.3f}ms "
                    f"plant={np.mean(plant_window):.2f}ms "
                    f"loop={np.mean(loop_window):.2f}ms",
                    flush=True,
                )
                phase1_window.clear()
                residual_window.clear()
                plant_window.clear()
                loop_window.clear()

            observation = next_observation
            phase1_ms = next_phase1_ms
            next_deadline += period
            remaining = next_deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -2.0 * period:
                next_deadline = time.monotonic()

        cycle_elapsed = time.monotonic() - cycle_started
        if cycle_errors:
            errors = np.asarray(cycle_errors)
            orientations = np.asarray(cycle_orientation_errors)
            achieved_rate = len(cycle_errors) / max(cycle_elapsed, 1.0e-9)
            print(
                "[phase2-live-summary] "
                f"cycle={cycle} mean={errors.mean():.2f}mm "
                f"p95={np.percentile(errors, 95):.2f}mm "
                f"max={errors.max():.2f}mm "
                f"orientation_mean={orientations.mean():.2f}deg "
                f"rate={achieved_rate:.2f}Hz",
                flush=True,
            )
        cycle += 1

    udp.close()
    print("[phase2-live] stopped", flush=True)
    sys.stdout.flush()
    del app
    os._exit(0)


def _reset_condition(env, torch, condition, q_noise, seed):
    env.reset()
    condition_id = int(condition["index"])
    env._condition_id.fill_(condition_id)
    disturbance = env._dist
    disturbance["wave_t"][:] = 0
    disturbance["wave_a"][:] = 1
    disturbance["amp_t"][:] = condition["amp_t"]
    disturbance["amp_z"][:] = condition["amp_z"]
    disturbance["z_phase"][:] = math.pi / 3.0
    disturbance["amp_a"][:] = math.radians(0.1)
    disturbance["freq_t"][:] = condition["freq"]
    disturbance["freq_a"].copy_(disturbance["freq_t"])
    disturbance["phase"][:] = 0.0
    env._start[:] = 0

    with open(condition["seed_csv"], newline="") as stream:
        row = next(csv.DictReader(stream))
    q_seed = torch.tensor(
        [[float(row[f"q_meas_{joint}"]) for joint in range(1, 8)]],
        device=env.device,
        dtype=torch.float32,
    )
    dq_seed = torch.tensor(
        [[float(row[f"dq_meas_{joint}"]) for joint in range(1, 8)]],
        device=env.device,
        dtype=torch.float32,
    )
    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed)
    q_seed += q_noise * torch.randn(
        q_seed.shape,
        device=env.device,
        dtype=q_seed.dtype,
        generator=generator,
    )
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

    _synchronize(torch, env._phase1_device)
    started = time.perf_counter()
    observation = env._get_observations()["policy"]
    _synchronize(torch, env._phase1_device)
    phase1_ms = (time.perf_counter() - started) * 1000.0
    return observation, phase1_ms


def _orientation_error_deg(env, torch):
    rotation_error = torch.bmm(
        env._R_ee.transpose(1, 2), env._R_tgt
    )
    cosine = (
        rotation_error[:, 0, 0]
        + rotation_error[:, 1, 1]
        + rotation_error[:, 2, 2]
        - 1.0
    ) * 0.5
    return float(
        torch.rad2deg(
            torch.arccos(torch.clamp(cosine, -1.0, 1.0))
        )[0].item()
    )


def _synchronize(torch, device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _tolist(tensor):
    return tensor.detach().cpu().tolist()


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
                1.0
                + rotation[0, 0]
                - rotation[1, 1]
                - rotation[2, 2]
            ) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(
                1.0
                - rotation[0, 0]
                + rotation[1, 1]
                - rotation[2, 2]
            ) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(
                1.0
                - rotation[0, 0]
                - rotation[1, 1]
                + rotation[2, 2]
            ) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.array([x, y, z, w])
    return quaternion / np.linalg.norm(quaternion)


if __name__ == "__main__":
    main()
