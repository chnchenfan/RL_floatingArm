#!/usr/bin/env python3
"""Run the selected Phase-1 policy online and stream each live control tick.

Isaac Lab runs in the project's Python 3.11 environment while ROS Humble uses
Python 3.10.  A small localhost UDP packet keeps those ABI-incompatible
processes separate without putting a file or prerecorded action sequence in
the control loop.
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

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ROOT = "/home/windylab/code/isaac_arm_rl"
SELECTED_CONFIG = f"{ROOT}/config/phase1_selected_policy.json"
CONTROL_HZ = 50.0

CONDITIONS = {
    "easy": {
        "amp_t": 0.02,
        "amp_z": 0.0,
        "freq": 1.0,
        "seed_csv": (
            "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
            "moving_base_mpc_loop_20260724_051521.csv"
        ),
    },
    "medium": {
        "amp_t": 0.02,
        "amp_z": 0.02,
        "freq": 1.0,
        "seed_csv": (
            "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
            "moving_base_mpc_loop_20260724_051912.csv"
        ),
    },
    "hard": {
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
        "--condition", choices=tuple(CONDITIONS), default="medium")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=45831)
    parser.add_argument("--rate-hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--cycle-steps", type=int, default=400)
    parser.add_argument(
        "--physics-substeps",
        type=int,
        choices=(1, 2),
        default=2,
        help="PhysX steps per 50 Hz control tick (evaluation uses 2)",
    )
    parser.add_argument(
        "--sim-device",
        choices=("cpu", "cuda:0"),
        default="cuda:0",
        help="PhysX device; the diffusion network remains on CUDA when present",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Stop after this many cycles; 0 runs until interrupted",
    )
    parser.add_argument("--q-noise", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20260724)
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
    return args


def main():
    args = parse_args()
    selected = _load_json(SELECTED_CONFIG)
    condition_cfg = CONDITIONS[args.condition]
    gain_by_condition = {
        name: item["gain"]
        for name, item in zip(
            CONDITIONS,
            selected["deployment"]["action_gain_by_condition"],
            strict=True,
        )
    }

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    import numpy as np
    import torch

    sys.path.insert(0, ROOT)
    from distill.diffusion_policy import make_policy_from_checkpoint
    from env.arm_track_env import ArmTrackEnv, ArmTrackEnvCfg

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    checkpoint_path = selected["checkpoint"]
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    observation_dim = len(checkpoint["obs_mean"])

    cfg = ArmTrackEnvCfg()
    cfg.scene.num_envs = 1
    cfg.decimation = args.physics_substeps
    cfg.sim.dt = 1.0 / (CONTROL_HZ * args.physics_substeps)
    cfg.sim.device = args.sim_device
    cfg.single_circle = True
    cfg.start_phase_random = False
    cfg.inertial_coupling = False
    cfg.extended_phase1_obs = observation_dim == 47
    cfg.observation_space = observation_dim
    env = ArmTrackEnv(cfg)
    policy_device = torch.device(
        "cuda:0" if torch.cuda.is_available() else env.device)
    policy = make_policy_from_checkpoint(checkpoint, policy_device)
    obs_mean = torch.as_tensor(
        checkpoint["obs_mean"], device=policy_device, dtype=torch.float32)
    obs_std = torch.as_tensor(
        checkpoint["obs_std"], device=policy_device, dtype=torch.float32)
    action_gain = float(gain_by_condition[args.condition])
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    destination = (args.host, args.port)

    should_run = True

    def request_stop(_signum, _frame):
        nonlocal should_run
        should_run = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(
        "[phase1-live] loaded "
        f"{checkpoint_path} condition={args.condition} gain={action_gain:.2f} "
        f"physics={args.physics_substeps}x"
        f"{CONTROL_HZ * args.physics_substeps:.0f}Hz@{env.device} "
        f"policy={policy_device} "
        f"stream=udp://{args.host}:{args.port} target_rate={args.rate_hz:.1f}Hz",
        flush=True,
    )

    global_seq = 0
    cycle = 0
    period = 1.0 / args.rate_hz
    next_deadline = time.monotonic()
    inference_window = []
    plant_window = []
    loop_window = []

    while should_run and (
        args.max_cycles == 0 or cycle < args.max_cycles
    ):
        history = _reset_condition(
            env=env,
            torch=torch,
            condition=condition_cfg,
            obs_mean=obs_mean,
            q_noise=args.q_noise,
            seed=args.seed + cycle,
        ).to(policy_device)
        cycle_errors = []
        cycle_started = time.monotonic()

        for step in range(args.cycle_steps):
            if not should_run:
                break
            loop_started = time.perf_counter()
            normalized = (
                (history - obs_mean) / obs_std
            ).clamp(-10.0, 10.0).flatten(1)
            inference_started = time.perf_counter()
            with torch.no_grad(), torch.autocast(
                device_type=policy_device.type,
                dtype=torch.bfloat16,
                enabled=policy_device.type == "cuda",
            ):
                chunk = policy.sample(
                    normalized,
                    inference_steps=selected["deployment"]["ddim_steps"],
                    deterministic=True,
                )
            inference_ms = (
                time.perf_counter() - inference_started) * 1000.0
            action = (
                action_gain * chunk[:, 0].float()
            ).clamp(-1.0, 1.0)
            plant_started = time.perf_counter()
            observation = _deployment_step(env, action).to(policy_device)
            plant_ms = (time.perf_counter() - plant_started) * 1000.0
            history = torch.stack([history[:, 1], observation], dim=1)

            base_position, base_rotation = env.targets.base_pose(
                env._phase() % env.targets.T, env._dist)
            ee_world = base_position + torch.bmm(
                base_rotation, env._p_ee[:, :, None])[:, :, 0]
            target_world = base_position + torch.bmm(
                base_rotation, env._p_tgt[:, :, None])[:, :, 0]
            error_mm = (
                torch.linalg.norm(env._p_tgt - env._p_ee, dim=-1)[0].item()
                * 1000.0
            )
            loop_ms = (time.perf_counter() - loop_started) * 1000.0
            packet = {
                "version": 1,
                "source": "phase1_diffusion_live",
                "checkpoint": os.path.basename(checkpoint_path),
                "condition": args.condition,
                "seq": global_seq,
                "cycle": cycle,
                "step": step,
                "sim_time": step / CONTROL_HZ,
                "wall_time_ns": time.time_ns(),
                "q": _tolist(env.robot.data.joint_pos[0]),
                "dq": _tolist(env.robot.data.joint_vel[0]),
                "base_pos": _tolist(base_position[0]),
                "base_quat": _rotation_to_xyzw(
                    base_rotation[0].detach().cpu().numpy()).tolist(),
                "ee": _tolist(ee_world[0]),
                "target": _tolist(target_world[0]),
                "error_mm": error_mm,
                "inference_ms": inference_ms,
                "loop_ms": loop_ms,
            }
            payload = json.dumps(
                packet, separators=(",", ":"), allow_nan=False).encode("utf-8")
            udp.sendto(payload, destination)

            global_seq += 1
            inference_window.append(inference_ms)
            plant_window.append(plant_ms)
            loop_window.append(loop_ms)
            cycle_errors.append(error_mm)
            if len(inference_window) >= round(args.rate_hz):
                print(
                    "[phase1-live] "
                    f"cycle={cycle} step={step:03d} error={error_mm:.2f}mm "
                    f"inference={np.mean(inference_window):.2f}ms "
                    f"plant={np.mean(plant_window):.2f}ms "
                    f"loop={np.mean(loop_window):.2f}ms",
                    flush=True,
                )
                inference_window.clear()
                plant_window.clear()
                loop_window.clear()

            next_deadline += period
            remaining = next_deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -2.0 * period:
                # Do not emit a catch-up burst after a debugger pause or a
                # transient system stall; resume from the current wall clock.
                next_deadline = time.monotonic()
        cycle_elapsed = time.monotonic() - cycle_started
        if cycle_errors:
            errors = np.asarray(cycle_errors)
            achieved_rate = len(cycle_errors) / max(cycle_elapsed, 1.0e-9)
            print(
                "[phase1-live-summary] "
                f"cycle={cycle} mean={errors.mean():.2f}mm "
                f"p95={np.percentile(errors, 95):.2f}mm "
                f"max={errors.max():.2f}mm rate={achieved_rate:.2f}Hz",
                flush=True,
            )
        cycle += 1

    udp.close()
    print("[phase1-live] stopped", flush=True)
    sys.stdout.flush()
    # Isaac's teardown is not reliable in all headless driver combinations;
    # the project exporters use the same clean process-boundary exit.
    del app
    os._exit(0)


def _reset_condition(
    env,
    torch,
    condition,
    obs_mean,
    q_noise,
    seed,
):
    env.reset()
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
    env._last_executed_action.zero_()
    env._prev_action.zero_()
    env.robot.write_joint_state_to_sim(q_seed, dq_seed)
    env.robot.set_joint_position_target(q_seed)
    observation = env._get_observations()["policy"]
    assert observation.shape[-1] == obs_mean.shape[-1]
    return torch.stack([observation, observation], dim=1)


def _deployment_step(env, action):
    """Step only the plant and observations needed by deployed inference."""
    action = action.to(env.device)
    env._pre_physics_step(action)
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    env.episode_length_buf += 1
    env.common_step_counter += 1
    return env._get_observations()["policy"]


def _load_json(path):
    with open(path, encoding="utf-8") as stream:
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
                1.0 + rotation[0, 0]
                - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(
                1.0 - rotation[0, 0]
                + rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(
                1.0 - rotation[0, 0]
                - rotation[1, 1] + rotation[2, 2]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.array([x, y, z, w])
    return quaternion / np.linalg.norm(quaternion)


if __name__ == "__main__":
    main()
