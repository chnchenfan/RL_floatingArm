#!/usr/bin/env python3
"""Run the whole-task Phase-1 (+ optional Phase-2 residual) controller live
and stream 50-Hz ticks over UDP for the ROS bridge.

Contract with the MPC trajectory visualizer (reused unchanged):
- the stream starts at whole phase 0 and paces to true 50 Hz wall time, so
  the visualizer's ``elapsed`` (from the first /student/joint_command stamp)
  matches the whole path time;
- every packet carries the final joint command (q_cmd) that the bridge
  republishes on /student/joint_command with frame_id 'moving_base_track';
- pen_down / stroke_id let the bridge break its own trails on pen-up.

    ./env_isaaclab/bin/python scripts/live_phase2_whole_stream.py \
        --checkpoint logs/phase2_whole_residual/<run>/model_950.pt
"""

from __future__ import annotations

import argparse
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
CONTROL_HZ = 50.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="",
        help="phase-2 rsl_rl checkpoint; empty = pure Phase-1 (zero residual)")
    parser.add_argument("--phase1-checkpoint", default="")
    parser.add_argument("--residual-scale", type=float, default=-1.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=45831)
    parser.add_argument("--rate-hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--q-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--physics-substeps", type=int,
                        choices=(1, 2), default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT))
    from env.residual_arm_track_env import (
        ResidualArmTrackEnv,
        WholeResidualArmTrackEnvCfg,
        _load_state_bank,
        _draw_bank_states,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = WholeResidualArmTrackEnvCfg()
    cfg.scene.num_envs = 1
    cfg.decimation = args.physics_substeps
    cfg.sim.dt = 1.0 / (CONTROL_HZ * args.physics_substeps)
    cfg.inertial_coupling = True
    cfg.divergence_threshold_m = 1.0e6
    cfg.episode_length_s = 1.0e6
    cfg.condition_jitter_fraction = 0.0
    if args.phase1_checkpoint:
        cfg.phase1_checkpoint = args.phase1_checkpoint
    if args.residual_scale > 0.0:
        cfg.residual_scale = args.residual_scale
    env = ResidualArmTrackEnv(cfg)
    device = env.device
    T = env.targets.T

    residual_policy = None
    if args.checkpoint:
        from rsl_rl.modules import ActorCritic
        actor_critic = ActorCritic(
            {"policy": torch.zeros(1, cfg.observation_space, device=device)},
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
        state = torch.load(
            args.checkpoint, map_location=device, weights_only=False)
        actor_critic.load_state_dict(state["model_state_dict"])
        actor_critic.eval()

        def residual_policy(observation):
            with torch.no_grad():
                return actor_critic.act_inference({"policy": observation})

    # ---- reset to whole phase 0 with the exact deployed base motion
    env.reset()
    exact = env.targets.sample_whole_disturbance(1, jitter=0.0)
    for key, value in exact.items():
        env._dist[key].copy_(value)
    env._start.zero_()
    env._condition_id.zero_()
    env.episode_length_buf.zero_()
    bank = _load_state_bank(cfg.state_bank_path, device)
    q0, dq0, prev0 = _draw_bank_states(
        bank, torch.zeros(1, dtype=torch.long, device=device))
    if args.q_noise > 0.0:
        q0 += args.q_noise * torch.randn_like(q0)
    env.robot.write_joint_state_to_sim(q0, dq0)
    env.robot.set_joint_position_target(q0)
    env.sync_command_state()
    env._last_executed_action.copy_(prev0)
    env._prev_action.copy_(prev0)
    env._history_needs_reset[:] = True
    observation = env._get_observations()["policy"]

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    destination = (args.host, args.port)
    should_run = True

    def request_stop(_signum, _frame):
        nonlocal should_run
        should_run = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    mode = "phase1+phase2" if residual_policy else "phase1_only"
    print(f"[whole-live] {mode} phase1={Path(cfg.phase1_checkpoint).name} "
          f"residual_scale={cfg.residual_scale} T={T} "
          f"stream=udp://{args.host}:{args.port} rate={args.rate_hz}Hz",
          flush=True)

    period = 1.0 / args.rate_hz
    next_deadline = time.monotonic()
    global_seq = 0
    step = 0
    window_err = []
    prev_q_cmd = q0.clone()
    while should_run:
        cycle, tick = divmod(step, T)
        if args.max_cycles and cycle >= args.max_cycles:
            break
        if residual_policy is not None:
            residual = residual_policy(observation)
        else:
            residual = torch.zeros(1, 7, device=device)

        env._pre_physics_step(residual)
        for _ in range(env.cfg.decimation):
            env._sim_step_counter += 1
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        env.episode_length_buf += 1
        env.common_step_counter += 1
        observation = env._get_observations()["policy"]

        idx = int((env._start + env.episode_length_buf)[0].item()) % T
        base_position, base_rotation = env.targets.base_pose(
            env._phase(), env._dist)
        ee_world = base_position + torch.bmm(
            base_rotation, env._p_ee[:, :, None])[:, :, 0]
        target_world = base_position + torch.bmm(
            base_rotation, env._p_tgt[:, :, None])[:, :, 0]
        q_cmd = env._q_command
        dq_cmd = (q_cmd - prev_q_cmd) / (1.0 / CONTROL_HZ)
        prev_q_cmd = q_cmd.clone()
        error_mm = float(torch.linalg.norm(
            env._p_tgt - env._p_ee, dim=-1)[0].item() * 1000.0)
        window_err.append(error_mm)

        packet = {
            "version": 2,
            "source": "phase2_whole_live",
            "mode": mode,
            "checkpoint": Path(args.checkpoint).name if args.checkpoint
            else "phase1_only",
            "seq": global_seq,
            "cycle": cycle,
            "step": tick,
            "sim_time": step / CONTROL_HZ,
            "wall_time_ns": time.time_ns(),
            "q": env.robot.data.joint_pos[0].detach().cpu().tolist(),
            "dq": env.robot.data.joint_vel[0].detach().cpu().tolist(),
            "q_cmd": q_cmd[0].detach().cpu().tolist(),
            "dq_cmd": dq_cmd[0].detach().cpu().tolist(),
            "base_pos": base_position[0].detach().cpu().tolist(),
            "base_quat": _rotation_to_xyzw(
                base_rotation[0].detach().cpu().numpy()).tolist(),
            "ee": ee_world[0].detach().cpu().tolist(),
            "target": target_world[0].detach().cpu().tolist(),
            "pen_down": bool(env.targets.pen[idx].item()),
            "stroke_id": int(env.targets.stroke_id[idx].item()),
            "task_id": int(env.targets.task_id[idx].item()),
            "error_mm": error_mm,
        }
        udp.sendto(json.dumps(
            packet, separators=(",", ":"), allow_nan=False).encode(),
            destination)
        global_seq += 1
        step += 1

        if len(window_err) >= 250:
            errs = np.asarray(window_err)
            print(f"[whole-live] cycle={cycle} tick={tick:4d} "
                  f"task={packet['task_id']} "
                  f"err mean={errs.mean():.2f} p95="
                  f"{np.percentile(errs, 95):.2f} max={errs.max():.2f}mm",
                  flush=True)
            window_err.clear()

        next_deadline += period
        remaining = next_deadline - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
        elif remaining < -2.0 * period:
            next_deadline = time.monotonic()

    udp.close()
    print("[whole-live] stopped", flush=True)
    sys.stdout.flush()
    del app
    os._exit(0)


def _rotation_to_xyzw(rotation):
    import numpy as np
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        nxt = [(1, 2), (2, 0), (0, 1)]
        i = index
        j, k = nxt[i]
        scale = math.sqrt(
            1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k]) * 2.0
        quat = [0.0, 0.0, 0.0, 0.0]
        quat[i] = 0.25 * scale
        quat[3] = (rotation[k, j] - rotation[j, k]) / scale
        quat[j] = (rotation[j, i] + rotation[i, j]) / scale
        quat[k] = (rotation[k, i] + rotation[i, k]) / scale
        x, y, z, w = quat
    quaternion = [x, y, z, w]
    norm = math.sqrt(sum(v * v for v in quaternion))
    return type(rotation)([v / norm for v in quaternion])


if __name__ == "__main__":
    main()
