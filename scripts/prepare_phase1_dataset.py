#!/usr/bin/env python3
"""Build Phase-1 observation histories and MPC action chunks from clean+DART CSVs.

Run with the system Python (it provides Pinocchio):
    python3 scripts/prepare_phase1_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np
import pinocchio as pin


ROOT = "/home/windylab/code/isaac_arm_rl"
DEFAULT_DATA = "/home/windylab/code/windylab_ws/src/arm-platform/demo/data"
ACTION_SCALE = 0.02


def _columns(rows, prefix):
    return np.array([
        [float(row[f"{prefix}_{i}"]) for i in range(1, 8)]
        for row in rows
    ], dtype=np.float64)


def _params_path(csv_path):
    return csv_path.replace("_loop_", "_params_").replace(".csv", ".json")


class ForwardKinematics:
    def __init__(self, urdf_path):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId("link7")
        if self.frame_id >= len(self.model.frames):
            raise ValueError("link7 frame not found")

    def pose(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.frame_id]
        return placement.translation.copy(), placement.rotation.copy()

    def position_jacobian(self, q):
        jacobian = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return np.asarray(jacobian[:3], dtype=np.float64).copy()


def _base_position(t, trajectory_config):
    radius = float(trajectory_config["BASE_CIRCLE_RADIUS"])
    period = float(trajectory_config["BASE_PERIOD_SEC"])
    height = float(trajectory_config["BASE_HEIGHT"])
    z_amp = float(trajectory_config["BASE_Z_AMPLITUDE"])
    z_phase = float(trajectory_config["BASE_Z_PHASE"])
    center = np.asarray(
        trajectory_config["BASE_CIRCLE_CENTER"], dtype=np.float64)
    angle = 2.0 * math.pi * t / period
    return center + np.array([
        radius * math.cos(angle),
        radius * math.sin(angle),
        height + z_amp * math.sin(angle + z_phase),
    ])


def build_rollout(csv_path, fk):
    params = json.load(open(_params_path(csv_path)))
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        raise ValueError(f"empty rollout: {csv_path}")
    q = _columns(rows, "q_meas")
    dq = _columns(rows, "dq_meas")
    if "u_expert_1" in rows[0]:
        u_expert = _columns(rows, "u_expert")
        u_exec = _columns(rows, "u_exec")
    else:
        q_pub = _columns(rows, "q_pub")
        u_expert = q_pub - q
        u_exec = u_expert.copy()

    tc = params["trajectory_config"]
    center = np.asarray(tc["EE_WORLD_CIRCLE_CENTER"], dtype=np.float64)
    radius = float(tc["EE_CIRCLE_RADIUS"])
    period = float(tc["EE_PERIOD_SEC"])
    target_R_world = pin.rpy.rpyToMatrix(
        float(tc["TARGET_ROLL_RAD"]),
        float(tc["TARGET_PITCH_RAD"]),
        float(tc["TARGET_YAW_RAD"]),
    )
    dt = float(params["control_dt"])
    obs = np.empty((len(rows), 47), dtype=np.float32)
    position_jacobian = np.empty((len(rows), 3, 7), dtype=np.float32)
    for k, row in enumerate(rows):
        base_pos = np.array([
            float(row["base_pos_x"]),
            float(row["base_pos_y"]),
            float(row["base_pos_z"]),
        ])
        base_rpy = np.array([
            float(row["base_roll"]),
            float(row["base_pitch"]),
            float(row["base_yaw"]),
        ])
        base_R = pin.rpy.rpyToMatrix(*base_rpy)
        lead = int(row["reference_lead_steps"])
        path_now = float(row["path_target_time"]) - lead * dt
        angle = 2.0 * math.pi * path_now / period
        target_world = center + radius * np.array(
            [math.cos(angle), math.sin(angle), 0.0])
        target_base = base_R.T @ (target_world - base_pos)
        target_R_base = base_R.T @ target_R_world
        ee_pos, _ = fk.pose(q[k])
        position_jacobian[k] = fk.position_jacobian(q[k]).astype(np.float32)
        prev_exec = np.zeros(7) if k == 0 else u_exec[k - 1] / ACTION_SCALE
        base_t = float(row["t_solve"])
        base_prev = _base_position(base_t - dt, tc)
        base_next = _base_position(base_t + dt, tc)
        base_vel = (base_next - base_prev) / (2.0 * dt)
        base_accel = (
            base_next - 2.0 * _base_position(base_t, tc) + base_prev
        ) / (dt * dt)
        condition_feature = np.array([
            float(tc["BASE_CIRCLE_RADIUS"]),
            float(tc["BASE_Z_AMPLITUDE"]),
            1.0 / float(tc["BASE_PERIOD_SEC"]),
        ])
        obs[k] = np.concatenate([
            q[k],
            0.1 * dq[k],
            target_base,
            target_R_base[:, 0],
            target_R_base[:, 1],
            ee_pos,
            target_base - ee_pos,
            np.clip(prev_exec, -1.0, 1.0),
            [math.sin(angle), math.cos(angle)],
            base_vel,
            base_accel,
            condition_feature,
        ]).astype(np.float32)

    action = np.clip(u_expert / ACTION_SCALE, -1.0, 1.0).astype(np.float32)
    sigma = float(params.get("ros_params", {}).get("dart_noise_std", 0.0))
    condition = (
        float(tc["BASE_CIRCLE_RADIUS"]),
        float(tc["BASE_PERIOD_SEC"]),
        float(tc["BASE_Z_AMPLITUDE"]),
        sigma,
    )
    return obs, action, position_jacobian, condition


def windows(obs, action, position_jacobian, obs_horizon, action_horizon,
            val_fraction):
    n = len(obs)
    split = int(round(n * (1.0 - val_fraction)))
    train_obs, train_action, train_jac = [], [], []
    val_obs, val_action, val_jac = [], [], []
    for t in range(obs_horizon - 1, n - action_horizon + 1):
        item_obs = obs[t - obs_horizon + 1:t + 1]
        item_action = action[t:t + action_horizon]
        if t + action_horizon <= split:
            train_obs.append(item_obs)
            train_action.append(item_action)
            train_jac.append(position_jacobian[t])
        elif t >= split + obs_horizon - 1:
            val_obs.append(item_obs)
            val_action.append(item_action)
            val_jac.append(position_jacobian[t])
    return train_obs, train_action, train_jac, val_obs, val_action, val_jac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument(
        "--output", default=f"{ROOT}/data/phase1_diffusion_dataset_v2.npz")
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    csvs = sorted(glob.glob(
        os.path.join(args.data_dir, "moving_base_mpc_loop_*.csv")))
    csvs = [path for path in csvs if os.path.exists(_params_path(path))]
    if not csvs:
        raise SystemExit("no MPC loop CSVs found")
    first_params = json.load(open(_params_path(csvs[0])))
    urdf = first_params.get(
        "urdf_path", f"{ROOT}/assets/arm_resolved.urdf")
    fk = ForwardKinematics(urdf)

    train_o, train_a, train_j, val_o, val_a, val_j = [], [], [], [], [], []
    train_condition_id, train_is_dart = [], []
    val_condition_id, val_is_dart = [], []
    condition_ids = {}
    manifest = []
    for path in csvs:
        obs, action, jacobian, condition = build_rollout(path, fk)
        tr_o, tr_a, tr_j, va_o, va_a, va_j = windows(
            obs, action, jacobian, args.obs_horizon, args.action_horizon,
            args.val_fraction)
        train_o.extend(tr_o); train_a.extend(tr_a); train_j.extend(tr_j)
        val_o.extend(va_o); val_a.extend(va_a); val_j.extend(va_j)
        condition_key = condition[:3]
        condition_id = condition_ids.setdefault(
            condition_key, len(condition_ids))
        is_dart = condition[3] > 0.0
        train_condition_id.extend([condition_id] * len(tr_o))
        train_is_dart.extend([is_dart] * len(tr_o))
        val_condition_id.extend([condition_id] * len(va_o))
        val_is_dart.extend([is_dart] * len(va_o))
        manifest.append({
            "file": os.path.basename(path),
            "rows": len(obs),
            "condition": condition,
            "train_windows": len(tr_o),
            "val_windows": len(va_o),
        })
        print(
            f"[data] {os.path.basename(path)} rows={len(obs)} "
            f"condition={condition} windows={len(tr_o)}/{len(va_o)}")

    train_o = np.asarray(train_o, dtype=np.float32)
    train_a = np.asarray(train_a, dtype=np.float32)
    train_j = np.asarray(train_j, dtype=np.float32)
    val_o = np.asarray(val_o, dtype=np.float32)
    val_a = np.asarray(val_a, dtype=np.float32)
    val_j = np.asarray(val_j, dtype=np.float32)
    obs_mean = train_o.reshape(-1, train_o.shape[-1]).mean(axis=0)
    obs_std = train_o.reshape(-1, train_o.shape[-1]).std(axis=0)
    obs_std = np.maximum(obs_std, 1.0e-4)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        train_obs=train_o,
        train_action=train_a,
        train_position_jacobian=train_j,
        train_condition_id=np.asarray(train_condition_id, dtype=np.int64),
        train_is_dart=np.asarray(train_is_dart, dtype=np.bool_),
        val_obs=val_o,
        val_action=val_a,
        val_position_jacobian=val_j,
        val_condition_id=np.asarray(val_condition_id, dtype=np.int64),
        val_is_dart=np.asarray(val_is_dart, dtype=np.bool_),
        obs_mean=obs_mean.astype(np.float32),
        obs_std=obs_std.astype(np.float32),
        obs_horizon=np.int64(args.obs_horizon),
        action_horizon=np.int64(args.action_horizon),
        action_scale=np.float32(ACTION_SCALE),
        manifest_json=np.array(json.dumps(manifest)),
    )
    print(
        f"[data] wrote {args.output} train={len(train_o)} val={len(val_o)} "
        f"obs={train_o.shape[1:]} action={train_a.shape[1:]}")


if __name__ == "__main__":
    main()
