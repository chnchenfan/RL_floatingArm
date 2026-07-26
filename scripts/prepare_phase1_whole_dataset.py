#!/usr/bin/env python3
"""Build whole-task Phase-1 observation/action-chunk dataset from MPC loop CSVs.

Differences from the single-circle ``prepare_phase1_dataset.py``:
- targets (position + rotation + phase) come from the frozen exact whole spec
  (``task_spec/whole_trajectory.py``), not a hardcoded circle;
- action scale is the whole-config MPC contract 0.04 rad/tick (asserted from
  each run's params snapshot);
- train/validation split is at ROLLOUT level (no window-level time leakage);
- every run must match the frozen schema (trajectory shape, period, source
  hash of the collection-time config snapshot is recorded in the manifest);
- also emits a per-phase state bank (q, dq indexed by whole phase tick) from
  the clean runs, used by the Phase-2 env for random-phase resets.

Run with the SYSTEM python3 (Pinocchio via ROS env):
    bash -c 'set +u; source /opt/ros/humble/setup.bash; \
             source ~/code/windylab_ws/install/setup.bash; \
             /usr/bin/python3 scripts/prepare_phase1_whole_dataset.py --runs ...'
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import pinocchio as pin

ROOT = "/home/windylab/code/isaac_arm_rl"
sys.path.insert(0, ROOT)
from task_spec import whole_trajectory as wt  # noqa: E402

DEFAULT_DATA = "/home/windylab/code/windylab_ws/src/arm-platform/demo/data"
ACTION_SCALE = 0.04          # whole-config MPC max_joint_step
CONTROL_DT = 0.02


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
            self.model, self.data, q, self.frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return np.asarray(jacobian[:3], dtype=np.float64).copy()


def _base_position(t, tc):
    radius = float(tc["BASE_CIRCLE_RADIUS"])
    period = float(tc["BASE_PERIOD_SEC"])
    height = float(tc["BASE_HEIGHT"])
    z_amp = float(tc["BASE_Z_AMPLITUDE"])
    z_phase = float(tc["BASE_Z_PHASE"])
    center = np.asarray(tc["BASE_CIRCLE_CENTER"], dtype=np.float64)
    angle = 2.0 * math.pi * t / period
    return center + np.array([
        radius * math.cos(angle),
        radius * math.sin(angle),
        height + z_amp * math.sin(angle + z_phase),
    ])


def _check_run(params, csv_path):
    rp = params.get("ros_params", {})
    tc = params.get("trajectory_config", {})
    if rp.get("trajectory_shape") != "whole":
        raise ValueError(f"{csv_path}: trajectory_shape != whole")
    period = float(tc.get("EE_TRAJECTORY_PERIOD_SEC", float("nan")))
    if abs(period - wt.WHOLE_PERIOD_SEC) > 1e-9:
        raise ValueError(
            f"{csv_path}: period {period!r} != frozen {wt.WHOLE_PERIOD_SEC!r}"
            " (old whole definition?)")
    step = float(rp.get("max_joint_step", float("nan")))
    if abs(step - ACTION_SCALE) > 1e-12:
        raise ValueError(f"{csv_path}: max_joint_step {step} != {ACTION_SCALE}")
    if not params.get("experiment_valid", False):
        raise ValueError(f"{csv_path}: experiment_valid is False")
    if abs(float(params.get("control_dt", CONTROL_DT)) - CONTROL_DT) > 1e-9:
        raise ValueError(f"{csv_path}: control_dt != 0.02")


def build_rollout(csv_path, fk):
    params = json.load(open(_params_path(csv_path)))
    _check_run(params, csv_path)
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        raise ValueError(f"empty rollout: {csv_path}")
    tc = params["trajectory_config"]
    rp = params["ros_params"]
    dt = float(params["control_dt"])

    q = _columns(rows, "q_meas")
    dq = _columns(rows, "dq_meas")
    u_expert = _columns(rows, "u_expert")
    u_exec = _columns(rows, "u_exec")

    # feedback-quality gate: stale /joint_states (FastDDS shm trap) shows up
    # as a large recv-vs-stamp gap; refuse silently degraded runs.
    js_delay_ms = np.array([
        (float(r["js_recv_wall_sec"]) - float(r["js_header_stamp_sec"]))
        * 1000.0 for r in rows])
    if np.median(js_delay_ms) > 10.0:
        raise ValueError(
            f"{csv_path}: median joint_states delay "
            f"{np.median(js_delay_ms):.1f} ms (stale-feedback run)")

    n = len(rows)
    obs = np.empty((n, 47), dtype=np.float32)
    position_jacobian = np.empty((n, 3, 7), dtype=np.float32)
    pen = np.zeros(n, dtype=np.bool_)
    task = np.zeros(n, dtype=np.int64)
    stroke = np.zeros(n, dtype=np.int64)
    phase_tick = np.zeros(n, dtype=np.int64)
    first_error = np.array([float(r["first_error_m"]) for r in rows])

    condition_feature = np.array([
        float(tc["BASE_CIRCLE_RADIUS"]),
        float(tc["BASE_Z_AMPLITUDE"]),
        1.0 / float(tc["BASE_PERIOD_SEC"]),
    ])
    for k, row in enumerate(rows):
        base_pos = np.array([
            float(row["base_pos_x"]),
            float(row["base_pos_y"]),
            float(row["base_pos_z"]),
        ])
        base_R = pin.rpy.rpyToMatrix(
            float(row["base_roll"]),
            float(row["base_pitch"]),
            float(row["base_yaw"]),
        )
        lead = int(row["reference_lead_steps"])
        path_now = float(row["path_target_time"]) - lead * dt
        sample = wt.sample(path_now)
        target_base = base_R.T @ (sample.position - base_pos)
        target_R_base = base_R.T @ sample.rotation
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
        angle = 2.0 * math.pi * path_now / wt.WHOLE_PERIOD_SEC
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
        pen[k] = sample.pen_down
        task[k] = sample.task_index
        stroke[k] = sample.stroke_id
        phase_tick[k] = int(math.floor(
            (path_now % wt.WHOLE_PERIOD_SEC) / dt)) % 3652

    action = np.clip(u_expert / ACTION_SCALE, -1.0, 1.0).astype(np.float32)
    sigma = float(rp.get("dart_noise_std", 0.0))
    meta = {
        "file": os.path.basename(csv_path),
        "rows": n,
        "dart_sigma": sigma,
        "dart_seed": int(rp.get("dart_noise_seed", 0)),
        "path_time_span": [
            float(rows[0]["path_target_time"]),
            float(rows[-1]["path_target_time"])],
        "first_error_mm_mean": float(first_error.mean() * 1000.0),
        "first_error_mm_p95": float(np.percentile(first_error, 95) * 1000.0),
        "js_delay_ms_p50": float(np.median(js_delay_ms)),
        "task_coverage": sorted(set(int(x) for x in task if x >= 0)),
        "schema_version": wt.WHOLE_SCHEMA_VERSION,
        "source_sha256": wt.SOURCE_SHA256,
    }
    prev_action = np.zeros_like(u_exec)
    prev_action[1:] = np.clip(u_exec[:-1] / ACTION_SCALE, -1.0, 1.0)
    extras = {
        "pen": pen, "task": task, "stroke": stroke,
        "phase_tick": phase_tick, "q": q, "dq": dq,
        "prev_action": prev_action,
        "first_error": first_error, "is_dart": sigma > 0.0,
    }
    return obs, action, position_jacobian, meta, extras


def windows(obs, action, jacobian, extras, obs_horizon, action_horizon):
    """All valid windows of one continuous rollout (no split here)."""
    n = len(obs)
    out_o, out_a, out_j, out_task = [], [], [], []
    for t in range(obs_horizon - 1, n - action_horizon + 1):
        out_o.append(obs[t - obs_horizon + 1:t + 1])
        out_a.append(action[t:t + action_horizon])
        out_j.append(jacobian[t])
        out_task.append(extras["task"][t])
    return out_o, out_a, out_j, out_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="loop CSV time stamps, e.g. 20260726_021045")
    parser.add_argument(
        "--val-runs", nargs="+", default=[],
        help="stamps held out entirely for validation")
    parser.add_argument(
        "--output", default=f"{ROOT}/data/phase1_whole_v2_dataset.npz")
    parser.add_argument(
        "--state-bank-output", default=f"{ROOT}/data/whole_v2_state_bank.npz")
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=8)
    args = parser.parse_args()

    stamps = list(args.runs) + [s for s in args.val_runs
                                if s not in args.runs]
    first_csv = os.path.join(
        args.data_dir, f"moving_base_mpc_loop_{stamps[0]}.csv")
    first_params = json.load(open(_params_path(first_csv)))
    urdf = first_params.get("urdf_path", f"{ROOT}/assets/arm_resolved.urdf")
    fk = ForwardKinematics(urdf)

    train, val = ([], [], [], [], []), ([], [], [], [], [])
    manifest = []
    bank_phase, bank_q, bank_dq, bank_prev = [], [], [], []
    for stamp in stamps:
        path = os.path.join(
            args.data_dir, f"moving_base_mpc_loop_{stamp}.csv")
        obs, action, jacobian, meta, extras = build_rollout(path, fk)
        o, a, j, tsk = windows(
            obs, action, jacobian, extras,
            args.obs_horizon, args.action_horizon)
        is_val = stamp in args.val_runs
        dest = val if is_val else train
        dest[0].extend(o); dest[1].extend(a); dest[2].extend(j)
        dest[3].extend([extras["is_dart"]] * len(o))
        dest[4].extend(tsk)
        meta["split"] = "val" if is_val else "train"
        meta["windows"] = len(o)
        manifest.append(meta)
        if not extras["is_dart"]:
            bank_phase.append(extras["phase_tick"])
            bank_q.append(extras["q"])
            bank_dq.append(extras["dq"])
            bank_prev.append(extras["prev_action"])
        print(f"[data] {meta['file']} rows={meta['rows']} "
              f"sigma={meta['dart_sigma']} split={meta['split']} "
              f"windows={len(o)} fe_mean={meta['first_error_mm_mean']:.3f}mm "
              f"tasks={meta['task_coverage']}")

    train_o = np.asarray(train[0], dtype=np.float32)
    train_a = np.asarray(train[1], dtype=np.float32)
    train_j = np.asarray(train[2], dtype=np.float32)
    val_o = np.asarray(val[0], dtype=np.float32)
    val_a = np.asarray(val[1], dtype=np.float32)
    val_j = np.asarray(val[2], dtype=np.float32)
    obs_mean = train_o.reshape(-1, train_o.shape[-1]).mean(axis=0)
    obs_std = np.maximum(
        train_o.reshape(-1, train_o.shape[-1]).std(axis=0), 1.0e-4)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        train_obs=train_o,
        train_action=train_a,
        train_position_jacobian=train_j,
        train_condition_id=np.zeros(len(train_o), dtype=np.int64),
        train_is_dart=np.asarray(train[3], dtype=np.bool_),
        train_task_id=np.asarray(train[4], dtype=np.int64),
        val_obs=val_o,
        val_action=val_a,
        val_position_jacobian=val_j,
        val_condition_id=np.zeros(len(val_o), dtype=np.int64),
        val_is_dart=np.asarray(val[3], dtype=np.bool_),
        val_task_id=np.asarray(val[4], dtype=np.int64),
        obs_mean=obs_mean.astype(np.float32),
        obs_std=obs_std.astype(np.float32),
        obs_horizon=np.int64(args.obs_horizon),
        action_horizon=np.int64(args.action_horizon),
        action_scale=np.float32(ACTION_SCALE),
        schema_version=np.array(wt.WHOLE_SCHEMA_VERSION),
        source_sha256=np.array(wt.SOURCE_SHA256),
        manifest_json=np.array(json.dumps(manifest)),
    )
    print(f"[data] wrote {args.output} train={len(train_o)} val={len(val_o)}")

    # per-phase state bank from clean runs (Phase-2 random-phase resets)
    phase = np.concatenate(bank_phase)
    qs = np.concatenate(bank_q)
    dqs = np.concatenate(bank_dq)
    prevs = np.concatenate(bank_prev)
    order = np.argsort(phase, kind="stable")
    np.savez_compressed(
        args.state_bank_output,
        phase_tick=phase[order].astype(np.int64),
        q=qs[order].astype(np.float32),
        dq=dqs[order].astype(np.float32),
        prev_action=prevs[order].astype(np.float32),
        ticks_per_cycle=np.int64(3652),
        schema_version=np.array(wt.WHOLE_SCHEMA_VERSION),
    )
    covered = len(set(phase.tolist()))
    print(f"[data] wrote {args.state_bank_output} samples={len(phase)} "
          f"distinct_phase_ticks={covered}/3652")


if __name__ == "__main__":
    main()
