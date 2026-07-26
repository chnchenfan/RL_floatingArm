#!/usr/bin/env python3
"""Compare selected Phase-2 results with the three clean acados MPC logs.

Run with the system Python because it provides Pinocchio:

    python3 scripts/compare_phase2_mpc.py

The report keeps the evaluation domains explicit.  Phase-2 metrics come from
the paired Isaac validation with inertial coupling; MPC metrics are
reconstructed from logged measured joints and targets.  They share task
parameters but are not measurements on the same plant.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pinocchio as pin


ROOT = Path("/home/windylab/code/isaac_arm_rl")
DEFAULT_CONFIG = ROOT / "config/phase2_selected_policy.json"
DEFAULT_LATENCY = ROOT / "reports/phase3/phase2_policy_latency.json"
DEFAULT_JSON = ROOT / "reports/phase3/phase2_vs_mpc.json"
DEFAULT_MARKDOWN = ROOT / "reports/phase3/PHASE2_VS_MPC.md"
MPC_LOGS = {
    "easy": Path(
        "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
        "moving_base_mpc_loop_20260724_051521.csv"
    ),
    "medium": Path(
        "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
        "moving_base_mpc_loop_20260724_051912.csv"
    ),
    "hard": Path(
        "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
        "moving_base_mpc_loop_20260724_052123.csv"
    ),
}


class ForwardKinematics:
    def __init__(self, urdf_path):
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId("link7")
        if self.frame_id >= len(self.model.frames):
            raise ValueError("link7 frame not found")

    def pose(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.frame_id]
        return placement.translation.copy(), placement.rotation.copy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--latency", type=Path, default=DEFAULT_LATENCY)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument(
        "--output-markdown", type=Path, default=DEFAULT_MARKDOWN
    )
    return parser.parse_args()


def _load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _params_path(csv_path):
    return Path(
        str(csv_path).replace("_loop_", "_params_").replace(".csv", ".json")
    )


def _distribution(values, unit_suffix):
    values = np.asarray(values, dtype=np.float64)
    return {
        f"mean_{unit_suffix}": float(values.mean()),
        f"rms_{unit_suffix}": float(np.sqrt(np.mean(values**2))),
        f"p95_{unit_suffix}": float(np.percentile(values, 95)),
        f"p99_{unit_suffix}": float(np.percentile(values, 99)),
        f"max_{unit_suffix}": float(values.max()),
    }


def _bool_fraction(rows, column):
    if column not in rows[0]:
        return None
    truthy = {"1", "true", "yes"}
    return float(
        np.mean([str(row[column]).strip().lower() in truthy for row in rows])
    )


def _analyze_mpc_log(path, fk):
    params = _load_json(_params_path(path))
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    trajectory = params["trajectory_config"]
    target_center = np.asarray(
        trajectory["EE_WORLD_CIRCLE_CENTER"], dtype=np.float64
    )
    target_radius = float(trajectory["EE_CIRCLE_RADIUS"])
    target_period = float(trajectory["EE_PERIOD_SEC"])
    target_rotation_world = pin.rpy.rpyToMatrix(
        float(trajectory["TARGET_ROLL_RAD"]),
        float(trajectory["TARGET_PITCH_RAD"]),
        float(trajectory["TARGET_YAW_RAD"]),
    )
    control_dt = float(params["control_dt"])
    position_errors_mm = []
    orientation_errors_deg = []
    for row in rows:
        q = np.asarray(
            [float(row[f"q_meas_{joint}"]) for joint in range(1, 8)]
        )
        base_position = np.asarray(
            [
                float(row["base_pos_x"]),
                float(row["base_pos_y"]),
                float(row["base_pos_z"]),
            ]
        )
        base_rotation = pin.rpy.rpyToMatrix(
            float(row["base_roll"]),
            float(row["base_pitch"]),
            float(row["base_yaw"]),
        )
        reference_lead = int(row["reference_lead_steps"])
        current_target_time = (
            float(row["path_target_time"]) - reference_lead * control_dt
        )
        target_angle = (
            2.0 * math.pi * current_target_time / target_period
        )
        target_world = target_center + target_radius * np.asarray(
            [math.cos(target_angle), math.sin(target_angle), 0.0]
        )
        target_base = base_rotation.T @ (target_world - base_position)
        target_rotation_base = base_rotation.T @ target_rotation_world
        ee_position, ee_rotation = fk.pose(q)
        position_errors_mm.append(
            1000.0 * np.linalg.norm(target_base - ee_position)
        )
        rotation_error = ee_rotation.T @ target_rotation_base
        cosine = np.clip((np.trace(rotation_error) - 1.0) * 0.5, -1.0, 1.0)
        orientation_errors_deg.append(math.degrees(math.acos(cosine)))

    solve_ms = np.asarray(
        [float(row["solve_time_ms"]) for row in rows], dtype=np.float64
    )
    publish_ms = np.asarray(
        [float(row["solve_to_publish_ms"]) for row in rows],
        dtype=np.float64,
    )
    result = {
        "source": str(path),
        "rows": len(rows),
        "task": {
            "base_radius_m": float(trajectory["BASE_CIRCLE_RADIUS"]),
            "base_period_s": float(trajectory["BASE_PERIOD_SEC"]),
            "base_z_amplitude_m": float(
                trajectory["BASE_Z_AMPLITUDE"]
            ),
        },
        "measured_position": _distribution(position_errors_mm, "mm"),
        "measured_orientation": _distribution(
            orientation_errors_deg, "deg"
        ),
        "acados_solve_latency": _distribution(solve_ms, "ms"),
        "solve_to_publish_latency": _distribution(publish_ms, "ms"),
        "status_fraction": {
            name: _bool_fraction(rows, name)
            for name in (
                "solver_success",
                "command_usable",
                "tracking_ok",
                "plan_feasible",
                "guard_block",
            )
        },
        "_position_values": np.asarray(position_errors_mm),
        "_orientation_values": np.asarray(orientation_errors_deg),
        "_solve_values": solve_ms,
        "_publish_values": publish_ms,
    }
    return result


def _public_copy(result):
    return {
        key: value
        for key, value in result.items()
        if not key.startswith("_")
    }


def _select_phase2_result(selection, summary):
    checkpoint_stem = Path(selection["checkpoint"]).stem
    residual_scale = float(selection["deployment"]["residual_scale"])
    matches = [
        result
        for result in summary["results"]
        if result["name"].startswith(checkpoint_stem)
        and math.isclose(
            float(result["residual_scale"]),
            residual_scale,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            "could not uniquely identify selected Phase-2 result: "
            f"checkpoint={checkpoint_stem} scale={residual_scale}"
        )
    return matches[0]


def _improvement(reference, candidate):
    return 100.0 * (reference - candidate) / reference


def _markdown(report):
    lines = [
        "# Phase 2 residual policy vs acados MPC",
        "",
        "## 结论",
        "",
        (
            f"- 三种工况合并后，Phase 2 在 Isaac 惯性耦合验证中的位置"
            f"均值为 **{report['comparison']['overall']['phase2_mean_mm']:.3f} "
            f"mm**；acados 日志按实测关节 FK 重建后的均值为 "
            f"**{report['comparison']['overall']['mpc_mean_mm']:.3f} mm**。"
        ),
        (
            f"- 这个跨域并列值对应 "
            f"**{report['comparison']['overall']['phase2_mean_improvement_percent']:.1f}%**"
            " 的较低均值，但不能单独证明同一真实机械臂上已经超越 MPC。"
        ),
        (
            f"- 当前完整 Phase 1+2 控制器单实例推理均值 "
            f"**{report['phase2_policy_latency']['full_controller']['mean_ms']:.2f} "
            f"ms**，50 Hz 截止期通过率 "
            f"**{100.0 * report['phase2_policy_latency']['full_controller_deadline_fraction']:.1f}%**。"
        ),
        (
            "- 完整策略当前并不比 acados 求解本身更快；Phase 2 残差网络很快，"
            "主要延迟来自 Phase 1 的 8 步 DDIM。"
        ),
        "",
        "## 位置跟踪",
        "",
        "| 工况 | acados 日志 mean / p95 / max (mm) | Phase 2 Isaac mean / p95 / max (mm) | 均值变化 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("easy", "medium", "hard"):
        item = report["comparison"]["conditions"][name]
        lines.append(
            f"| {name} | {item['mpc_mean_mm']:.3f} / "
            f"{item['mpc_p95_mm']:.3f} / {item['mpc_max_mm']:.3f} | "
            f"{item['phase2_mean_mm']:.3f} / "
            f"{item['phase2_p95_mm']:.3f} / "
            f"{item['phase2_max_mm']:.3f} | "
            f"{item['phase2_mean_improvement_percent']:+.1f}% |"
        )
    lines.extend(
        [
            "",
            "正的“均值变化”表示 Phase 2 数值更低；负值表示 acados 更低。",
            "",
            "## 延迟",
            "",
            "| 路径 | mean (ms) | p95 (ms) | max (ms) |",
            "|---|---:|---:|---:|",
        ]
    )
    latency_rows = report["comparison"]["latency"]
    for label, item in latency_rows.items():
        lines.append(
            f"| {label} | {item['mean_ms']:.3f} | "
            f"{item['p95_ms']:.3f} | {item['max_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 口径与限制",
            "",
            "- acados 位置误差不是求解器内部预测误差；它由每行 `q_meas`、记录的基座位姿和当前圆轨迹目标重新计算。",
            "- Phase 2 来自 4096 环境、400 tick、随机 MPC handoff、关节噪声 0.002 rad、开启惯性耦合的 Isaac 验证。",
            "- 两组共享三套任务参数，但执行系统和初始状态分布不同。因此当前报告适合判断方向，不是严格的同平台 A/B 证明。",
            "- 严格结论还需要在同一真实 ROS 控制管线上运行导出的 Phase 2 策略并记录同样字段。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    selection = _load_json(args.config)
    summary = _load_json(selection["validation"]["summary"])
    latency = _load_json(args.latency)
    phase2 = _select_phase2_result(selection, summary)

    first_params = _load_json(_params_path(next(iter(MPC_LOGS.values()))))
    fk = ForwardKinematics(first_params["urdf_path"])
    mpc_internal = {
        name: _analyze_mpc_log(path, fk)
        for name, path in MPC_LOGS.items()
    }
    aggregate_position = np.concatenate(
        [item["_position_values"] for item in mpc_internal.values()]
    )
    aggregate_orientation = np.concatenate(
        [item["_orientation_values"] for item in mpc_internal.values()]
    )
    aggregate_solve = np.concatenate(
        [item["_solve_values"] for item in mpc_internal.values()]
    )
    aggregate_publish = np.concatenate(
        [item["_publish_values"] for item in mpc_internal.values()]
    )
    mpc_aggregate = {
        "rows": int(sum(item["rows"] for item in mpc_internal.values())),
        "measured_position": _distribution(aggregate_position, "mm"),
        "measured_orientation": _distribution(
            aggregate_orientation, "deg"
        ),
        "acados_solve_latency": _distribution(aggregate_solve, "ms"),
        "solve_to_publish_latency": _distribution(
            aggregate_publish, "ms"
        ),
    }

    condition_comparison = {}
    for name in ("easy", "medium", "hard"):
        mpc = mpc_internal[name]["measured_position"]
        rl = phase2["conditions"][name]
        condition_comparison[name] = {
            "mpc_mean_mm": mpc["mean_mm"],
            "mpc_p95_mm": mpc["p95_mm"],
            "mpc_max_mm": mpc["max_mm"],
            "phase2_mean_mm": rl["mean_mm"],
            "phase2_p95_mm": rl["p95_mm"],
            "phase2_max_mm": rl["max_mm"],
            "phase2_mean_improvement_percent": _improvement(
                mpc["mean_mm"], rl["mean_mm"]
            ),
        }

    full_latency = latency["full_controller"]
    comparison = {
        "conditions": condition_comparison,
        "overall": {
            "mpc_mean_mm": mpc_aggregate["measured_position"]["mean_mm"],
            "mpc_p95_mm": mpc_aggregate["measured_position"]["p95_mm"],
            "phase2_mean_mm": phase2["overall"]["mean_mm"],
            "phase2_p95_mm": phase2["overall"]["p95_mm"],
            "phase2_mean_improvement_percent": _improvement(
                mpc_aggregate["measured_position"]["mean_mm"],
                phase2["overall"]["mean_mm"],
            ),
        },
        "latency": {
            "acados solve": mpc_aggregate["acados_solve_latency"],
            "acados solve→publish": mpc_aggregate[
                "solve_to_publish_latency"
            ],
            "Phase 1+2 policy": full_latency,
            "Phase 2 residual only": latency[
                "phase2_residual_and_composition"
            ],
        },
    }
    report = {
        "schema_version": 1,
        "scope": {
            "phase2": (
                "Isaac paired random-handoff validation with inertial coupling"
            ),
            "mpc": (
                "measured-joint FK reconstruction from clean acados ROS logs"
            ),
            "strict_same_plant_ab_test": False,
        },
        "phase2_selection": {
            "checkpoint": selection["checkpoint"],
            "residual_scale": selection["deployment"]["residual_scale"],
            "validation_summary": selection["validation"]["summary"],
            "num_envs": summary["num_envs"],
            "steps": summary["steps"],
            "overall": phase2["overall"],
            "conditions": phase2["conditions"],
        },
        "phase2_policy_latency": latency,
        "mpc": {
            "aggregate": mpc_aggregate,
            "conditions": {
                name: _public_copy(item)
                for name, item in mpc_internal.items()
            },
        },
        "comparison": comparison,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    with args.output_markdown.open("w", encoding="utf-8") as stream:
        stream.write(_markdown(report))
    print(_markdown(report))
    print(f"[compare] wrote {args.output_json}")
    print(f"[compare] wrote {args.output_markdown}")


if __name__ == "__main__":
    main()
