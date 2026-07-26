#!/usr/bin/env python3
"""Validate Phase-0 DART rollouts and quantify expert recovery response."""

import argparse
import csv
import glob
import os

import numpy as np


def _vec(rows, prefix):
    return np.array([
        [float(row[f"{prefix}_{i}"]) for i in range(1, 8)]
        for row in rows
    ])


def analyze(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    required = {
        "q_expert_1", "q_pub_1", "u0_1", "u_expert_1", "u_exec_1",
        "dart_noise_applied_1", "dart_noise_std_rad", "dart_noise_clipped",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path}: missing Phase-0 fields {sorted(missing)}")

    q = _vec(rows, "q_meas")
    q_next = _vec(rows, "q_next")
    q_expert = _vec(rows, "q_expert")
    q_pub = _vec(rows, "q_pub")
    u0 = _vec(rows, "u0")
    u_expert = _vec(rows, "u_expert")
    u_exec = _vec(rows, "u_exec")
    noise = _vec(rows, "dart_noise_applied")
    clipped = np.array([int(row["dart_noise_clipped"]) for row in rows])
    error_mm = np.array([
        float(row["first_error_m"]) * 1000.0 for row in rows
        if row["first_error_m"] not in ("", "nan")
    ])

    checks = {
        "raw_label": np.max(np.abs((q_next - q) - u0)),
        "expert_label": np.max(np.abs((q_expert - q) - u_expert)),
        "executed_action": np.max(np.abs((q_pub - q) - u_exec)),
        "noise_reconstruction": np.max(np.abs((q_expert + noise) - q_pub)),
    }
    if max(checks.values()) > 1e-9:
        raise ValueError(f"{path}: action reconstruction failed: {checks}")
    if np.max(np.abs(u_exec)) > 0.020000001:
        raise ValueError(f"{path}: executed action violates 0.02-rad contract")

    recovery = []
    for lag in range(1, min(7, len(rows))):
        x = noise[:-lag]
        y = u_expert[lag:]
        recovery.append({
            "lag": lag,
            "covariance": float(np.mean(np.sum(
                (x - x.mean(0)) * (y - y.mean(0)), axis=1))),
            "opposing": float(np.mean(np.sum(x * y, axis=1) < 0.0)),
        })
    best = min(recovery, key=lambda item: item["covariance"])
    return {
        "path": path,
        "rows": len(rows),
        "sigma": float(rows[0]["dart_noise_std_rad"]),
        "noise_std": float(noise.std()),
        "noise_nonzero": float(np.mean(np.abs(noise) > 1e-12)),
        "clipped": float(clipped.mean()),
        "error_mean_mm": float(error_mm.mean()),
        "error_p95_mm": float(np.percentile(error_mm, 95)),
        "error_max_mm": float(error_mm.max()),
        "recovery_lag": best["lag"],
        "recovery_opposing": best["opposing"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths", nargs="*",
        default=glob.glob(
            "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
            "moving_base_mpc_loop_*.csv"))
    args = parser.parse_args()

    results = []
    for path in sorted(args.paths):
        try:
            result = analyze(path)
        except ValueError as exc:
            if "missing Phase-0 fields" in str(exc):
                continue
            raise
        if result["sigma"] > 0.0:
            results.append(result)
    if not results:
        raise SystemExit("no DART rollouts found")

    print("file rows sigma noise_std clip% err_mean/p95/max(mm) recovery")
    for r in results:
        print(
            f"{os.path.basename(r['path'])} {r['rows']} "
            f"{r['sigma']:.4f} {r['noise_std']:.5f} "
            f"{100*r['clipped']:.1f}% "
            f"{r['error_mean_mm']:.2f}/{r['error_p95_mm']:.2f}/"
            f"{r['error_max_mm']:.2f} "
            f"lag={r['recovery_lag']} oppose={100*r['recovery_opposing']:.1f}%"
        )


if __name__ == "__main__":
    main()
