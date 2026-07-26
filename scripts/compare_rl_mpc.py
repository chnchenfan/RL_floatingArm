#!/usr/bin/env python3
"""RL vs MPC(=pinocchio IK) comparison on the SAME config-matched disturbance.

Tracking error: from data/rollout_rl.csv (RL) and data/rollout_task.csv (IK/MPC),
both world-frame ee vs tgt over the drawing phases.
Latency: RL from logs/eval2.log; MPC by timing the pinocchio IK solve here.

    python3 scripts/compare_rl_mpc.py
"""
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/windylab/code/windylab_ws/src/arm-platform/demo")
from rollout_io import read_rollout                       # noqa: E402

RL = "/home/windylab/code/isaac_arm_rl/data/rollout_rl.csv"
MPC = "/home/windylab/code/isaac_arm_rl/data/rollout_task.csv"
EVAL_LOG = "/home/windylab/code/isaac_arm_rl/logs/eval2.log"


def track_stats(path):
    r = read_rollout(path)
    if r["ee"] is None or r["tgt"] is None:
        return None
    err = np.linalg.norm(r["ee"] - r["tgt"], axis=1) * 1000.0
    m = r["drawing"] if r["drawing"] is not None else np.ones(len(err), bool)
    e = err[m]
    return dict(mean=e.mean(), p95=np.percentile(e, 95), max=e.max())


def mpc_latency():
    """Time the pinocchio IK solve on the config-matched task (MPC-equivalent)."""
    import pinocchio as pin
    from task_spec.task_model import TaskModel
    from pinocchio_ik import PinocchioIK
    tm = TaskModel(); ik = PinocchioIK()
    T = int(tm.period * 50)
    q = None; lat = []
    for k in range(T):
        p_b, R_b = tm.ee_target_base(k / 50.0)
        t0 = time.perf_counter()
        out = ik.solve(p_b, target_rotation=R_b, q_init=q, rot_weight=0.2)
        lat.append((time.perf_counter() - t0) * 1e3)
        q = out[0] if isinstance(out, (tuple, list)) else out
    lat = np.array(lat)
    return dict(mean=lat.mean(), p95=np.percentile(lat, 95), max=lat.max())


def rl_latency():
    if not os.path.isfile(EVAL_LOG):
        return None
    txt = open(EVAL_LOG).read()
    m = re.search(r"inference latency: mean=([\d.]+)ms p95=([\d.]+)ms max=([\d.]+)ms", txt)
    if not m:
        return None
    return dict(mean=float(m[1]), p95=float(m[2]), max=float(m[3]))


def main():
    rl_t = track_stats(RL)
    mpc_t = track_stats(MPC)
    print("=" * 66)
    print("RL vs MPC  (config disturbance 1Hz/2cm/0.1deg, drawing phases)")
    print("=" * 66)
    print(f"{'':14}{'tracking mean':>16}{'p95':>10}{'max':>10}")
    if mpc_t:
        print(f"{'MPC (IK)':14}{mpc_t['mean']:>13.2f}mm{mpc_t['p95']:>8.2f}{mpc_t['max']:>10.2f}")
    if rl_t:
        print(f"{'RL policy':14}{rl_t['mean']:>13.2f}mm{rl_t['p95']:>8.2f}{rl_t['max']:>10.2f}")
    print("-" * 66)
    print("Latency per control tick:")
    ml = mpc_latency(); rl_l = rl_latency()
    print(f"{'':14}{'mean':>16}{'p95':>10}{'max':>10}")
    print(f"{'MPC (IK solve)':14}{ml['mean']:>14.3f}ms{ml['p95']:>8.3f}{ml['max']:>10.3f}")
    if rl_l:
        print(f"{'RL (net fwd)':14}{rl_l['mean']:>14.3f}ms{rl_l['p95']:>8.3f}{rl_l['max']:>10.3f}")
        print("-" * 66)
        print(f"latency speedup (mean): {ml['mean']/rl_l['mean']:.1f}x  "
              f"(p95): {ml['p95']/rl_l['p95']:.1f}x")
    print("=" * 66)


if __name__ == "__main__":
    main()
