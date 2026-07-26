#!/usr/bin/env python3
"""Pointwise parity between the frozen RL whole spec and the authoritative
MPC trajectory source.

Run with the SYSTEM python3 (needs Pinocchio, provided by the ROS install):

    python3 tests/test_whole_trajectory_parity.py

Checks position, rotation, pen_down, stroke_id, base nominal yaw on
- a dense uniform grid over one period (plus wraparound into cycle 2),
- both sides of every internal segment boundary,
and verifies the frozen source hash and period.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys

import numpy as np

RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = "/home/windylab/code/windylab_ws/src/arm-platform/demo"
sys.path.insert(0, RL_ROOT)
sys.path.insert(0, DEMO_DIR)

from task_spec import whole_trajectory as wt  # noqa: E402
import moving_base_trajectory_config as auth  # noqa: E402

EPS = 1e-7          # seconds, boundary probe offset
ATOL = 1e-9         # metres / rotation entries


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def check_source_hash():
    digest = hashlib.sha256(
        open(os.path.join(DEMO_DIR, "moving_base_trajectory_config.py"),
             "rb").read()).hexdigest()
    if digest != wt.SOURCE_SHA256:
        fail(
            "authoritative source hash changed: "
            f"{digest} != frozen {wt.SOURCE_SHA256}; re-freeze the schema "
            "before collecting data or training")
    print(f"[parity] source sha256 ok: {digest}")


def check_periods():
    auth_period = auth.ee_trajectory_period("whole")
    if abs(auth_period - wt.WHOLE_PERIOD_SEC) > 1e-12:
        fail(f"period mismatch {auth_period!r} != {wt.WHOLE_PERIOD_SEC!r}")
    if abs(auth_period - 73.03949045888578) > 1e-9:
        fail(f"authoritative whole period drifted: {auth_period!r}")
    once = auth._WINDYLAB_ONCE_PERIOD_SEC
    if abs(once - wt.WINDYLAB_ONCE_PERIOD_SEC) > 1e-12:
        fail(f"windylab once period mismatch {once!r}")
    rep = auth._WINDYLAB_PERIOD_SEC
    if abs(rep - wt.WINDYLAB_PERIOD_SEC) > 1e-12:
        fail(f"windylab repeat period mismatch {rep!r}")
    print(f"[parity] period ok: {auth_period!r} "
          f"(windylab once {once!r}, standalone {rep!r})")


def compare_at(t):
    ref = auth.ee_trajectory_sample(t, "whole")
    got = wt.sample(t)
    if not np.allclose(ref.position, got.position, atol=ATOL, rtol=0.0):
        fail(f"position mismatch at t={t!r}: {ref.position} vs {got.position} "
             f"(|d|={np.linalg.norm(ref.position - got.position):.3e})")
    if not np.allclose(ref.rotation, got.rotation, atol=1e-8, rtol=0.0):
        fail(f"rotation mismatch at t={t!r}:\n{ref.rotation}\nvs\n"
             f"{got.rotation}")
    if bool(ref.pen_down) != bool(got.pen_down):
        fail(f"pen_down mismatch at t={t!r}: {ref.pen_down} vs {got.pen_down}")
    if int(ref.stroke_id) != int(got.stroke_id):
        fail(f"stroke_id mismatch at t={t!r}: "
             f"{ref.stroke_id} vs {got.stroke_id}")
    yaw_ref = auth.base_nominal_yaw(t, "whole")
    if abs(yaw_ref - got.base_yaw) > 1e-12:
        fail(f"base yaw mismatch at t={t!r}: {yaw_ref} vs {got.base_yaw}")


def main():
    check_source_hash()
    check_periods()

    period = wt.WHOLE_PERIOD_SEC
    times = []
    # dense uniform grids: one for structure, one anti-aliased (irrational step)
    times.extend(np.linspace(0.0, period, 7301, endpoint=False))
    times.extend(np.arange(0.0, 2.0 * period, math.pi / 20.0))
    # both sides of every internal boundary, in cycles 0 and 1
    for b in wt.segment_boundaries():
        for cycle_offset in (0.0, period):
            for probe in (b - EPS, b, b + EPS):
                if probe + cycle_offset >= 0.0:
                    times.append(probe + cycle_offset)
    # exact period boundary fall-through and a large time
    times.extend([period, period + EPS, 10.0 * period + 1.2345])

    for t in times:
        compare_at(float(t))
    print(f"[parity] {len(times)} sample points ok "
          f"(pos atol {ATOL}, rot atol 1e-8)")

    # pen/stroke census over one cycle: 14 pen-down strokes across 5 tasks
    tab = wt.build_tables(dt=0.02)
    strokes = set(int(s) for s, p in zip(tab["stroke"], tab["pen"]) if p)
    tasks = sorted(set(int(x) for x in tab["task"] if x >= 0))
    if tasks != [0, 1, 2, 3, 4]:
        fail(f"expected 5 tasks, got {tasks}")
    per_task = {k: sorted(s % 100 for s in strokes
                          if (s % 10000) // 100 == k) for k in range(5)}
    expected = {0: [0], 1: [0], 2: list(range(10)), 3: [0], 4: [0]}
    if per_task != expected:
        fail(f"stroke census mismatch: {per_task} != {expected}")
    if len(strokes) != 14:
        fail(f"expected 14 pen-down strokes, got {len(strokes)}")
    # one-shot windylab: no pen-down return to the 'w' start after final 'b'
    ends = [k for k in range(tab["T"]) if tab["task"][k] == 2]
    last_windylab_tick = max(ends)
    following = tab["pen"][last_windylab_tick + 1:last_windylab_tick + 150]
    if following.any():
        fail("pen goes down again right after windylab one-shot end")
    print(f"[parity] stroke census ok: {sum(len(v) for v in expected.values())}"
          f" strokes, tasks {tasks}, T={tab['T']} ticks @50Hz")
    print("PARITY OK")


if __name__ == "__main__":
    main()
