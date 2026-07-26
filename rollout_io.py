#!/usr/bin/env python3
"""Shared rollout format between the Isaac (py3.11) and ROS (py3.10) worlds.

A rollout is a time series of the 7 arm joint angles produced by running a
policy (or any controller) in Isaac. The ROS replay node publishes it on
/joint_states so the EXISTING robot_state_publisher + RViz +
trajectory_visualizer_demo.py show the arm and its end-effector tracking, with
zero changes to the visualizer.

File format: plain CSV, human-inspectable, no external deps.
  line 1: `# rollout meta: <json>`   (joint_names, dt, source, note)
  line 2: header row  ->  t,q1,q2,q3,q4,q5,q6,q7[,ee_x,ee_y,ee_z,tgt_x,tgt_y,tgt_z]
  rows  : one sample per control tick

Only `t` and `q1..q7` are required. The optional ee_*/tgt_* columns let the
exporter stash the Isaac-side actual/target EE for later offline comparison;
the replay node ignores them (the visualizer re-derives EE via FK).
"""
import csv
import json
from typing import Optional

import numpy as np

JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
_Q_COLS = [f"q{i}" for i in range(1, 8)]
_EE_COLS = ["ee_x", "ee_y", "ee_z"]
_TGT_COLS = ["tgt_x", "tgt_y", "tgt_z"]
# world->base_link pose (so RViz can show the base shaking)
_BASE_COLS = ["bx", "by", "bz", "bqx", "bqy", "bqz", "bqw"]


def write_rollout(path, t, q, joint_names=None, ee=None, tgt=None,
                  base_pos=None, base_quat=None, drawing=None,
                  source="isaac", note=""):
    """Write a rollout CSV.

    t:  (T,) seconds. q: (T,7) joint angles [rad], urdf order joint1..joint7.
    ee/tgt: optional (T,3) end-effector actual/target position [m], WORLD frame.
    base_pos/base_quat: optional world->base_link pose, (T,3) and (T,4) xyzw,
        so the replay node can drive the base TF and RViz shows it shaking.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float)
    assert q.shape[0] == t.shape[0], f"t/q length mismatch {t.shape} {q.shape}"
    assert q.shape[1] == 7, f"expected 7 joints, got {q.shape[1]}"
    joint_names = list(joint_names) if joint_names else list(JOINT_NAMES)

    cols = ["t"] + _Q_COLS
    if ee is not None:
        ee = np.asarray(ee, dtype=float); cols += _EE_COLS
    if tgt is not None:
        tgt = np.asarray(tgt, dtype=float); cols += _TGT_COLS
    if base_pos is not None:
        base_pos = np.asarray(base_pos, dtype=float)
        base_quat = np.asarray(base_quat, dtype=float); cols += _BASE_COLS
    if drawing is not None:
        drawing = np.asarray(drawing).astype(int); cols += ["drawing"]

    dt = float(np.mean(np.diff(t))) if t.shape[0] > 1 else 0.0
    meta = {"joint_names": joint_names, "dt": dt, "n": int(t.shape[0]),
            "has_base": base_pos is not None, "source": source, "note": note}
    with open(path, "w", newline="") as f:
        f.write("# rollout meta: " + json.dumps(meta) + "\n")
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(t.shape[0]):
            row = [f"{t[i]:.6f}"] + [f"{v:.8f}" for v in q[i]]
            if ee is not None:
                row += [f"{v:.6f}" for v in ee[i]]
            if tgt is not None:
                row += [f"{v:.6f}" for v in tgt[i]]
            if base_pos is not None:
                row += [f"{v:.6f}" for v in base_pos[i]] + \
                       [f"{v:.8f}" for v in base_quat[i]]
            if drawing is not None:
                row += [str(int(drawing[i]))]
            w.writerow(row)
    return path


def read_rollout(path):
    """Return dict: meta, t (T,), q (T,7), joint_names, ee (T,3|None), tgt(...)."""
    meta = {}
    with open(path, "r") as f:
        first = f.readline()
        if first.startswith("# rollout meta:"):
            meta = json.loads(first[len("# rollout meta:"):].strip())
            reader = csv.DictReader(f)
        else:
            f.seek(0)
            reader = csv.DictReader(f)
        rows = list(reader)

    t = np.array([float(r["t"]) for r in rows])
    q = np.array([[float(r[c]) for c in _Q_COLS] for r in rows])
    ee = (np.array([[float(r[c]) for c in _EE_COLS] for r in rows])
          if all(c in rows[0] for c in _EE_COLS) else None) if rows else None
    tgt = (np.array([[float(r[c]) for c in _TGT_COLS] for r in rows])
           if all(c in rows[0] for c in _TGT_COLS) else None) if rows else None
    has_base = bool(rows) and all(c in rows[0] for c in _BASE_COLS)
    base_pos = (np.array([[float(r[c]) for c in _BASE_COLS[:3]] for r in rows])
                if has_base else None)
    base_quat = (np.array([[float(r[c]) for c in _BASE_COLS[3:]] for r in rows])
                 if has_base else None)
    drawing = (np.array([int(float(r["drawing"])) for r in rows]).astype(bool)
               if rows and "drawing" in rows[0] else None)
    joint_names = meta.get("joint_names", JOINT_NAMES)
    return {"meta": meta, "t": t, "q": q, "joint_names": joint_names,
            "ee": ee, "tgt": tgt, "base_pos": base_pos, "base_quat": base_quat,
            "drawing": drawing}
