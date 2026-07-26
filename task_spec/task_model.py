#!/usr/bin/env python3
"""Full task model: shapes in WORLD frame + a base that (a) slowly yaws to face
each drawing station and (b) jitters at high frequency (the disturbance). The
end-effector must stay on the world-frame shape while the base shakes.

Timeline is explicit DRAW / TURN phases so the turning between stations is a
visible, smooth motion (not a jump):
    DRAW circle @front  ->  TURN 0->90  ->  DRAW sine @left  ->  TURN 90->180
    ->  DRAW windylab @back  ->  TURN 180->270  ->  DRAW triangle @right
    ->  TURN 270->360 (closes the loop seamlessly)

During DRAW: the shape is anchored in WORLD at the station; the base jitters
around the facing orientation and the arm rejects the jitter (inertial stab),
built exactly like move_base_circle_mpc_ik_demo.py:
    ee_target_base = R_base_world^T @ (ee_world - base_pos_world)
During TURN: no drawing; the EE holds a forward "ready" pose in the BASE frame
(so it stays reachable while the platform yaws).

Shared by the pinocchio IK demo and the Isaac Lab RL env.
"""
import math

import numpy as np

from .shapes import SHAPES, AZIMUTH, ORIENTATION_TARGET, default_schedule
from .base_disturbance import BaseDisturbance


def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rpy_to_matrix(r, p, y):
    return _Rz(y) @ np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0],
                              [-math.sin(p), 0, math.cos(p)]]) @ \
        np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)],
                  [0, math.sin(r), math.cos(r)]])


class TaskModel:
    def __init__(self, schedule=None, disturbance=None,
                 station_dist=0.30, station_height=0.10,
                 base_nominal=(0.0, 0.0, 0.0), turn_dur=2.0):
        self.sch = schedule or default_schedule()
        self.dist = disturbance or BaseDisturbance(
            trans_wave="circle", att_wave="sine",
            trans_amp=0.02, att_amp_deg=0.1, trans_freq=1.0, att_freq=1.0)
        self.D = station_dist
        self.H = station_height
        self.base_nominal = np.asarray(base_nominal, dtype=float)
        self.turn_dur = turn_dur

        # unwrapped, monotonically increasing station azimuths (rad): 0,90,180,270
        segs = self.sch.segments
        raw = [math.radians(AZIMUTH[s["azimuth"]] if isinstance(s["azimuth"], str)
                            else float(s["azimuth"])) for s in segs]
        unwrapped, acc = [], 0.0
        prev = 0.0
        for i, a in enumerate(raw):
            if i == 0:
                unwrapped.append(a)
            else:
                d = (a - prev + math.pi) % (2 * math.pi) - math.pi
                if d < 0:            # force one consistent (CCW) direction
                    d += 2 * math.pi
                acc += d
                unwrapped.append(unwrapped[0] + acc)
            prev = a
        self._az = unwrapped                        # per-segment facing yaw (rad)
        self._az_end = unwrapped[0] + 2 * math.pi   # close the loop back to start

        # build phase timeline: DRAW_i then TURN_i, i=0..n-1.
        # Single-segment task (e.g. single circle) has NO turns -- base just
        # faces that station the whole time.
        self.phases = []
        t = 0.0
        n = len(segs)
        for i, seg in enumerate(segs):
            self.phases.append({"kind": "draw", "seg": i, "az": self._az[i],
                                "t0": t, "dur": seg["dur"]})
            t += seg["dur"]
            if n > 1:
                az_to = self._az[i + 1] if i + 1 < n else self._az_end
                self.phases.append({"kind": "turn", "az0": self._az[i],
                                    "az1": az_to, "t0": t, "dur": self.turn_dur})
                t += self.turn_dur
        self.period = t

    def _phase_at(self, t):
        t = t % self.period
        for ph in self.phases:
            if t < ph["t0"] + ph["dur"] + 1e-9:
                return ph, t - ph["t0"]
        return self.phases[-1], self.phases[-1]["dur"]

    # ---- base ----
    def _face_yaw(self, t):
        ph, lt = self._phase_at(t)
        if ph["kind"] == "draw":
            return ph["az"]
        s = min(max(lt / max(ph["dur"], 1e-6), 0.0), 1.0)
        # smoothstep for a natural ease-in/out turn
        s = s * s * (3 - 2 * s)
        return ph["az0"] + (ph["az1"] - ph["az0"]) * s

    def base_pose_world(self, t):
        dpos, drpy = self.dist.at(t)
        R = _Rz(self._face_yaw(t)) @ _rpy_to_matrix(*drpy)
        pos = self.base_nominal + np.array([dpos[0], dpos[1], dpos[2]])
        return pos, R

    # ---- EE target ----
    def _ready_base(self):
        return np.array([self.D, 0.0, self.H]), np.asarray(ORIENTATION_TARGET)

    def ee_target_world(self, t):
        ph, lt = self._phase_at(t)
        if ph["kind"] == "turn":
            # ready pose fixed in base frame -> rotate to world
            p_b, R_b = self._ready_base()
            bp, bR = self.base_pose_world(t)
            return bp + bR @ p_b, bR @ R_b
        seg = self.sch.segments[ph["seg"]]
        s = min(max(lt / max(seg["dur"], 1e-6), 0.0), 1.0)
        uv = SHAPES[seg["shape"]](np.array([s]), **seg.get("kwargs", {}))[0]
        yaw = ph["az"]
        Rz = _Rz(yaw)
        tang, up = Rz[:, 1], np.array([0.0, 0.0, 1.0])
        center = self.base_nominal + self.D * Rz[:, 0] + self.H * up
        pos = center + uv[0] * tang + uv[1] * up
        return pos, Rz @ np.asarray(ORIENTATION_TARGET)

    def ee_target_base(self, t):
        ph, lt = self._phase_at(t)
        bp, bR = self.base_pose_world(t)
        if ph["kind"] == "turn":
            p_b, R_b = self._ready_base()          # already base-frame, reachable
            return p_b, R_b
        p_w, R_w = self.ee_target_world(t)
        return bR.T @ (p_w - bp), bR.T @ R_w

    def is_drawing(self, t):
        return self._phase_at(t)[0]["kind"] == "draw"


def default_task():
    return TaskModel()
