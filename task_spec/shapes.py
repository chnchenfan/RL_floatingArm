#!/usr/bin/env python3
"""World-frame end-effector target library for the Isaac arm RL task.

Authoritative task spec (see memory project-isaac-arm-rl-task-spec):
  - link7 tracks shapes DEFINED IN WORLD FRAME while the base shakes; the arm
    must inertially stabilize against the base motion.
  - shapes: circle, sine, triangle, and writing "windylab".
  - shapes are placed around the base at front/left/right/back and drawn in
    sequence (rotate around the arm one turn).
  - link7 orientation locked at 90 deg (soft) -> ORIENTATION_TARGET below.

This module is pure geometry (numpy only) so BOTH the pinocchio IK demo and the
Isaac Lab env query the exact same targets. No ROS, no Isaac deps.

Frames: each shape lives in a local 2D plane (u=horizontal-tangential,
v=vertical=world +z). place() maps (u,v) to world at a given azimuth/radius/
height around the base. Azimuth: front=0, left=+90, right=-90, back=180 (deg),
measured in the base x-y plane (x forward, y left).
"""
import math

import numpy as np

# link7 orientation lock: matches windylab_ws moving_base_trajectory_config
# (roll=0, pitch=90, yaw=90 deg). Stored as a 3x3 rotation (world<-ee).
def _rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


ORIENTATION_TARGET = _rpy_to_matrix(0.0, math.pi / 2.0, math.pi / 2.0)


# ---------------------------------------------------------------------------
# 2D shape primitives:  s in [0,1] -> (u, v) in a unit-ish local plane.
# Scales are chosen small (~0.12 m span) so they stay in the arm workspace.
# ---------------------------------------------------------------------------
def circle_uv(s, r=0.06):
    a = 2.0 * math.pi * np.asarray(s)
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=-1)


def sine_uv(s, width=0.14, amp=0.05, cycles=2.0):
    s = np.asarray(s)
    u = (s - 0.5) * width
    v = amp * np.sin(2.0 * math.pi * cycles * s)
    return np.stack([u, v], axis=-1)


def triangle_uv(s, size=0.12):
    # equilateral, closed, param by perimeter fraction
    h = size * math.sqrt(3.0) / 2.0
    verts = np.array([[-size / 2, -h / 3],
                      [size / 2, -h / 3],
                      [0.0, 2 * h / 3],
                      [-size / 2, -h / 3]])
    seg = np.linspace(0, 3, len(verts))
    s = np.asarray(s) * 3.0
    u = np.interp(s, seg, verts[:, 0])
    v = np.interp(s, seg, verts[:, 1])
    return np.stack([u, v], axis=-1)


# ---- stroke font for "windylab" -------------------------------------------
# Each glyph: list of strokes; each stroke a polyline in a [0,1]x[0,1] cell.
_GLYPHS = {
    "w": [[(0.0, 1.0), (0.15, 0.0), (0.5, 0.65), (0.85, 0.0), (1.0, 1.0)]],
    "i": [[(0.5, 0.0), (0.5, 0.62)], [(0.5, 0.85), (0.5, 1.0)]],
    "n": [[(0.05, 0.0), (0.05, 0.7)],
          [(0.05, 0.55), (0.25, 0.72), (0.6, 0.72), (0.75, 0.55), (0.75, 0.0)]],
    "d": [[(0.75, 0.0), (0.75, 1.0)],
          [(0.75, 0.6), (0.4, 0.75), (0.1, 0.6), (0.05, 0.3),
           (0.15, 0.05), (0.45, 0.0), (0.75, 0.12)]],
    "y": [[(0.05, 0.7), (0.45, 0.0)],
          [(0.85, 0.7), (0.25, -0.35)]],
    "l": [[(0.5, 0.0), (0.5, 1.0)]],
    "a": [[(0.7, 0.6), (0.4, 0.72), (0.1, 0.55), (0.1, 0.2),
           (0.4, 0.02), (0.7, 0.18)],
          [(0.7, 0.6), (0.7, 0.0)]],
    "b": [[(0.1, 0.0), (0.1, 1.0)],
          [(0.1, 0.6), (0.4, 0.72), (0.7, 0.55), (0.72, 0.28),
           (0.6, 0.05), (0.3, 0.0), (0.1, 0.12)]],
}
_WORD = "windylab"


def _word_polylines(word=_WORD, cell_w=0.62, gap=0.28, height=0.11):
    """Return list of strokes (each (M,2)) for the whole word, laid out along u,
    centered on origin, scaled so total height == `height`."""
    strokes = []
    x = 0.0
    for ch in word:
        for st in _GLYPHS[ch]:
            pts = np.array(st, dtype=float)
            pts[:, 0] = pts[:, 0] * cell_w + x
            strokes.append(pts)
        x += cell_w + gap
    total_w = x - gap
    # center and scale (cell v-range ~[-0.35,1.0]) so height maps to `height`
    allpts = np.vstack(strokes)
    cx = 0.5 * (allpts[:, 0].min() + allpts[:, 0].max())
    cy = 0.5 * (allpts[:, 1].min() + allpts[:, 1].max())
    vspan = allpts[:, 1].max() - allpts[:, 1].min()
    scale = height / vspan
    out = []
    for st in strokes:
        st = st.copy()
        st[:, 0] = (st[:, 0] - cx) * scale
        st[:, 1] = (st[:, 1] - cy) * scale
        out.append(st)
    return out, total_w * scale


def windylab_uv(s, height=0.08):   # <=0.09 stays inside the arm workspace (~0.1mm IK)
    """Continuous pen-down path over the word (transit lines between strokes
    are included; acceptable for a tracking target)."""
    strokes, _ = _word_polylines(height=height)
    # concatenate strokes into one polyline, arc-length parametrized
    poly = np.vstack(strokes)
    seg_len = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    seg_len /= seg_len[-1]
    s = np.asarray(s)
    u = np.interp(s, seg_len, poly[:, 0])
    v = np.interp(s, seg_len, poly[:, 1])
    return np.stack([u, v], axis=-1)


SHAPES = {
    "circle": circle_uv,
    "sine": sine_uv,
    "triangle": triangle_uv,
    "windylab": windylab_uv,
}

# azimuth presets (deg): front / left / right / back
AZIMUTH = {"front": 0.0, "left": 90.0, "right": -90.0, "back": 180.0}


def place(uv, azimuth_deg, radius, height, base_pos=(0.0, 0.0, 0.0)):
    """Map local (u,v) [.,2] to world (.,3): vertical plane at given azimuth,
    `radius` out from base, centered at `height`. u tangential, v up (+z)."""
    th = math.radians(azimuth_deg)
    radial = np.array([math.cos(th), math.sin(th), 0.0])
    tang = np.array([-math.sin(th), math.cos(th), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    center = np.asarray(base_pos) + radial * radius + up * height
    uv = np.atleast_2d(uv)
    return center[None, :] + uv[:, 0:1] * tang[None, :] + uv[:, 1:2] * up[None, :]


class ShapeSchedule:
    """Sequence of shapes placed around the base, drawn one after another.

    segments: list of dicts {shape, azimuth(name or deg), dur, kwargs}.
    Query world_target(t) -> (pos3, R33). Total period = sum(dur).
    """

    def __init__(self, segments, radius=0.30, height=0.30, base_pos=(0, 0, 0)):
        self.segments = segments
        self.radius = radius
        self.height = height
        self.base_pos = np.asarray(base_pos, dtype=float)
        self.durations = np.array([seg["dur"] for seg in segments], dtype=float)
        self.starts = np.r_[0.0, np.cumsum(self.durations)]
        self.period = float(self.starts[-1])

    def _az(self, a):
        return AZIMUTH[a] if isinstance(a, str) else float(a)

    def world_target(self, t):
        t = float(t) % self.period
        i = int(np.searchsorted(self.starts, t, side="right") - 1)
        i = max(0, min(i, len(self.segments) - 1))
        seg = self.segments[i]
        s = (t - self.starts[i]) / max(self.durations[i], 1e-6)
        uv = SHAPES[seg["shape"]](np.array([s]), **seg.get("kwargs", {}))[0]
        pos = place(uv, self._az(seg["azimuth"]), self.radius, self.height,
                    self.base_pos)[0]
        return pos, ORIENTATION_TARGET

    def sample(self, n_per_segment=200):
        """Dense world-frame path (N,3) + segment index, for viz/IK/reachability."""
        pts, segidx, ts = [], [], []
        for i, seg in enumerate(self.segments):
            s = np.linspace(0, 1, n_per_segment, endpoint=False)
            uv = SHAPES[seg["shape"]](s, **seg.get("kwargs", {}))
            w = place(uv, self._az(seg["azimuth"]), self.radius, self.height,
                      self.base_pos)
            pts.append(w)
            segidx.append(np.full(n_per_segment, i))
            ts.append(self.starts[i] + s * self.durations[i])
        return np.vstack(pts), np.concatenate(segidx), np.concatenate(ts)


def single_circle_schedule(radius=0.30, height=0.10, ee_radius=0.08, dur=4.0):
    """One circle at the front (config-matched moving_base_circle): EE traces a
    circle of radius `ee_radius` in the vertical plane `radius` ahead, period
    `dur` s. No turns -- base always faces front. For the high-precision /
    imitation task."""
    return ShapeSchedule(
        segments=[{"shape": "circle", "azimuth": "front", "dur": dur,
                   "kwargs": {"r": ee_radius}}],
        radius=radius, height=height,
    )


def default_schedule(radius=0.30, height=0.10, seg_dur=5.0):
    """The canonical task: draw a shape at each of the four sides while turning
    ONE way around the arm. Visit order is monotonic in azimuth
    (front 0 -> left 90 -> back 180 -> right 270) so the base never turns more
    than 90 deg between stations (a 180 deg swing pushes the target behind the
    arm mid-turn -> unreachable). Each shape stays at its side; only the drawing
    order goes around smoothly."""
    return ShapeSchedule(
        segments=[
            {"shape": "circle", "azimuth": "front", "dur": seg_dur},
            {"shape": "sine", "azimuth": "left", "dur": seg_dur},
            {"shape": "windylab", "azimuth": "back", "dur": seg_dur * 1.6},
            {"shape": "triangle", "azimuth": "right", "dur": seg_dur},
        ],
        radius=radius, height=height,
    )


if __name__ == "__main__":
    sch = default_schedule()
    pts, seg, ts = sch.sample(150)
    print(f"period={sch.period}s, samples={len(pts)}")
    for i, s in enumerate(sch.segments):
        m = seg == i
        p = pts[m]
        print(f"  {s['shape']:10s}@{s['azimuth']:6}: "
              f"x[{p[:,0].min():.2f},{p[:,0].max():.2f}] "
              f"y[{p[:,1].min():.2f},{p[:,1].max():.2f}] "
              f"z[{p[:,2].min():.2f},{p[:,2].max():.2f}]")
