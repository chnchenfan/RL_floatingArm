#!/usr/bin/env python3
"""Frozen exact ``whole`` trajectory spec (pure NumPy, no Pinocchio).

This module re-implements, bit-for-bit, the authoritative one-shot ``whole``
task defined by

    /home/windylab/code/windylab_ws/src/arm-platform/demo/moving_base_trajectory_config.py

at the frozen source revision recorded in ``SOURCE_SHA256`` below.  The RL
Isaac environment cannot import Pinocchio reliably, so the three Pinocchio
calls used by the authoritative source (``rpy.rpyToMatrix``, ``log3``,
``exp3``) are re-implemented here in NumPy.  Parity is enforced by
``tests/test_whole_trajectory_parity.py`` which compares this module pointwise
against the authoritative source on dense grids and on both sides of every
segment boundary.

Task order (canonical): front circle -> left triangle -> back one-shot
``windylab`` -> right sine -> top square, with a 0.6/2.4/0.6 s
lift/transfer/lower transition between consecutive stations (5 transitions per
cycle).  Total period 73.03949045888578 s.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

WHOLE_SCHEMA_VERSION = "whole_v2_oneshot_73.0394904589"
AUTHORITATIVE_SOURCE = (
    "/home/windylab/code/windylab_ws/src/arm-platform/demo/"
    "moving_base_trajectory_config.py")
SOURCE_SHA256 = (
    "0d0181b736f8b00f82e6559ce55358c36fab8f263792d5667e279c49bb8042e5")

# ---------------------------------------------------------------- constants
# Values below mirror the authoritative module's defaults (no MB_* overrides).
EE_WORLD_CIRCLE_CENTER = np.array([0.3, 0.01, 0.1])
EE_CIRCLE_RADIUS = 0.08
EE_PERIOD_SEC = 4.0
EE_DRAW_PLANE_Z = float(EE_WORLD_CIRCLE_CENTER[2])
EE_PEN_LIFT_HEIGHT = 0.03
EE_DRAW_SPEED = 0.04
EE_TRAVEL_SPEED = 0.08
EE_WINDYLAB_LETTER_WIDTH = 0.016
EE_WINDYLAB_LETTER_GAP = 0.016
EE_WINDYLAB_LETTER_HEIGHT = 0.050
EE_WINDYLAB_MIRROR_X = True

_TRIANGLE_PERIOD_SEC = 6.0
_SQUARE_PERIOD_SEC = 8.0
_SINE_DRAW_SEC = 5.0
_SINE_LIFT_SEC = 0.5
_SINE_TRANSFER_SEC = 2.0
_SINE_LOWER_SEC = 0.5

_SIDE_STATION_RADIUS = 0.30
_SIDE_STATION_HEIGHT = 0.12
_TOP_STATION_FORWARD = 0.22
_TOP_STATION_HEIGHT = 0.30
WHOLE_TASKS = (
    ('circle', 'front'),
    ('triangle', 'left'),
    ('windylab', 'back'),
    ('sine', 'right'),
    ('square', 'top'),
)
STATION_AZIMUTH = {
    'front': 0.0,
    'left': 0.5 * math.pi,
    'back': math.pi,
    'right': 1.5 * math.pi,
    'top': 2.0 * math.pi,
}
WHOLE_LIFT_SEC = 0.6
WHOLE_TRANSFER_SEC = 2.4
WHOLE_LOWER_SEC = 0.6
WHOLE_TRANSITION_SEC = WHOLE_LIFT_SEC + WHOLE_TRANSFER_SEC + WHOLE_LOWER_SEC

# Base-motion constants (circle_z defaults) used by the MPC demo; the RL env
# randomizes around these nominal values.
BASE_CIRCLE_RADIUS = 0.01
BASE_PERIOD_SEC = 0.5
BASE_HEIGHT = 0.01
BASE_Z_AMPLITUDE = 0.01
BASE_Z_PHASE = math.pi / 3.0
BASE_ATTITUDE_AMPLITUDE_RAD = math.radians(0.1)


class WholeSample(NamedTuple):
    position: np.ndarray      # (3,) world frame
    pen_down: bool
    stroke_id: int            # cycle*10000 + task_index*100 + local id; -1 pen-up
    rotation: np.ndarray      # (3,3) world frame tool rotation
    task_index: int           # 0..4 during draw; -1 during whole transitions
    base_yaw: float           # nominal base heading for this path time


# ----------------------------------------------------------- SO(3) helpers
def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Match pinocchio.rpy.rpyToMatrix: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def so3_log(R: np.ndarray) -> np.ndarray:
    """Match pinocchio.log3 for the rotations used here (angle < pi)."""
    cos = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(cos)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-8:
        return 0.5 * v
    return v * (angle / (2.0 * math.sin(angle)))


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Match pinocchio.exp3 (Rodrigues)."""
    angle = float(np.linalg.norm(w))
    K = np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])
    if angle < 1e-8:
        return np.eye(3) + K + 0.5 * (K @ K)
    return (
        np.eye(3)
        + (math.sin(angle) / angle) * K
        + ((1.0 - math.cos(angle)) / (angle * angle)) * (K @ K)
    )


TARGET_ROTATION = rpy_to_matrix(0.0, math.radians(90.0), math.radians(90.0))


# ----------------------------------------------------------- path helpers
def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value ** 3 * (10.0 + value * (-15.0 + 6.0 * value))


def _line_sample(start, end, phase: float) -> np.ndarray:
    return np.asarray(start, dtype=float) + _smoothstep(phase) * (
        np.asarray(end, dtype=float) - np.asarray(start, dtype=float))


def _closed_polyline_sample(t, vertices, period):
    cycle = math.floor(float(t) / period)
    local = float(t) - cycle * period
    edge_count = len(vertices) - 1
    edge_phase = local / period * edge_count
    edge_index = min(int(edge_phase), edge_count - 1)
    position = _line_sample(
        vertices[edge_index], vertices[edge_index + 1],
        edge_phase - edge_index)
    return position, True, int(cycle)


def _letter_strokes():
    glyphs = {
        'w': [
            [(0.00, 0.72), (0.20, 0.00), (0.50, 0.48),
             (0.80, 0.00), (1.00, 0.72)],
        ],
        'i': [
            [(0.50, 0.00), (0.50, 0.70)],
            [(0.50, 0.88), (0.50, 1.00)],
        ],
        'n': [
            [(0.00, 0.00), (0.00, 0.70), (0.45, 0.70),
             (0.82, 0.55), (1.00, 0.00)],
        ],
        'd': [
            [(1.00, 0.00), (0.28, 0.00), (0.00, 0.20),
             (0.00, 0.52), (0.28, 0.70), (1.00, 0.70),
             (1.00, 0.00), (1.00, 1.00)],
        ],
        'y': [
            [(0.00, 0.70), (0.50, 0.20), (1.00, 0.70)],
            [(0.50, 0.20), (0.50, -0.25)],
        ],
        'l': [
            [(0.50, 1.00), (0.50, 0.00)],
        ],
        'a': [
            [(1.00, 0.00), (0.28, 0.00), (0.00, 0.20),
             (0.00, 0.52), (0.28, 0.70), (1.00, 0.70),
             (1.00, 0.00)],
        ],
        'b': [
            [(0.00, 1.00), (0.00, 0.00), (0.68, 0.00),
             (1.00, 0.20), (1.00, 0.50), (0.68, 0.70),
             (0.00, 0.70)],
        ],
    }
    word = 'windylab'
    letter_width = EE_WINDYLAB_LETTER_WIDTH
    letter_gap = EE_WINDYLAB_LETTER_GAP
    letter_height = EE_WINDYLAB_LETTER_HEIGHT
    total_width = len(word) * letter_width + (len(word) - 1) * letter_gap
    x_origin = EE_WORLD_CIRCLE_CENTER[0] - 0.5 * total_width
    baseline_y = EE_WORLD_CIRCLE_CENTER[1] - 0.022

    strokes = []
    for letter_index, letter in enumerate(word):
        letter_x = x_origin + letter_index * (letter_width + letter_gap)
        for points in glyphs[letter]:
            stroke = np.array([
                [
                    letter_x + letter_width * u,
                    baseline_y + letter_height * v,
                    EE_DRAW_PLANE_Z,
                ]
                for u, v in points
            ], dtype=float)
            if EE_WINDYLAB_MIRROR_X:
                stroke[:, 0] = 2.0 * EE_WORLD_CIRCLE_CENTER[0] - stroke[:, 0]
            strokes.append(stroke)
    return strokes


def _motion_duration(start, end, speed):
    distance = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
    return max(distance / max(float(speed), 1e-6), 0.12)


def _build_windylab_timeline(include_cycle_return: bool = True):
    strokes = _letter_strokes()
    segments = []

    def add(start, end, speed, pen_down, stroke_id):
        segments.append({
            'start': np.asarray(start, dtype=float),
            'end': np.asarray(end, dtype=float),
            'duration': _motion_duration(start, end, speed),
            'pen_down': bool(pen_down),
            'stroke_id': int(stroke_id),
        })

    for stroke_id, stroke in enumerate(strokes):
        for index in range(len(stroke) - 1):
            add(stroke[index], stroke[index + 1],
                EE_DRAW_SPEED, True, stroke_id)
        if stroke_id == len(strokes) - 1 and not include_cycle_return:
            continue
        next_stroke = strokes[(stroke_id + 1) % len(strokes)]
        current = stroke[-1]
        next_start = next_stroke[0]
        lifted_current = current + np.array([0.0, 0.0, EE_PEN_LIFT_HEIGHT])
        lifted_next = next_start + np.array([0.0, 0.0, EE_PEN_LIFT_HEIGHT])
        add(current, lifted_current, EE_TRAVEL_SPEED, False, -1)
        add(lifted_current, lifted_next, EE_TRAVEL_SPEED, False, -1)
        add(lifted_next, next_start, EE_TRAVEL_SPEED, False, -1)

    cumulative = np.cumsum([0.0] + [s['duration'] for s in segments])
    return strokes, segments, cumulative


_WINDYLAB_STROKES, _WINDYLAB_SEGMENTS, _WINDYLAB_CUMULATIVE = (
    _build_windylab_timeline(include_cycle_return=True))
WINDYLAB_PERIOD_SEC = float(_WINDYLAB_CUMULATIVE[-1])
_, _WINDYLAB_ONCE_SEGMENTS, _WINDYLAB_ONCE_CUMULATIVE = (
    _build_windylab_timeline(include_cycle_return=False))
WINDYLAB_ONCE_PERIOD_SEC = float(_WINDYLAB_ONCE_CUMULATIVE[-1])


def _local_trajectory_period(shape: str) -> float:
    return {
        'circle': EE_PERIOD_SEC,
        'triangle': _TRIANGLE_PERIOD_SEC,
        'square': _SQUARE_PERIOD_SEC,
        'sine': (_SINE_DRAW_SEC + _SINE_LIFT_SEC
                 + _SINE_TRANSFER_SEC + _SINE_LOWER_SEC),
        'windylab': WINDYLAB_PERIOD_SEC,
    }[shape]


def whole_task_draw_period(shape: str) -> float:
    if shape == 'windylab':
        return WINDYLAB_ONCE_PERIOD_SEC
    return _local_trajectory_period(shape)


WHOLE_PERIOD_SEC = (
    sum(whole_task_draw_period(task_shape) for task_shape, _ in WHOLE_TASKS)
    + len(WHOLE_TASKS) * WHOLE_TRANSITION_SEC
)


def _windylab_sample(t: float, repeat: bool = True):
    if repeat:
        period = WINDYLAB_PERIOD_SEC
        segments = _WINDYLAB_SEGMENTS
        cumulative = _WINDYLAB_CUMULATIVE
        cycle = math.floor(float(t) / period)
        local = float(t) - cycle * period
    else:
        period = WINDYLAB_ONCE_PERIOD_SEC
        segments = _WINDYLAB_ONCE_SEGMENTS
        cumulative = _WINDYLAB_ONCE_CUMULATIVE
        cycle = 0
        local = float(np.clip(float(t), 0.0, period))
    index = int(np.searchsorted(cumulative, local, side='right') - 1)
    index = int(np.clip(index, 0, len(segments) - 1))
    segment = segments[index]
    phase = (local - cumulative[index]) / segment['duration']
    stroke_id = segment['stroke_id']
    if stroke_id >= 0:
        stroke_id += int(cycle) * len(_WINDYLAB_STROKES)
    return (
        _line_sample(segment['start'], segment['end'], phase),
        segment['pen_down'],
        stroke_id,
    )


def _local_trajectory_sample(t: float, shape: str, repeat: bool = True):
    """Sample a primitive in the canonical local drawing plane.

    Returns (position, pen_down, stroke_id) exactly like the authoritative
    ``_local_trajectory_sample`` (rotation is attached by the caller).
    """
    t = float(t)
    center = EE_WORLD_CIRCLE_CENTER

    if shape == 'circle':
        period = EE_PERIOD_SEC
        cycle = math.floor(t / period)
        angle = 2.0 * math.pi * (t - cycle * period) / period
        position = center + EE_CIRCLE_RADIUS * np.array([
            math.cos(angle), math.sin(angle), 0.0])
        return position, True, int(cycle)

    if shape == 'triangle':
        vertices = center + np.array([
            [-0.070, -0.050, 0.0],
            [0.000, 0.070, 0.0],
            [0.070, -0.050, 0.0],
            [-0.070, -0.050, 0.0],
        ])
        return _closed_polyline_sample(t, vertices, _TRIANGLE_PERIOD_SEC)

    if shape == 'square':
        vertices = center + np.array([
            [-0.060, -0.060, 0.0],
            [-0.060, 0.060, 0.0],
            [0.060, 0.060, 0.0],
            [0.060, -0.060, 0.0],
            [-0.060, -0.060, 0.0],
        ])
        return _closed_polyline_sample(t, vertices, _SQUARE_PERIOD_SEC)

    if shape == 'sine':
        period = _local_trajectory_period('sine')
        cycle = math.floor(t / period)
        local = t - cycle * period
        start = center + np.array([-0.080, 0.0, 0.0])
        end = center + np.array([0.080, 0.0, 0.0])
        lift = np.array([0.0, 0.0, EE_PEN_LIFT_HEIGHT])
        if local < _SINE_DRAW_SEC:
            progress = _smoothstep(local / _SINE_DRAW_SEC)
            position = np.array([
                start[0] + (end[0] - start[0]) * progress,
                center[1] + 0.050 * math.sin(2.0 * math.pi * progress),
                EE_DRAW_PLANE_Z,
            ])
            return position, True, int(cycle)
        local -= _SINE_DRAW_SEC
        if local < _SINE_LIFT_SEC:
            return (_line_sample(end, end + lift, local / _SINE_LIFT_SEC),
                    False, -1)
        local -= _SINE_LIFT_SEC
        if local < _SINE_TRANSFER_SEC:
            return (_line_sample(end + lift, start + lift,
                                 local / _SINE_TRANSFER_SEC), False, -1)
        local -= _SINE_TRANSFER_SEC
        return (_line_sample(start + lift, start, local / _SINE_LOWER_SEC),
                False, -1)

    return _windylab_sample(t, repeat=repeat)


def _rz(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _station_geometry(station: str):
    if station == 'top':
        center = np.array([_TOP_STATION_FORWARD, 0.0, _TOP_STATION_HEIGHT])
        axis_u = np.array([1.0, 0.0, 0.0])
        axis_v = np.array([0.0, 1.0, 0.0])
        lift_direction = np.array([0.0, 0.0, 1.0])
        rotation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ])
        return center, axis_u, axis_v, lift_direction, rotation

    azimuth = STATION_AZIMUTH[station]
    radial = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
    axis_u = np.array([math.sin(azimuth), -math.cos(azimuth), 0.0])
    axis_v = np.array([0.0, 0.0, 1.0])
    center = (_SIDE_STATION_RADIUS * radial
              + np.array([0.0, 0.0, _SIDE_STATION_HEIGHT]))
    lift_direction = -radial
    rotation = _rz(azimuth) @ TARGET_ROTATION
    return center, axis_u, axis_v, lift_direction, rotation


def _place_local_sample(position, pen_down, stroke_id, station):
    center, axis_u, axis_v, lift_direction, rotation = (
        _station_geometry(station))
    local_offset = np.asarray(position) - EE_WORLD_CIRCLE_CENTER
    placed = (
        center
        + local_offset[0] * axis_u
        + local_offset[1] * axis_v
        + local_offset[2] * lift_direction
    )
    return placed, pen_down, stroke_id, rotation.copy()


def _interpolate_rotation(rotation_start, rotation_end, phase):
    phase = _smoothstep(phase)
    delta = so3_log(rotation_start.T @ rotation_end)
    return rotation_start @ so3_exp(phase * delta)


def base_nominal_yaw(t: float) -> float:
    """Whole-task nominal base heading (authoritative ``base_nominal_yaw``)."""
    period = WHOLE_PERIOD_SEC
    cycle = math.floor(float(t) / period)
    local_time = float(t) - cycle * period
    for task_index, (task_shape, station) in enumerate(WHOLE_TASKS):
        draw_period = whole_task_draw_period(task_shape)
        current_yaw = STATION_AZIMUTH[station]
        if local_time < draw_period:
            return current_yaw
        local_time -= draw_period

        next_station = WHOLE_TASKS[(task_index + 1) % len(WHOLE_TASKS)][1]
        next_yaw = STATION_AZIMUTH[next_station]
        if task_index == len(WHOLE_TASKS) - 1:
            next_yaw = 2.0 * math.pi
        if local_time < WHOLE_TRANSITION_SEC:
            phase = _smoothstep(local_time / WHOLE_TRANSITION_SEC)
            return current_yaw + phase * (next_yaw - current_yaw)
        local_time -= WHOLE_TRANSITION_SEC
    return 2.0 * math.pi


def sample(t: float) -> WholeSample:
    """Exact whole-task sample at path time ``t`` (world frame)."""
    period = WHOLE_PERIOD_SEC
    cycle = math.floor(float(t) / period)
    local_time = float(t) - cycle * period
    task_count = len(WHOLE_TASKS)
    yaw = base_nominal_yaw(t)

    for task_index, (task_shape, station) in enumerate(WHOLE_TASKS):
        draw_period = whole_task_draw_period(task_shape)
        if local_time < draw_period:
            pos, pen, stroke = _local_trajectory_sample(
                local_time, task_shape, repeat=(task_shape != 'windylab'))
            placed, pen, stroke, rotation = _place_local_sample(
                pos, pen, stroke, station)
            if stroke >= 0:
                stroke += int(cycle) * 10000 + task_index * 100
            return WholeSample(placed, pen, stroke, rotation, task_index, yaw)
        local_time -= draw_period
        if local_time >= WHOLE_TRANSITION_SEC:
            local_time -= WHOLE_TRANSITION_SEC
            continue

        next_shape, next_station = WHOLE_TASKS[(task_index + 1) % task_count]
        end_pos, end_pen, end_stroke = _local_trajectory_sample(
            draw_period - 1e-9, task_shape,
            repeat=(task_shape != 'windylab'))
        cur_pos, _, _, cur_rot = _place_local_sample(
            end_pos, end_pen, end_stroke, station)
        nxt_pos0, _, _ = _local_trajectory_sample(0.0, next_shape)
        nxt_pos, _, _, nxt_rot = _place_local_sample(
            nxt_pos0, True, 0, next_station)
        current_lift = _station_geometry(station)[3]
        next_lift = _station_geometry(next_station)[3]
        lifted_current = cur_pos + EE_PEN_LIFT_HEIGHT * current_lift
        lifted_next = nxt_pos + EE_PEN_LIFT_HEIGHT * next_lift

        if local_time < WHOLE_LIFT_SEC:
            phase = local_time / WHOLE_LIFT_SEC
            position = _line_sample(cur_pos, lifted_current, phase)
            rotation = cur_rot
        elif local_time < WHOLE_LIFT_SEC + WHOLE_TRANSFER_SEC:
            phase = (local_time - WHOLE_LIFT_SEC) / WHOLE_TRANSFER_SEC
            position = _line_sample(lifted_current, lifted_next, phase)
            rotation = _interpolate_rotation(cur_rot, nxt_rot, phase)
        else:
            phase = (
                local_time - WHOLE_LIFT_SEC - WHOLE_TRANSFER_SEC
            ) / WHOLE_LOWER_SEC
            position = _line_sample(lifted_next, nxt_pos, phase)
            rotation = nxt_rot
        return WholeSample(position, False, -1, rotation.copy(), -1, yaw)

    return sample(0.0)


def segment_boundaries():
    """All internal time boundaries of one whole cycle, sorted ascending.

    Includes task draw start/end, lift/transfer/lower boundaries, and every
    windylab segment boundary (mapped into whole time).
    """
    bounds = [0.0]
    t = 0.0
    for task_shape, _ in WHOLE_TASKS:
        draw = whole_task_draw_period(task_shape)
        if task_shape == 'windylab':
            bounds.extend(t + c for c in _WINDYLAB_ONCE_CUMULATIVE)
        elif task_shape == 'sine':
            local = [_SINE_DRAW_SEC, _SINE_DRAW_SEC + _SINE_LIFT_SEC,
                     _SINE_DRAW_SEC + _SINE_LIFT_SEC + _SINE_TRANSFER_SEC]
            bounds.extend(t + b for b in local)
        elif task_shape == 'triangle':
            bounds.extend(t + _TRIANGLE_PERIOD_SEC * k / 3.0
                          for k in range(1, 3))
        elif task_shape == 'square':
            bounds.extend(t + _SQUARE_PERIOD_SEC * k / 4.0
                          for k in range(1, 4))
        t += draw
        bounds.append(t)
        bounds.append(t + WHOLE_LIFT_SEC)
        bounds.append(t + WHOLE_LIFT_SEC + WHOLE_TRANSFER_SEC)
        t += WHOLE_TRANSITION_SEC
        bounds.append(t)
    return sorted(set(bounds))


def build_tables(dt: float = 0.02, ticks: int = None):
    """Precompute per-tick lookup tables over one whole cycle.

    Returns a dict of numpy arrays with keys: pos (T,3), rot (T,3,3),
    pen (T,), stroke (T,), task (T,), yaw (T,), plus scalars period/dt/T.
    """
    if ticks is None:
        ticks = int(round(WHOLE_PERIOD_SEC / dt))
    pos = np.zeros((ticks, 3))
    rot = np.zeros((ticks, 3, 3))
    pen = np.zeros(ticks, dtype=bool)
    stroke = np.zeros(ticks, dtype=np.int64)
    task = np.zeros(ticks, dtype=np.int64)
    yaw = np.zeros(ticks)
    for k in range(ticks):
        s = sample(k * dt)
        pos[k] = s.position
        rot[k] = s.rotation
        pen[k] = s.pen_down
        stroke[k] = s.stroke_id
        task[k] = s.task_index
        yaw[k] = s.base_yaw
    return {
        'pos': pos, 'rot': rot, 'pen': pen, 'stroke': stroke,
        'task': task, 'yaw': yaw,
        'period': WHOLE_PERIOD_SEC, 'dt': dt, 'T': ticks,
        'schema_version': WHOLE_SCHEMA_VERSION,
        'source_sha256': SOURCE_SHA256,
    }


if __name__ == '__main__':
    print(f'schema={WHOLE_SCHEMA_VERSION}')
    print(f'period={WHOLE_PERIOD_SEC!r}')
    print(f'windylab_once={WINDYLAB_ONCE_PERIOD_SEC!r}')
    print(f'windylab_repeat={WINDYLAB_PERIOD_SEC!r}')
    print(f'ticks@50Hz={int(round(WHOLE_PERIOD_SEC / 0.02))}')
    print(f'boundaries={len(segment_boundaries())}')
