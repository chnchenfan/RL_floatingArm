#!/usr/bin/env python3
"""Vectorized (torch, batched over envs) base-frame EE target for the RL env.

The shape schedule is COMMON to all envs (fixed four-shape task); what varies per
env is the randomized base disturbance. So we precompute the common world-frame
shape target + facing yaw + drawing flag as lookup tables (from the numpy
TaskModel), and per step compute each env's base pose from its sampled
disturbance and express the world target in that (moving) base frame -- exactly
`ee_target_base = R_base^T (p_world - p_base)`.

v1: fixed-base sim. The arm's base_link is fixed; the disturbance enters ONLY as
this moving base-frame target (kinematic). Inertial coupling is a v2 upgrade.
"""
import math

import numpy as np
import torch

from .task_model import TaskModel
from .shapes import ORIENTATION_TARGET


def _rpy_to_R(rpy):
    """rpy (...,3) -> R (...,3,3) torch."""
    r, p, y = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    z = torch.zeros_like(r); o = torch.ones_like(r)
    Rz = torch.stack([cy, -sy, z, sy, cy, z, z, z, o], -1).reshape(*r.shape, 3, 3)
    Ry = torch.stack([cp, z, sp, z, o, z, -sp, z, cp], -1).reshape(*r.shape, 3, 3)
    Rx = torch.stack([o, z, z, z, cr, -sr, z, sr, cr], -1).reshape(*r.shape, 3, 3)
    return Rz @ Ry @ Rx


# disturbance waveform ids
WAVES = ["circle", "sine", "oscillate", "white_noise"]


class TaskTargets:
    """Precomputes common tables; produces per-env base-frame targets.

    Randomization ranges (wide, per task-spec sign-off):
      trans amp 1-5 cm, att amp 0.5-3 deg, freq 0.5-5 Hz, waveform in WAVES.
    """

    def __init__(self, device, dt=1.0 / 50.0,
                 amp_trans=(0.01, 0.05), amp_att_deg=(0.5, 3.0),
                 freq=(0.5, 5.0), schedule=None,
                 base_nominal=(0.0, 0.0, 0.0),
                 moving_base_circle=False):
        self.device = device
        self.dt = dt
        self.amp_trans = amp_trans
        self.amp_att = (math.radians(amp_att_deg[0]), math.radians(amp_att_deg[1]))
        self.freq = freq

        # ----- common tables from the numpy TaskModel (no base jitter here; we
        # add the per-env disturbance ourselves) -----
        tm = TaskModel(schedule=schedule, base_nominal=base_nominal)
        self.period = 4.0 if moving_base_circle else tm.period
        self.T = int(round(self.period / dt))
        ws_pos = np.zeros((self.T, 3)); ws_R = np.zeros((self.T, 3, 3))
        yaw = np.zeros(self.T); draw = np.zeros(self.T)
        if moving_base_circle:
            # Exact match to moving_base_trajectory_config.py: a horizontal XY
            # circle in WORLD, not ShapeSchedule's vertical front-plane circle.
            center = np.array([0.30, 0.01, 0.10])
            for k in range(self.T):
                a = 2.0 * math.pi * (k * dt) / self.period
                ws_pos[k] = center + 0.08 * np.array(
                    [math.cos(a), math.sin(a), 0.0])
                ws_R[k] = ORIENTATION_TARGET
                draw[k] = 1.0
        else:
            # query world target with a DISTURBANCE-FREE base (nominal facing only)
            tm.dist.trans_amp = 0.0; tm.dist.att_amp = 0.0
            for k in range(self.T):
                t = k * dt
                p_w, R_w = tm.ee_target_world(t)
                ws_pos[k] = p_w; ws_R[k] = R_w
                yaw[k] = tm._face_yaw(t)
                draw[k] = 1.0 if tm.is_drawing(t) else 0.0
        self.ws_pos = torch.tensor(ws_pos, dtype=torch.float32, device=device)
        self.ws_R = torch.tensor(ws_R, dtype=torch.float32, device=device)
        self.yaw = torch.tensor(yaw, dtype=torch.float32, device=device)
        self.draw = torch.tensor(draw, dtype=torch.float32, device=device)
        self.base_nominal = torch.tensor(base_nominal, dtype=torch.float32,
                                         device=device)

    # ----- per-env disturbance params -----
    def sample_disturbance(self, n):
        d = self.device
        def u(lo, hi): return lo + (hi - lo) * torch.rand(n, device=d)
        return {
            "wave_t": torch.randint(0, len(WAVES), (n,), device=d),
            "wave_a": torch.randint(0, len(WAVES), (n,), device=d),
            "amp_t": u(*self.amp_trans),
            "amp_z": torch.zeros(n, device=d),
            "z_phase": torch.full((n,), math.pi / 3.0, device=d),
            "amp_a": u(*self.amp_att),
            "freq_t": u(*self.freq),
            "freq_a": u(*self.freq),
            "phase": u(0.0, 2 * math.pi),
        }

    def sample_circle_disturbance(self, n):
        """Config-matched moving_base_circle disturbance (xy circle r~0.02, ~1Hz)
        with small randomization, for the single-circle precision/imitation task."""
        d = self.device
        def u(lo, hi): return lo + (hi - lo) * torch.rand(n, device=d)
        return {
            "wave_t": torch.zeros(n, dtype=torch.long, device=d),   # circle (xy)
            "wave_a": torch.ones(n, dtype=torch.long, device=d),    # sine attitude
            "amp_t": u(0.015, 0.025),
            "amp_z": u(0.0, 0.02),
            "z_phase": torch.full((n,), math.pi / 3.0, device=d),
            "amp_a": u(math.radians(0.05), math.radians(0.15)),
            "freq_t": u(0.8, 1.2),
            "freq_a": u(0.8, 1.2),
            "phase": u(0.0, 2 * math.pi),
        }

    def _wave(self, wave_id, t, freq, phase):
        """All args (E,); returns (E,3) unit offset. t is per-env time [s]."""
        E = wave_id.shape[0]
        out = torch.zeros(E, 3, device=self.device)
        wt = 2 * math.pi * freq * t
        m = wave_id == 0                                   # circle
        a = wt[m] + phase[m]
        out[m, 0] = torch.cos(a); out[m, 1] = torch.sin(a)
        m = wave_id == 1                                   # sine
        for ax, ph in enumerate((0.0, 2.094, 4.189)):
            out[m, ax] = torch.sin(wt[m] + phase[m] + ph)
        m = wave_id == 2                                   # oscillate
        for ax, ph in enumerate((0.0, 2.094, 4.189)):
            out[m, ax] = (torch.sin(wt[m]) + 0.4 * torch.sin(3 * wt[m] + ph)) / 1.4
        m = wave_id == 3                                   # white noise (per time-bin)
        b = torch.floor(t[m] / max(self.dt, 1e-6))
        for ax in range(3):
            r = torch.sin(12.9898 * b + 78.233 * (ax + 1) + phase[m]) * 43758.5453
            out[m, ax] = 2 * (r - torch.floor(r)) - 1
        m = wave_id == 4                                   # 3D sphere: shaking that
        # fills a ball of radius 1 (per-axis 1/sqrt3, decorrelated phases -> |.|<=1)
        for ax, ph in enumerate((0.0, 2.094, 4.189)):
            out[m, ax] = torch.sin(wt[m] + phase[m] + ph) / math.sqrt(3.0)
        return out

    def set_sphere_sweep(self, R, f, att_deg=0.0):
        """Set per-env disturbance to a 3D sphere of radius R[env] shaking at
        f[env] Hz (for the (R,f) envelope sweep). R,f are (E,) tensors."""
        n = R.shape[0]
        z = torch.zeros(n, device=self.device)
        self._sweep = {
            "wave_t": torch.full((n,), 4, dtype=torch.long, device=self.device),
            "wave_a": torch.full((n,), 1, dtype=torch.long, device=self.device),
            "amp_t": R.to(self.device),
            "amp_z": z,
            "z_phase": torch.full((n,), math.pi / 3.0, device=self.device),
            "amp_a": torch.full((n,), math.radians(att_deg), device=self.device),
            "freq_t": f.to(self.device),
            "freq_a": f.to(self.device),
            "phase": z,
        }
        return self._sweep

    def base_pose(self, idx, dist):
        """idx (E,) long episode-phase index. Returns pos (E,3), R (E,3,3)."""
        t = idx.float() * self.dt
        dpos = dist["amp_t"][:, None] * self._wave(dist["wave_t"], t,
                                                   dist["freq_t"], dist["phase"])
        if "amp_z" in dist:
            dpos[:, 2] += dist["amp_z"] * torch.sin(
                2 * math.pi * dist["freq_t"] * t + dist["z_phase"])
        drpy = dist["amp_a"][:, None] * self._wave(dist["wave_a"], t,
                                                   dist["freq_a"], dist["phase"])
        yaw = self.yaw[idx]
        z = torch.zeros_like(yaw)
        R = _rpy_to_R(torch.stack([z, z, yaw], -1)) @ _rpy_to_R(drpy)
        pos = self.base_nominal[None, :] + dpos
        return pos, R

    def base_accel(self, step_env, dist):
        """World-frame linear acceleration (E,3) of the base from the translation
        disturbance. For sinusoidal components d2/dt2 = -w^2 * pos (exact for
        circle/sine, bounded approx for oscillate/noise)."""
        idx = step_env % self.T
        t = idx.float() * self.dt
        dpos = dist["amp_t"][:, None] * self._wave(dist["wave_t"], t,
                                                   dist["freq_t"], dist["phase"])
        w = 2 * math.pi * dist["freq_t"]
        if "amp_z" in dist:
            dpos[:, 2] += dist["amp_z"] * torch.sin(
                w * t + dist["z_phase"])
        return -(w ** 2)[:, None] * dpos

    def base_velocity(self, step_env, dist):
        """World-frame base linear velocity (E,3).

        A centered difference intentionally uses the same target generator as
        ``base_pose``. This keeps the feature valid for every waveform and
        makes the offline and Isaac observation contracts easy to reproduce.
        """
        p_next, _ = self.base_pose((step_env + 1) % self.T, dist)
        p_prev, _ = self.base_pose((step_env - 1) % self.T, dist)
        return (p_next - p_prev) / (2.0 * self.dt)

    def ee_target_base(self, step_env, dist):
        """step_env (E,) long. Returns base-frame target pos (E,3), R (E,3,3),
        drawing flag (E,)."""
        idx = step_env % self.T
        bp, bR = self.base_pose(idx, dist)
        p_w = self.ws_pos[idx]
        R_w = self.ws_R[idx]
        RT = bR.transpose(1, 2)
        p_b = torch.bmm(RT, (p_w - bp)[:, :, None])[:, :, 0]
        R_b = torch.bmm(RT, R_w)
        return p_b, R_b, self.draw[idx]
