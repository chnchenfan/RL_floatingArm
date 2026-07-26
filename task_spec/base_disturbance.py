#!/usr/bin/env python3
"""Base disturbance library: the high-frequency/high-speed base shaking the arm
must inertially reject (see task spec). Pure numpy; shared by the pinocchio
demo and the Isaac Lab env (as the base root motion + domain randomization).

Each generator maps time t -> (base_pos[3], base_rpy[3]) offset from nominal.
Compose translation + attitude, pick a waveform, randomize freq/amplitude for RL.
"""
import math

import numpy as np


class BaseDisturbance:
    """Configurable base motion. waveform in {none, circle, sine, oscillate,
    white_noise}. Translation and attitude are generated independently."""

    def __init__(self, trans_wave="circle", att_wave="sine",
                 trans_amp=0.02, att_amp_deg=1.0,
                 trans_freq=1.0, att_freq=1.5, seed=0, dt=1.0 / 100.0):
        self.trans_wave = trans_wave
        self.att_wave = att_wave
        self.trans_amp = trans_amp
        self.att_amp = math.radians(att_amp_deg)
        self.trans_freq = trans_freq
        self.att_freq = att_freq
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        # pre-integrated band-limited noise state (per axis) for white_noise mode
        self._noise_t = {"trans": None, "att": None}

    # --- waveforms: t (scalar) -> unit 3-vector offset ---
    def _wave(self, kind, t, freq, phase=(0.0, 2.0, 4.0)):
        w = 2.0 * math.pi * freq
        if kind == "none":
            return np.zeros(3)
        if kind == "circle":
            return np.array([math.cos(w * t), math.sin(w * t), 0.0])
        if kind == "sine":
            return np.array([math.sin(w * t + phase[k]) for k in range(3)])
        if kind == "oscillate":  # sharper, multi-harmonic shake
            return np.array([
                math.sin(w * t) + 0.4 * math.sin(3 * w * t + phase[k])
                for k in range(3)]) / 1.4
        if kind == "white_noise":
            # deterministic-per-call pseudo white noise via hashed time bins
            b = int(t / max(self.dt, 1e-6))
            r = np.array([
                math.sin(12.9898 * b + 78.233 * (k + 1)) * 43758.5453
                for k in range(3)])
            return 2.0 * (r - np.floor(r)) - 1.0
        raise ValueError(f"unknown waveform {kind}")

    def at(self, t):
        """Return (base_pos_offset[3], base_rpy_offset[3]) at time t."""
        pos = self.trans_amp * self._wave(self.trans_wave, t, self.trans_freq)
        rpy = self.att_amp * self._wave(self.att_wave, t, self.att_freq,
                                        phase=(0.0, 2.094, 4.189))
        return pos, rpy

    @staticmethod
    def randomized(rng):
        """Sample a disturbance config for domain randomization (RL)."""
        waves = ["circle", "sine", "oscillate", "white_noise"]
        return BaseDisturbance(
            trans_wave=rng.choice(waves), att_wave=rng.choice(waves),
            trans_amp=float(rng.uniform(0.01, 0.05)),
            att_amp_deg=float(rng.uniform(0.5, 3.0)),
            trans_freq=float(rng.uniform(0.5, 5.0)),
            att_freq=float(rng.uniform(0.5, 6.0)),
            seed=int(rng.integers(0, 1 << 30)),
        )


if __name__ == "__main__":
    for wave in ["circle", "sine", "oscillate", "white_noise"]:
        d = BaseDisturbance(trans_wave=wave, att_wave=wave)
        p, r = d.at(0.37)
        print(f"{wave:12s} pos={np.round(p,4)} rpy_deg={np.round(np.degrees(r),3)}")
