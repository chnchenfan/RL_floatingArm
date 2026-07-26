#!/usr/bin/env python3
"""Phase-2 residual-RL environment built on the selected Phase-1 policy.

The frozen diffusion policy supplies the nominal action. PPO controls only a
small bounded residual:

    a_final = clamp(a_phase1 + residual_scale * a_rl, -1, 1)

The plant includes the moving-base inertial force that the kinematic MPC
teacher and Phase-1 behavior-cloning policy did not model.
"""

from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np
import torch

from isaaclab.utils import configclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.diffusion_policy import make_policy_from_checkpoint  # noqa: E402
from env.arm_track_env import ArmTrackEnv, ArmTrackEnvCfg  # noqa: E402


ROOT = "/home/windylab/code/isaac_arm_rl"
PHASE1_CHECKPOINT = f"{ROOT}/logs/phase1_diffusion/policy_precision.pt"
SEED_CSVS = (
    "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
    "moving_base_mpc_loop_20260724_051521.csv",
    "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
    "moving_base_mpc_loop_20260724_051912.csv",
    "/home/windylab/code/windylab_ws/src/arm-platform/demo/data/"
    "moving_base_mpc_loop_20260724_052123.csv",
)


@configclass
class ResidualArmTrackEnvCfg(ArmTrackEnvCfg):
    """Configuration for residual learning around the frozen Phase-1 policy."""

    observation_space = 54  # normalized Phase-1 obs (47) + nominal action (7)
    episode_length_s = 8.0
    single_circle = True
    extended_phase1_obs = True
    start_phase_random = False
    inertial_coupling = True

    phase1_checkpoint: str = PHASE1_CHECKPOINT
    phase1_ddim_steps: int = 8
    phase1_policy_device: str = "cuda:0"
    residual_scale: float = 0.25
    initial_joint_noise_std: float = 0.002
    condition_jitter_fraction: float = 0.10
    divergence_threshold_m: float = 0.05

    precision_sigma_m: float = 0.003
    tracking_sigma_m: float = 0.010
    # Keep the MPC/Phase-1 orientation lock while improving position.  The
    # original 0.30-rad, 0.25-weight term made a 10--20 degree orientation
    # error cheap relative to a sub-millimetre position gain.
    orientation_sigma_rad: float = 0.15
    orientation_limit_rad: float = math.radians(5.0)
    w_precision: float = 4.0
    w_tracking: float = 2.0
    w_orientation: float = 1.0
    w_orientation_limit: float = 2.0
    w_residual: float = 0.10
    w_residual_rate: float = 0.03
    w_saturation: float = 0.10


@configclass
class WholeResidualArmTrackEnvCfg(ResidualArmTrackEnvCfg):
    """Residual RL on the frozen exact whole task (whole_v2 schema).

    Training uses random-phase truncated windows over the 73.039-s cycle;
    resets are seeded from the per-phase state bank built out of clean MPC
    whole rollouts.  Final validation must still run one full continuous
    cycle (see scripts/eval_phase2_whole.py)."""

    single_circle = False
    whole_task = True
    start_phase_random = True
    episode_length_s = 12.0
    # whole-config MPC step contract (0.04 rad/tick at 50 Hz)
    max_joint_step = 0.04
    # plant model calibrated against the 2026-07-26 whole loop CSVs:
    # velocity-feedforward ramp tracking with 80000/4000 PD.
    # command_delay_ticks MUST stay 0: with delay=1 the BC loop (2-tick
    # action->obs latency) oscillates and 100% diverges; with delay=0 the
    # r3 policy holds 1.80mm mean / 0% divergence over 800-tick windows
    # (measured 2026-07-26, 256 envs, random phases).
    command_delay_ticks = 0
    plant_stiffness = 80000.0
    plant_damping = 4000.0

    phase1_checkpoint: str = (
        f"{ROOT}/logs/phase1_whole_diffusion/policy_whole.pt")
    phase1_condition_gain: float = 1.0
    state_bank_path: str = f"{ROOT}/data/whole_v2_state_bank.npz"


class ResidualArmTrackEnv(ArmTrackEnv):
    cfg: ResidualArmTrackEnvCfg

    def __init__(self, cfg: ResidualArmTrackEnvCfg, render_mode=None, **kwargs):
        # These guards are needed because DirectRLEnv construction may dispatch
        # to subclass reset/observation hooks before Phase-2 buffers exist.
        self._phase1_policy = None
        self._phase1_history = None
        self._history_needs_reset = None
        self._condition_id = None
        super().__init__(cfg, render_mode, **kwargs)

        if not os.path.isfile(cfg.phase1_checkpoint):
            raise FileNotFoundError(
                f"Phase-1 checkpoint not found: {cfg.phase1_checkpoint}")
        checkpoint = torch.load(
            cfg.phase1_checkpoint, map_location="cpu", weights_only=False)
        if len(checkpoint["obs_mean"]) != 47:
            raise ValueError("Phase-2 requires the 47-D Phase-1 v2 observation")

        requested_device = torch.device(cfg.phase1_policy_device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device(self.device)
        self._phase1_device = requested_device
        self._phase1_policy = make_policy_from_checkpoint(
            checkpoint, self._phase1_device)
        for parameter in self._phase1_policy.parameters():
            parameter.requires_grad_(False)
        self._phase1_obs_mean = torch.as_tensor(
            checkpoint["obs_mean"],
            device=self._phase1_device,
            dtype=torch.float32,
        )
        self._phase1_obs_std = torch.as_tensor(
            checkpoint["obs_std"],
            device=self._phase1_device,
            dtype=torch.float32,
        )

        self._phase1_history = torch.zeros(
            self.num_envs, 2, 47, device=self._phase1_device)
        self._history_needs_reset = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._condition_id = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        if cfg.whole_task:
            gain = float(getattr(cfg, "phase1_condition_gain", 1.0))
            self._phase1_gain = torch.full((3,), gain, device=self.device)
        else:
            self._phase1_gain = torch.tensor(
                [0.87, 0.90, 0.93], device=self.device)
        self._base_action = torch.zeros(
            self.num_envs, 7, device=self.device)
        self._residual_action = torch.zeros_like(self._base_action)
        self._previous_residual_action = torch.zeros_like(self._base_action)
        self._residual_delta = torch.zeros_like(self._base_action)
        self._final_action = torch.zeros_like(self._base_action)
        if cfg.whole_task:
            self._state_bank = _load_state_bank(
                cfg.state_bank_path, self.device)
            self._seed_q = self._seed_dq = None
        else:
            self._state_bank = None
            self._seed_q, self._seed_dq = _load_seed_states(self.device)

        all_envs = torch.arange(
            self.num_envs, device=self.device, dtype=torch.long)
        self._set_phase2_condition(all_envs)
        self._history_needs_reset[:] = True

    def _set_phase2_condition(self, env_ids: torch.Tensor):
        """Sample the episode's base-motion condition.

        whole task: one config-matched condition with small jitter, random
        start phase kept from the base-class reset.  circle task: legacy even
        easy/medium/hard mixture with phase forced to 0."""
        count = len(env_ids)
        if self.cfg.whole_task:
            self._condition_id[env_ids] = 0
            d = self.targets.sample_whole_disturbance(
                count, jitter=self.cfg.condition_jitter_fraction)
            for key, value in d.items():
                self._dist[key][env_ids] = value
            return
        condition_id = torch.randint(
            0, 3, (count,), device=self.device)
        self._condition_id[env_ids] = condition_id

        amp_t_nominal = torch.tensor(
            [0.02, 0.02, 0.01], device=self.device)[condition_id]
        amp_z_nominal = torch.tensor(
            [0.00, 0.02, 0.01], device=self.device)[condition_id]
        freq_nominal = torch.tensor(
            [1.00, 1.00, 2.00], device=self.device)[condition_id]
        jitter = self.cfg.condition_jitter_fraction

        def factor():
            return 1.0 + jitter * (
                2.0 * torch.rand(count, device=self.device) - 1.0)

        disturbance = self._dist
        disturbance["wave_t"][env_ids] = 0
        disturbance["wave_a"][env_ids] = 1
        disturbance["amp_t"][env_ids] = amp_t_nominal * factor()
        disturbance["amp_z"][env_ids] = amp_z_nominal * factor()
        disturbance["z_phase"][env_ids] = math.pi / 3.0
        disturbance["amp_a"][env_ids] = (
            math.radians(0.05)
            + math.radians(0.10) * torch.rand(count, device=self.device)
        )
        disturbance["freq_t"][env_ids] = freq_nominal * factor()
        disturbance["freq_a"][env_ids].copy_(
            disturbance["freq_t"][env_ids])
        disturbance["phase"][env_ids] = 0.0
        self._start[env_ids] = 0

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self._condition_id is None:
            return

        self._set_phase2_condition(env_ids)
        if self.cfg.whole_task:
            q, dq, prev_action = _draw_bank_states(
                self._state_bank, self._start[env_ids])
            q = q + self.cfg.initial_joint_noise_std * torch.randn_like(q)
        else:
            condition_id = self._condition_id[env_ids]
            q = self._seed_q[condition_id].clone()
            q += self.cfg.initial_joint_noise_std * torch.randn_like(q)
            dq = self._seed_dq[condition_id].clone()
            prev_action = None
        self.robot.write_joint_state_to_sim(q, dq, env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        # the base-class reset synced command anchors to the home posture;
        # re-anchor them to the just-written hand-off states
        self.sync_command_state(env_ids)

        self._history_needs_reset[env_ids] = True
        self._base_action[env_ids] = 0.0
        self._residual_action[env_ids] = 0.0
        self._previous_residual_action[env_ids] = 0.0
        self._residual_delta[env_ids] = 0.0
        self._final_action[env_ids] = 0.0
        if prev_action is not None:
            self._last_executed_action[env_ids] = prev_action
            self._prev_action[env_ids] = prev_action

    def _get_observations(self):
        raw_observation = super()._get_observations()["policy"]
        if self._phase1_policy is None:
            # Construction-time fallback; normal reset occurs after the frozen
            # policy has been loaded.
            padding = torch.zeros(
                raw_observation.shape[0],
                7,
                device=raw_observation.device,
                dtype=raw_observation.dtype,
            )
            return {"policy": torch.cat([raw_observation, padding], dim=-1)}

        current = raw_observation.to(self._phase1_device)
        reset_mask = self._history_needs_reset.to(self._phase1_device)
        self._phase1_history[:, 0] = torch.where(
            reset_mask[:, None],
            current,
            self._phase1_history[:, 1],
        )
        self._phase1_history[:, 1] = current
        self._history_needs_reset[:] = False

        normalized_history = (
            (self._phase1_history - self._phase1_obs_mean)
            / self._phase1_obs_std
        ).clamp(-10.0, 10.0)
        with torch.inference_mode(), torch.autocast(
            device_type=self._phase1_device.type,
            dtype=torch.bfloat16,
            enabled=self._phase1_device.type == "cuda",
        ):
            chunk = self._phase1_policy.sample(
                normalized_history.flatten(1),
                inference_steps=self.cfg.phase1_ddim_steps,
                deterministic=True,
            )
        gain = self._phase1_gain[
            self._condition_id
        ].to(self._phase1_device)[:, None]
        base_action = (gain * chunk[:, 0].float()).clamp(-1.0, 1.0)
        self._base_action.copy_(base_action.to(self.device))

        normalized_current = normalized_history[:, 1].to(self.device)
        residual_observation = torch.cat(
            [normalized_current, self._base_action], dim=-1)
        return {"policy": residual_observation}

    def _pre_physics_step(self, residual_action: torch.Tensor):
        residual = residual_action.to(self.device).clamp(-1.0, 1.0)
        self._residual_delta.copy_(
            residual - self._previous_residual_action)
        self._previous_residual_action.copy_(residual)
        self._residual_action.copy_(residual)
        final_action = (
            self._base_action + self.cfg.residual_scale * residual
        ).clamp(-1.0, 1.0)
        self._final_action.copy_(final_action)
        super()._pre_physics_step(final_action)

    def _get_rewards(self):
        position_error = torch.linalg.norm(
            self._p_tgt - self._p_ee, dim=-1)
        rotation_error = torch.bmm(
            self._R_ee.transpose(1, 2), self._R_tgt)
        cosine = (
            rotation_error[:, 0, 0]
            + rotation_error[:, 1, 1]
            + rotation_error[:, 2, 2]
            - 1.0
        ) * 0.5
        orientation_error = torch.arccos(
            torch.clamp(cosine, -1.0, 1.0))

        precision_reward = torch.exp(
            -(position_error / self.cfg.precision_sigma_m).square())
        tracking_reward = torch.exp(
            -(position_error / self.cfg.tracking_sigma_m).square())
        orientation_reward = torch.exp(
            -(orientation_error / self.cfg.orientation_sigma_rad).square())
        orientation_limit_cost = torch.relu(
            (
                orientation_error - self.cfg.orientation_limit_rad
            ) / self.cfg.orientation_limit_rad
        ).square()
        residual_cost = self._residual_action.square().mean(dim=-1)
        residual_rate_cost = self._residual_delta.square().mean(dim=-1)
        saturation_cost = (
            self._final_action.abs() > 0.999
        ).float().mean(dim=-1)

        reward = (
            self.cfg.w_precision * precision_reward
            + self.cfg.w_tracking * tracking_reward
            + self.cfg.w_orientation * orientation_reward
            - self.cfg.w_orientation_limit * orientation_limit_cost
            - self.cfg.w_residual * residual_cost
            - self.cfg.w_residual_rate * residual_rate_cost
            - self.cfg.w_saturation * saturation_cost
        )
        self.extras["log"] = {
            "tracking/position_error_mm": position_error.mean() * 1000.0,
            "tracking/max_error_mm": position_error.max() * 1000.0,
            "tracking/orientation_error_deg": (
                torch.rad2deg(orientation_error).mean()
            ),
            "tracking/orientation_limit_fraction": (
                orientation_error > self.cfg.orientation_limit_rad
            ).float().mean(),
            "residual/rms": torch.sqrt(residual_cost.mean()),
            "residual/saturation_fraction": saturation_cost.mean(),
            "reward/mean": reward.mean(),
        }
        return reward

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if hasattr(self, "_p_tgt") and hasattr(self, "_p_ee"):
            diverged = torch.linalg.norm(
                self._p_tgt - self._p_ee, dim=-1
            ) > self.cfg.divergence_threshold_m
        else:
            diverged = torch.zeros_like(time_out)
        return diverged, time_out


def _load_state_bank(path, device):
    """Load the per-phase (q, dq, prev_action) bank from clean whole logs.

    Returns tensors plus, per phase tick 0..T-1, the [start, end) sample range
    (falling back to the nearest populated tick)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"whole state bank not found: {path}; run "
            "scripts/prepare_phase1_whole_dataset.py first")
    data = np.load(path)
    phase = data["phase_tick"].astype(np.int64)          # sorted
    ticks = int(data["ticks_per_cycle"])
    starts = np.searchsorted(phase, np.arange(ticks), side="left")
    ends = np.searchsorted(phase, np.arange(ticks), side="right")
    empty = starts == ends
    if empty.any():
        populated = np.flatnonzero(~empty)
        for tick in np.flatnonzero(empty):
            nearest = populated[np.argmin(np.abs(populated - tick))]
            starts[tick] = starts[nearest]
            ends[tick] = ends[nearest]
    prev_action = data["prev_action"] if "prev_action" in data else (
        np.zeros_like(data["q"]))
    return {
        "q": torch.tensor(data["q"], dtype=torch.float32, device=device),
        "dq": torch.tensor(data["dq"], dtype=torch.float32, device=device),
        "prev_action": torch.tensor(
            prev_action, dtype=torch.float32, device=device),
        "range_start": torch.tensor(
            starts, dtype=torch.long, device=device),
        "range_end": torch.tensor(ends, dtype=torch.long, device=device),
        "ticks": ticks,
    }


def _draw_bank_states(bank, start_phase):
    """start_phase (E,) long -> (q, dq, prev_action) sampled from the bank."""
    lo = bank["range_start"][start_phase]
    hi = bank["range_end"][start_phase]
    pick = lo + (torch.rand_like(lo, dtype=torch.float32)
                 * (hi - lo).float()).long().clamp(min=0)
    pick = torch.minimum(pick, hi - 1)
    return (
        bank["q"][pick].clone(),
        bank["dq"][pick].clone(),
        bank["prev_action"][pick].clone(),
    )


def _load_seed_states(device):
    q_rows = []
    dq_rows = []
    for path in SEED_CSVS:
        with open(path, newline="") as stream:
            row = next(csv.DictReader(stream))
        q_rows.append([
            float(row[f"q_meas_{joint}"]) for joint in range(1, 8)
        ])
        dq_rows.append([
            float(row[f"dq_meas_{joint}"]) for joint in range(1, 8)
        ])
    return (
        torch.tensor(q_rows, device=device, dtype=torch.float32),
        torch.tensor(dq_rows, device=device, dtype=torch.float32),
    )
