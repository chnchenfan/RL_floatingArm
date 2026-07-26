"""Geometry regression tests for the config-matched Phase-1 circle target."""

import math

import torch

from task_spec.target_torch import TaskTargets


def test_world_circle_is_horizontal_and_has_config_radius():
    targets = TaskTargets(
        "cpu",
        dt=0.02,
        base_nominal=(0.0, 0.0, 0.01),
        moving_base_circle=True,
    )
    center = torch.tensor([0.30, 0.01, 0.10])
    relative = targets.ws_pos - center
    torch.testing.assert_close(
        relative[:, 2], torch.zeros(targets.T), rtol=0.0, atol=1.0e-7)
    torch.testing.assert_close(
        torch.linalg.norm(relative[:, :2], dim=-1),
        torch.full((targets.T,), 0.08),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_base_z_nominal_and_independent_sinusoid():
    targets = TaskTargets(
        "cpu",
        dt=0.02,
        base_nominal=(0.0, 0.0, 0.01),
        moving_base_circle=True,
    )
    disturbance = {
        "wave_t": torch.tensor([0]),
        "wave_a": torch.tensor([1]),
        "amp_t": torch.tensor([0.02]),
        "amp_z": torch.tensor([0.01]),
        "z_phase": torch.tensor([math.pi / 3.0]),
        "amp_a": torch.tensor([0.0]),
        "freq_t": torch.tensor([1.0]),
        "freq_a": torch.tensor([1.0]),
        "phase": torch.tensor([0.0]),
    }
    position, _ = targets.base_pose(torch.tensor([0]), disturbance)
    expected = torch.tensor(
        [[0.02, 0.0, 0.01 + 0.01 * math.sin(math.pi / 3.0)]])
    torch.testing.assert_close(position, expected, rtol=1.0e-6, atol=1.0e-6)
    velocity = targets.base_velocity(torch.tensor([0]), disturbance)
    omega = 2.0 * math.pi
    centered_scale = math.sin(omega * targets.dt) / targets.dt
    expected_velocity = torch.tensor([[
        0.0,
        0.02 * centered_scale,
        0.01 * centered_scale * math.cos(math.pi / 3.0),
    ]])
    torch.testing.assert_close(
        velocity, expected_velocity, rtol=1.0e-5, atol=1.0e-5)
