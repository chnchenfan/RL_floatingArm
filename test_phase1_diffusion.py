"""CPU contract tests for the Phase-1 diffusion action policy."""

import torch

from distill.diffusion_policy import DiffusionChunkPolicy


def make_policy():
    torch.manual_seed(7)
    policy = DiffusionChunkPolicy(
        cond_dim=72,
        action_dim=7,
        horizon=8,
        diffusion_steps=20,
        d_model=32,
        layers=1,
        heads=4,
        dropout=0.0,
    )
    policy.eval()
    return policy


def test_sample_shape_and_bounds():
    policy = make_policy()
    output = policy.sample(torch.randn(5, 72), inference_steps=4)
    assert output.shape == (5, 8, 7)
    assert torch.isfinite(output).all()
    assert output.abs().max() <= 1.0


def test_deterministic_mode_is_repeatable():
    policy = make_policy()
    condition = torch.randn(3, 72)
    a = policy.sample(condition, inference_steps=4, deterministic=True)
    b = policy.sample(
        condition,
        inference_steps=4,
        generator=torch.Generator().manual_seed(999),
        deterministic=True,
    )
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


def test_stochastic_mode_respects_generator():
    policy = make_policy()
    condition = torch.randn(3, 72)
    a = policy.sample(
        condition,
        inference_steps=4,
        generator=torch.Generator().manual_seed(11),
    )
    b = policy.sample(
        condition,
        inference_steps=4,
        generator=torch.Generator().manual_seed(11),
    )
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


def test_precision_loss_has_finite_cartesian_gradient():
    policy = make_policy()
    action = torch.randn(5, 8, 7).clamp(-1.0, 1.0)
    condition = torch.randn(5, 72)
    jacobian = torch.randn(5, 3, 7) * 0.2
    losses = policy.loss(
        action,
        condition,
        first_position_jacobian=jacobian,
        bc_weight=0.2,
        cartesian_weight=0.02,
        return_components=True,
    )
    assert set(losses) == {"total", "diffusion", "first_bc", "cartesian"}
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )
