"""Conditional diffusion action-chunk policy for Phase 1."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freq = torch.exp(
            -scale * torch.arange(half, device=t.device, dtype=torch.float32))
        phase = t.float()[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        action_dim: int = 7,
        horizon: int = 8,
        d_model: int = 128,
        layers: int = 4,
        heads: int = 8,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.action_in = nn.Linear(action_dim, d_model)
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.position = nn.Parameter(torch.zeros(1, horizon, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, action_dim))

    def forward(
        self, noisy_action: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        x = self.action_in(noisy_action)
        context = self.cond_encoder(cond) + self.time_encoder(timestep)
        x = x + self.position + context[:, None, :]
        return self.out(self.transformer(x))


class DiffusionChunkPolicy(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        action_dim: int = 7,
        horizon: int = 8,
        diffusion_steps: int = 100,
        d_model: int = 128,
        layers: int = 4,
        heads: int = 8,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.diffusion_steps = diffusion_steps
        self.denoiser = DiffusionTransformer(
            cond_dim=cond_dim,
            action_dim=action_dim,
            horizon=horizon,
            d_model=d_model,
            layers=layers,
            heads=heads,
            dropout=dropout,
        )
        betas = torch.linspace(1.0e-4, 2.0e-2, diffusion_steps)
        alphas = 1.0 - betas
        self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0))

    def loss(
        self,
        action: torch.Tensor,
        cond: torch.Tensor,
        first_position_jacobian: torch.Tensor | None = None,
        action_scale: float = 0.02,
        bc_weight: float = 0.0,
        cartesian_weight: float = 0.0,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        batch = action.shape[0]
        t = torch.randint(
            0, self.diffusion_steps, (batch,), device=action.device)
        noise = torch.randn_like(action)
        ab = self.alpha_bar[t].view(batch, 1, 1)
        noisy = ab.sqrt() * action + (1.0 - ab).sqrt() * noise
        predicted = self.denoiser(noisy, t, cond)
        horizon_weight = torch.pow(
            torch.tensor(0.92, device=action.device),
            torch.arange(self.horizon, device=action.device),
        ).view(1, self.horizon, 1)
        diffusion = ((predicted - noise).square() * horizon_weight).mean()

        # Recover x_0 from the same noisy sample. The first action is the one
        # executed by the receding-horizon controller, so supervise it directly
        # instead of relying only on the indirect noise-prediction objective.
        predicted_action = (
            noisy - (1.0 - ab).sqrt() * predicted
        ) / ab.sqrt()
        predicted_action = predicted_action.clamp(-1.5, 1.5)
        first_bc = F.smooth_l1_loss(
            predicted_action[:, 0],
            action[:, 0],
            beta=0.05,
        )

        cartesian = diffusion.new_zeros(())
        if first_position_jacobian is not None:
            joint_delta = (
                predicted_action[:, 0] - action[:, 0]
            ) * action_scale
            cartesian_delta_mm = torch.bmm(
                first_position_jacobian,
                joint_delta[:, :, None],
            )[:, :, 0] * 1000.0
            cartesian = F.smooth_l1_loss(
                cartesian_delta_mm,
                torch.zeros_like(cartesian_delta_mm),
                beta=0.5,
            )

        total = (
            diffusion
            + bc_weight * first_bc
            + cartesian_weight * cartesian
        )
        if return_components:
            return {
                "total": total,
                "diffusion": diffusion,
                "first_bc": first_bc,
                "cartesian": cartesian,
            }
        return total

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        inference_steps: int = 10,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """DDIM action sampling.

        ``deterministic=True`` starts from the center of the Gaussian instead
        of drawing a fresh sample. This is the deployment mode for the
        deterministic MPC teacher: it selects a repeatable central action and
        avoids injecting diffusion sampling jitter into the control loop.
        """
        batch = cond.shape[0]
        shape = (batch, self.horizon, self.action_dim)
        if deterministic:
            x = torch.zeros(shape, device=cond.device, dtype=cond.dtype)
        else:
            x = torch.randn(
                shape,
                device=cond.device,
                dtype=cond.dtype,
                generator=generator,
            )
        # Build the short DDIM schedule on the CPU. Calling ``.item()`` on a
        # CUDA scalar inside every denoising iteration forces a device
        # synchronization and adds avoidable latency to the live controller.
        indices_tensor = torch.linspace(
            self.diffusion_steps - 1,
            0,
            inference_steps,
        ).round().long()
        indices = torch.unique_consecutive(indices_tensor).tolist()
        for i, timestep_index in enumerate(indices):
            t = torch.full(
                (batch,),
                timestep_index,
                device=cond.device,
                dtype=torch.long,
            )
            eps = self.denoiser(x, t, cond)
            ab = self.alpha_bar[timestep_index].to(dtype=x.dtype)
            x0 = (x - (1.0 - ab).sqrt() * eps) / ab.sqrt()
            x0 = x0.clamp(-1.5, 1.5)
            if i + 1 == len(indices):
                x = x0
            else:
                next_ab = self.alpha_bar[indices[i + 1]].to(dtype=x.dtype)
                x = next_ab.sqrt() * x0 + (1.0 - next_ab).sqrt() * eps
        return x.clamp(-1.0, 1.0)


def make_policy_from_checkpoint(checkpoint: dict, device: torch.device | str):
    cfg = checkpoint["model_config"]
    policy = DiffusionChunkPolicy(**cfg).to(device)
    policy.load_state_dict(checkpoint["ema_state_dict"])
    policy.eval()
    return policy
