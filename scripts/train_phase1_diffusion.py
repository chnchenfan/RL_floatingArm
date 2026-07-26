#!/usr/bin/env python3
"""Train the robust-first Phase-1 diffusion action-chunk policy."""

from __future__ import annotations

import copy
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = "/home/windylab/code/isaac_arm_rl"
sys.path.insert(0, ROOT)
from distill.diffusion_policy import DiffusionChunkPolicy  # noqa: E402


DATASET = os.environ.get(
    "PHASE1_DATASET", f"{ROOT}/data/phase1_diffusion_dataset_v2.npz")
OUTPUT = os.environ.get(
    "PHASE1_OUTPUT", f"{ROOT}/logs/phase1_diffusion/policy_precision.pt")
RESUME = os.environ.get("PHASE1_RESUME", "")
STEPS = int(os.environ.get("PHASE1_STEPS", "3000"))
BATCH = int(os.environ.get("PHASE1_BATCH", "1024"))
LR = float(os.environ.get("PHASE1_LR", "3e-4"))
SEED = int(os.environ.get("PHASE1_SEED", "42"))
BC_WEIGHT = float(os.environ.get("PHASE1_BC_WEIGHT", "0.2"))
CARTESIAN_WEIGHT = float(os.environ.get("PHASE1_CARTESIAN_WEIGHT", "0.02"))
DART_FRACTION = float(os.environ.get("PHASE1_DART_FRACTION", "0.3"))
CONDITION_WEIGHTS_TEXT = os.environ.get("PHASE1_CONDITION_WEIGHTS", "")


def update_ema(ema, model, decay=0.999):
    with torch.no_grad():
        for ema_p, model_p in zip(ema.parameters(), model.parameters()):
            ema_p.lerp_(model_p, 1.0 - decay)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")
    data = np.load(DATASET)
    obs_mean = torch.tensor(data["obs_mean"], device=device)
    obs_std = torch.tensor(data["obs_std"], device=device)
    train_obs = torch.tensor(data["train_obs"], device=device)
    train_action = torch.tensor(data["train_action"], device=device)
    train_jacobian = torch.tensor(
        data["train_position_jacobian"], device=device)
    train_condition_id = torch.tensor(
        data["train_condition_id"], device=device)
    train_is_dart = torch.tensor(data["train_is_dart"], device=device)
    val_obs = torch.tensor(data["val_obs"], device=device)
    val_action = torch.tensor(data["val_action"], device=device)
    val_jacobian = torch.tensor(
        data["val_position_jacobian"], device=device)
    obs_horizon = int(data["obs_horizon"])
    action_horizon = int(data["action_horizon"])

    def condition(obs):
        normalized = ((obs - obs_mean) / obs_std).clamp(-10.0, 10.0)
        return normalized.flatten(1)

    condition_count = int(train_condition_id.max().item()) + 1
    if CONDITION_WEIGHTS_TEXT:
        condition_weights = np.asarray(
            [float(value) for value in CONDITION_WEIGHTS_TEXT.split(",")],
            dtype=np.float64,
        )
        if len(condition_weights) != condition_count:
            raise ValueError(
                f"expected {condition_count} condition weights, got "
                f"{len(condition_weights)}")
        if np.any(condition_weights <= 0.0):
            raise ValueError("condition weights must be positive")
    else:
        condition_weights = np.ones(condition_count, dtype=np.float64)
    condition_weights /= condition_weights.sum()
    groups = {}
    for condition_id in range(condition_count):
        for is_dart in (False, True):
            groups[(condition_id, is_dart)] = torch.nonzero(
                (train_condition_id == condition_id)
                & (train_is_dart == is_dart),
                as_tuple=False,
            )[:, 0]

    def sample_balanced_indices():
        dart_n = int(round(BATCH * DART_FRACTION))
        type_counts = {False: BATCH - dart_n, True: dart_n}
        pieces = []
        for is_dart, total in type_counts.items():
            raw_counts = total * condition_weights
            counts = np.floor(raw_counts).astype(np.int64)
            for condition_id in np.argsort(
                    -(raw_counts - counts))[:total - int(counts.sum())]:
                counts[condition_id] += 1
            for condition_id in range(condition_count):
                count = int(counts[condition_id])
                candidates = groups[(condition_id, is_dart)]
                if len(candidates) == 0:
                    raise ValueError(
                        f"empty sampling group condition={condition_id} "
                        f"dart={is_dart}")
                pick = torch.randint(
                    0, len(candidates), (count,), device=device)
                pieces.append(candidates[pick])
        indices = torch.cat(pieces)
        return indices[torch.randperm(len(indices), device=device)]

    config = {
        "cond_dim": obs_horizon * train_obs.shape[-1],
        "action_dim": train_action.shape[-1],
        "horizon": action_horizon,
        "diffusion_steps": 100,
        "d_model": 128,
        "layers": 4,
        "heads": 8,
        "dropout": 0.05,
    }
    model = DiffusionChunkPolicy(**config).to(device)
    ema = copy.deepcopy(model).eval()
    resume_step = 0
    if RESUME:
        checkpoint = torch.load(RESUME, map_location=device, weights_only=False)
        if checkpoint["model_config"] != config:
            raise ValueError("resume checkpoint model_config does not match dataset")
        model.load_state_dict(checkpoint["model_state_dict"])
        ema.load_state_dict(checkpoint["ema_state_dict"])
        resume_step = int(checkpoint.get("step", 0))
        print(
            f"[phase1] resumed {RESUME} at checkpoint step={resume_step}",
            flush=True,
        )
    for p in ema.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STEPS, eta_min=LR * 0.05)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    print(
        f"[phase1] train={len(train_obs)} val={len(val_obs)} batch={BATCH} "
        f"steps={STEPS} params={sum(p.numel() for p in model.parameters()):,} "
        f"bc_w={BC_WEIGHT} cart_w={CARTESIAN_WEIGHT} "
        f"dart_fraction={DART_FRACTION} "
        f"condition_weights={condition_weights.tolist()}",
        flush=True,
    )
    best_val = float("inf")
    best_selection = float("inf")
    start = time.time()
    for step in range(1, STEPS + 1):
        idx = sample_balanced_indices()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = model.loss(
                train_action[idx],
                condition(train_obs[idx]),
                first_position_jacobian=train_jacobian[idx],
                action_scale=float(data["action_scale"]),
                bc_weight=BC_WEIGHT,
                cartesian_weight=CARTESIAN_WEIGHT,
                return_components=True,
            )
            loss = losses["total"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        update_ema(ema, model)

        total_step = resume_step + step
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                vidx = torch.randint(
                    0, len(val_obs), (min(BATCH, len(val_obs)),), device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    val_losses = ema.loss(
                        val_action[vidx],
                        condition(val_obs[vidx]),
                        first_position_jacobian=val_jacobian[vidx],
                        action_scale=float(data["action_scale"]),
                        bc_weight=BC_WEIGHT,
                        cartesian_weight=CARTESIAN_WEIGHT,
                        return_components=True,
                    )
                    val_loss = val_losses["total"].item()
            best_val = min(best_val, val_loss)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                deployment_prediction = ema.sample(
                    condition(val_obs),
                    inference_steps=8,
                    deterministic=True,
                ).float()
            deployment_action_mae = (
                deployment_prediction[:, 0] - val_action[:, 0]
            ).abs().mean().item()
            deployment_joint_delta = (
                deployment_prediction[:, 0] - val_action[:, 0]
            ) * float(data["action_scale"])
            deployment_cartesian_mm = torch.bmm(
                val_jacobian,
                deployment_joint_delta[:, :, None],
            )[:, :, 0] * 1000.0
            selection_score = torch.linalg.norm(
                deployment_cartesian_mm, dim=-1).mean().item()
            if selection_score < best_selection:
                best_selection = selection_score
                torch.save({
                    "ema_state_dict": ema.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "model_config": config,
                    "obs_mean": data["obs_mean"],
                    "obs_std": data["obs_std"],
                    "obs_horizon": obs_horizon,
                    "action_horizon": action_horizon,
                    "action_scale": float(data["action_scale"]),
                    "step": total_step,
                    "val_loss": val_loss,
                    "selection_cartesian_mean_mm": selection_score,
                    "selection_first_action_mae": deployment_action_mae,
                    "seed": SEED,
                    "observation_version": 2,
                    "bc_weight": BC_WEIGHT,
                    "cartesian_weight": CARTESIAN_WEIGHT,
                    "dart_fraction": DART_FRACTION,
                    "condition_weights": condition_weights.tolist(),
                }, OUTPUT)
            extra = ""
            if step % 500 == 0:
                sample_n = min(256, len(val_obs))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = ema.sample(
                        condition(val_obs[:sample_n]),
                        inference_steps=10,
                        deterministic=True,
                    )
                mae = (prediction.float() - val_action[:sample_n]).abs().mean().item()
                first_mae = (
                    prediction[:, 0].float() - val_action[:sample_n, 0]
                ).abs().mean().item()
                extra = f" sample_mae={mae:.4f} first={first_mae:.4f}"
            print(
                f"[phase1] step={total_step:5d} train={loss.item():.5f} "
                f"diff={losses['diffusion'].item():.5f} "
                f"bc={losses['first_bc'].item():.5f} "
                f"cart={losses['cartesian'].item():.4f} "
                f"val={val_loss:.5f} best={best_val:.5f} "
                f"deploy_cart={selection_score:.4f}mm "
                f"deploy_action={deployment_action_mae:.5f} "
                f"best_cart={best_selection:.4f}mm "
                f"lr={scheduler.get_last_lr()[0]:.2e}{extra}",
                flush=True,
            )
    print(
        f"[phase1] DONE output={OUTPUT} best_val={best_val:.5f} "
        f"best_cart={best_selection:.4f}mm "
        f"elapsed={time.time()-start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
