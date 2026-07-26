#!/usr/bin/env python3
"""Train PPO (rsl_rl) on ArmTrackEnv.

    source env_isaaclab/bin/activate
    python scripts/train_ppo.py                 # full run
    TRAIN_ITERS=3 TRAIN_ENVS=64 python scripts/train_ppo.py   # quick smoke

Checkpoints + tensorboard logs -> logs/rsl_rl/arm_track/<timestamp-ish>/.
"""
import os
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ITERS = int(os.environ.get("TRAIN_ITERS", "1500"))
ENVS = int(os.environ.get("TRAIN_ENVS", "1024"))
LOG_ROOT = "/home/windylab/code/isaac_arm_rl/logs/rsl_rl/arm_track"


def main():
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False).app

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from env.arm_track_env import ArmTrackEnv, ArmTrackEnvCfg
    from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg,
                                    RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg)
    from rsl_rl.runners import OnPolicyRunner

    cfg = ArmTrackEnvCfg()
    cfg.scene.num_envs = ENVS
    env = ArmTrackEnv(cfg)
    env = RslRlVecEnvWrapper(env)

    agent = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=ITERS,
        save_interval=50,
        experiment_name="arm_track",
        empirical_normalization=False,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
            entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
            learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
            desired_kl=0.01, max_grad_norm=1.0,
        ),
    )

    os.makedirs(LOG_ROOT, exist_ok=True)
    tag = os.environ.get("TRAIN_TAG", f"envs{ENVS}")
    log_dir = os.path.join(LOG_ROOT, tag)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=log_dir,
                            device=env.unwrapped.device)
    print(f"[train] envs={ENVS} iters={ITERS} log_dir={log_dir}", flush=True)
    runner.learn(num_learning_iterations=ITERS, init_at_random_ep_len=True)
    print("[train] DONE", flush=True)

    import sys as _s; _s.stdout.flush()
    # NOTE: app.close() hangs on Kit shutdown here (spins a core at 100% -> the
    # laptop-overheating zombies). Skip it and hard-exit; the OS reclaims GPU.
    os._exit(0)


if __name__ == "__main__":
    main()
