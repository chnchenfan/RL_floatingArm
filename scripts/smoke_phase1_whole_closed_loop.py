from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=False).app
import sys, torch, numpy as np
sys.path.insert(0, "/home/windylab/code/isaac_arm_rl")
from distill.diffusion_policy import make_policy_from_checkpoint
from env.arm_track_env import ArmTrackEnv, ArmTrackEnvCfg
from env.residual_arm_track_env import _load_state_bank, _draw_bank_states

E = 256
STEPS = 800
cfg = ArmTrackEnvCfg()
cfg.scene.num_envs = E
cfg.single_circle = False
cfg.whole_task = True
cfg.max_joint_step = 0.04
cfg.extended_phase1_obs = True
cfg.observation_space = 47
cfg.inertial_coupling = False
cfg.episode_length_s = 1e6
cfg.command_delay_ticks = 1
cfg.plant_stiffness = 80000.0
cfg.plant_damping = 4000.0
env = ArmTrackEnv(cfg)
env.reset()
dev = env.device
ck = torch.load('/home/windylab/code/isaac_arm_rl/logs/phase1_whole_diffusion/'
                'policy_whole_r3.pt', map_location='cpu', weights_only=False)
policy = make_policy_from_checkpoint(ck, dev)
mean = torch.as_tensor(ck['obs_mean'], device=dev)
std = torch.as_tensor(ck['obs_std'], device=dev)
bank = _load_state_bank(
    '/home/windylab/code/isaac_arm_rl/data/whole_v2_state_bank.npz', dev)

gen = torch.Generator(device=dev)
gen.manual_seed(1)
start0 = torch.randint(0, env.targets.T, (E,), device=dev, generator=gen)
q0, dq0, prev0 = _draw_bank_states(bank, start0)


def run(gain):
    env.reset()
    ex = env.targets.sample_whole_disturbance(E, jitter=0.0)
    for k, v in ex.items():
        env._dist[k].copy_(v)
    env._start.copy_(start0)
    env.episode_length_buf.zero_()
    env._last_executed_action.copy_(prev0)
    env._prev_action.copy_(prev0)
    env.robot.write_joint_state_to_sim(q0.clone(), dq0.clone())
    env.robot.set_joint_position_target(q0.clone())
    env.sync_command_state()
    obs = env._get_observations()["policy"]
    hist = torch.stack([obs, obs], dim=1)
    g = torch.Generator(device=dev)
    g.manual_seed(2)
    errs = []
    for s in range(STEPS):
        cond = ((hist - mean) / std).clamp(-10, 10).flatten(1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            chunk = policy.sample(cond, inference_steps=8, generator=g,
                                  deterministic=True)
        a = (gain * chunk[:, 0].float()).clamp(-1, 1)
        obs = env.step(a)[0]["policy"]
        hist = torch.stack([hist[:, 1], obs], dim=1)
        errs.append(torch.linalg.norm(env._p_tgt - env._p_ee, dim=-1))
    e = torch.stack(errs[100:]).flatten() * 1000
    tail = torch.stack(errs[-100:]).flatten() * 1000
    div = (torch.stack(errs).max(dim=0).values > 0.05).float().mean()
    print(f"[sweep] gain={gain}: mean={e.mean():.2f}mm "
          f"p95={torch.quantile(e, .95):.2f} tail_mean={tail.mean():.2f} "
          f"diverged={100 * div:.1f}%", flush=True)


for gain in (1.0, 0.95, 1.05):
    run(gain)
print("DONE", flush=True)
