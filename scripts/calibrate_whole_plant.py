from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=False).app
import sys, csv, torch, numpy as np
sys.path.insert(0, "/home/windylab/code/isaac_arm_rl")
from env.arm_track_env import ArmTrackEnv, ArmTrackEnvCfg

CSV = ('/home/windylab/code/windylab_ws/src/arm-platform/demo/data/'
      'moving_base_mpc_loop_20260726_023830.csv')
rows = list(csv.DictReader(open(CSV)))
N = 2000
qpub = np.array([[float(r[f'q_pub_{i}']) for i in range(1, 8)]
                 for r in rows[:N + 3]], dtype=np.float32)
qmeas = np.array([[float(r[f'q_meas_{i}']) for i in range(1, 8)]
                  for r in rows[:N + 3]], dtype=np.float32)
dqmeas = np.array([[float(r[f'dq_meas_{i}']) for i in range(1, 8)]
                   for r in rows[:N + 3]], dtype=np.float32)

cfg = ArmTrackEnvCfg()
cfg.scene.num_envs = 1
cfg.single_circle = False; cfg.whole_task = True; cfg.max_joint_step = 0.04
cfg.extended_phase1_obs = True; cfg.observation_space = 47
cfg.inertial_coupling = False; cfg.episode_length_s = 1e6
cfg.command_delay_ticks = 1
env = ArmTrackEnv(cfg)
env.reset()
dev = env.device


def replay(stiff, damp, delay):
    env.cfg.command_delay_ticks = delay
    env.robot.write_joint_stiffness_to_sim(
        torch.full((1, 7), float(stiff), device=dev))
    env.robot.write_joint_damping_to_sim(
        torch.full((1, 7), float(damp), device=dev))
    q0 = torch.tensor(qmeas[0:1], device=dev)
    env.robot.write_joint_state_to_sim(
        q0.clone(), torch.tensor(dqmeas[0:1], device=dev))
    env.robot.set_joint_position_target(q0.clone())
    env.sync_command_state()
    errs = []
    verrs = []
    for k in range(N):
        qn = env.robot.data.joint_pos
        a = ((torch.tensor(qpub[k:k + 1], device=dev) - qn)
             / cfg.max_joint_step).clamp(-1, 1)
        env.step(a)
        errs.append((env.robot.data.joint_pos
                     - torch.tensor(qmeas[k + 1:k + 2], device=dev)
                     ).abs().max().item())
        verrs.append((env.robot.data.joint_vel
                      - torch.tensor(dqmeas[k + 1:k + 2], device=dev)
                      ).abs().max().item())
    e = np.array(errs); v = np.array(verrs)
    print(f"[cal] stiff={int(stiff)} damp={int(damp)} delay={delay}: "
          f"q mean={e.mean():.5f} p95={np.percentile(e, 95):.5f} "
          f"max={e.max():.5f} | dq mean={v.mean():.4f}", flush=True)
    return e.mean()


best = None
for delay in (1, 0):
    for stiff, damp in ((40000, 6000), (20000, 3000), (80000, 12000),
                        (13333, 2000), (40000, 2000), (80000, 4000)):
        m = replay(stiff, damp, delay)
        if best is None or m < best[0]:
            best = (m, stiff, damp, delay)
print(f"BEST stiff={best[1]} damp={best[2]} delay={best[3]} "
      f"mean={best[0]:.5f}", flush=True)
