#!/usr/bin/env python3
"""Minimal Isaac smoke test: N arm envs, GPU PhysX, random joint targets, step.
De-risks headless GPU multi-env training on this Blackwell laptop (RTX renderer
crashes here; PhysX GPU must still work). No RL, no reward -- just physics.

    source env_isaaclab/bin/activate
    python scripts/smoke_env.py
"""
import os
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import os as _os
USD = "/home/windylab/code/isaac_arm_rl/usd/arm.usd"
N_ENVS = int(_os.environ.get("SMOKE_N_ENVS", "16"))
STEPS = int(_os.environ.get("SMOKE_STEPS", "60"))


def main():
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    simulation_app = app_launcher.app

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext, SimulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils import configclass

    arm_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Arm",
        spawn=sim_utils.UsdFileCfg(usd_path=USD),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={f"joint{i}": 0.0 for i in range(1, 8)}),
        actuators={"all": ImplicitActuatorCfg(
            joint_names_expr=[".*"], stiffness=80.0, damping=8.0)},
    )

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        arm = arm_cfg   # fixed-base arm: no ground/light needed for a physics test

    import time as _t
    def stamp(msg, t0=[_t.time()]):
        print(f"[stage] +{_t.time()-t0[0]:6.1f}s {msg}", flush=True)

    device = _os.environ.get("SMOKE_DEVICE", "cuda:0")
    stamp(f"begin (device={device})")
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=device))
    stamp("SimulationContext created")
    scene = InteractiveScene(SceneCfg(num_envs=N_ENVS, env_spacing=2.0))
    stamp("InteractiveScene created")
    sim.reset()
    stamp("sim.reset() done")
    arm = scene["arm"]
    print(f"[smoke] device=cuda:0 envs={N_ENVS} dof={arm.num_joints} "
          f"bodies={arm.num_bodies}")

    q0 = arm.data.default_joint_pos.clone()
    torch.manual_seed(0)
    stamp("entering step loop")
    for k in range(STEPS):
        # random small joint position targets around home
        tgt = q0 + 0.3 * torch.randn_like(q0)
        arm.set_joint_position_target(tgt)
        scene.write_data_to_sim()
        sim.step()
        scene.update(1.0 / 120.0)
        if k % 10 == 0:
            stamp(f"step {k}")

    jp = arm.data.joint_pos
    ok = torch.isfinite(jp).all().item()
    stamp(f"DONE ok={ok} final joint_pos mean={jp.mean().item():+.3f}")
    import sys as _sys; _sys.stdout.flush()
    os._exit(0 if ok else 1)  # skip hanging Kit shutdown


if __name__ == "__main__":
    main()
