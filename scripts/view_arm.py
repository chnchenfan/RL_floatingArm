#!/usr/bin/env python3
"""Open the imported arm in an Isaac Sim GUI window to eyeball the import.

    source env_isaaclab/bin/activate
    python scripts/view_arm.py            # all joints slow-sweep
    python scripts/view_arm.py --static   # hold home pose, no motion

Uses the same AppLauncher path that works headless here (lighter than the raw
`isaacsim` GUI). Gravity is OFF and joints are driven kinematically so the arm
holds/moves regardless of drive gains. If the RTX viewport crashes on this 8GB
Blackwell laptop, fall back to the `isaacsim` command (see README).
"""
import argparse
import math
import os
import re
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJ = "/home/windylab/code/isaac_arm_rl"
USD_PATH = os.path.join(PROJ, "usd", "arm.usd")

parser = argparse.ArgumentParser()
parser.add_argument("--static", action="store_true", help="hold home pose")
parser.add_argument(
    "--force-unsupported-driver",
    action="store_true",
    help="attempt GUI startup even when the installed driver is known to crash",
)
args = parser.parse_args()


def main() -> None:
    driver_version = _nvidia_driver_version()
    if (
        driver_version is not None
        and driver_version.startswith("595.")
        and not args.force_unsupported_driver
    ):
        print(
            "[view] refusing known-incompatible GUI combination:\n"
            f"       Isaac Sim 5.1.0 + NVIDIA {driver_version} + RTX viewport\n"
            "       Driver 595.x is known to crash in librtx.scenedb on "
            "Blackwell GPUs.\n"
            "       Headless Isaac and RViz remain usable. For native GUI, "
            "install a validated 580-open driver and reboot.\n"
            "       Use --force-unsupported-driver only to reproduce the crash.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=False, enable_cameras=False)
    simulation_app = app_launcher.app

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext, SimulationCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg

    sim = SimulationContext(
        SimulationCfg(dt=1.0 / 120.0, device="cpu", gravity=(0.0, 0.0, 0.0))
    )

    # ground + dome light so the arm is visible
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2500.0)
    )

    arm = Articulation(ArticulationCfg(
        prim_path="/World/arm",
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),
        actuators={"all": ImplicitActuatorCfg(
            joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    ))
    sim.reset()
    sim.set_camera_view(eye=(1.1, 1.1, 0.8), target=(0.3, 0.0, 0.3))

    n = len(arm.data.joint_names)
    print(f"[view] joints={arm.data.joint_names}")
    print(f"[view] bodies={arm.data.body_names}")
    print("[view] window is up — close it or Ctrl-C to quit.")

    t = 0.0
    dq0 = torch.zeros((1, n), dtype=torch.float32)
    while simulation_app.is_running():
        if not args.static:
            q = torch.zeros((1, n), dtype=torch.float32)
            for j in range(n):
                # small, phase-shifted sweep so each joint is distinguishable
                q[0, j] = 0.5 * math.sin(0.6 * t + j * math.pi / 4.0)
            arm.write_joint_state_to_sim(q, dq0)
            arm.write_data_to_sim()
        sim.step()
        arm.update(1.0 / 120.0)
        t += 1.0 / 120.0

    simulation_app.close()


def _nvidia_driver_version():
    path = "/proc/driver/nvidia/version"
    try:
        with open(path, encoding="utf-8") as stream:
            contents = stream.read()
    except OSError:
        return None
    match = re.search(
        r"NVRM version:.*?([0-9]{3}(?:\.[0-9]+)+)",
        contents,
    )
    return match.group(1) if match else None


if __name__ == "__main__":
    main()
