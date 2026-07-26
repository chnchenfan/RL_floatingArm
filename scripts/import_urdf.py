#!/usr/bin/env python3
"""Phase 1, step B: convert arm.urdf (7DOF) -> USD, headless, via Isaac Lab.

Must run inside the Isaac Lab py3.11 venv:
    source env_isaaclab/bin/activate
    python scripts/import_urdf.py

Steps:
  1. Resolve `package://dummy_description/...` mesh refs to absolute paths and
     write assets/arm_resolved.urdf (Isaac's URDF importer does not know ROS
     package roots).
  2. Launch a headless Kit app and run isaaclab.sim.converters.UrdfConverter.
  3. Emit usd/arm.usd.
"""
import os
import re

# Accept the Omniverse EULA non-interactively (headless first boot).
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJ = "/home/windylab/code/isaac_arm_rl"
SRC_URDF = "/home/windylab/code/windylab_ws/src/arm-platform/config/arm.urdf"
PKG_ROOT = "/home/windylab/code/windylab_ws/src/dummy_description"
RESOLVED_URDF = os.path.join(PROJ, "assets", "arm_resolved.urdf")
USD_DIR = os.path.join(PROJ, "usd")
USD_NAME = "arm.usd"


def resolve_package_paths() -> str:
    with open(SRC_URDF, "r") as f:
        txt = f.read()
    # package://dummy_description/<rest>  ->  <PKG_ROOT>/<rest>
    resolved = re.sub(r"package://dummy_description/", PKG_ROOT + "/", txt)
    os.makedirs(os.path.dirname(RESOLVED_URDF), exist_ok=True)
    with open(RESOLVED_URDF, "w") as f:
        f.write(resolved)
    # verify referenced meshes exist
    missing = [m for m in re.findall(r'filename="([^"]+)"', resolved)
               if not os.path.isfile(m)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} mesh(es) not found, e.g. {missing[0]}")
    print(f"[ok] resolved URDF -> {RESOLVED_URDF} (all meshes exist)")
    return RESOLVED_URDF


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--floating", action="store_true",
                    help="import with a movable (floating) base -> arm_floating.usd")
    args = ap.parse_args()
    fix_base = not args.floating
    usd_name = "arm.usd" if fix_base else "arm_floating.usd"

    urdf = resolve_package_paths()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    os.makedirs(USD_DIR, exist_ok=True)

    # Isaac Lab 2.3 requires joint_drive.gains.stiffness (no default). Gains are
    # irrelevant to FK validation (we teleport joints and read poses), but the
    # converter validates them; use plain position-control PD gains.
    JointDriveCfg = UrdfConverterCfg.JointDriveCfg
    cfg = UrdfConverterCfg(
        asset_path=urdf,
        usd_dir=USD_DIR,
        usd_file_name=USD_NAME,
        fix_base=True,               # 7DOF arm: base is fixed for FK validation
        merge_fixed_joints=False,
        force_usd_conversion=True,
        joint_drive=JointDriveCfg(
            target_type="position",
            gains=JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=10.0),
        ),
    )
    # Free-space tracking task: NO collision needed. Cooking collision from the
    # detailed STL visual meshes made PhysX load pathologically slow (minutes),
    # so keep collision OFF.
    for attr, val in (("collision_from_visuals", False),
                      ("replace_cylinders_with_capsules", False)):
        if hasattr(cfg, attr):
            try:
                setattr(cfg, attr, val)
            except Exception:
                pass

    converter = UrdfConverter(cfg)
    print(f"[ok] USD written: {converter.usd_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
