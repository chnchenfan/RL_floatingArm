#!/usr/bin/env python3
"""Phase 1, step D: compare Isaac FK vs Pinocchio FK reference.

Pure numpy, runs anywhere:
    python3 scripts/compare_fk.py

Pass criterion (kinematic import correctness):
    max position error < 1.0 mm  AND  max orientation error < 0.5 deg
"""
import json
import os

import numpy as np

PROJ = "/home/windylab/code/isaac_arm_rl"
REF = os.path.join(PROJ, "assets", "fk_reference_pinocchio.json")
ISAAC = os.path.join(PROJ, "assets", "fk_isaac.json")
POS_TOL_MM = 1.0
ANG_TOL_DEG = 0.5


def quat_angle_deg(q1, q2):
    q1 = np.asarray(q1) / np.linalg.norm(q1)
    q2 = np.asarray(q2) / np.linalg.norm(q2)
    d = abs(float(np.dot(q1, q2)))
    d = min(1.0, d)
    return float(np.degrees(2.0 * np.arccos(d)))


def main() -> None:
    ref = {p["index"]: p for p in json.load(open(REF))["poses"]}
    isa = {p["index"]: p for p in json.load(open(ISAAC))["poses"]}
    idxs = sorted(set(ref) & set(isa))

    rows = []
    pos_errs, ang_errs = [], []
    for i in idxs:
        dp = (np.asarray(isa[i]["pos"]) - np.asarray(ref[i]["pos"])) * 1000.0
        pe = float(np.linalg.norm(dp))
        ae = quat_angle_deg(isa[i]["quat_xyzw"], ref[i]["quat_xyzw"])
        pos_errs.append(pe); ang_errs.append(ae)
        rows.append((i, pe, ae))

    pos_errs = np.array(pos_errs); ang_errs = np.array(ang_errs)
    print(f"{'idx':>4} {'pos_err_mm':>12} {'ang_err_deg':>12}")
    for i, pe, ae in rows:
        flag = "" if (pe < POS_TOL_MM and ae < ANG_TOL_DEG) else "  <-- OUT"
        print(f"{i:>4} {pe:>12.4f} {ae:>12.4f}{flag}")

    print("-" * 34)
    print(f"n={len(idxs)}  pos max={pos_errs.max():.4f}mm mean={pos_errs.mean():.4f}mm")
    print(f"          ang max={ang_errs.max():.4f}deg mean={ang_errs.mean():.4f}deg")
    ok = pos_errs.max() < POS_TOL_MM and ang_errs.max() < ANG_TOL_DEG
    print(f"RESULT: {'PASS' if ok else 'FAIL'} "
          f"(tol pos<{POS_TOL_MM}mm ang<{ANG_TOL_DEG}deg)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
