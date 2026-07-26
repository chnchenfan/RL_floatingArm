# Phase 1: robust MPC imitation

Phase 1 trains a conditional diffusion action-chunk policy from the clean and
DART-perturbed MPC rollouts produced in Phase 0.

## Artifacts

- Legacy 36-D dataset: `data/phase1_diffusion_dataset.npz`
- Precision 47-D dataset: `data/phase1_diffusion_dataset_v2.npz`
- Selected precision/robust checkpoint:
  `logs/phase1_diffusion/policy_precision.pt`
- Selected deployment configuration:
  `config/phase1_selected_policy.json`
- Legacy maximum-tail-robust checkpoint: `logs/phase1_diffusion/policy.pt`
- Rejected experimental refinements:
  `policy_precision_refined.pt` and `policy_precision_balanced.pt`
- Final 4096-env log:
  `logs/phase1_diffusion/eval/eval_4096_precision_calibrated_q002.log`

The selected model has two observations of 47 values, predicts eight 7-DOF
actions, and uses a four-layer conditional diffusion Transformer. The eleven
additional observable values are path phase, base linear velocity and
acceleration, and the base-motion condition `(R, z_amplitude, frequency)`.
Training adds direct first-action supervision and a position-Jacobian Cartesian
loss to the diffusion objective.

Deployment
uses deterministic DDIM with eight denoising steps and replans every control
tick. Deterministic inference is intentional because the MPC teacher is a
single-valued controller; it removes action jitter caused by repeatedly drawing
fresh diffusion noise.

## Reproduce

Build the dataset with the system Python, which provides Pinocchio:

```bash
python3 scripts/prepare_phase1_dataset.py
```

Train with the Isaac/Torch environment:

```bash
source env_isaaclab/bin/activate
PHASE1_STEPS=5000 PHASE1_BATCH=1024 \
  python3 scripts/train_phase1_diffusion.py
```

Run the robust 4096-env validation:

```bash
source env_isaaclab/bin/activate
PHASE1_EVAL_ENVS=4096 \
PHASE1_EVAL_STEPS=400 \
PHASE1_Q_SEED_NOISE=0.002 \
PHASE1_DETERMINISTIC=1 \
PHASE1_DDIM_STEPS=8 \
PHASE1_EXEC_HORIZON=1 \
PHASE1_ACTION_GAINS=0.87,0.90,0.93 \
  python3 scripts/eval_phase1_4096.py
```

Run the lightweight policy contract tests:

```bash
source env_isaaclab/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q test_phase1_diffusion.py
```

## Final robust validation

The final test used 4096 environments, 400 control ticks, and a 0.002-rad
joint-state perturbation at hand-off.

| Base condition | Mean | p95 | p99 | Maximum | Diverged (>50 mm) |
|---|---:|---:|---:|---:|---:|
| R=.02 m, T=1 s, z=0 | 0.79 mm | 1.63 mm | 1.84 mm | 3.56 mm | 0.00% |
| R=.02 m, T=1 s, z=.02 m | 1.36 mm | 2.19 mm | 2.55 mm | 7.44 mm | 0.00% |
| R=.01 m, T=.5 s, z=.01 m | 2.03 mm | 3.33 mm | 4.28 mm | 9.09 mm | 0.00% |

Against the legacy robust checkpoint, mean error improved by about 78%, 66%,
and 28% on the three conditions. The gain schedule is an execution-layer
calibration selected with a 256-env sweep; the quoted results are from the
separate 4096-env validation.

As an Isaac control-floor check, directly following the absolute clean-MPC
`q_pub` sequence achieved 0.39/0.80/2.49 mm mean on the three conditions.
This confirms that the remaining policy gap is mostly closed-loop imitation
error rather than an FK or simulator floor.

Single-environment DDIM-8 latency on the development GPU was 6.45 ms mean and
7.29 ms p95. A 4096-env inference batch took 260.81 ms; this is a stress-test
throughput measurement, not the single-robot deployment latency.

Phase 1 deliberately disables fictitious inertial coupling. That disturbance is
reserved for Phase 2, where residual RL will learn the correction on top of
this stable imitation policy.
