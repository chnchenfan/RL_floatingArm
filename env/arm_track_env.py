#!/usr/bin/env python3
"""DirectRLEnv: 7DOF arm (fixed base, real dynamics) tracks the world-frame
shapes while a randomized base disturbance moves the base-frame target it must
follow (v1 kinematic disturbance; inertial coupling is a v2 floating-base upgrade).

Action  : Delta-q per joint, scaled by max_joint_step (aligns with the MPC step).
Obs     : joint pos/vel, base-frame target pos + orientation(6D), current EE
          pos, EE->target error, prev action.
Reward  : exp position tracking + soft 90-deg orientation + action effort/rate.
"""
from __future__ import annotations

import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_spec.target_torch import TaskTargets  # noqa: E402

USD = "/home/windylab/code/isaac_arm_rl/usd/arm.usd"
CONTROL_HZ = 50.0


@configclass
class ArmTrackEnvCfg(DirectRLEnvCfg):
    decimation = 2                       # sim 100Hz / 2 -> 50Hz control
    episode_length_s = 31.0              # one full turn around the body
    action_space = 7
    observation_space = 36   # legacy policy observation; Phase-1 v2 uses 47
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 100.0, render_interval=2)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.5)

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Arm",
        spawn=sim_utils.UsdFileCfg(usd_path=USD),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={f"joint{i}": 0.0 for i in range(1, 8)}),
        actuators={"all": ImplicitActuatorCfg(
            # Calibrated to the logged 50-Hz hardware response. The previous
            # 2000/100 setting realized only about one third of each joint
            # command before the next control tick and created artificial lag.
            joint_names_expr=[".*"], stiffness=5000.0, damping=50.0)},
    )

    # task / reward
    # Match the deployed 50 Hz acados MPC contract: 1 rad/s * 0.02 s.
    max_joint_step: float = 0.02
    pos_sigma: float = 0.03              # exp reward length scale [m]
    ori_sigma: float = 0.30             # exp reward length scale [rad] (~17deg)
    w_pos: float = 2.0                  # sharp exp bonus near target
    w_pos_lin: float = 2.0              # dense -||err|| gradient at any distance
    w_ori: float = 1.0                  # soft 90-deg orientation exp bonus
    w_ori_lin: float = 0.5              # dense -angle gradient (fixes 111deg fail)
    w_effort: float = 0.005
    w_rate: float = 0.005
    start_phase_random: bool = True
    # inertial coupling: fictitious force -m_i*a(t) on each link (shaking base)
    inertial_coupling: bool = True
    # single-circle (config-matched moving_base_circle) for the precision task
    single_circle: bool = False
    # frozen exact whole task (five stations, one-shot windylab, 73.039 s)
    whole_task: bool = False
    # multiplicative jitter around the config-matched whole base motion
    whole_disturbance_jitter: float = 0.10
    # apply the position target one control tick late, matching the observed
    # MPC-plant latency (q_obs(k+1) = q_cmd(k-1) in the loop CSVs)
    command_delay_ticks: int = 0
    # override the actuator PD at startup (0 = keep the USD/actuator default).
    # The whole task uses 80000/4000: high damping realizes the ramp velocity
    # feedforward like the deployed smooth position controller (tau = kd/kp)
    plant_stiffness: float = 0.0
    plant_damping: float = 0.0
    ik_damping: float = 0.03           # damped-least-squares lambda for the expert
    ik_gain: float = 1.5               # step gain for the IK expert (lead the lag)
    ik_lookahead: int = 1              # aim N ticks ahead to cancel tracking lag
    w_ori_ik: float = 0.3              # orientation weight in the IK expert (soft)
    extended_phase1_obs: bool = False  # +phase/base motion/condition = 47 dims


class ArmTrackEnv(DirectRLEnv):
    cfg: ArmTrackEnvCfg

    def __init__(self, cfg: ArmTrackEnvCfg, render_mode=None, **kw):
        super().__init__(cfg, render_mode, **kw)
        self.robot = self.scene["robot"]
        self._ee_idx = self.robot.body_names.index("link7")
        self._dt = 1.0 / CONTROL_HZ
        if cfg.whole_task and cfg.single_circle:
            raise ValueError("whole_task and single_circle are exclusive")
        self.targets = TaskTargets(
            self.device,
            dt=self._dt,
            base_nominal=(
                (0.0, 0.0, 0.01) if (cfg.single_circle or cfg.whole_task)
                else (0.0, 0.0, 0.0)),
            moving_base_circle=cfg.single_circle,
            whole=cfg.whole_task,
        )

        if cfg.plant_stiffness > 0.0:
            self.robot.write_joint_stiffness_to_sim(
                torch.full((self.num_envs, 7), cfg.plant_stiffness,
                           device=self.device))
        if cfg.plant_damping > 0.0:
            self.robot.write_joint_damping_to_sim(
                torch.full((self.num_envs, 7), cfg.plant_damping,
                           device=self.device))

        # joint limits (fall back to +-pi where infinite)
        lim = self.robot.data.joint_pos_limits[0]          # (nJ, 2)
        self.q_lo = lim[:, 0].clone()
        self.q_hi = lim[:, 1].clone()

        # per-body masses for the fictitious inertial force -m_i*a(t)
        self._mass = self.robot.data.default_mass[0].to(self.device)   # (nBodies,)
        self._nbodies = self._mass.shape[0]

        n = self.num_envs
        self._dist = self._sample_dist(n)
        self._start = (torch.randint(0, self.targets.T, (n,), device=self.device)
                       if cfg.start_phase_random
                       else torch.zeros(n, dtype=torch.long, device=self.device))
        self._prev_action = torch.zeros(n, 7, device=self.device)
        self.actions = torch.zeros(n, 7, device=self.device)
        self._delayed_target = None
        self._ramp_goal = None
        self._substep = 0
        # This is the action already executed by the plant.  It appears in the
        # next observation as a_{t-1}; keeping a separate, explicit buffer avoids
        # accidentally pairing obs_t with label a_t in Phase-1 offline datasets.
        self._last_executed_action = torch.zeros(n, 7, device=self.device)

    def _sample_dist(self, n):
        if self.cfg.whole_task:
            return self.targets.sample_whole_disturbance(
                n, jitter=self.cfg.whole_disturbance_jitter)
        if self.cfg.single_circle:
            return self.targets.sample_circle_disturbance(n)
        return self.targets.sample_disturbance(n)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

    # --------------------------------------------------------------- stepping
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-1.0, 1.0)
        self._last_executed_action.copy_(self.actions)
        # Compute the position target ONCE per control tick.  Recomputing it
        # from the moving joint state inside every physics substep executed
        # ~1.25x the commanded delta per tick (measured), which breaks the
        # `q_{k+1} = q_k + a*max_joint_step` contract the MPC data follows.
        q = self.robot.data.joint_pos
        target = q + self.actions * self.cfg.max_joint_step
        target = torch.clamp(target, self.q_lo, self.q_hi)
        # final joint position command; the live RViz bridge republishes this
        # as /student/joint_command for the MPC trajectory visualizer
        self._q_command = target
        if self._ramp_goal is None:
            self._ramp_goal = q.clone()
        if self.cfg.command_delay_ticks > 0:
            if self._delayed_target is None:
                self._delayed_target = target.clone()
            goal = self._delayed_target
            self._delayed_target = target.clone()
        else:
            goal = target
        # The plant tracks a within-tick linear ramp from last tick's goal to
        # this tick's goal (the ROS smooth position controller follows the
        # velocity-feedforward ramp, not a step).  This realizes 100% of each
        # commanded delta per tick with a smooth velocity ~ delta/dt.
        self._ramp_from = self._ramp_goal
        self._ramp_goal = goal.clone()
        self._substep = 0

    def _apply_action(self):
        self._substep = min(self._substep + 1, self.cfg.decimation)
        fraction = self._substep / self.cfg.decimation
        target = self._ramp_from + fraction * (
            self._ramp_goal - self._ramp_from)
        self.robot.set_joint_position_target(target)
        # velocity feedforward along the ramp: with kd >> 0 the PhysX drive
        # realizes v = v_ff + (kp/kd)(q_t - q), the same control law as the
        # deployed smooth position controller (kp/kd = 1/tau)
        self.robot.set_joint_velocity_target(
            (self._ramp_goal - self._ramp_from) / self._dt)

        if self.cfg.inertial_coupling:
            # shaking-base inertial coupling: fictitious force -m_i * a(t) on every
            # link (world frame). a(t) is the base linear acceleration this tick.
            a = self.targets.base_accel(self._phase(), self._dist)
            forces = -(self._mass.view(1, self._nbodies, 1)
                       * a.view(self.num_envs, 1, 3))       # (E, nBodies, 3)
            torques = torch.zeros_like(forces)
            self.robot.set_external_force_and_torque(
                forces,
                torques,
                is_global=True,
            )

    # --------------------------------------------------------------- targets
    def _phase(self):
        return self.episode_length_buf + self._start

    def _target(self):
        p_b, R_b, draw = self.targets.ee_target_base(self._phase(), self._dist)
        return p_b, R_b, draw

    def _ee_pose(self):
        # fixed base at origin -> world frame == base_link frame
        p = self.robot.data.body_pos_w[:, self._ee_idx, :] \
            - self.scene.env_origins
        R = _quat_to_R(self.robot.data.body_quat_w[:, self._ee_idx, :])
        return p, R

    # ----------------------------------------------- MPC-equivalent IK expert
    def expert_action(self):
        """Vectorized damped-least-squares IK toward the base-frame target — the
        MPC-equivalent teacher for imitation. Returns action in [-1,1]^7.
        Aims `ik_lookahead` ticks ahead to cancel the actuator/tracking lag."""
        la = self.cfg.ik_lookahead
        p_tgt, R_tgt, _ = self.targets.ee_target_base(
            self._phase() + la, self._dist)
        p_ee, R_ee = self._ee_pose()
        pos_err = p_tgt - p_ee                                   # (N,3)
        Rerr = torch.bmm(R_tgt, R_ee.transpose(1, 2))            # ee->tgt rotation
        ori_err = _so3_log(Rerr) * self.cfg.w_ori_ik            # (N,3)
        err6 = torch.cat([pos_err, ori_err], dim=-1)            # (N,6)
        J = self._ee_jacobian()                                 # (N,6,7)
        lam2 = self.cfg.ik_damping ** 2
        JT = J.transpose(1, 2)
        A = torch.bmm(J, JT) + lam2 * torch.eye(6, device=self.device)[None]
        dq = torch.bmm(JT, torch.linalg.solve(A, err6[:, :, None]))[:, :, 0]
        dq = self.cfg.ik_gain * dq                              # (N,7)
        return torch.clamp(dq / self.cfg.max_joint_step, -1.0, 1.0)

    def _ee_jacobian(self):
        # PhysX jacobian for link7 (fixed base -> body index shifted by 1)
        jac = self.robot.root_physx_view.get_jacobians()        # (N, nB-?, 6, nDof)
        return jac[:, self._ee_idx - 1, :, :]

    # ------------------------------------------------------------------- obs
    def _get_observations(self):
        p_tgt, R_tgt, draw = self._target()
        p_ee, R_ee = self._ee_pose()
        parts = [
            self.robot.data.joint_pos,                     # 7
            self.robot.data.joint_vel * 0.1,               # 7
            p_tgt,                                         # 3
            R_tgt[:, :, 0], R_tgt[:, :, 1],                # 6 (orient 6D)
            p_ee,                                          # 3
            p_tgt - p_ee,                                  # 3 (pos error)
            self._last_executed_action,                     # 7: a_(t-1) at next decision
        ]
        if self.cfg.extended_phase1_obs:
            phase = 2.0 * math.pi * (
                (self._phase() % self.targets.T).float() / self.targets.T)
            base_vel = self.targets.base_velocity(self._phase(), self._dist)
            base_accel = self.targets.base_accel(self._phase(), self._dist)
            condition = torch.stack([
                self._dist["amp_t"],
                self._dist["amp_z"],
                self._dist["freq_t"],
            ], dim=-1)
            parts.extend([
                torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1),  # 2
                base_vel,                                                   # 3
                base_accel,                                                 # 3
                condition,                                                  # 3
            ])
        obs = torch.cat(parts, dim=-1)
        # store for reward
        self._p_tgt, self._R_tgt, self._draw = p_tgt, R_tgt, draw
        self._p_ee, self._R_ee = p_ee, R_ee
        return {"policy": obs}

    # ---------------------------------------------------------------- reward
    def _get_rewards(self):
        pos_err = torch.linalg.norm(self._p_tgt - self._p_ee, dim=-1)
        r_pos = torch.exp(-(pos_err / self.cfg.pos_sigma) ** 2)
        # orientation error angle
        Rerr = torch.bmm(self._R_ee.transpose(1, 2), self._R_tgt)
        cos = ((Rerr[:, 0, 0] + Rerr[:, 1, 1] + Rerr[:, 2, 2]) - 1) * 0.5
        ang = torch.arccos(torch.clamp(cos, -1.0, 1.0))
        r_ori = torch.exp(-(ang / self.cfg.ori_sigma) ** 2)
        effort = torch.sum(self.actions ** 2, dim=-1)
        rate = torch.sum((self.actions - self._prev_action) ** 2, dim=-1)
        self._prev_action = self.actions.clone()
        rew = (self.cfg.w_pos * r_pos
               - self.cfg.w_pos_lin * pos_err          # dense gradient to target
               + self.cfg.w_ori * r_ori
               - self.cfg.w_ori_lin * ang              # dense gradient on 90deg lock
               - self.cfg.w_effort * effort - self.cfg.w_rate * rate)
        return rew

    # ----------------------------------------------------------------- dones
    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out)
        return died, time_out

    # ----------------------------------------------------------------- reset
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        # reset joints to home + small noise
        q = self.robot.data.default_joint_pos[env_ids] \
            + 0.05 * torch.randn(n, 7, device=self.device)
        qd = torch.zeros(n, 7, device=self.device)
        self.robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
        # resample disturbance + phase for these envs
        d = self._sample_dist(n)
        for k, v in d.items():
            self._dist[k][env_ids] = v
        if self.cfg.start_phase_random:
            self._start[env_ids] = torch.randint(
                0, self.targets.T, (n,), device=self.device)
        self._prev_action[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self._last_executed_action[env_ids] = 0.0
        self.sync_command_state(env_ids)

    def sync_command_state(self, env_ids=None):
        """Re-anchor the delayed/ramp command buffers to the current joint
        state.  Must be called after writing joint states manually (resets,
        eval seeding), or the first tick would ramp from a stale target."""
        q = self.robot.data.joint_pos
        if env_ids is None:
            if self._delayed_target is not None:
                self._delayed_target.copy_(q)
            if self._ramp_goal is not None:
                self._ramp_goal.copy_(q)
            return
        if self._delayed_target is not None:
            self._delayed_target[env_ids] = q[env_ids]
        if self._ramp_goal is not None:
            self._ramp_goal[env_ids] = q[env_ids]


def _so3_log(R):
    """Rotation matrix (N,3,3) -> axis-angle vector (N,3)."""
    cos = torch.clamp(((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) - 1) * 0.5, -1.0, 1.0)
    ang = torch.arccos(cos)                                     # (N,)
    v = torch.stack([R[:, 2, 1] - R[:, 1, 2],
                     R[:, 0, 2] - R[:, 2, 0],
                     R[:, 1, 0] - R[:, 0, 1]], dim=-1)          # 2*sin*axis
    s = torch.sin(ang)
    scale = torch.where(s.abs() > 1e-6, ang / (2 * s), torch.ones_like(ang) * 0.5)
    return v * scale[:, None]


def _quat_to_R(q):
    """quat (w,x,y,z) (N,4) -> R (N,3,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = torch.empty(N, 3, 3, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R
