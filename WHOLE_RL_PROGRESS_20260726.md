# whole 任务 RL 实施记录（2026-07-26 会话）

状态：**用户要求暂停**。Phase 1 已在截断窗口闭环稳定（1.80mm / 0% 发散），
完整周期评测与 Phase 2 训练尚未执行。本文档记录本会话全部完成项、
关键发现（尤其是闭环发散的 5 层根因排查）与精确的续作步骤。

## 1. 一句话状态

exact whole（`73.03949045888578 s` one-shot 定义）的规格冻结、parity 测试、
clean+DART 数据采集、Phase 1 Diffusion BC 训练均已完成；经过对 Isaac 植物
模型的三项修复与两轮数据/训练迭代，Phase 1（r3 checkpoint）在 800-tick
随机相位闭环中达到 **mean 1.80mm / p95 3.24mm / 0% 发散**（256 env）。
**尚未做**：完整 3652-tick 周期评测、Phase 2 residual RL 训练、RViz 实时
验证（代码已就绪未运行）、最终报告。

## 2. 已完成并验证的交付物

### A. 规格冻结（whole_v2_oneshot_73.0394904589）

- `task_spec/whole_trajectory.py`：纯 NumPy exact whole（位置/旋转/pen_down/
  stroke_id/task_index/base nominal yaw），自实现 pinocchio 兼容的
  rpyToMatrix/log3/exp3。
- `tests/test_whole_trajectory_parity.py`：与权威源
  `windylab_ws/src/arm-platform/demo/moving_base_trajectory_config.py`
  （sha256 `0d0181b7…`，已冻结）逐点对比 **8761 个采样点**（含全部边界
  两侧 ±1e-7s、周期回绕、笔画普查 14 笔），**PARITY OK**。
  运行方式：`set +u; source ROS+ws; /usr/bin/python3 tests/test_whole_trajectory_parity.py`
  （注意：裸 `python3` 是 uv 的 3.11，import 不了 ROS 的 pinocchio）。
- `config/whole_task_schema.json`：schema/hash/几何/周期/MPC 基线快照。

### B. 数据（全部为当前 73.0394s 定义 + 定版 MPC 配置 max_joint_step=0.04）

MPC 侧 clean 数据（用户当日已采，经质检）：

| stamp | σ | 覆盖 | fe mean/p95 | 结论 |
|---|---|---|---|---|
| 20260726_021045 | 0 | 103.6s 全周期 | 0.184/0.327mm | ✓ train（官方基线 run）|
| 20260726_023717 | 0 | 46.9s 部分 | 0.191/0.325mm | ✓ train |
| 20260726_023830 | 0 | 115.5s 全周期 | 0.240/0.610mm | ✓ **val** |
| 022959 / 024550 | 0 | — | 24mm/25mm | ✗ FastDDS 共享内存故障（jsdelay 93ms）|
| 024254 | 0 | 75s | 0.26mm | ✗ experiment_valid=False |
| 024655 | 0 | 82s | 0.38mm, 尖峰 11mm | ✗ 反馈延迟散布 |
| 051842 | 0 | 22s | — | ✗ 杂散短 run，忽略 |

本会话新采 DART 数据（`scratchpad/collect_whole_dart.sh`，plant sim +
MPC demo 无头，每个 ≥ 完整周期，全部 valid、jsdelay < 1ms）：

| stamp | σ (rad) | seed | fe mean/p95/max (mm) | 用途 |
|---|---|---|---|---|
| 043233 | 0.001 | 11 | 0.44/0.82/— | train |
| 043435 | 0.001 | 12 | 0.45/0.93/— | **val** |
| 043637 | 0.002 | 21 | 0.45/0.84/— | train |
| 043840 | 0.002 | 22 | 0.56/1.02/— | train |
| 054851 | 0.004 | 41 | 0.98/2.04/19.6 | train（恢复包络）|
| 055053 | 0.006 | 61 | 1.75/4.10/17.1 | train（恢复包络）|
| 055255 | 0.008 | 81 | 2.55/5.95/21.5 | train（恢复包络）|

- 数据集：`data/phase1_whole_v2_dataset_r2.npz`（train 41340 / val 11541 窗口，
  **按整条 rollout 划分 train/val**，无窗口级时间泄露）。
  旧的窄包络版本 `data/phase1_whole_v2_dataset.npz` 保留。
- 相位状态库：`data/whole_v2_state_bank.npz`（clean run 的 q/dq/prev_action，
  覆盖全部 3652/3652 相位 tick，Phase 2 随机相位 reset 用）。
- 预处理脚本：`scripts/prepare_phase1_whole_dataset.py`（whole 目标由冻结
  规格生成、断言 period/max_joint_step/experiment_valid/jsdelay、action
  scale 0.04）。

### C. Phase 1 Diffusion BC

- **选定：`logs/phase1_whole_diffusion/policy_whole_r3.pt`**
  （配置 `config/phase1_whole_v2_selected_policy.json`）。
  训练：r2 数据集、6000 步、DART_FRACTION=0.5、**速度分桶平衡采样**
  （PHASE1_SPEED_BALANCE=1，桶边界 |a|=0.25/0.5，比例 0.4/0.3/0.3）、
  无 obs 噪声。离线 held-out deploy_cart **0.369mm**。
- 历史版本（保留勿删，用于对照）：
  - `policy_whole.pt`（r1：窄 DART 30%，离线 0.34mm，闭环发散）
  - `policy_whole_r2.pt`（r2：+宽 DART+z 噪声 0.05，闭环更糟——
    结论：**z-空间噪声会洗掉毫米级 err 特征的反馈增益，勿再用**）
- 训练脚本改动：`scripts/train_phase1_diffusion.py` 新增
  `PHASE1_OBS_NOISE`（默认 0）与 `PHASE1_SPEED_BALANCE`（默认 0）。

### D. Isaac 环境扩展（`env/arm_track_env.py`, `env/residual_arm_track_env.py`）

- `TaskTargets(whole=True)`：从冻结规格建表（pos/rot/yaw/pen/stroke/task），
  `sample_whole_disturbance()` 精确匹配 base 工况（r=0.01m、z=0.01m、2Hz、
  0.1°，jitter 可调）；基座抖动相位改跑连续时钟（不随周期回绕跳变）。
- `ArmTrackEnvCfg` 新增：`whole_task`、`whole_disturbance_jitter`、
  `command_delay_ticks`、`plant_stiffness/plant_damping`。
- `WholeResidualArmTrackEnvCfg`：whole 版 Phase 2 配置（12s 随机相位窗口、
  状态库 reset、单工况 gain=1.0、max_joint_step=0.04、标定植物参数）。
- `sync_command_state()`：手动写关节态后必须调用（评测/live 脚本已接）。

### E. 评测 / 训练 / 桥接脚本（已写好，部分未跑）

- `scripts/eval_phase1_whole.py`：全周期（3700 tick）闭环评测，
  overall/pen-down/per-task/transition + 姿态 + 发散 + 延迟。**未跑完整版**。
- `scripts/train_phase2_whole_residual.py`：whole Phase 2 PPO（rsl_rl 配方
  与单圆一致，log 到 `logs/phase2_whole_residual/`）。**未跑**。
- `scripts/eval_phase2_whole.py`：成对全周期评测（model_init = 同惯性
  Phase 1 基线）。**未跑**。
- `scripts/live_phase2_whole_stream.py` + `ros_bridge/live_stream_node.py`
  （升级：发布 `/student/joint_command` frame_id=moving_base_track、trail 按
  pen/stroke 分段 LINE_LIST）+ `ros_bridge/view_whole_live_rviz.sh`
  （复用原版 trajectory_visualizer_demo.py，启动顺序 plant→visualizer→Isaac）。
  **未实测**。

## 3. 闭环发散问题的完整排查记录（本会话核心）

现象：Phase 1 离线拟合极好（deploy_cart 0.34mm）但闭环 100% 发散（60-170mm）。
逐层排查结论（每条都有实验证据，勿重复排查）：

1. **obs 一致性 ✓**：同状态下数据集 obs 与 env obs 逐特征一致
   （仅 base_vel/base_accel 有 ~17% 固定相位差，源于 t_solve 与 path tick
   的 ~0.65 tick 钟差；不致命）。
2. **κ=1.25 bug（已修）**：旧 `_apply_action` 每个物理子步用当前 q 重算
   目标，一拍实际执行 1.25×命令量。这也是旧单圆需要 0.87-0.93 condition
   gain 的真正原因。修复：目标每拍算一次（`_pre_physics_step`）。
3. **植物响应模型（已修）**：ROS 新植物（命令逆变换+速度前馈）的可观测
   动力学是 `q_obs(k+2)=q_pub(k)`（两拍全量落点、平滑速度）。Isaac 改为
   **拍内线性斜坡目标 + set_joint_velocity_target 前馈**，PD=80000/4000
   （τ=kd/kp=0.05）。对 2000 tick 专家命令流 oracle 回放：q 误差 0.0042 rad、
   dq 误差 0.082 rad/s（修复前 0.014 rad / 0.60 rad/s）。
4. **恢复数据缺失（已补）**：原 DART σ=0.001/0.002 只覆盖 ±0.8mm 邻域，
   闭环滞后 3-10mm 即离分布（法证：策略在快段输出偏小、q 零空间漂移
   0.2-0.4 rad 后崩溃）。补采 σ=0.004/0.006/0.008（fe p95 达 6mm）。
   注意：**光加恢复数据反而更糟**（r2/r3 with delay=1 比 r1 差），因为
   高增益反馈×2 拍延迟=振荡，见下条。
5. **command_delay 是最后一块拼图**：`command_delay_ticks=1`（模拟数据的
   2 拍动作→观测延迟）下所有策略 100% 发散；**delay=0 时 r3 完全稳定**。
   解释：BC 无法从 2 帧历史完全学会 MPC Smith 预测器的延迟补偿，缩短
   环路延迟后其学到的高增益纠偏（来自宽 DART 数据）变为稳定反馈。
   消融数据（256 env × 800 tick 随机相位，精确工况）：

   | 策略 | delay=1 | delay=0 |
   |---|---|---|
   | r1（窄 DART）| 93mm / 88% 发散 | 54mm / 98% 发散 |
   | **r3（宽 DART+速度平衡）** | 151mm / 100% | **1.80mm / p95 3.24 / 0%** |

   → 宽 DART 恢复数据和 delay=0 **缺一不可**。

其他结论：action gain 调节（0.8-1.05）无效；EXEC_HORIZON>1 更糟（策略
需要每拍反馈）；obs z-噪声有害。

## 4. 下一步（按序，恢复执行时从 ① 开始）

① **全周期 Phase 1 评测**（~15 分钟 GPU）：
```bash
cd /home/windylab/code/isaac_arm_rl
PHASE1_POLICY=$PWD/logs/phase1_whole_diffusion/policy_whole_r3.pt \
PHASE1_EVAL_ENVS=1024 PHASE1_EVAL_STEPS=3700 \
PHASE1_EVAL_OUT=$PWD/logs/phase1_whole_diffusion/eval_fullcycle_r3.npz \
OMNI_KIT_ACCEPT_EULA=YES ./env_isaaclab/bin/python scripts/eval_phase1_whole.py
```
验收：无发散、给出 overall/pen-down/per-task/transition 分解。注意 800-tick
窗口的 1.8mm 未覆盖最长的 windylab 连续段，全周期可能暴露新问题。

② **Phase 2 residual PPO**（~80-90 分钟 GPU，先 64 env 冒烟测显存）：
```bash
OMNI_KIT_ACCEPT_EULA=YES ./env_isaaclab/bin/python \
  scripts/train_phase2_whole_residual.py --num-envs 4096 --iterations 1000 \
  --phase1-checkpoint $PWD/logs/phase1_whole_diffusion/policy_whole_r3.pt \
  --residual-scale 0.10 --tag r3base
```
Phase 1 基线 1.8mm 与单圆 Phase 1（1.4mm）相近，原 residual_scale=0.10 配方
应可用；若提升不足可试 0.25。

③ **成对全周期 Phase 2 评测**：`scripts/eval_phase2_whole.py
--run-dir logs/phase2_whole_residual/<run> --checkpoints model_init.pt,model_950.pt`。
必须同 seeds/初态/相位，报告 paired improvement。

④ **RViz 实时验证**：`bash ros_bridge/view_whole_live_rviz.sh [phase2_ckpt]`。
验收：5 任务 14 笔画齐全、抬笔不连线、top square 位姿正确、visualizer KPI
正常（先清 `/dev/shm/fastrtps_*`）。

⑤ **报告与发布**：`reports/phase3_whole_v2/`（目录已建），指标口径见交接
文档 §F；circle / old whole / current whole 严格分开；导出
`exports/phase2_whole_v2_robust/`；独立 commit。

## 5. 风险与注意

- **不要把 800-tick 1.80mm 当成最终成绩**——完整周期、4096 env 未验证。
- MPC 基线（同定义同口径）是 RMS 0.27mm/P95 0.43mm——RL 目前 1.8mm 均值，
  Phase 2 的目标是逼近毫米内；对比表述须谨慎。
- `command_delay_ticks` 已在 whole cfg/评测脚本中固定为 0（delay=1 会发散，
  代码里有注释）；若未来想更忠实地建模 2 拍延迟，需同时给策略更长历史
  或显式 action 队列特征，属于新实验。
- 环境代码改动（每拍单目标、速度前馈、可调 PD）会影响旧单圆 env 的物理
  行为；旧 circle checkpoint/配置未动，但旧评测数字不可直接复现，重评需
  注意。
- 有复用价值的临时脚本已在暂停时拷入仓库（可安全关机）：
  `scripts/collect_whole_dart.sh`（DART 数据采集 wrapper）、
  `scripts/smoke_phase1_whole_closed_loop.py`（256env×800tick 闭环冒烟，
  即验证 r3 1.80mm 的脚本）、`scripts/calibrate_whole_plant.py`（植物
  PD/延迟标定 oracle 回放）。其余一次性诊断脚本（forensic/dart_cmp/
  kappa_test）在 /tmp，关机即失，无需保留。
- 未提交的 git 变更（本仓库）：env/、task_spec/、scripts/、ros_bridge/、
  config/、tests/、data/、logs/phase1_whole_diffusion/ 及本文档；
  windylab_ws 仓库**没有任何修改**（只新增了 demo/data 下的数据文件）。
