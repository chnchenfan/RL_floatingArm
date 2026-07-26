# RL 浮动机械臂 whole 任务工作交接

更新时间：2026-07-26  
接手目标：在现有单圆 Phase 0–3 成果基础上，实现并验证与 MPC `trajectory_shape:=whole` 完全一致的 RL 任务，同时复用现有 `trajectory_visualizer_demo.py` 做实时 RViz 对比。

## 1. 一句话状态

现有 RL 已完成并验证的是**单个水平圆轨迹**，Phase 2 在 Isaac 4096 环境评测中达到约 `0.646 mm` 平均位置误差；**exact whole 尚未完成训练、评测或实时展示**。当前首要工作不是直接继续训练，而是冻结目前 MPC 的 whole 轨迹定义、重新采集与该定义匹配的 clean/DART 专家数据，再分别训练 whole 专用的 Phase 1 和 Phase 2。

严禁把旧的 `0.646 mm` 写成 whole 成绩，也不要把旧 `76.5895 s` whole 数据和当前 `73.0395 s` one-shot whole 定义混合训练。

## 2. 仓库和 Git 状态

### RL 仓库

- 路径：`/home/windylab/code/isaac_arm_rl`
- 私有远端：`git@github.com:chnchenfan/RL_floatingArm.git`
- 分支：`main`
- 交接前提交：`f2ab4bb Initial RL floating-arm implementation`
- 已上传内容：代码、配置、选定模型和报告。
- 未上传的大目录：`env_isaaclab/` 等本地环境，已由 `.gitignore` 排除。
- 本交接文档创建后会成为本地未提交文件，除非之后另行提交和推送。

### MPC/ROS 工作区

- 路径：`/home/windylab/code/windylab_ws`
- 分支：`position-only-baseline`
- 当前 HEAD：`06873d4 发布运动基座多轨迹二阶NMPC基线`
- 工作区目前有大量用户未提交修改，涉及 MPC、轨迹配置、测试和调参文件。
- **不要执行 `git reset --hard`、`git checkout --`、批量覆盖或把无关修改一起提交。**
- MPC 专项补充交接：`/home/windylab/code/windylab_ws/MPC_TRACKING_HANDOFF_20260726.md`

接手后应先执行：

```bash
git -C /home/windylab/code/isaac_arm_rl status --short --branch
git -C /home/windylab/code/windylab_ws status --short --branch
git -C /home/windylab/code/windylab_ws diff -- \
  src/arm-platform/demo/moving_base_trajectory_config.py \
  src/arm-platform/demo/move_base_circle_mpc_ik_demo.py
```

## 3. 已经完成的部分：单圆 Phase 0–3

以下成果都属于旧的**单水平圆**任务，可作为代码、网络和训练流程的基线，但不能直接当作 whole 结果。

### Phase 1：鲁棒优先的 Diffusion BC

- 选定 checkpoint：`/home/windylab/code/isaac_arm_rl/logs/phase1_diffusion/policy_precision.pt`
- 配置：`/home/windylab/code/isaac_arm_rl/config/phase1_selected_policy.json`
- 观测版本：v2，维度 47。
- observation horizon：2。
- action horizon：8。
- action scale：`0.02 rad`。
- 部署推理：确定性 DDIM 8 步。
- 三种圆工况 condition gain：`0.87 / 0.90 / 0.93`。
- 原验证规模：4096 环境、400 ticks。

### Phase 2：Residual RL

- 选定配置：`/home/windylab/code/isaac_arm_rl/config/phase2_selected_policy.json`
- 选定 checkpoint：
  `/home/windylab/code/isaac_arm_rl/logs/phase2_residual/20260725_011102_robust_worldforce_orientation/model_950.pt`
- 组合方式：

```text
action = clip(phase1_action + 0.10 * residual_action)
```

- 正式训练完成标志：
  `/home/windylab/code/isaac_arm_rl/logs/phase2_residual/formal_robust_4096.status`
- 正式训练：
  - 4096 并行环境。
  - 1000 iterations，最终记录到 iteration 999。
  - 总步数 `98,304,000`。
  - 吞吐约 `19,984 steps/s`。
  - 最后训练位置误差约 `0.7079 mm`。

4096 环境、400 ticks 的旧单圆评测：

| 策略 | 位置 mean | p95 | p99 | 发散 |
|---|---:|---:|---:|---:|
| Phase 1 Diffusion | 1.407225 mm | 2.935521 mm | 3.641907 mm | 0 |
| Phase 2 Residual | 0.645898 mm | 1.388681 mm | 1.686385 mm | 0 |

Phase 2 姿态误差：

- mean：`3.405411 deg`
- p95：`6.157974 deg`

### Phase 3：部署和报告

- 部署包：`/home/windylab/code/isaac_arm_rl/exports/phase2_robust/`
- 其中包含 Phase 1 checkpoint、Phase 2 checkpoint、TorchScript、ONNX、manifest 和 SHA256。
- 报告：`/home/windylab/code/isaac_arm_rl/reports/phase3/PHASE3_REPORT.md`

旧单圆延迟：

| 部分 | mean | p95 | max |
|---|---:|---:|---:|
| Phase 1 DDIM-8 | 5.776 ms | 6.448 ms | 6.844 ms |
| Residual + compose | 0.299 ms | 0.905 ms | 1.299 ms |
| 完整策略 | 6.090 ms | 6.850 ms | 7.106 ms |

这些数值只能用于单圆基线。whole 必须重新测量精度和延迟。

## 4. 当前 exact whole 的权威定义

权威源码：

```text
/home/windylab/code/windylab_ws/src/arm-platform/demo/moving_base_trajectory_config.py
```

当前文件包含尚未提交的新 one-shot windylab/whole 行为。交接检查时其修改时间约为 `2026-07-25 20:00:54`；接手者应以**当前磁盘内容**重新计算轨迹周期，并在采集数据前保存源码哈希。

另一个正在频繁变化的文件：

```text
/home/windylab/code/windylab_ws/src/arm-platform/demo/move_base_circle_mpc_ik_demo.py
```

交接检查时其修改时间约为 `2026-07-26 01:00`。不要只依赖 Git HEAD 中的旧版本。

### 当前任务顺序

| 顺序 | 图形/工位 | 绘制时长 |
|---:|---|---:|
| 1 | front circle | 4.0 s |
| 2 | left triangle | 6.0 s |
| 3 | back one-shot `windylab` | 29.0394904589 s |
| 4 | right sine | 8.0 s |
| 5 | top square | 8.0 s |

每两个工位之间的 transition：

- lift：`0.6 s`
- transfer：`2.4 s`
- lower：`0.6 s`
- 合计：`3.6 s`
- 一整个循环共有 5 个 transition。

当前 whole 总周期：

```text
73.03949045888578 s
```

50 Hz 时约为：

```text
3652 control ticks
```

评测至少覆盖一个完整周期，建议用不低于 3655 ticks，并明确 warm-up、reset 和周期起点。旧的 400 ticks 只覆盖 8 秒，完全不足以验证 whole。

### 几何与姿态

- 侧面工位半径：`0.30 m`
- 侧面工位高度：`0.12 m`
- 顶面工位 forward：`0.22 m`
- 顶面工位高度：`0.30 m`
- 工位方位：
  - front：`0`
  - left：`π/2`
  - back：`π`
  - right：`3π/2`
  - top：`2π`
- 每个工位有自己的目标旋转和 nominal yaw。
- transfer 中位置与旋转都会插值。
- 当前 smoothstep 是五次 C2：

```text
x^3 * (10 + x * (-15 + 6*x))
```

- `pen_down` 和 `stroke_id` 是任务语义的一部分，不能只生成一条连续折线。
- `windylab` 在 whole 中是 one-shot：最后一个 `b` 结束后不会为了重复周期回到开头的 `w`。
- standalone repeating windylab 的周期仍约为 `32.58951836053247 s`，不要把它当成 whole 内 windylab 时长。

权威源码使用 Pinocchio 的 `rpyToMatrix`、`log3`、`exp3`。RL 的 `env_isaaclab` 环境此前不能稳定直接 import Pinocchio，因此建议在 RL 仓库中实现一个纯 NumPy/SciPy 的版本或生成冻结表，并通过回归测试与权威源码逐点比较。

## 5. 当前已有 whole 数据及兼容性

### 旧 clean whole 专家日志

参数：

```text
/home/windylab/code/windylab_ws/src/arm-platform/demo/data/moving_base_mpc_params_20260725_063105.json
```

循环数据：

```text
/home/windylab/code/windylab_ws/src/arm-platform/demo/data/moving_base_mpc_loop_20260725_063105.csv
```

已知信息：

- 4683 行。
- `trajectory_shape=whole`
- `dart_noise_std=0.0`
- `experiment_valid=true`
- 参数快照中的 whole period 是 `76.58951836053248 s`。
- 这属于旧的 repeating-windylab-in-whole 定义，与当前 `73.0394904589 s` one-shot whole 不一致。
- `path_target_time` 约从 `0.02` 到 `93.663 s`。
- `t_solve` 约从 `0` 到 `93.643 s`。
- 旧日志的 `path_time` 列一直是 `0.0`，不能把它当成真实轨迹相位。
- 若只做旧日志诊断，应从 `path_target_time`、lead 和 dt 重建相位；不要静默使用 `path_time`。
- 旧日志内部 `first_error_m`：
  - mean：约 `4.332 mm`
  - p95：约 `8.813 mm`
  - p99：约 `15.567 mm`
  - max：约 `41.090 mm`

结论：该数据可以用于理解字段、调试 parser 或复现旧任务，但不能直接作为当前 exact whole 的训练数据。

### 旧 visualizer/MPC 输出

```text
/home/windylab/code/windylab_ws/src/arm-platform/demo/data/moving_base_circle_whole_trajectory_20260725_080239.csv
```

- 3343 samples，可能只覆盖旧任务的一部分。
- 根据实际点与目标点重建的旧参考误差：

| 范围 | mean | RMSE | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| 全部样本 | 6.402 mm | 7.909 mm | 15.426 mm | 26.404 mm | 34.388 mm |
| pen-down | 5.009 mm | 5.639 mm | 9.132 mm | 13.181 mm | 20.081 mm |

这只是旧版本参考，不是当前 whole 的 MPC 正式基线，更不是 RL 结果。

### 数据决策

优先方案：

1. 冻结当前 `73.0394904589 s` exact whole 的 schema/version/hash。
2. 用当前 MPC 重新采集 clean 和 DART 数据。
3. 每份数据同时记录轨迹版本、源码 hash、period、dt、DART sigma 和随机种子。

不推荐的备选：

- 若必须使用旧数据，应显式复现并命名旧 `76.5895183605 s` 任务，所有模型、报告和可视化都标成 old whole v1。
- 不能把旧数据裁剪或重标时间后冒充当前 whole。

## 6. RL 代码中的主要缺口

### 环境仍固定为单圆

```text
/home/windylab/code/isaac_arm_rl/env/residual_arm_track_env.py
```

目前配置中有：

```python
single_circle = True
```

基础环境：

```text
/home/windylab/code/isaac_arm_rl/env/arm_track_env.py
```

目标构造仍由 `single_circle` / `moving_base_circle` 路径控制。

### target 实现不是当前 exact whole

```text
/home/windylab/code/isaac_arm_rl/task_spec/target_torch.py
```

- `moving_base_circle=True` 时只精确匹配旧单圆。
- generic schedule 是项目早期的近似四图形任务，不是当前 MPC exact whole。

```text
/home/windylab/code/isaac_arm_rl/task_spec/shapes.py
/home/windylab/code/isaac_arm_rl/task_spec/task_model.py
```

这些文件定义的是旧的 circle/sine/windylab/triangle 四侧面方案：

- 缺少当前第五个 top square。
- 任务顺序不一致。
- one-shot windylab 不一致。
- transition、旋转、pen/stroke 语义也不完全一致。

不要仅把 `single_circle=False` 就当作完成 exact whole。

### Phase 1 数据预处理硬编码圆

```text
/home/windylab/code/isaac_arm_rl/scripts/prepare_phase1_dataset.py
```

当前使用圆心、半径、周期生成水平圆 target。需要改为：

- 从 exact whole schema 生成位置、旋转、pen 状态、stroke id、phase 和 base 状态。
- 处理当前日志的目标时间字段。
- window 不能跨 reset、轨迹版本变化或不连续边界。
- train/validation 要按完整 rollout 分组，不能把同一条长轨迹的相邻窗口随机分到两边，否则会产生观测/时间泄露。

### 评测和实时脚本硬编码单圆

需要检查和扩展：

```text
/home/windylab/code/isaac_arm_rl/scripts/eval_phase1_4096.py
/home/windylab/code/isaac_arm_rl/scripts/export_phase1_rviz_rollouts.py
/home/windylab/code/isaac_arm_rl/scripts/live_phase1_stream.py
/home/windylab/code/isaac_arm_rl/scripts/live_phase2_stream.py
```

共同问题：

- 默认 `--cycle-steps=400`，whole 不够。
- 缺少 `--trajectory-shape whole` 或对应 schema 选择。
- 输出包缺少完整的 `pen_down`、`stroke_id`、whole phase 和 command q。
- 旧 live 脚本只证明单圆可实时播放。

### ROS bridge 还不能完全复用现有 visualizer

当前 bridge 主要发布：

- `/joint_states`
- 简单的 `/rollout_ee_actual`
- 简单的 `/rollout_ee_target`

它还没有完整发布 MPC visualizer 所需的 `/student/joint_command`，而自己的 line strip 也没有正确按 pen-up 和 stroke 切段。

## 7. 建议的剩余实施顺序

### A. 冻结和测试 exact whole 规格

建议新增：

```text
task_spec/whole_trajectory.py
tests/test_whole_trajectory_parity.py
config/whole_task_schema.json
```

要求：

1. 在 RL 仓库实现纯 NumPy/SciPy 的 exact whole：
   - position
   - rotation
   - nominal yaw
   - pen_down
   - stroke_id
   - task/station id
   - total period
2. 写入 `WHOLE_SCHEMA_VERSION`。
3. 写入权威源路径和采集时源码 SHA256。
4. 在许多普通时间点和所有边界两侧，与当前
   `moving_base_trajectory_config.py` 输出逐点对比。
5. 检查：
   - 周期连续性
   - transfer 首尾
   - top square 姿态
   - windylab 最后一笔不回到开头
   - pen/stroke 分段

如果权威源继续变化，先更新 schema/version 和测试，再采数据；不要让任务定义在训练中途漂移。

### B. 重新采集当前 whole 的 clean + DART 数据

至少采集：

- clean：`sigma=0`
- DART：`sigma=0.001`
- DART：`sigma=0.002`

更稳健的方案：

- 每个 sigma 多个随机种子。
- 覆盖项目已有的三种 base/圆工况或明确新的 whole 工况分布。
- 每次至少完整覆盖一个当前 73.04 秒周期。
- 运行时间建议大于 90 秒，以包含初始化、warm-up、完整循环和结尾缓冲。
- 记录 reset、控制 tick、目标时间、轨迹版本和 hash。

数据质检：

- clean 与 DART 分开统计。
- 确认保存的是**被扰状态下 MPC 的纠偏动作**。
- 检查动作到底是位置增量、速度还是绝对目标，统一单位。
- 检查 DART 扰动没有直接污染 supervision label。
- 逐任务画出目标/实际/误差和控制量。
- 确认所有任务和所有笔画都出现。

### C. 训练 whole 专用 Phase 1 Diffusion BC

- 保留“鲁棒优先”，继续使用 Diffusion Policy。
- 不要覆盖现有单圆 checkpoint 和配置。
- 建议命名：

```text
logs/phase1_whole_diffusion/
config/phase1_whole_selected_policy.json
```

训练数据应包含 exact whole 的目标位置和姿态上下文。需要特别防止：

- 同一 rollout 相邻窗口跨 train/validation 的时间泄露。
- window 跨 reset。
- window 跨不可解释的 logging gap。
- 用未来实际状态作为当前观测。
- 绝对仿真时间成为轨迹答案的捷径；如果需要 phase，显式提供受控、部署时同样可获得的周期 phase。

评测：

- 4096 环境可以继续使用。
- 随机化 whole 起始 phase，不能所有环境都从 circle 的 t=0 开始。
- 至少做一次完整 73.04 秒闭环 rollout。
- 同时按 overall、pen-down、task、stroke、transition 报告。

### D. 训练 whole 专用 Phase 2 Residual RL

新增或扩展环境配置：

```text
trajectory_shape = "whole"
```

不要继续依赖：

```python
single_circle = True
```

建议：

- 从 whole Phase 1 策略开始，Residual 仍接近零初始化。
- 可把旧圆 Phase 2 当作初始化实验，但不能直接作为 whole 已完成模型。
- 初始状态/phase 要覆盖整个周期，尤其是 station handoff 和 top square。
- 若单 episode 无法承受 73 秒，可使用随机 phase 的截断窗口训练，但最终必须做完整连续周期验证。
- handoff 状态可来自当前 MPC 日志或离线 IK state bank。
- 新日志与模型建议命名：

```text
logs/phase2_whole_residual/
config/phase2_whole_selected_policy.json
exports/phase2_whole_robust/
```

- 不覆盖现有 circle 模型。

### E. 实时 RViz 和现有 visualizer 对接

用户要求视觉效果与：

```text
trajectory_visualizer_demo.py
```

一致。最可靠的方式是**复用它，不另写一套近似 visualizer**。

RL bridge 应以约 50 Hz 发布：

1. `/joint_states`
   - 实际 7 关节位置/速度。
2. `/student/joint_command`
   - RL 最终关节 command position/velocity。
   - `JointState.header.frame_id = "moving_base_track"`。
   - 名称为 `joint1` 到 `joint7`。
3. TF：
   - `world -> base_link`。

环境的 `_apply_action` 应保存最终 joint position target，并把 `q_command` 加入 live packet。不要只传抽象 action。

时间语义：

- 使用真实发布时间戳。
- whole 的 `t=0` 对齐 visualizer 接收到的第一条有效 `moving_base_track` command。
- 处理暂停、超时、reset 和重新开始。
- visualizer 的 stale command timeout 默认约 0.2 秒，bridge 不能长时间停更。

现有 visualizer 主要话题：

- `/trajectory/expected`
- `/trajectory/actual`
- `/trajectory/command_fk`
- `/trajectory/current_target`
- `/trajectory/current_actual`

其默认输入：

- actual：`/joint_states`
- command：`/student/joint_command`
- `start_on_command=true`
- `use_message_stamp=true`

如果保留 RL bridge 自己的 Marker，必须按 `pen_down` 和 `stroke_id` 分段，不能在抬笔时连接不同字母或不同工位。也可以隐藏 bridge 的轨迹 Marker，只保留现有 visualizer 的话题。

### F. 最终评测和发布

至少报告：

- position：
  - mean
  - RMSE
  - p95
  - p99
  - max
- orientation：
  - mean
  - p95
  - max
- divergence：
  - 建议阈值 `>50 mm`
- action saturation ratio
- inference latency：
  - mean
  - p95
  - p99
  - max

所有位置指标都分别给出：

- overall
- pen-down only
- 每个 task/station
- 每个 stroke
- transition

对比规则：

- Phase 1 和 Phase 2 使用相同 seeds、初态和 phase，做 paired comparison。
- MPC 与 RL 必须使用相同 exact whole 版本、相同周期、相同目标定义和相同误差重建方式。
- 不满足这些条件前，不宣称“RL 超过 MPC”。
- 延迟评测应与 Isaac GUI/GPU 渲染竞争分开测量。

视觉验收：

- 五个任务都完整出现。
- 14 个预期绘图片段/笔画均可见。
- 抬笔时没有跨工位或跨字母连线。
- top square 位于正确工位并有正确姿态。
- 运行一个完整周期后轨迹闭合/重启语义正确。

## 8. MPC 与 visualizer 的当前运行命令

机械臂：

```bash
source /opt/ros/humble/setup.bash
source ~/code/windylab_ws/install/setup.bash
ros2 launch manipulator student_arm.launch.py velocity_feedforward_gain:=1.0
```

MPC whole：

```bash
cd ~/code/windylab_ws/src/arm-platform/demo
source /opt/ros/humble/setup.bash
source ~/code/windylab_ws/install/setup.bash
source ~/code/windylab_ws/src/arm-platform/scripts/setup_acados_env.sh
python3 move_base_circle_mpc_ik_demo.py --ros-args \
  -p trajectory_shape:=whole
```

现有 visualizer：

```bash
source /opt/ros/humble/setup.bash
source ~/code/windylab_ws/install/setup.bash
ros2 run manipulator trajectory_visualizer_demo.py --ros-args \
  -p trajectory_mode:=moving_base_circle \
  -p trajectory_shape:=whole \
  -p frame_id:=world \
  -p record_control_signal:=true \
  -p ignore_initial_sec:=3.0
```

注意：在带 `set -u` 的 wrapper 中直接 source ROS Humble 曾触发：

```text
AMENT_TRACE_SETUP_FILES: unbound variable
```

wrapper 应在 source ROS/工作区环境时临时 `set +u`，随后再恢复需要的 shell 选项。

## 9. Isaac/硬件运行注意事项

- 本机 GPU：NVIDIA RTX 5060 Laptop，约 8 GiB。
- 旧 Phase 2 的 4096 环境训练约占用 `3.8 GiB`，GPU 接近 100%。
- 旧正式 Phase 2 的 1000 iterations 用时约 82 分钟。
- 交接时没有训练或 live Isaac 进程在运行，GPU 应为空闲状态。
- 4096 可作为 whole 的起始规模，但 whole 增加的 horizon、缓存和 target 状态会改变显存占用，必须先用小步 smoke test 测显存再决定是否增加环境数。
- 不要只因为旧单圆还能剩余显存就盲目增加 whole 环境数；吞吐、OOM 余量和 reset 成本都要实际测。

Isaac Sim GUI 在当前 RTX 5060 + 595.x 驱动组合上曾在 `librtx.scenedb` 附近崩溃或黑屏，但 headless 训练正常。推荐：

```text
AppLauncher(headless=True, enable_cameras=False)
```

训练过程实时观察优先使用 RViz bridge。RViz 能显示被抽样/镜像出来的一个或少量环境，不等于同时渲染全部 4096 环境。

## 10. 接手者建议先做的只读检查

```bash
cd /home/windylab/code/isaac_arm_rl
git status --short --branch
git log -1 --oneline
git remote -v

sed -n '1,240p' config/phase1_selected_policy.json
sed -n '1,240p' config/phase2_selected_policy.json
sed -n '1,260p' reports/phase3/PHASE3_REPORT.md

rg -n "single_circle|cycle.steps|cycle_steps|moving_base_circle" \
  env task_spec scripts ros_bridge

cd /home/windylab/code/windylab_ws
git status --short --branch
rg -n "_whole_task_draw_period|WINDYLAB_ONCE|whole|stroke_id|pen_down" \
  src/arm-platform/demo/moving_base_trajectory_config.py
```

然后：

1. 从当前权威源码重新打印 whole period 和各阶段边界。
2. 计算并保存 `moving_base_trajectory_config.py` 的 SHA256。
3. 在写训练代码前先完成 exact whole parity test。
4. 在大规模采集前先跑一个短 clean smoke test，并核对 ROS 日志字段。
5. 在正式训练前确认新数据的轨迹版本都是当前版本。

## 11. 文件命名和不可覆盖项

建议所有 whole 产物都显式含 `whole` 和 schema version，例如：

```text
data/phase0_whole_v2_*.npz
logs/phase1_whole_v2_diffusion/
logs/phase2_whole_v2_residual/
config/phase1_whole_v2_selected_policy.json
config/phase2_whole_v2_selected_policy.json
exports/phase2_whole_v2_robust/
reports/phase3_whole_v2/
```

不要覆盖：

- `logs/phase1_diffusion/policy_precision.pt`
- `config/phase1_selected_policy.json`
- `config/phase2_selected_policy.json`
- `logs/phase2_residual/20260725_011102_robust_worldforce_orientation/`
- `exports/phase2_robust/`
- 原始 MPC CSV/JSON 日志
- `windylab_ws` 中用户未提交的 MPC 修改

## 12. 完成交付定义

只有满足以下条件，才能把 whole 标为完成：

- exact whole 规格已冻结，带版本和源码 hash。
- parity test 覆盖普通采样点和所有 phase 边界。
- 当前版本 clean/DART 数据完成且通过质量检查。
- whole Phase 1 完整周期不发散。
- whole Phase 2 在相同 seeds/初态下稳定优于 Phase 1，或如实报告未优于。
- 4096 环境统计覆盖完整周期和所有任务。
- RViz 使用现有 `trajectory_visualizer_demo.py` 实时显示目标、实际和 command FK。
- `/student/joint_command` 的 frame、时间戳、关节顺序和频率符合 visualizer 约定。
- pen-up/stroke 没有错误连线。
- 精度、姿态、发散、饱和和延迟都有可复现报告。
- circle 成绩、old whole 成绩和 current exact whole 成绩严格分开。
- 新模型、配置、数据清单和报告已提交到独立 commit；大文件是否上传按仓库策略明确记录。

## 13. 最重要的三个风险

1. **任务版本漂移**：当前 whole 源码是未提交修改，旧数据的周期与当前定义不同。先冻结版本，再采集和训练。
2. **“能运行”被误认为“exact whole 完成”**：当前 generic target 只是旧近似任务，简单关闭 `single_circle` 不够。
3. **评测口径错误**：400 ticks、全部点平均、同 rollout 随机切 train/val，都可能给出好看但无效的数字。必须完整周期、pen-down/per-task 指标和 rollout-level split。

