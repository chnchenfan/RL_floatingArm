# ros_bridge — Isaac 在线策略 / rollout → RViz

把 Isaac 里跑出的策略动作(7 关节角时间序列)拉进你现有的 RViz / trajectory_visualizer
看效果,**不改动可视化器**。训练在 py3.11 headless,这里在 py3.10 + ROS Humble。

## 数据流
```
Isaac 策略 rollout ──write_rollout()──▶ data/rollout_*.csv
                                             │ read
                          replay_rollout_node.py ──▶ /joint_states (joint1..7)
                                             │
                    robot_state_publisher(arm.urdf) ──▶ TF ──▶ RViz 渲染臂
                                             │
                    trajectory_visualizer_demo.py ──▶ 期望 vs 实际末端跟踪图
```

## Phase 2 最终策略在线验收（推荐）

下面的命令现场运行 Phase 1 扩散基座策略、Phase 2 残差网络和 Isaac
物理，并把每一个最新 tick 通过本机 UDP 发给 ROS/RViz。它不是录像，也不读取
预生成动作：

```bash
cd /home/windylab/code/isaac_arm_rl
bash ros_bridge/view_phase2_live_rviz.sh medium
```

把 `medium` 换成 `easy` 或 `hard`。红线是实际末端，绿线是目标；关闭 RViz
会自动结束其余后台进程。Isaac 仍使用已验证可工作的 headless 路径，因此不受
本机 RTX viewport 崩溃影响。

当前完整策略脱离 Isaac 物理后的单实例纯推理均值约 6.1 ms，满足 50 Hz 的
20 ms 截止期。可视化命令把 PhysX 和策略放在同一张笔记本 GPU 上，实测约
22 Hz 墙钟刷新；这是在线仿真争用造成的显示速度，不是策略本身的推理延迟。

## Phase 1 最终策略人工验收

### 在线策略（推荐）

Isaac 每个 tick 现场运行选定的扩散策略和物理仿真，并通过本机 UDP
送入 ROS；该路径不读取预生成动作或 CSV：

```bash
cd /home/windylab/code/isaac_arm_rl
bash ros_bridge/view_phase1_live_rviz.sh medium
```

把 `medium` 换成 `easy` 或 `hard` 即可。初次 GPU/Isaac 启动需要等待一会儿，
RViz 可能先显示网格，收到第一个在线控制 tick 后机械臂开始运动。
在线验收保持评估使用的两次 100 Hz PhysX 子步，PhysX 和扩散网络都使用
GPU。脚本中的 50 Hz 是控制/仿真时间口径；墙钟刷新率取决于 GPU 负载。

### 离线回放（用于复现实验）

最终策略的 easy/medium/hard 三条轨迹也已经导出。直接选择工况启动：

```bash
cd /home/windylab/code/isaac_arm_rl
bash ros_bridge/view_phase1_rviz.sh medium
```

把 `medium` 换成 `easy` 或 `hard` 即可。RViz 固定坐标系已经设为
`world`：

- 红线：策略实际末端轨迹 `/rollout_ee_actual`
- 绿线：期望圆轨迹 `/rollout_ee_target`
- 机械臂和基座 TF：按 50 Hz CSV 回放

关闭 RViz 后，回放节点和 `robot_state_publisher` 会自动停止。

如果要重新从选定 checkpoint 生成三条 CSV：

```bash
source env_isaaclab/bin/activate
python3 scripts/export_phase1_rviz_rollouts.py
```

## 用法
```bash
source /opt/ros/humble/setup.bash
source ~/code/windylab_ws/install/setup.bash        # package:// 网格解析

# 1) 生成一条测试轨迹(末端画圆,CLIK 跟踪误差 0.01mm)—— 无策略时的冒烟测
python3 scripts/make_synthetic_rollout.py

# 2) RViz 里看臂动
ros2 launch ros_bridge/view_rollout.launch.py \
    rollout:=data/rollout_synth.csv rate:=50.0 use_rviz:=true

# 3)(可选)另开终端看期望 vs 实际跟踪图
ros2 run manipulator trajectory_visualizer_demo.py --ros-args \
    -p trajectory_mode:=circle -p frame_id:=base_link
```

## 格式(`rollout_io.py`,两侧共用)
CSV:`# rollout meta: {json}` + 表头 `t,q1..q7[,ee_x,ee_y,ee_z,tgt_x,tgt_y,tgt_z]`。
必需列只有 `t` + `q1..q7`;`ee_*/tgt_*` 可选(存 Isaac 侧实际/目标末端,供离线对比)。

## Phase 2 模型导出

可部署 bundle 位于 `exports/phase2_robust/`，其中包括 Phase 1 checkpoint、
Phase 2 TorchScript/ONNX 残差网络、组合公式、配置和 SHA256 校验。重新导出：

```bash
env_isaaclab/bin/python scripts/export_phase2_policy.py
```

离线回放侧仍兼容 `rollout_io.write_rollout(...)`，无需修改 RViz 节点。
