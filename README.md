# isaac_arm_rl — 7DOF 臂 MPC 模仿学习 + 残差强化学习

在笔记本(RTX 5060 Laptop, Blackwell sm_120)上训练 7DOF 浮动基座臂的 RL 策略,
导出 ONNX 部署到 ROS 2 Humble / Jetson,对标现有 NMPC。训练与 ROS 工作区完全隔离。

## 环境(Phase 0,已装好)
- Python **3.11.15** venv:`env_isaaclab/`(uv 创建,勿与 ROS 的 3.10 混用)
- `isaaclab[isaacsim,all]==2.3.2.post1` → 实际拉入 **Isaac Sim 5.1.0**
- `torch==2.7.0+cu128`(CUDA 12.8,支持 Blackwell sm_120,已验证 GPU matmul)
- 激活:`source env_isaaclab/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES`

## 目标机械臂
- URDF:`src/arm-platform/config/arm.urdf`(7 转动关节 joint1..joint7)
- 末端帧:`link7`(与 `pinocchio_ik.py` 的 EE_FRAME 一致)
- meshes:`package://dummy_description/` → `windylab_ws/src/dummy_description/`

## 脚本
| 脚本 | 解释器 | 作用 |
|---|---|---|
| `scripts/install_isaac.sh` | — | Phase 0 安装(已跑完) |
| `scripts/gen_fk_reference.py` | 系统 py3.10 + pinocchio | 生成 28 组关节配置 + link7 位姿基准(已跑通) |
| `scripts/import_urdf.py` | venv py3.11 | 解析 package:// → 绝对路径,UrdfConverter 转 `usd/arm.usd` |
| `scripts/fk_isaac.py` | venv py3.11 | Isaac 里对同一批配置算 base→link7 相对位姿 |
| `scripts/compare_fk.py` | 系统 py3.10 | 对照,判据:位置<1mm 且姿态<0.5° |
| `scripts/finish_phase01.sh` | — | 重启后一键收尾(import→fk→compare) |

## 复现 Phase 1(已通过)
```bash
cd /home/windylab/code/isaac_arm_rl
/usr/bin/python3 scripts/gen_fk_reference.py     # pinocchio 基准(py3.10)
bash scripts/finish_phase01.sh                   # 导入 USD + Isaac FK + 对照
```

## 关键坑(已解决,备查)
1. **驱动模块不匹配**:2026-07-23 unattended-upgrade 升级 595.71.05→595.84,内核仍跑旧模块,
   Vulkan 建不了设备。GPU 被桌面占用不能热 rmmod → **重启**解决。以后 CUDA/Isaac 突然报
   NVML mismatch 都是这个,`cat /proc/driver/nvidia/version` vs `modinfo nvidia` 一比即知。
2. **EULA 卡首启**:脚本里设 `OMNI_KIT_ACCEPT_EULA=YES`。
3. **RTX GUI 与 595.x 驱动不兼容**:Isaac Sim 5.1 在 Blackwell + NVIDIA 595.x 上会于
   `librtx.scenedb` 段错误；headless 不走 RTX viewport，因此训练不受影响。原生 GUI 需换回
   Isaac Sim 5.1 验证的 580-open 驱动并重启。FK/训练继续使用
   `AppLauncher(headless=True, enable_cameras=False)`。
4. **UrdfConverterCfg 必填项**:2.3 版 `joint_drive.gains.stiffness` 无默认值,须显式给。
5. **FK 重力漂移**:自由关节步进物理被重力拉动 → ~1mm 误差。设 `gravity=(0,0,0)` 纯运动学 FK
   → 误差降到 float32 噪声地板。

## 进度
- [x] Phase 0 安装(uv/venv/isaaclab 2.3.2/isaacsim 5.1/torch cu128)
- [x] Phase 0 验收:torch 看到 sm_120 且 GPU matmul 通过;Isaac headless 启动 + 物理步进 OK;
      RL 栈(rsl_rl/skrl 2.1/gymnasium 1.2/tensordict)可导入
- [x] Phase 1:pinocchio FK 基准(28 组配置)
- [x] Phase 1:URDF→USD 导入(`usd/arm.usd` + 分层配置)
- [x] **Phase 1 验收:Isaac vs Pinocchio FK 对照 PASS,max 0.0002mm / 0.0001°**
- [x] Phase 0:DART 噪声采集、观测泄露和动作尺度问题修复
- [x] Phase 1:鲁棒优先的 deterministic DDIM-8 diffusion BC
- [x] Phase 2:冻结 Phase 1 + bounded residual PPO，选定 `model_950.pt`
      与部署残差尺度 `0.10`
- [x] Phase 2:4096 环境成对验证，位置均值 0.646mm、p95 1.389mm、
      divergence 0%
- [x] Phase 3:模型 bundle、TorchScript/ONNX、在线 RViz、单实例延迟基准、
      acados 日志对照和 1024 场景 `(R,f)` 球形扰动包络

## 可视化桥(已就绪,`ros_bridge/`)
Isaac 无 GUI(RTX 崩溃),改走 **rollout → RViz 回放**:Isaac 导出关节轨迹 CSV →
`replay_rollout_node.py` 发 `/joint_states` → 你现有 robot_state_publisher/RViz/
trajectory_visualizer 直接看,可视化器零改动。已 headless 验证(TF base_link→link7 精确复现
圆轨迹)。用法见 `ros_bridge/README.md`;格式 `rollout_io.py` 两侧共用。

## 当前入口

```bash
# Phase 2 在线 Isaac → RViz
bash ros_bridge/view_phase2_live_rviz.sh medium

# 重新导出部署 bundle
env_isaaclab/bin/python scripts/export_phase2_policy.py

# 重新测纯策略推理延迟
env_isaaclab/bin/python scripts/benchmark_phase2_policy.py

# 用 q_meas FK 重建 acados 跟踪误差并生成对照
python3 scripts/compare_phase2_mpc.py
```

Phase 3 结果和口径见 `reports/phase3/PHASE3_REPORT.md`。真正的同平台最终 A/B
仍需把导出策略接到与 acados 相同的 ROS/机械臂控制管线再采一组日志；当前
Isaac 结果和 acados 日志并列用于判断方向，不能冒充同平台证明。
