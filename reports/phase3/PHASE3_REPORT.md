# Phase 3 deployment and comparison report

## 当前选定控制器

- Phase 1：`policy_precision.pt`，deterministic DDIM-8，每 tick 重新规划。
- Phase 2：`model_950.pt` 的确定性 PPO mean actor。
- 鲁棒部署尺度：`residual_scale = 0.10`。
- 组合：`clip(a_phase1 + 0.10 * clip(a_residual, -1, 1), -1, 1)`。

4096 环境、400 tick、随机 MPC handoff、0.002 rad 关节扰动、开启 world-frame
惯性力的成对验证结果：

| 指标 | Phase 1 基线 | Phase 1 + Phase 2 |
|---|---:|---:|
| position mean | 1.407 mm | 0.646 mm |
| position p95 | 2.936 mm | 1.389 mm |
| position p99 | 3.642 mm | 1.686 mm |
| orientation mean | 3.570° | 3.405° |
| orientation p95 | 6.813° | 6.158° |
| divergence (>50 mm) | 0% | 0% |

## 导出

部署 bundle：`exports/phase2_robust/`

- `residual_actor.ts`：已验证的 TorchScript 残差 actor。
- `residual_actor.onnx`：ONNX opset 17，动态 batch，ONNX checker 通过。
- `phase1_diffusion_policy.pt`：Phase 1 基座策略。
- `deployment_manifest.json`：张量维度、DDIM 步数、组合公式和控制频率。
- `SHA256SUMS`：所有 bundle 文件的完整性校验。

TorchScript 对 eager actor 的最大绝对误差为 0。

## 在线 RViz

```bash
cd /home/windylab/code/isaac_arm_rl
bash ros_bridge/view_phase2_live_rviz.sh medium
```

这是在线闭环，不是 CSV 或录像。红线为实际末端，绿线为目标。

纯控制器在 RTX 5060 Laptop GPU 上的 500 次单实例基准：

| 路径 | mean | p95 | max |
|---|---:|---:|---:|
| Phase 1 DDIM-8 | 5.776 ms | 6.448 ms | 6.844 ms |
| Phase 2 residual + compose | 0.299 ms | 0.905 ms | 1.299 ms |
| 完整 Phase 1+2 | 6.090 ms | 6.850 ms | 7.106 ms |

500/500 次均满足 50 Hz 的 20 ms 截止期。Isaac GPU PhysX 与策略同时运行时，
在线 RViz 测试约 22 Hz；这反映同卡物理仿真争用，不是脱离 Isaac 后的控制器延迟。

## acados MPC 对照

MPC 的实际位置误差由日志中每行 `q_meas`、基座位姿和当前轨迹目标做
Pinocchio FK 重建，不使用求解器内部 `first_error_m`：

| 工况 | acados mean / p95 / max | Phase 2 mean / p95 / max |
|---|---:|---:|
| easy | 0.217 / 0.539 / 5.160 mm | 0.342 / 0.679 / 3.207 mm |
| medium | 0.946 / 4.522 / 7.327 mm | 0.670 / 1.228 / 6.991 mm |
| hard | 2.572 / 5.665 / 9.196 mm | 0.925 / 1.573 / 9.084 mm |

合并均值为 acados 1.221 mm、Phase 2 0.646 mm。Phase 2 数值低 47.1%，但
easy 工况的 acados 均值更好。当前是相同任务参数、不同执行平台/初始状态协议的
并列，不是严格同平台 A/B。

延迟方面，acados solve 均值 3.368 ms，solve-to-publish 均值 5.266 ms；
完整策略均值 6.090 ms。因此当前完整扩散策略满足实时截止期，但尚未比 acados
求解本身更快；如果目标是 `<1 ms`，仍需对 Phase 1 做一致性/单步策略蒸馏并重新
闭环验收，不能只引用 0.299 ms 的残差层。

## 3-D 球形基座扰动包络

扫描了 32×32 = 1024 个并行场景：

- 半径 `R = 0–0.05 m`
- 频率 `f = 0.25–5 Hz`
- 开启惯性耦合
- 先 warmup 200 tick，再测量 200 tick
- 全图固定 Phase 1 gain 0.90，避免离散 gain 切换制造假边界

严格可行判据：position RMS ≤ 2 mm、position max ≤ 10 mm、
orientation mean ≤ 7°。可行格点占全扫描的 10.25%。约 1 Hz 时观测到的
最大可行半径约 1.61 cm；约 1.94 Hz 时约 0.48 cm；约 3 Hz 仅 R=0 的格点
通过，3.16 Hz 以上没有格点通过。

这个球形波形超出了 Phase 1/2 主要训练的圆形工况，所以图中存在非单调和孤立
可行格点。它是 OOD 压力测试，不应把孤立格点外包络当成连续安全保证。

## 尚需真实系统授权的最后一步

把 bundle 接到和 acados 相同的 ROS 控制管线，增加：

1. 47-D Phase 1 观测的在线构造和两帧历史；
2. Phase 1 DDIM + Phase 2 residual 的动作组合；
3. 与 MPC 完全相同的 joint/velocity/Cartesian safety guard；
4. 同样的 `q_meas`、目标、solve/inference-to-publish 时间戳日志。

完成这一项后，才能给出真正的 “RL vs acados MPC” 同平台最终结论。
