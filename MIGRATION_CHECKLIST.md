# Nero → xArm 迁移清单

实施前记录：目标基线 `xarm_ws/main@61eef3b9c88c57f3f8e6aaa3cac1989cd10ed0ca`；参考 Nero master `bc7f7c3fe03b69deda2171711913dea253154274`、pi0_wm `55681547766c1f0cbe9e3b3c80bc5ee3a71c022a`。当前实现分支为 `feat/xarm-nero-migration`，保留了基线工作区已有修改。状态只按代码和无硬件验证填写；`mock` 或 headless 验证不代表真机已验证。

| 源文件/入口 | 目标模块 | 硬件前提 | 实现状态 | 验证状态 |
|---|---|---|---|---|
| `nero_collection/teleop/master_slave.py`、`nero_collection/teleop/bilateral.py` | 主从、位置跟随、双边控制、episode 生命周期 | 主从两端；MIT 仅限有合法关节力矩后端 | 已迁移；xArm 默认位置伺服，MIT 请求会拒绝 | Mock episode、SDK mock；未做真机动作 |
| `nero_collection/cli.py`、`nero_collection/h5_writer.py` | 开始/暂停/保存/丢弃/结束、H5、命令因果记录 | 相机和输入设备按配置 | 已迁移；`q_cmd` 为成功下发的保持值，`delta_q=q_cmd-q_follower` | H5 round-trip 脚本；未接真实相机 |
| `nero_collection/cameras.py` | 多相机绑定、裁剪、缩放、预览 | V4L2/RealSense/Orbbec 设备 | 已迁移，预览与策略帧分离 | mock camera/headless；设备绑定未验证 |
| `nero_collection/inverse_dynamics.py`、`dynamics_processing.py` | FK、重力、RNEA、残差和滤波 | 与型号匹配的 URDF/辨识参数 | 算法和离线流程可用；未提供 xArm 默认参数 | 无硬件离线单测；Nero URDF 不作为 xArm 默认参数 |
| `nero_collection/realtime_dynamics.py`、`realtime_plot.py` | 在线动力学曲线 | 有效力矩/电流报告 | 可选；缺失反馈不会伪造为零 | mock/离线；真机报告语义待用户确认 |
| `ufactory_devices/robot/xarm_adapter.py` | 统一生命周期、状态、位置伺服、夹爪 | xArm SDK、xArm5/6/7 型号和 IP | 已实现；`connect` 只读，真实执行需 `execution_enabled: true`；MIT/直接 torque 明确不支持 | fake SDK 测试；未做真机动作 |
| `nero_collection/arms/process.py`、`xarm_stack/remote_arm.py` | 隔离进程和 ZMQ 状态/RPC | SDK 或 xArm 服务 | 已实现，状态向量支持 5/6/7 轴 | mock/fake SDK；网络服务未在真机验证 |
| `inference/policies/dp/*`、`inference/pipeline.py` | DP/LeRobot DP | 对应 checkpoint、相机和模型契约 | 保留独立 DP 入口 | 仅导入/静态检查，checkpoint 闭环未验证 |
| `inference/pi0_wm/*`、`inference/configs/pi0_wm_xarm.yaml` | 独立 π0-WM、OpenPI、prefetch、迟到/保持处理 | OpenPI 服务、xArm 训练兼容权重 | 独立配置和计划调度已保留；WM 开关可切换连续 IK/WM | Mock worker/计划逻辑；真实权重和真机未验证 |
| `inference/mujoco_*`、`inference/pi0_wm/visualization.py` | MuJoCo FK、完整 horizon 显示、Rerun/回放 | 匹配 xArm URDF、可选 GUI | 显示进程与控制解耦；预测状态不会回灌真机历史 | headless 取决于本机 mujoco 安装 |
| `scripts/view_episode.py`、`scripts/rerun_episode.py` | H5 清单和可选 Rerun 时间线 | `h5py`；Rerun 需 `rerun-sdk` | 已实现为独立离线工具，不进入控制循环 | H5 清单已验证，Rerun 未安装 |
| `xarm_stack/*.py` | 真机服务、Gello/夹爪/力反馈入口 | xArm SDK、Gello/设备 SDK | 保留入口；力反馈仅在有效报告和显式映射后开启 | 协议和导入静态检查 |

## 能力边界

- 我核对了官方 `xArm-Developer/xArm-Python-SDK` 当前 master（commit `d911319cebc45142613e18086aabb8067b18ab7f`，2026-09-23）：xArm7 有专用示例和冗余 IK 参数，但 `set_joints_torque` 在 `xarm/x3/xarm.py` 与 `xarm/wrapper/xarm_api.py` 中都明确标为 **This interface is no longer supported**；`get_joints_torque` 仍保留用于读取。公开 SDK/公开 GitHub 搜索中未找到 UFACTORY xArm7 固件源码，固件实现应视为厂商闭源二进制。
- xArm 公共 SDK 当前提供连续关节位置伺服（优先 `mode=1 + set_servo_angle_j`）。它不提供经核实的 Nero MIT `q/v/kp/kd/tau_ff` 或直接七关节力矩下发；请求这些模式会在使能前报错。
- 若安装官方六维末端力传感器且固件满足 SDK 要求（当前文档标注 `firmware_version >= 1.8.3`），SDK 另有 `set_ft_sensor_admittance_parameters`、`set_ft_sensor_force_parameters` 和 `set_ft_sensor_mode`。这是内部笛卡尔导纳/末端力控制，不是关节力矩或 Nero MIT；本迁移未启用、未验证。
- xArm+Gello 三服务中已实现从 xArm 有效力矩/已标定电流到 Gello 力矩请求的计算路径，包含偏置、死区、低通、幅值/变化率限制和启动渐入；示例默认关闭。Gello 驱动的力矩/电流映射与真机闭环效果仍未验证，也不等于 xArm MIT。
- `feedback_signal` 必须明确为 `none`、`torque` 或 `current`。电流不会被重命名为力矩；缺失反馈用 `NaN + *_valid=false` 表示。
- 夹爪命令默认使用物理宽度（米），由 endpoint 的 `gripper_open_position`、`gripper_close_position`、`gripper_min_width_m` 和 `gripper_max_width_m` 显式标定；`raw_position` 只用于已确认的原始 SDK 单位。
- 真实 xArm 连接默认是只读 dry-run。`connect` 不使能、不清故障、不移动、不初始化夹爪、不更改碰撞设置。
- `MockArm`、协议验证、FK/MuJoCo、H5 查看/回放、动力学计算可在无硬件时运行。它们不构成真机验证。
- Nero URDF、Nero 惯量/限位/辨识结果和模型权重不会自动作为 xArm 参数；请提供与实际 xArm 型号匹配的 URDF、工具坐标和 checkpoint 契约。
- π0/WM 的 `coordinate_frame` 由配置和 checkpoint action contract 显式核对；xArm 示例使用 `tcp`，旧 Nero 示例使用 `link7`，二者不匹配时启动会拒绝。

## 安装和命令

基础无硬件检查：

```bash
python -m compileall -q ufactory_devices nero_collection xarm_stack inference
python -m nero_collection.cli --config /path/to/mock_collection.yaml --backend mock --dry-run-duration 1 --episode-limit 1 --auto-save
```

H5 查看和离线回放使用 `nero_collection/h5_writer.py` 生成的 `episode_*.h5`；推理离线入口按 `inference/README.md` 的 DP 配置执行。π0-WM 使用独立配置：

```bash
python scripts/run_pi0_wm.py --config inference/configs/pi0_wm_xarm.yaml --mock --mock-wm --steps 260
```

```bash
python scripts/view_episode.py /path/to/episode.h5
python scripts/rerun_episode.py /path/to/episode.h5 --camera wrist  # 可选 rerun-sdk
```

DP 配置检查（会加载配置声明的 checkpoint，需安装对应 PyTorch/模型依赖）：

```bash
python -m inference.cli --config inference/configs/dp_xarm.yaml --check --backend mock
```

ZMQ xArm 服务（只读 dry-run；不会使能机器人）：

```bash
python -m xarm_stack.xarm_server --config configs/xarm_three_server.yaml
```

π0-WM 的三类模式严格区分：`--mock` 使用模拟臂和模拟相机；不传
`--enable-commands` 是真实反馈 dry-run；只有在服务端 endpoint 已人工设置
`execution_enabled: true` 后，才可使用 `--enable-commands` 发送连续位置目标：

```bash
python scripts/run_pi0_wm.py --config inference/configs/pi0_wm_xarm.yaml --dry-run
python scripts/run_pi0_wm.py --config inference/configs/pi0_wm_xarm.yaml --enable-commands
```

允许真机位置执行前，用户必须在 endpoint 中填写真实型号、`robot_ip`、`dof`、匹配的 `rest_q`/限位、`model_path`、`base_frame`/`tcp_frame`、夹爪型号和 `feedback_signal`，并显式设置 `execution_enabled: true`。只有完成人工安全检查后才运行 `enable` 或遥操命令。本仓库自动化验证不会向真实机械臂发送动作。

## 待用户填写

1. xArm5、xArm6 或 xArm7 型号及固件版本、IP。
2. 实际 TCP/工具偏置、base/TCP frame、与型号匹配的 URDF 和辨识结果。
3. 夹爪型号、原始位置范围与物理宽度标定。
4. 相机序列号、设备绑定、裁剪/缩放和 RGB 约定。
5. Gello/Pika/UMI/SpaceMouse/Xbox 设备路径及运动映射。
6. OpenPI server 地址、训练配置名、embodiment/关节布局、归一化统计量和与 xArm 匹配的 DP/π0/WM checkpoint。

## 本次实际验证

- `python -m compileall -q .`：通过。
- `pytest -q`：7 个 xArm 生命周期、失败下发保持值、gripper 启动只读及 xArm+Gello 采集因果/反馈计算与拒绝测试通过。
- fake xArm SDK：xArm6 维度、毫米到米 EE 转换、`connect` 无运动副作用、gripper 服务启动只读、SDK 错误码和成功后 `last_commanded_q` 更新均通过。
- `python scripts/run_pi0_wm.py --mock --mock-wm --steps 5`：通过启动预热、π0/WM 延迟测量和调度；输出仅为模拟臂，未发送真机命令。
- DP CLI circular import：已修复为惰性 compatibility runner 导入；`dp_xarm.yaml --check` 已进入 checkpoint 加载阶段，但当前环境缺少可选 `lerobot==0.4.0`，因此模型恢复未完成。
- mock collection：采集→H5→`scripts/view_episode.py` 通过；H5 包含 `q_cmd`、`delta_q`、相机/时间和 robot profile 元数据。
- 未验证：真实 xArm SDK/固件、真实相机、现场 OpenPI/DP/WM 权重、CAN/Gello、任何实体运动和真实 100 Hz 达标率。
