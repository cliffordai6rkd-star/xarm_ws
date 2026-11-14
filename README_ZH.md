# UFACTORY 遥操作系统

本项目为 UFACTORY (深圳市众为创造科技有限公司) 机械臂提供遥操作解决方案，包含三套独立方案：

1. **基于 Pika Sense 的遥操作**：利用 Agilex Robotics 的 Pika Sense 技术实现精准运动跟踪与控制。
[![观看视频](assets/pika_teleoperation_system.jpg)](https://www.bilibili.com/video/BV1791rB4Egk/?spm_id_from=333.1387)
2. **UMI 遥操作**：使用 FAST UMI 设备进行运动捕捉与控制。
[![观看视频](assets/fastumi.jpg)](https://www.bilibili.com/video/BV1qAGm6fEZE/?spm_id_from=333.1387)
3. **基于 GELLO 的遥操作框架**：基于开源 GELLO 框架（https://wuphilipp.github.io/gello_site/）的理念  
[![观看视频](assets/gello.png)](https://www.bilibili.com/video/BV12xFjzzEaX/?spm_id_from=333.1387)

## 概述

UFACTORY 遥操作系统通过先进的运动跟踪技术，实现对 UFACTORY 机械臂的直观远程控制。该系统旨在降低高质量示教数据采集的门槛，服务于机器人学习和操作任务。

## 项目结构

```
ufactory_teleop/
├── ufactory_devices/         # 共享核心库
│   ├── transformations.py    # 位姿数学工具（四元数、RPY、轴角、齐次矩阵）
│   ├── robot/                # xArm 机械臂封装（连接、运动、夹爪控制）
│   ├── pika/                 # Pika Sense 与 Pika Gripper 串口驱动
│   ├── umi/                  # UMI 设备 SDK 绑定（基于 ctypes 的 XVLib 封装）和 HTC Vive Tracker 驱动（pysurvive）
├── pika_teleop/              # Pika Sense 遥操作方案
│   ├── uf_robot_pika_teleop.py
│   ├── calibrate.py
│   ├── config/
│   └── rules/
├── umi_teleop/               # UMI 遥操作方案
│   ├── uf_robot_umi_teleop.py
│   ├── uf_robot_umi_teleop_dual.py
│   ├── calibrate.py
│   ├── config/
│   ├── rules/
│   └── xvsdk/
└── gello_teleop/             # Gello 遥操作方案
    ├── uf_robot_gello_teleop.py
    ├── config/
    └── rules/
```

## 遥操作方案

### 基于 Pika Sense 的方案

使用 Agilex Robotics 的 Pika Sense —— 一款集成 Vive Tracker 安装位的手持式夹爪控制器 —— 实时捕捉 6 自由度手部运动，并将其映射为机械臂末端执行器的运动。

- **跟踪方式**：HTC Vive Tracker + Lighthouse 基站（通过 Pika SDK）
- **控制方式**：单臂遥操作，通过夹爪开合切换启停状态
- **夹爪支持**：Pika Gripper、xArm Gripper (G1/G2)、BIO Gripper G2、Robotiq Gripper
- **运动模式**：伺服笛卡尔模式（模式 1）与在线轨迹规划模式（模式 7）

详见 [pika_teleop/README_ZH.md](pika_teleop/README_ZH.md)。

### UMI 遥操作方案

使用 FastUMI 设备实现直观的遥操作。支持单臂和双臂配置，提供灵活的跟踪选项。

- **跟踪方式**：UMI 内置 SLAM 或 HTC Vive Tracker + Lighthouse 基站（可配置切换）
- **控制方式**：单臂（`uf_robot_umi_teleop.py`）与双臂（`uf_robot_umi_teleop_dual.py`）遥操作
- **夹爪支持**：xArm Gripper G2、xArm Gripper、BIO Gripper G2、Pika Gripper、Robotiq Gripper
- **运动模式**：伺服笛卡尔模式（模式 1）与在线轨迹规划模式（模式 7）
- **双臂协作**：两个 UMI 设备通过独立线程同时控制两台 xArm 机械臂

详见 [umi_teleop/README_ZH.md](umi_teleop/README_ZH.md)。

### Gello 遥操作方案

使用 Gello 主手（基于 Dynamixel 伺服的触觉输入设备）在关节空间对 xArm 机械臂进行遥操作控制。通过串口读取 Gello 关节位置，自动计算关节偏置后映射为机器人目标关节角。

- **控制空间**：关节空间（robot_mode: 6），直接映射主手关节运动
- **支持机型**：xArm5、xArm6、xArm7（提供对应示例配置）
- **关节映射**：可通过 `joint_ids` 和 `joint_signs` 灵活配置 Gello 关节到机器人关节的映射关系
- **自动偏置**：启动时自动读取 Gello 当前姿态并计算关节偏置
- **力矩模式**：不参与映射的 Dynamixel 关节可通过 `torque_joint_ids` 开启力矩模式以保持固定
- **夹爪支持**：可选 Gello 侧 Dynamixel 夹爪，支持 xArm Gripper (G1/G2)、Pika Gripper、Robotiq Gripper

详见 [gello_teleop/README_ZH.md](gello_teleop/README_ZH.md)。


## 功能特性

- **直观操控**：直接操作界面，缩小用户与机器人本体之间的隔阂
- **经济高效**：采用成熟的商用跟踪技术与现成组件
- **高质量示教**：为模仿学习采集精准的示教数据
- **多机型兼容**：适配多种 UFACTORY 机械臂型号（xArm 5/6/7、Lite 6、850）
- **夹爪灵活可选**：支持 xArm Gripper (G1/G2)、BIO Gripper G2、Pika Gripper、Robotiq Gripper
- **双臂协同操作**：UMI 方案支持同步双臂遥操作，适用于双手协作任务

## 快速开始

### Pika Sense 遥操作

详见 [pika_teleop/README_ZH.md](pika_teleop/README_ZH.md)。

快速启动：

```bash
git clone https://github.com/xArm-Developer/ufactory_teleop
cd ufactory_teleop/pika_teleop
conda create --name py39 python=3.9
conda activate py39
pip install -r requirements.txt
pip install pysurvive agx-pypika --no-deps
sudo cp rules/*.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
python uf_robot_pika_teleop.py --config config/xarm6_pika_teleop.yaml
```

### UMI 遥操作

详见 [umi_teleop/README_ZH.md](umi_teleop/README_ZH.md)。

快速启动（单臂）：

```bash
cd ufactory_teleop/umi_teleop
conda create --name py39 python=3.9
conda activate py39
sudo dpkg -i xvsdk/XVSDK_focal_amd64.deb
sudo apt install -y --fix-broken
pip install -r requirements.txt
pip install pysurvive agx-pypika --no-deps
sudo cp rules/*.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
python uf_robot_umi_teleop.py --config config/xarm6_umi_teleop.yaml
```

快速启动（双臂）：

```bash
python uf_robot_umi_teleop_dual.py --config config/xarm6_umi_teleop_dual.yaml
```

### Gello 遥操作

详见 [gello_teleop/README_ZH.md](gello_teleop/README_ZH.md)。

快速启动：

```bash
cd ufactory_teleop/gello_teleop
conda create --name py39 python=3.9
conda activate py39
pip install -r requirements.txt
pip install pysurvive agx-pypika --no-deps
sudo usermod -aG dialout $USER
# 重新登录后运行
python uf_robot_gello_teleop.py --config config/xarm7_gello_teleop.yaml
```

## 参考资料

## 功能迁移

统一数采、动力学、H5、xArm 适配层和独立 π0-WM 运行时见
[MIGRATION_CHECKLIST.md](MIGRATION_CHECKLIST.md)。无硬件 mock 验证：

```bash
pip install -r requirements-xarm.txt
python -m nero_collection.cli --config configs/mock_collection.yaml --backend mock \
  --dry-run-duration 1 --episode-limit 1 --auto-save
python -m inference.cli --config inference/configs/pi0_wm_mock.yaml --mock --auto-start --steps 260
```

xArm SDK 默认只读；只有 endpoint 显式设置 `execution_enabled: true` 后才允许使能和位置下发。SDK 没有经核实的 Nero/MIT 关节力矩接口。硬件字段、模型契约和未验证范围请以迁移清单为准。

- [Agilex Robotics Pika Sense](https://global.agilex.ai/products/pika)
- [UFACTORY 协作机械臂](https://www.ufactory.cc/xarm-collaborative-robot/)
- [LuMos FastUMI](https://www.fastumi.com/)
- [GELLO: 通用低成本遥操作框架](https://wuphilipp.github.io/gello_site/)


- [dual_gello_collect.py](/home/rei/mnt/code/xarm_ws/gello_teleop/dual_gello_collect.py)：双臂连接、低速复位、人工对齐、接管、100 Hz 位置控制、ZOH、episode 采集。
- [xarm7_gello_dual_dataset.yaml](/home/rei/mnt/code/xarm_ws/gello_teleop/config/xarm7_gello_dual_dataset.yaml)：双臂配置。
- [test_joint_current.py](/home/rei/mnt/code/xarm_ws/xarm_stack/test_joint_current.py)：独立电流观测、阶段标记、HDF5/CSV/PNG 输出。
- [xarm_joint_current_test.yaml](/home/rei/mnt/code/xarm_ws/xarm_stack/config/xarm_joint_current_test.yaml)：电流测试配置。
- 修复 xArm set_servo_angle_j(angles=...) 参数。
- 按 SDK 真实格式解析 get_joint_states()；effort 根据 set_report_tau_or_i(0/1) 分别记录为 torque 或 current。
- 增加 GELLO 型号探测、电流阻尼、速度滤波、输出限幅、弱保持和 watchdog。
- HDF5 增加双臂关节、命令时间、序号、数据年龄、dq/ddq、有效性、反馈来源、相机时间轴和统计信息。
运行：
、、、
python -m gello_teleop.dual_gello_collect \
  -c gello_teleop/config/xarm7_gello_dual_dataset.yaml
  、、、
流程为连接检查 → xArm 复位 → GELLO 插值对齐 → 自动采样核对 → 接管；接管后按 r 采集，Enter/空格结束本集采样并提示确认，y 保存、n 丢弃，等待确认时遥操继续，t 重新对齐后接管。

GELLO/xArm 的 H5 每臂数据统一使用 `侧别_量_设备`，例如 `left_q_xarm`、`left_q_gello`、`right_q_xarm`、`right_q_gello`；单臂也保留侧别。末端位姿保存为 `left_q_eepose_xarm` 和 `right_eepose_xarm`，每帧为基座到 TCP 的 `4×4` 变换矩阵，平移单位米。完整字段和旧文件转换命令见 [数采数据字段说明](gello_teleop/README_ZH.md)。
URDF 力矩窗口按 `active_arms` 显示每臂的 `tau_ext_l1` 和 J1～J7 的 `tau_ext`；双臂并排两列，单臂只显示所选侧。
按 q 或 Ctrl+C 时，先停止遥操，再按 `alignment.reset_*` 配置返回本次启动的 `reset_q`，
确认到位后保留 xArm 的位置保持和电机使能，再关闭连接退出。复位失败会报错并尝试原地保持。
电流测试：
、、、
python -m xarm_stack.test_joint_current \
  -c xarm_stack/config/xarm_joint_current_test.yaml \
  --arm both --duration 30 --output current_test.h5
  、、、
默认不运动；需要当前位置保持时增加 --hold，并将对应 execution_enabled 改为 true。运行中 b/p/l 标记 baseline、press、release。

URDF 动力学离线验证使用 `xarm_stack.validate_urdf`，只读取 H5，不连接机械臂。
入口读取右臂有效的实测 `q/dq/tau`，使用反馈来源时间戳去掉重复报告，
按无效样本和时间间隙分段，在各段内重采样并用相同的 Savitzky–Golay 窗口平滑，
从速度独立求导得到加速度；滤波边缘不计入误差统计。
Pinocchio 与 MuJoCo 从相同的 URDF、状态、力矩计算正向动力学，
另将速度求导得到的加速度用于逆动力学力矩对比。

```bash
conda activate gello
# 先分析目录排序后的第一集，保存七轴曲线、NPZ 和 JSON。
python -m xarm_stack.validate_urdf runs/cotrain/peel_cucumber
# 分析全部集；七轴曲线和 NPZ 保存第一集，JSON 包含所有集和加权汇总。
python -m xarm_stack.validate_urdf runs/cotrain/peel_cucumber \
  --limit 0 --output runs/urdf_validation/cotrain_g2_all
# MuJoCo 实测回放 + 半透明蓝色力矩仿真。每 100 ms 重置预测，窗口内独立积分。
python -m xarm_stack.validate_urdf runs/cotrain/peel_cucumber --viewer
# 无窗口的 MuJoCo 姿态对比图；EGL 环境需要可用的图形驱动。
MUJOCO_GL=egl python -m xarm_stack.validate_urdf \
  runs/cotrain/peel_cucumber --render
```

默认使用 `gello_teleop/models/xarm7_dynamics_g2.urdf`，不额外添加负载，
关闭 MuJoCo 接触/关节限位约束和自动惯性修复。默认不启用尚未标定的摩擦；
`--friction urdf` 在两个引擎中显式使用相同的粘性＋平滑库仑摩擦模型。
`--urdf` 可选择裸臂或其他七轴模型。`--window-s` 默认 0.11 s，
用于离线零相位平滑，不能据此推断实时性能。

削皮数据的接触状态默认标为未知。低速片段不被当作无接触片段。
对单个 H5 文件，只有在人工确认无接触区间后才使用
`--free-space 起始秒 结束秒`，时间从该文件第一条反馈开始计算。
这项统计单独列出，不改变全记录结果。不能把模型估算的外力矩扣掉后再验证同一模型，
也不能用主机虚拟 MIT 力矩代替上报关节力矩。
结果包含外部接触、未标定摩擦、实际工具负载及力矩上报误差；
两引擎一致只能验证模型转换与计算一致，不能证明实机 URDF 准确。

对于人工确认整集无外部接触、末端只有 G2 的记录，可以明确标注假设，
并用 `--arm left/right` 分别选择旧格式 H5 的前/后七列及每臂来源时间戳：

```bash
python -m xarm_stack.validate_urdf \
  runs/test_data/bg_data/dual_gello_xarm_0001_20260930_211754.h5 \
  --arm right --no-contact --tool-configuration g2_only \
  --output runs/urdf_validation/bg0001_right_g2
```

`--no-contact` 记录人工确认，不从动力学残差推断接触。
`--tool-configuration` 仅记录实际末端配置，不改所选 URDF 的质量和惯性。
报告另列静止样本（七轴速度均小于 0.01 rad/s、加速度均小于 0.1 rad/s²）、
运动样本、实测加速度 RMS 和位置求导与记录速度的一致性。
`--render --render-time 100` 保存 100 s 之后约 100 ms 独立积分的姿态对比；
`--viewer` 回放完整所选集。长记录可先用 JSON 指标和静态图评估，
避免仅凭位置回放动画判断动力学准确。

若要检查“记录速度求导得到 ddq → URDF 逆动力学 tau → MIT → 动力学反算 ddq”，
在上述命令中加入 `--roundtrip`。此路径用 `Kp=Kd=0`、`t_ff=tau_model`，
不另加重力、不限幅，输出 `model_tau_roundtrip.png` 和 JSON 中的
`model_tau_roundtrip` 指标；NPZ 包含两种反算加速度及实际 MIT 输入力矩。
原来的 `Acceleration RMSE` 仍表示记录力矩作为输入的独立对比，
新增控制台行 `Model-tau roundtrip ... RMSE` 才表示同模型逆算/反算的误差。
同一个有效 URDF 的逆动力学与正向动力学在数学上互相抵消，
因此此指标接近零只验证计算一致性，不能证明质量、质心和惯性符合真机。

已有实验模型的力矩输出使用独立入口，避免与 `tau_model` 混淆：

```bash
python -m xarm_stack.validate_learned_tau \
  runs/test_data/bg_data/dual_gello_xarm_0001_20260930_211754.h5 \
  --no-contact --output runs/urdf_validation/bg0001_right_experimental_tau
```

默认读取 `model/tau_offline/residual_bilstm_zero5_50/model.pt`，
在所选 H5 上用冻结的实验模型重新预测 `tau_free`，不使用别集的旧 NPZ。
按实验原来的 100 Hz → 零相位 5 Hz → 50 Hz 流程处理状态，
从同一滤波速度求导得到 ddq，在原生 50 Hz 网格上比较。
输出是模型预测的总关节力矩，MIT 仅回放该力矩（Kp=Kd=0、t_ff=tau_free），
没有用当前待验证 URDF 的逆动力学力矩替换实验输出。
JSON 分别记录预测力矩与实测力矩的误差、预测力矩反算加速度的误差、
以及预测误差对反算结果的贡献。`--checkpoint` 可换兼容的已导出实验模型。
默认实验的辨识先验为右臂，本入口只验证右臂。
`--viewer` 打开 MuJoCo 回放；`--render --render-time 100` 保存短时预测姿态对比图。
零相位处理及双向网络使用未来上下文，这些是离线对比结果。

入口会读取实验档案的划分，标明训练重放或原验证集。
默认模型的 0001 左右臂参与训练，0002 右臂是档案验证集；
001 的预测结果不能当作独立数据上的泛化精度。
实验模型自身包含 URDF 辨识先验和学习残差，模型误差、摩擦和驱动动力学
都会影响反算结果，仍不能仅凭该复合对比认证 URDF 的真实惯性参数。

直接用记录力矩验证位置增量，使用 `xarm_stack.validate_torque_replay`：

```bash
python -m xarm_stack.validate_torque_replay \
  runs/test_data/bg_data/dual_gello_xarm_0001_20260930_211754.h5 \
  --output runs/urdf_validation/bg0001_recorded_torque_q_raw
# 可选噪声敏感性对照：只滤波输入力矩，q/dq 参考保持原样。
python -m xarm_stack.validate_torque_replay \
  runs/test_data/bg_data/dual_gello_xarm_0001_20260930_211754.h5 \
  --torque-cutoff-hz 5 --output runs/urdf_validation/bg0001_recorded_torque_q_lowpass5
```

本入口从 H5 读取 `tau`，以记录 `q/dq` 为单步起始状态，计算
`a=solve(M(q),tau-h(q,dq)-friction)`，再以实际反馈间隔 `dt` 计算
`delta_q=dq*dt+0.5*a*dt²`、`q_next=q+delta_q`，与记录的下一步 q 比较。
并用 MuJoCo 每 1 ms 重算动力学、积分同一段记录力矩。
不使用实验预测力矩、不生成逆动力学输入力矩，也不对 q/dq 求导生成参考加速度。
必须保留起始速度贡献 `dq*dt`，不能只积分加速度项。

单步对比每次从记录状态初始化；JSON 同时给出仅保留起始速度的基线，
以及默认约 100 ms 的多步独立积分误差，多步窗口内不覆盖仿真 q/dq。
图使用中文说明，单步图左列直接叠加数据 q 和 URDF 积分 q，右列显示两者差值。
`j7_direct_q_comparison.png` 是 J7 的一秒放大对照，
`q_direct_comparison.csv` 每行保存同一时刻七轴的数据 q、URDF 反推 q 和差值。
另提供单个连续积分窗口图。
所有角度误差按未折叠的关节角统计，整圈转动误差不按 2π 消除。
`--physics-dt-s` 可检查数值积分步长，`--friction urdf` 是未标定摩擦的显式对照。
关节限位、碰撞约束及驱动惯量默认关闭，以检查所声明的刚体模型；
结论仍取决于上报力矩是否能解释为实际驱动力矩以及驱动/摩擦参数是否完整。

H5 离线回看使用 `scripts/browse_h5_rerun.py`，支持 `dual_gello_xarm_NNNN_*.h5`
和旧 `episode_NNNN_*.h5` 文件名。当前机器的 `gello` 环境已安装 Rerun 0.26.2 和 Pinocchio：

```bash
cd /home/eid/code/xarm_ws
conda activate gello
python scripts/browse_h5_rerun.py \
  runs/peel_cucumber_25hzcam --arm right
```

输入集号 `0`～`47` 回看双路 RGB、右臂 TCP 三维轨迹、J1～J7 实际/目标关节角
及七轴 `tau_ext_cal`。图像使用各自的采集时间戳，与低维曲线共享时间轴。
输入下一集号切换，`q` 退出；`d` 会将当前文件移动到 `runs/trash`。

已录制的七轴 `tau_ext_cal` 优先显示。缺失时默认按 H5 的 `force_feedback` 配置，
使用当前 URDF、已录制的关节状态及实测力矩离线估算，曲线和面板标明 `offline estimate`，
不修改原 H5。`peel_cucumber_25hzcam` 的 48 集均未保存七轴外力矩，使用动量观测器估算；
控制器 TCP 负载未随数据保存，因此仅使用当前 URDF 的惯量，结果可能包含负载和模型误差。
无效及观测器启动样本保留为空值。面板显示 URDF 路径、SHA256、输入字段和摩擦配置。
增加 `--no-estimate-tau-ext` 可以关闭估算。

LeRobot v3 清洗使用独立入口（`pinn` 环境含 PyAV 和 Rerun 0.26.2）：

```bash
conda activate pinn
python scripts/rerun_lerobotv3.py --clip
# 也可直接传入数据集根目录：
python scripts/rerun_lerobotv3.py \
  runs/3cam_data/v3/peel_cucumber_25hz_3cam_lerobotv3 --clip
```

先输入包含 `meta/info.json` 的 v3 文件夹，再输入 episode 编号。只读取所选
episode 的低维数据，视频按需逐帧解码，以每秒 60 个原始帧播放；25 fps 数据约为
2.4 倍速，解码/显示跟不上时降低播放速度，不跳帧。加载完所选 episode 后，
在 Rerun 的 **`episode_time`** 时间轴上拖动查看，图像和曲线随游标一起变化。

- 确定截断位置后，在终端输入该 episode 内的秒数，例如 `12.34`，也支持 `00:12.34`。
- 终端显示对应帧号、实际保留到的时间及删除帧数。保留时间小于或等于输入时间的帧；
  输入时间落在两帧之间时，保留前一帧。可以重新输入时间调整选择。
- 输入 `y` 并回车确认裁剪，保存后回到 episode 输入。`c` 并回车跳过当前集，
  `q` / Ctrl-C 退出。不再使用终端空格暂停来选择截断点。

默认写入同级 `<数据集名>_clipped`，可用 `--output` 指定独立目录；原数据不修改。
未裁剪的 episode 保留，episode 编号不变。工具更新 Parquet 行、连续全局 index、
episode 长度/起止索引、视频有效区间和统计量。共享 MP4 保留原编码，被移除的尾段
不再被训练索引引用；不重新编码视频。结果分阶段写入并原子发布，失败保留上次结果。
`meta/clip_history.json` 记录选择；再次用相同源目录和输出目录启动时自动继续已有结果。
Rerun 用于查看时间；实际裁剪以终端输入并确认的时间为准，不自动读取查看器游标。

不加 `--clip` 只回看，不保存；`--camera front` 可只看一路，也支持
`--camera front/left_wrist/right_wrist`。保存始终裁剪整条统一帧记录，包含所有相机。
后续训练应使用 `_clipped` 目录，并重新生成训练所需的归一化统计；本工具不修改
现有模型 checkpoint 或 openpi 的 `norm_stats.json`。

直接检查或导出某集（不会启动 GUI）：

```bash
python scripts/visualize_h5_rerun.py \
  runs/peel_cucumber_25hzcam --episode 0 --check --no-images
python scripts/visualize_h5_rerun.py \
  runs/peel_cucumber_25hzcam --episode 0 --save /tmp/peel_cucumber_0.rrd
```

直接查看命令也接受 `--urdf /path/to/model.urdf`，用于指定离线估算模型。

conda activate gello
cd /home/eid/code/xarm_ws

# 可选：只做配置/标定完整性检查，不连接硬件
python -m gello_teleop.dual_gello_collect \
  -c gello_teleop/config/xarm7_gello_dual_dataset.yaml --check-config

# 正式数采
python -m gello_teleop.dual_gello_collect \
  -c gello_teleop/config/xarm7_gello_dual_dataset.yaml
