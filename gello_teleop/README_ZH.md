# Gello 遥操作系统

使用 Gello 主手对 UFACTORY xArm 机械臂进行关节空间遥操作控制。

GitHub: https://github.com/xArm-Developer/ufactory_teleop

本文档根据 `gello_teleop/uf_robot_gello_teleop.py` 编写，说明如何使用 Gello/Dynamixel 主手和 UFACTORY xArm 机械臂进行关节遥操作。

## 1. 简介

入口脚本通过 Dynamixel 串口读取 Gello 主手关节位置，使用 `JointMapper` 将主手关节映射为机器人目标关节角，并通过 `UFRobot.send_action()` 发送给机械臂执行。

主要组件：

- `gello_teleop/uf_robot_gello_teleop.py`：Gello 遥操作入口脚本。
- `gello_teleop/config/*.yaml`：xArm5、xArm6、xArm7 示例配置文件。
- `ufactory_devices/robot/uf_robot.py`：xArm 连接、初始化、关节运动和夹爪控制封装。

支持能力：

- 支持 xArm5、xArm6、xArm7 关节空间遥操作。
- 启动时自动读取 Gello 当前姿态并计算关节偏置。
- 支持通过 `joint_ids`、`joint_signs` 配置 Gello 关节映射。
- 支持可选 Gello 夹爪关节，夹爪动作作为 action 最后一维发送。
- 支持将不参与映射的 Dynamixel 关节配置到 `torque_joint_ids`，以位置模式固定其当前位置。
- 机器人侧统一使用 `send_action` 接口，Gello 入口要求 `robot_mode: 6`。

## 2. 环境与硬件要求

推荐环境：

- Ubuntu 20.04、Ubuntu 22.04 或 Ubuntu 24.04。
- Python 3.8、3.9 或 3.10。
- UFACTORY xArm 机械臂。
- Gello 主手。
- Dynamixel USB 串口适配器。
- 可选：Gello 侧 Dynamixel 夹爪关节。
- 可选：xArm Gripper、xArm Gripper G2 或其他 `UFRobot` 支持的机器人侧夹爪。

运行前请确认：

- 控制电脑可以访问 xArm 控制器 IP。
- Gello Dynamixel 串口设备可以被当前用户访问。
- xArm 处于可使能状态，急停已释放。

## 3. 安装

克隆项目并进入 Gello 遥操作目录：

```bash
git clone https://github.com/xArm-Developer/ufactory_teleop
cd ufactory_teleop/gello_teleop
```

创建并激活虚拟环境：

```bash
conda create --name py39 python=3.9
conda activate py39
```

安装 Python 依赖：

```bash
pip install -r requirements.txt
pip install pysurvive agx-pypika --no-deps
```

安装后需要保证以下导入可以成功：

```bash
python -c "from gello.dynamixel.driver import DynamixelDriver; from gello.agents.gello_agent import GelloAgent;from xarm.wrapper import XArmAPI"
```

配置串口权限：

```bash
sudo usermod -aG dialout $USER
```

执行后建议重新登录系统，或重新插拔 Gello Dynamixel USB 串口适配器。

## 4. 配置文件

示例配置文件：

```bash
gello_teleop/config/xarm7_gello_teleop.yaml
gello_teleop/config/xarm6_gello_teleop.yaml
gello_teleop/config/xarm5_gello_teleop.yaml
```

xArm7 示例：

```yaml
RobotConfig:
  robot_ip: "192.168.1.29"
  robot_mode: 6
  robot_speed: 90
  robot_acc: 500
  gripper_type: 1
  start_joints: [0, 0, 0, 1.5708, 0, 1.5708, 0]

TeleoperatorConfig:
  port: "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAJZYC7-if00-port0"
  start_joints: [0, 0, 0, 1.5708, 0, 1.5708, 0]
  gripper_id: 8
```

### RobotConfig

| 参数 | 说明 |
| --- | --- |
| `robot_ip` | 机械臂控制器 IP 地址。 |
| `robot_mode` | 机器人运动模式。Gello 入口脚本必须使用 `6`，表示关节控制模式。 |
| `robot_speed` | 关节速度，单位为 deg/s。 |
| `robot_acc` | 关节加速度，单位为 deg/s^2。 |
| `gripper_type` | 机器人侧夹爪类型。`0` 无夹爪，`1` xArm Gripper，`2` xArm Gripper G2，`10` Pika Gripper，`11` Robotiq Gripper。 |
| `gripper_port` | Pika Gripper 串口，仅 `gripper_type: 10` 时使用。 |
| `gripper_speed` | 夹爪速度，`-1` 表示使用默认值。 |
| `gripper_force` | 夹爪力控参数，`-1` 表示使用默认值。 |
| `start_joints` | 启动时机械臂先移动到的关节角，单位 rad，长度需要与机器人轴数一致。 |
| `start_tcp_pose` | 可选。到达 `start_joints` 后再移动到的 TCP 位姿，格式为 `[x, y, z, roll, pitch, yaw]`，位置单位 mm，姿态单位 rad。 |

### TeleoperatorConfig

| 参数 | 说明 |
| --- | --- |
| `fps` | 控制循环频率，默认 `30`。 |
| `port` | Gello Dynamixel 串口路径，例如 `/dev/serial/by-id/...`。 |
| `joint_ids` | 参与机器人关节映射的 Gello Dynamixel ID。xArm7 未填写时默认 `[1, 2, 3, 4, 5, 6, 7]`。 |
| `joint_signs` | 每个 Gello 关节的方向符号，长度必须与 `joint_ids` 一致。xArm7 未填写时默认全为 `1`。 |
| `start_joints` | Gello 启动参考姿态对应的机器人关节角，单位 rad，长度必须与 `joint_ids` 一致。 |
| `gripper_id` | Gello 夹爪 Dynamixel ID。设置为 `-1` 表示不使用 Gello 夹爪。 |
| `torque_joint_ids` | 不参与映射、需要固定在当前位置的额外 Dynamixel ID；要求位置模式 (3)。这不是重力补偿，也不能与 `joint_ids` 重复。 |

## 5. Gello 姿态对齐

主从关节采用逐轴线性映射，按 `joint_ids` 的顺序对应 xArm 的 J1…Jn：

```python
q_gello_ordered[i] = encoder_angle_of(joint_ids[i])
q_robot[i] = joint_signs[i] * (q_gello_ordered[i] - joint_offsets[i])
joint_offsets[i] = leader_reference_q[i] - reset_q[i] / joint_signs[i]
```

不需要主手 URDF。先逐轴确认哪个 Dynamixel ID 对应哪一个 xArm 关节，
按 xArm J1…Jn 的顺序填写 `joint_ids`；再用小角度试动判断正负方向，最后在
双方对齐的参考姿态记录编码器零点。URDF 本身不能确定实际接线的电机 ID 或编码器零点。
双臂运行会读取保存的 `joint_signs`、`joint_offsets`、`leader_reference_q`，
不会每次开机重新计算零点。首次用未标定的双臂配置运行并完成两侧人工对齐后，
程序会在同目录自动生成 `*_calibrated.yaml`；下次继续用原始 `--config`，
程序会自动加载该文件中的映射。端口、电机 ID、机器人 IP 或参考姿态不匹配时
会在运动前拒绝复用。其它参数（如 `hold_pwm_by_joint`）仍以原始配置为准。
可先不连接硬件检查当前是否会加载保存的标定：

```bash
python gello_teleop/uf_robot_gello_teleop_dual.py \
  --config gello_teleop/config/xarm7_gello_teleop_dual.yaml --check-config
```

注意事项：

- `RobotConfig.reset_q` 是机器人启动参考姿态。
- `TeleoperatorConfig.leader_reference_q` 是 Gello 对应的编码器参考角。
- 更换电机、齿轮安装角、接线、关节顺序或机器人参考姿态后需要重新标定。
- xArm5/xArm6 示例中，部分 Gello 物理关节不参与映射，会通过 `torque_joint_ids` 固定。

## 6. 运行方法

从仓库根目录运行：

```bash
cd ~/code/xarm_ws
# xArm7
python gello_teleop/uf_robot_gello_teleop.py --config gello_teleop/config/xarm7_gello_teleop.yaml

# xArm6
python gello_teleop/uf_robot_gello_teleop.py --config gello_teleop/config/xarm6_gello_teleop.yaml

# xArm5
python gello_teleop/uf_robot_gello_teleop.py --config gello_teleop/config/xarm5_gello_teleop.yaml
```

### 双臂逐轴标定和复用

双臂配置使用 `left` 和 `right` 两个段。首次保存方式取决于现有映射：

如果现有 `joint_ids`、`joint_signs` 已能正确一对一跟随，只需从仓库根目录
运行一次双臂入口并完成两侧人工对齐，程序会自动保存参考零点：

```bash
python gello_teleop/uf_robot_gello_teleop_dual.py \
  --config gello_teleop/config/xarm7_gello_teleop_dual.yaml
```

重新标定并用于双臂数采时，分别运行：

```bash
python gello_teleop/calibrate_joint_directions.py --left
python gello_teleop/calibrate_joint_directions.py --right
```

每次只连接所选侧。先用控制界面摆好 xArm 参考姿态，再手动把 GELLO 摆成
对应原点并打开主手夹爪。记录原点后，在终端输入 `1`～`7` 选择轴；xArm
该轴正向运动 15°，操作者让 GELLO 同轴按同一物理方向转动约 15°，静止后
按 Enter。程序根据原始角度增减推断方向，并同步重算偏置、覆盖保存。
GELLO 记录原点后复用采集配置中的纯电流阻尼；`--no-damping` 可关闭。
每轴试动后 xArm 返回起点，GELLO 由人返回原点。
可任意顺序、重复选择轴；输入 `q` 退出。重跑某侧先在内存中记录新原点，首个轴成功后才覆盖该侧结果并保存本次
方向确认进度；尚无成功轴时旧文件保持不变，另一侧已保存结果保留。

唯一结果文件为 `config/xarm7_gello_calibration.yaml`。原点或任一方向未确认时，
数采入口会拒绝连接硬件。两侧均完成后，直接运行：

```bash
python -m gello_teleop.dual_gello_collect \
  -c gello_teleop/config/xarm7_gello_dual_dataset.yaml
```

该配置直接读取上述结果，不会重新估计零点；启用双臂、两台 D405 和 G1 夹爪。
只读查看保存的映射：

```bash
python gello_teleop/calibrate_joint_directions.py --show-mapping
```

详细流程见 [RECALIBRATION_AND_PIPELINE_ZH.md](RECALIBRATION_AND_PIPELINE_ZH.md)。
旧 `_calibrated`、`zero_auto*` 和已测范围结果已清理；历史零点/保持工具为独立入口。

### Gello 保持力不足时

`hold_pwm_by_joint` 是 Dynamixel 位置模式的 **Goal PWM 输出上限**，不是
持续施加的重力补偿力矩；填写时它会覆盖统一的 `hold_pwm`。它不能超过电机
EEPROM 中的 PWM Limit，也不能超过电机的持续力矩和温度能力。停止其他占用
GELLO 串口的程序后，可只读查看型号、工作模式、PWM 上限与硬件错误：

```bash
python gello_teleop/diagnose_gello_pwm.py \
  --config gello_teleop/config/xarm7_gello_teleop_dual.yaml --side left
```

某个 ID 超时（`-3001`，未收到状态包）时，脚本仍会读取后面的 ID，最后以非零
状态退出。先检查该电机的供电、串接线、ID 和波特率；遥操程序同样要求所有关节
都能读到。此脚本独占串口，不能与遥操程序同时运行。

脚本不使能或关闭电机，也不下发目标。断电或扭矩关闭后 `present PWM` 常为零，
此时只能用它检查 EEPROM 上限、配置值和故障状态。确认所有 ID 通讯正常后，
可在正常双臂启动命令后加 `--hold-diagnostics`，程序在主手保持参考姿态后、
开始遥操前，通过同一个串口只读打印每个电机的 `torque`、`goal`、`present` PWM
和位置误差。此命令会照常连接并复位两台 xArm，必须按正常启动流程确认安全。
看完状态可按 Ctrl+C 退出。

```bash
python gello_teleop/uf_robot_gello_teleop_dual.py \
  --config gello_teleop/config/xarm7_gello_teleop_dual.yaml --hold-diagnostics
```

保持时若 J2 的 `goal` 不是配置的 600，需检查加载的配置和启动路径。若
`goal=600` 而 `present` 的绝对值接近 600 且仍下垂，输出可能已达上限；
需检查电机持续力矩、负载与机械配重。若输出远低于上限却无法保持，应再检查
位置环参数、工作模式和供电。仅提高 `hold_pwm_by_joint` 不会持续施加更大力矩。

### 交互测试 GELLO J4 的保持输出

如果 J4 在位置保持时下垂，可以使用 `tune_gello_j4_pwm.py` 逐步测试
`Goal PWM` 上限。这个值不是 Nm 力矩；脚本让 J4 保持当前单圈位置，按 `d`
每次增加一个小步长，终端实时显示位置误差、实际 PWM、电流和温度，按 `y`
接受当前值，按 `q` 退出。脚本默认将温度限制在 50°C、电流限制在 0.6A，
并且不会连接或移动 xArm：

```bash
python gello_teleop/tune_gello_j4_pwm.py \
  --config gello_teleop/config/xarm7_gello_calibration.yaml \
  --side left \
  --output gello_teleop/config/xarm7_gello_teleop_dual_left_j4_tuned.yaml \
  --step 25
```

请托住主手、停止其它 GELLO 程序，并为输出文件选择新路径。脚本只操作
`joint_ids` 中第 4 个电机；最大值取 `min(EEPROM PWM Limit, 885)`，不会修改
EEPROM。确认后，接受的值会写入输出 YAML 的
`TeleoperatorConfig.hold_pwm_by_joint[3]`，后续使用该输出配置时会在自动回位
和对齐的位置保持阶段使用它。遥操正式跟随阶段会释放映射关节力矩，所以这个
设置不会把 J4 变成 Gello 到 xArm 的力反馈通道。

### MIT 力矩计算后下发位置

每侧 `TeleoperatorConfig.control_mode` 可设为 `mit_to_position`。该模式仍向
xArm 下发关节位置：先以实测 `q`、相邻两次反馈估计的 `dq` 和 GELLO 目标
计算 `tau = kp * (q_des - q) - kd * dq + gravity(q)`，再用匹配的动力学模型
求 `ddq = M(q)^-1 * (tau - h(q,dq))`。控制器先积分更新虚拟关节速度
`dq_cmd += ddq * dt`，再积分更新位置 `q_cmd += dq_cmd * dt`，并将
`q_cmd` 作为目标发送。积分步长最多为一帧；它不代表 xArm 实际输出了 `tau`。

启用前，为左右侧分别填写以下字段（示例值请换成各自实际参数）：

```yaml
TeleoperatorConfig:
  control_mode: mit_to_position
  dynamics_urdf: ../models/xarm7_dynamics.urdf
  dynamics_joint_names: [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
  dynamics_locked_joint_names: []
  mit_kp: [0, 0, 0, 0, 0, 0, 0]
  mit_kd: [0, 0, 0, 0, 0, 0, 0]
  mit_torque_limit_nm: [1, 1, 1, 1, 1, 1, 1]
  mit_acceleration_limit: 1.0
  mit_tracking_error_limit: 0.15
```

上面的零增益和力矩限值仅展示字段格式，不能当作可运行参数。仓库附带的
`models/xarm7_dynamics.urdf` 使用 UFACTORY `xarm_ros2` 官方 xArm7
`xarm7_type7_HT_BR2` 连杆参数，只保留关节、连杆和惯量，不包含已安装的夹爪或
其他末端负载。真机使用前应补齐实际负载的动力学参数。`dynamics_joint_names`
必须与 SDK 关节顺序一致。
若 URDF 含可动夹爪关节，填入 `dynamics_locked_joint_names` 以得到七轴模型。
配置缺失、关节顺序不符或复位姿态的重力超出力矩限值时，程序会在连接机器人前
报错。还需根据模型和真机反馈标定增益、力矩限值与控制频率后才能启用。

## 7. 安全注意事项

- 启动脚本前确认机械臂工作空间无障碍物。
- 机械臂在第一个 Enter 提示前只连接并检查；确认后才移动到 `RobotConfig.reset_q`。
- 首次运行建议降低 `robot_speed` 和 `robot_acc`。
- 确认 Gello 主手已摆放到 `TeleoperatorConfig.start_joints` 对应姿态后再启动脚本。
- 按 Enter 进入遥操作前，再次确认机器人和 Gello 都处于安全姿态。
- 调试时安排人员看守急停。
- 如果不使用 Gello 夹爪，请设置 `gripper_id: -1`。
- 如果不使用机器人侧夹爪，请设置 `gripper_type: 0`。
- 如果机器人返回非 0 错误码，控制循环会退出。

## 8. 常见问题

### 报错 `Gello teleop requires robot_mode=6 joint control mode`

Gello 入口脚本只支持关节控制模式。请将配置改为：

```yaml
RobotConfig:
  robot_mode: 6
```

### 报错 `joint_signs and joint_ids length mismatch`

`TeleoperatorConfig.joint_signs` 和 `TeleoperatorConfig.joint_ids` 的长度不一致。请检查两个列表的元素数量。

### 报错 `start_joints and joint_ids length mismatch`

`TeleoperatorConfig.start_joints` 和 `TeleoperatorConfig.joint_ids` 的长度不一致。请确认每个参与映射的 Gello 关节都有一个对应的参考关节角。

### 报错 `Joint action length must be ...`

`GelloAgent` 返回的 action 维度与脚本期望的 `_action_dim` 不一致。请检查 `joint_ids`、`gripper_id` 和 Gello 硬件配置。

### 无法打开 Gello 串口

请检查：

- `TeleoperatorConfig.port` 是否正确。
- USB 串口适配器是否插入。
- 当前用户是否有串口权限。
- 是否有其他进程占用了该串口。

### 按 Enter 后机器人不运动

请检查：

- xArm 控制器 IP 是否正确。
- 机器人是否处于错误或急停状态。
- UFACTORY SDK 是否可以单独控制机械臂。
- Gello 是否返回有效 action。

### 一次零点标定，以后 GELLO 自动回起始位置

首次只手动定义参考姿态，不再让 GELLO 自动逐轴试转：

```bash
conda activate gello
python gello_teleop/calibrate_zero.py \
  --config gello_teleop/config/xarm7_gello_teleop_dual.yaml \
  --output gello_teleop/config/xarm7_gello_zero_auto.yaml
```

先通过从臂控制界面把两台 xArm 放到安全、容易对应的起始姿态并停止。
脚本不移动、不使能 xArm；连接后确认释放 GELLO 力矩，再手动将主手摆到
对应姿态、打开夹爪并托住。按 Enter 采集稳定角度，输入 y 确认该侧。
左右完成后才写入新文件；输出文件已存在会在连接硬件前拒绝覆盖。
这就是软件定义的零点：不需要把编码器数值写成 0，也不要求从臂所有关节为 0。

映射为 `q_xarm = q_xarm_ref + joint_signs * (q_gello - q_gello_ref)`。
沿用输入文件的 joint_ids 和 joint_signs；当前双臂配置 J2/J6 为 -1。
一个参考姿态只能确定偏移，不能自动识别关节顺序、方向或机械装配错误。
脚本将累计圈数转换为位置模式的单圈坐标，并同步计算偏移，不让电机追赶 530° 等目标。

以后每次启动使用新文件：

```bash
python gello_teleop/uf_robot_gello_teleop_dual.py \
  --config gello_teleop/config/xarm7_gello_zero_auto.yaml
```

确认复位路径后，xArm 先回记录姿态；两个 GELLO 依次低速自动回位并保持。
无需每次手动精确对齐或重新标定。打开主手夹爪、托住主手，确认开始后
主手释放力矩，从臂开始跟随。配置默认如下（TeleoperatorConfig）：

```yaml
leader_passive: false
leader_reset_speed_deg: 5.0
leader_reset_timeout: 30.0
leader_reset_max_travel_deg: 90.0
```

回位按单圈坐标内的直线路径逐步下发目标，不选择跨编码器边界的绕圈捷径。
若所需行程超过每轴 90°、跟踪偏差过大或超时，程序停止，不会继续硬拉线缆。
这时需要检查机械限制或托住主手靠近起始姿态；自动回位不等于避障规划或重力补偿。
更换电机、舵盘或安装方向后需重标。

可选：标定命令加 `--manual-start` 可生成 `leader_passive: true` 的手动回位配置。
仅此模式在每次启动时需要大致摆回参考姿态，默认每轴允许偏差
`passive_start_tolerance_deg: 10.0`，开始后从臂限速追赶，保存零点保持不变。

自动回位期间终端用左右两行原地更新诊断（最多约 4 Hz，完成/失败时额外刷新）。
各数组按 J1..J7 排列：Edeg 为实际位置减当前指令，Rdeg 为实际位置减最终参考，
I_mA 为 XL330 实际电流，PWM 为实际输出。电流/PWM 都不是实测 Nm 力矩。
宽终端同时显示四组数据，窄终端在相同两行内轮换四组数据；重定向输出只保留最终两行。

## 双臂遥操数采入口

人工新参考姿态、方向重标定，以及双臂＋G1 夹爪＋RealSense RGB/深度全链路
验收流程见 [RECALIBRATION_AND_PIPELINE_ZH.md](RECALIBRATION_AND_PIPELINE_ZH.md)。

新的入口是 `dual_gello_collect.py`，配置示例为
`config/xarm7_gello_dual_dataset.yaml`：

```bash
python -m gello_teleop.dual_gello_collect \
  -c gello_teleop/config/xarm7_gello_dual_dataset.yaml
```

该配置引用已经保存 `joint_signs`、`joint_offsets` 和
`leader_reference_q` 的 calibrated GELLO 文件。没有保存标定时入口会在连接前拒绝，
不会用当前摆放姿态重新计算 offset。实际顺序固定为：连接检查、两台 xArm 低速到
`reset_q`、GELLO 电流位置插值对齐、恢复阻尼并自动采样核对、全部通过后接管。
取消人手微调等待：`alignment_samples` 指定核对次数，任一次误差超过
`alignment_tolerance_rad` 或采样失败都立即报错停止接管。
终端 `r` 开始新的 episode，Enter 或空格结束本集采样并提示输入 `y/n`：
`y` 保存、`n` 丢弃当前 episode，大小写均可。等待确认时遥操继续，
输入 `y/n` 即生效，无需再次按 Enter；`F/o/t/q` 仍可使用。
保存日志包含 episode 长度（秒）、样本数和绝对保存路径。
停止后由传输线程分块发送已冻结的 episode，H5 数组整理、写入和工作副本释放在
独立进程执行，减少保存期间与控制线程争用 Python GIL。每块最多 1024 行或
256 KiB（单张图像超过该大小时按一张发送），不会一次序列化整个 episode。
实机连接前预启动保存进程，日志打印其 PID；H5 元数据中的 `writer_process_id` 和
`collector_process_id` 可以核对写入与采集确实在不同进程。
终端、遥操和相机预览继续运行，可按 `r` 录制下一集；保存队列按顺序写入并
预留不同 episode 编号。退出时先停止并关闭设备，
再等待已提交的保存任务完成。保存失败会报出目标路径并保留失败任务的内存数据。
当前未录制时按 Enter 或空格也会打印提示。键盘操作需在启动程序的终端进行。
相机按 `fps: 30` 采集，RGB 和对齐深度在采集进程内按 `output_size: [224, 224]`
缩放后保存；深度使用最近邻缩放并保留 uint16 原始深度值。
`output.camera_compression: null` 直接写 H5 原始数组，不做 gzip 压缩或视频编码。
保存按小批次写入，日志报告每路相机的帧数、进度和最终写入耗时。
关节数据按 `control.sample_rate_hz: 100` 保存；相机使用独立的 30 Hz 时间戳，
不复制图像以匹配关节数据行数。预览尺寸可独立由 `preview_output_size` 设置。
`control.deadline_policy: skip` 让运动控制线程在延迟后跳过过期时刻，
按主机单调时钟使用最新有效主手目标和从臂状态继续下发，不补发历史指令。
平滑位置参考使用真实经过的时间更新，仍受已有步长、速度和加速度限制。
采集配置启用 `skip`；省略该设置或设为 `strict` 时，控制延迟超过
`control.maximum_lateness_s` 仍会停止。`skip` 模式下该值作为警告阈值。
H5 元数据中的 `control_late_tick_count`、`control_skipped_tick_count` 和
`control_max_lateness_s` 记录运动控制调度间隙。
这项设置不处理串口坏包或回包超时；主手/从臂反馈过期、GELLO 读写失败或
xArm 控制器故障仍会停止。时间戳用于主机调度及记录，接口不向 xArm 下发未来执行时间。
独立录制线程超过该阈值时打印警告，跳过过期时刻后继续采样，不补造样本。
每行的采样延迟及 H5 元数据中的 `sampling_late_tick_count`、
`sampling_skipped_tick_count`、`sampling_max_lateness_s` 可用于检查会话内的调度间隙。
`F/f` 让 xArm 和夹爪保持当前位置，GELLO 阻尼继续运行；`o/O` 让 xArm 按复位速度
返回 `reset_q`，随后等待 `t/T`。`t` 按启动时的电流位置插值流程将 GELLO 对齐到
xArm 当前保持姿态，恢复阻尼并执行同样的自动采样核对，通过后接管，无需 Enter 或人手微调。
`F/t/o` 不启停录制：执行这些操作时，已开始的 episode 持续记录状态与图像，
操作时间保存在 `teleop_events`；对齐期间 GELLO 数据的有效性单独标记。

`q` 和 Ctrl+C 都先停止遥操，再让两台 xArm 返回本次启动读取的 `reset_q`，
使用 `alignment.reset_speed_rad_s`、`reset_acc_rad_s2`、`reset_timeout_s`、
`reset_tolerance_rad` 和 `reset_samples`。实际关节反馈连续到位、控制器停止运动后，
保留模式 0 的位置保持并断开连接，不调用 `motion_enable(False)`。
退出复位超时、反馈无效或设备故障会报错，并尝试保持当前位置；不自动清除故障。
退出复位期间再次按 Ctrl+C 可取消回位，仍尝试保持当前位置后关闭连接。
异常退出直接尝试原地保持；连接尚未完成或仅 `--check-config` 时不复位。

碰撞保护灵敏度可在数采配置中调整：

```yaml
control:
  collision_sensitivity: 3
arms:
  left:
    collision_sensitivity: 3  # 可选：覆盖公共值，右臂同理
```

支持整数 0～5：0 关闭碰撞检测，1 最不敏感、5 最敏感；`null` 保留控制器当前设置。
[官方建议启用检测时不要低于 3](https://docs.ufactory.cc/user_manual/ufactoryStudio/7.settings.html)。
下次启动、使能机械臂时下发；只检查配置或连接设备不会改变它，不写入控制器持久配置。
该值是控制器碰撞检测等级，不能直接换算成固定的 N/Nm 上限。
设为 0 时使能日志会提示碰撞检测已关闭；力矩图的低通滤波不影响控制器的碰撞检测。
Bio Gripper G2 的 TCP 负载质量、质心和机械臂安装方向仍需与实机一致，设置不准确会误报。
当前仍使用位置伺服；降低灵敏度不能保证持续接触，也不提供恒力或柔顺控制。
触发 C31 时停止发送运动指令，日志同时报告侧别、控制器 C31 和 SDK 返回码，
不自动清错、复位或重新使能；处理故障后需重新启动并对齐接管。

数采配置默认开启 **URDF 外力矩 L1 范数和七轴曲线图**。
每个启用臂显示一列八张曲线：上方为 `tau_ext_l1 = ||tau_ext||₁`，
下方依次为 J1～J7 的 `tau_ext`，单位均为 Nm。
`active_arms: [left, right]` 时左右臂并排，单臂配置只显示所选臂的一列；
`--left/--right/--both` 覆盖配置后，窗口也跟随实际选择。
七轴曲线复用实际用于 GELLO 反馈的滑动均值残差，范数为这些值的绝对值之和，
取值位于阈值、增益和电流限幅之前。窗口未满、暂停或数据无效/过期时显示断线和 `-- Nm`。
`force_feedback.source: checkpoint` 保留原有七轴网络 / URDF 残差对比布局。
窗口在连接检查时启动，启动复位和 GELLO 对齐期间也更新，
未录制以及 `F/t/o` 期间持续运行；关窗或窗口内按 q/Esc 只关闭绘图。
GUI 在独立进程，通过有界非阻塞队列接收状态和已完成的反馈计算结果，
队列满时丢弃绘图样本，不阻塞控制或 H5 采样。图中不额外标注模型参数或近似说明。

可用 `--left`、`--right`、`--both` 覆盖 YAML 的 `active_arms`，例如：

```bash
python -m gello_teleop.dual_gello_collect \
  -c gello_teleop/config/xarm7_gello_dual_dataset.yaml --left
```

只连接所选侧的 xArm 和 GELLO。`active_arms` 和 `--left/--right/--both` 不影响相机：
所有相机独立按 `cameras` 配置连接并保存，`enabled: false` 关闭对应相机，
`visualize: true/false` 控制该相机预览。名称 `left wrist`、`right wrist` 和
兼容字段 `arm: left/right` 都不会按机械臂选择过滤相机。
GELLO 底层串口断连会报出侧别、串口路径、电机 ID 和操作，停止当次采集。

遇到 `code=-3001`（`COMM_RX_TIMEOUT`），表示当前读取没有收到完整回包；
`code=-3002`（`COMM_RX_CORRUPT`）可能是 CRC 错误，也可能是收到
不完整回包后超时，不能仅凭 ID 判断电机损坏。停止数采及其他 GELLO 程序后运行：

```bash
python -m gello_teleop.diagnose_gello_serial \
  -c gello_teleop/config/xarm7_gello_dual_dataset.yaml --side right \
  --samples 600 --rate-hz 10 --output /tmp/gello_right_serial.json
```

该工具只读位置和 Goal Current，不使能力矩、不写目标、不连接 xArm。
输出耗时分位数、跳过的采样时刻、USB 延迟、失败 ID/寄存器和坏包原始字节，
区分无回包、不完整包与完整坏包；`--position-only` 可单独测试位置读取。
无负载只读测试通过不能排除启用电流后的供电压降、接插件接触不良或干扰。
若复发，优先检查报错 ID 及其前后级连接、电源和 USB 线缆。

57600 baud、8 个位置反馈和 7 个电流写入/回读，单轮仅线路传输至少约
50.2 ms，另有电机回包延迟和 USB/线程调度开销，20 Hz 无通信余量。
当前采集配置的 `gello_damping.sample_rate_hz` 为 10 Hz，并覆盖标定文件的
`fps`；右侧通过 `gello_damping.sides.right.sample_rate_hz` 设为 12 Hz，
用于 USB `latency_timer=1` 的总线。xArm 控制和录制仍为 100 Hz。
主手一轮读写超过周期时，下一轮立即读取新的位置并重新安排周期，避免额外空等
一个周期；不会积压或补发过去的请求。控制和录制线程仍跳过过期时刻。
`leader_sample_status()` 的 `skipped_ticks` 显示跳过次数，失败时的
`read_ms` 显示最近一次位置读取耗时，`io_ms` 显示最近一次结束的整轮耗时；
`io_stage` 和 `in_flight_ms` 显示当前阶段及本轮已用时间。
坏包仍触发原有停止和释放阻尼流程。

有效的位置读完后立即供控制和夹爪线程使用，保持读取时的原始时间戳；
阻尼电流写入与回读随后完成。写入或回读失败仍立即停止并撤销当前主手目标，
启动接管仍要求连续完成读写校验。`leader_max_age_s: 0.15` 保持不变。
`read_ms` 与 `io_ms` 在读取进行中可能来自不同轮次，应结合 I/O 阶段判断。
启用独立相机预览时，采集进程只接收保存尺寸的图像和深度，原始预览图直接
送往预览进程，减少图像传输与反序列化对采样的影响。

GELLO 串口在没有数据时最多等待 1 ms，让出 CPU，已有字节立即读取；
SDK 回包超时使用单调时钟，保留完整包、CRC、电机错误和 Goal Current 回读校验。
Linux 的 USB `latency_timer=1` 减少驱动缓冲延迟；SDK 内的超时裕量不等于
每次读取的固定等待时间，不能仅因 USB 已设为 1 ms 就压缩回包期限。
xArm 状态优先从同一次 `get_joint_states` 回复取得 q、dq 和所选 effort，
然后读取末端位姿，将每轮状态查询由 3 次减少到 2 次；旧 SDK 或无效位置回复
仍回退到角度查询。遥操夹爪指令同时设置 `wait=False` 和 `wait_motion=False`，
避免额外等待机械臂动作占用共享通讯锁。

URDF 反馈调用 Pinocchio `rnea(q, dq, ddq_filtered)`，计算
`tau_model = M(q)ddq_filtered + C(q,dq)dq + g(q)`。
单位为 rad、rad/s、rad/s² 和 Nm。加速度来自采集状态 `ddq`（xArm 适配器对速度反馈差分），
先按 `acceleration_cutoff_hz` 做一阶低通，再送入 RNEA；RNEA 输出不再做力矩低通，
也不再通过动量观测器计算。静止时输出重力力矩 `g(q)`。
算法说明见 [Pinocchio 官方 RNEA 文档](https://github.com/stack-of-tasks/pinocchio/blob/devel/doc/a-features/g-dynamic.md)。

`tau_ext_cal` 使用一阶低通后的 SDK 实测力矩减去上述 RNEA 力矩，
随后逐轴取滑动均值得到 `tau_ext`，图中展示这七轴值和对应的 L1 范数。
URDF 模式的滤波参数和均值窗口使用 `force_feedback` 配置，绘图不重复计算 RNEA 或滤波。
实测力矩截止频率为 `measured_torque_cutoff_hz`。
两项截止频率独立，默认均为 3 Hz（时间常数约 53 ms），支持标量或 J1..J7 七元素数组，
可在 `sides.left/right` 分别覆盖。滤波使用实际反馈时间间隔 `dt`：
`alpha = 1-exp(-2*pi*cutoff_hz*dt)`，`filtered = previous+alpha*(raw-previous)`。
首个有效样本直接初始化，避免从零开始的过渡；重复样本不更新滤波状态。
反馈缺失、无效或间隔超过有效期时显示断线并重置计算状态。
位置、速度、加速度或实测力矩不可用时，范数显示 `-- Nm`。
不能将空图上的零轴视为计算结果。示例配置的 `arms.<side>.feedback_signal: torque`
明确选择 SDK 力矩报告字段，而非电流。URDF 残差同时用于主手反馈和范数绘图。

切换为 `source: checkpoint` 时，`tau_ext_pred` 复用 `force_feedback` 的异步预测，
不在绘图进程重复加载网络；另一列独立计算 URDF 残差。
实测力矩来自同一次推理的 `tau_measured`，其滤波方式由
`force_feedback.measured_tau_filter` 决定；当前 `checkpoint` 对应训练目标的 10 Hz 低通。
网络列按推理源状态时间戳绘制，不用最新实测值减去较早的预测。
它显示阈值判断、增益、电流限幅之前的完整残差，可用于观察空载误差和接触信号。
网络关闭、历史不足、暂停、结果无效或过期时显示断线和 `-- Nm`；URDF 列独立更新。

首次使用需在数采 Python 环境安装 `matplotlib>=3.7` 和 `pin>=3,<4`，并使用可用的 GUI 后端。
无桌面运行时加 `--no-torque-plot`；也可用 `--torque-plot` 覆盖配置开启。
配置示例（相对路径以数采 YAML 所在目录为基准）：

```yaml
torque_visualization:
  enabled: true
  urdf_path: ../models/xarm7_dynamics.urdf
  measured_torque_cutoff_hz: 3.0  # 实测力矩一阶低通，Hz
  acceleration_cutoff_hz: 3.0     # ddq 一阶低通，Hz
  window_s: 30.0
  update_hz: 10.0
  stale_s: 0.15
  payload:
    source: controller  # 只读 SDK rich report 中的 TCP 负载
  sides:
    left:
      gravity_m_s2: [0, 0, -9.81]  # 各臂基座坐标系中的重力
    right:
      gravity_m_s2: [0, 0, -9.81]
```

内置 `xarm7_dynamics.urdf` 使用官方 `xarm7_type7_HT_BR2` 惯量 YAML，七个运动连杆
总质量为 10.4315 kg。源文件固定到
[xarm_ros2 62936f7 的参数](https://github.com/xArm-Developer/xarm_ros2/blob/62936f7ea1846a85f7350de2c4c18f39e6d19715/xarm_description/config/link_inertial/xarm7_type7_HT_BR2.yaml)，
副本为 `models/xarm7_type7_HT_BR2.yaml`；旧版 2.76914 kg 连杆模型保留为
`models/xarm7_dynamics_legacy.urdf`，不再用于默认计算。这是官方当前默认的参数版本，
不同实机版本仍可能需要替换对应 URDF，不能通过整体乘一个系数校正。

Bio Gripper G2 官方质量为含转接板 0.79 kg，见
[官方规格](https://docs.accessories.ufactory.cc/Bio_Gripper_G2/6.technical_specifications.html)。
数采默认从 SDK `tcp_load` 属性读取 Studio 已设置的负载质量和法兰坐标系质心，
将 mm 转为 m，再加到七轴模型末端；不发送 `set_tcp_load`，也不修改机器人设置。
在 Studio 中选择官方 Bio-G2 预设即可，见
[官方 TCP 设置说明](https://docs.accessories.ufactory.cc/Bio_Gripper_G2/3.control.html)。
SDK 未报告正质量时，模型仅包含裸臂。负载变化时重置观测器，避免切换造成力矩尖峰。
未提供夹爪自身的转动惯量时，末端按质点建模：计入重力和质心平动惯性，不包含夹爪绕质心
的转动惯性。官方通用 BIO Xacro 的 G2 分支共用旧惯量，其质量总和约 0.4125 kg，
与 G2 含转接板质量不符，因此没有将该惯量当作准确的 G2 参数直接使用。

也可在 `sides.left/right.payload` 显式设置 `mass_kg`、`com_m`（法兰坐标系）和可选的
`inertia_kg_m2`（质心处 3×3 矩阵），或用包含完整固定末端惯量的 URDF 并关闭附加 payload，
避免重复计入。活动关节必须保持 `joint1..joint7` 顺序。
模型不包含摩擦、电机偏置或未知末端相机负载。
`torque_visualization` 的两项滤波用于网络模式下的独立 URDF 对比列。
当前 URDF 模式的曲线复用反馈结果；H5 外力矩结果只保存 `<side>_tau_ext_l1_xarm`，
读取失败时以 NaN 和有效性标记保存。
离线验证布局可使用模拟运动，不连接硬件：

```bash
python -m xarm_stack.torque_visualization --demo --headless \
  --duration 3 --save-plot /tmp/dual_torque.png
```

加 `--left` 或 `--right` 可预览单臂布局。

双臂七轴电流可在另一个终端实时查看：

```bash
python -m xarm_stack.plot_dual_joint_current \
  -c gello_teleop/config/xarm7_gello_dual_dataset.yaml
```

一个窗口内七行两列：左列为左 xArm J1～J7，右列为右 xArm J1～J7，纵轴单位 A。
脚本从同一配置及其 `gello_config` 读取两侧 IP，仅接收 TCP 30002 详细状态报文，
使用独立的电流字段，不切换数采使用的力矩/电流报告选择器，不操作使能、模式或复位。
因此可在遥操与录制期间运行。断连或反馈超过 `--stale-s` 时显示缺口和状态，不用零替代。
报文频率通常为 5 Hz，增加窗口刷新频率不会增加反馈频率；字段定义见
[UFACTORY 官方 TCP 报文说明](https://docs.supportarticle.ufactory.cc/support_articles/developer/firmware/data-description-of-tcp-port.html)。
默认显示最近 30 秒；可加 `--window-s 10 --update-hz 10`，`--ylim 1.5` 固定全部纵轴为 ±1.5 A。
窗口按 `q`/Esc、关闭窗口或终端 Ctrl+C 退出；退出只关闭接收连接。
首次使用需要 `python -m pip install "matplotlib>=3.7"`。
`--demo` 使用模拟电流检查布局；`--demo --headless --duration 5 --save-plot /tmp/dual_currents.png`
可在无桌面环境保存示例图，`--check-config` 只检查两侧 IP。

`cameras` 中的 `visualize: true` 视角在一个 `Nero cameras` 窗口里横向拼接并标注
配置的 `name`；预览在独立进程运行，复位、保持和未录制时也持续显示。
`left wrist` 对应序列号 `419122271564`，`right wrist` 对应 `419122270270`。
HDF5 中的相机分组也使用这些名称，`serial_number` 仍用于选择设备。
当前 OpenCV 没有 GUI 支持时，预览自动使用 Tk，无需替换 OpenCV。
夹爪默认 `gripper.control_mode: absolute_position`：主手按标定的打开/关闭角度计算
闭合比例，直接映射到 xArm 的绝对开口；保持相同主手开合度不会累计闭合。
需要对目标开口限速时可改为 `rate_limited_position`，此时使用 `max_speed_m_s`。

夹爪硬件速度在数采配置的 `arms.left.gripper_speed` 和 `arms.right.gripper_speed`
分别设置；示例为 `5000`，单位是 G1 SDK 的电机 r/min，不是开口 m/s。
数采初始化夹爪时调用 `set_gripper_speed`，失败会直接报错；省略、`null` 或 `-1`
保留控制器原速度。修改后重启数采生效。`gripper.sample_rate_hz` 控制夹爪目标/反馈
循环频率（当前 20 Hz，上限 30 Hz），不能替代硬件速度设置。
绝对映射模式下修改 `max_speed_m_s` 不影响跟随速度；关节的 `kp/kd` 也不控制夹爪。
速度接口及单位见 [官方 SDK 文档](https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/doc/api/xarm_api.md#set_gripper_speed)，
5000 的示例见 [G1 控制示例](https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/example/wrapper/common/5004-set_gripper.py)。

若按压某侧主手夹爪没有反应，先检查实际开合角。保持新版数采运行，在另一终端执行：

```bash
python gello_teleop/inspect_gello_grippers.py -c gello_teleop/config/xarm7_gello_dual_dataset.yaml --left --watch
```

该工具复用数采缓存，不额外打开串口，不写标定或机器人；显示 ID8 原始角度、
闭合百分比、映射开口、最近成功下发的开口及实际反馈。`--right` 检查右侧，
省略左右参数显示两侧，`--json` 查看完整数据。修改代码前已启动的数采进程须先重启。
可在数采按 `F` 保持从臂与夹爪，再手动松开/按压 GELLO，读取两个端点；
此时显示“保持/未下发”，映射值会变化，实际夹爪保持不动。

初始七轴标定只记录主手夹爪打开角，关闭角曾按 `gripper_travel_deg: -42` 推算，
不保证适用于实际机构。松开和按到底的两个读数分别填写统一标定文件
`xarm7_gello_calibration.yaml` 对应侧的 `TeleoperatorConfig.gripper_open_deg` 和
`gripper_close_deg`；两个显式端点优先于 `gripper_travel_deg`。正负方向都支持，
应使用实测角度；不要因“左侧”而直接猜测符号。松开应为 0% 闭合，按到底应为 100%。
若原始角度变化而闭合百分比始终为 0，端点/方向不匹配；若原始角度完全不变，
检查主手 ID8 及机构；若目标与下发开口变化而实际反馈不变化，检查 xArm 夹爪执行端。
保存端点后重启数采；已有七轴原点/方向无需重标。

### 将当前遥操位姿保存为下一次启动参考

直接使用数采遥操入口的参考位姿模式：

```bash
python -m gello_teleop.dual_gello_collect \
  -c gello_teleop/config/xarm7_gello_dual_dataset.yaml --reset-q
```

程序先执行原来的完整遥操 pipeline。遥操到希望的初始位姿后，在同一个终端
按 `s/S`，立即读取左/右 xArm 与 GELLO 四条臂最新的七轴实际反馈，后台覆盖
保存 `gello_config` 指向的统一标定文件。无需按 `F`、Enter，不等待连续静止
采样，也不需要退出遥操或启动另一个脚本。保存后打印四组 `q_rad` 和文件路径，
可以继续遥操，或到另一个位置再次按 `s` 覆盖；上一份仍在写入时会提示等待其完成。

两侧 `RobotConfig.reset_q`、`TeleoperatorConfig.start_joints`、`leader_reference_q`
和 `joint_offsets` 一次性原子覆盖。偏移按 `raw - xarm_q / joint_signs` 重算，
保留原有方向核对结果、关节方向和夹爪端点。主从反馈由独立线程采集，保存的是
按键时缓存内最新的实际角度及各自时间戳；单次快照不等于四设备硬件同步采样。
反馈无效/过期、复位对齐中、设备方向不符或文件被其它进程写入时，报错且不覆盖。

`s` 不停止遥操，不改变阻尼、夹爪、相机预览或录制状态；后台写文件时仍处理
其它按键、相机和采样队列。录制中保存参考的时刻记入 `teleop_events`。
当前进程继续使用原来的映射与复位目标，避免运行中改变关节目标；下一次启动
数采时，两台 xArm 返回新 `reset_q`，GELLO 插值到新记录的位置并自动对齐接管。
没有 `--reset-q` 时 `s` 不写文件，会提示添加该参数。

独立的 `capture_reference_pose.py` 保留为额外检查工具；`--dry-run` 只打印。
它要求遥操进程仍在运行，并做多次静止采样；日常在线保存使用上述 `--reset-q` 与 `s`。

xArm 控制线程使用 monotonic 100 Hz tick；GELLO 按各自 `fps` 在独立线程读取，控制
线程只使用最近有效目标的 ZOH。`position_mode: direct` 是带步长和跟踪误差上限的
位置参考；`second_order` 使用独立参考状态，单位为 rad、rad/s、rad/s²，公式为
`ddq_ref = clip(kp*(q_target-q_ref)-kd*dq_ref)`，再按真实 dt 积分并限幅。`kp/kd`
是参考轨迹参数，不是电机刚度或 MIT 力矩参数。

GELLO/xArm 的 HDF5 每臂字段统一使用 `侧别_量_设备`，例如
`left_q_xarm`、`left_q_gello`、`right_q_xarm`、`right_q_gello`。
七轴数据为 `(N,7)`，不拼成 `(N,14)`；设备名始终放在最后。
新采集的 H5 固定包含左右臂各五个必需字段：`q_xarm`、`dq_xarm`、`tau_xarm`、
`q_cmd_xarm`、`tau_ext_l1_xarm`。关节量为 `(N,7)`，范数为 `(N,1)`。
同时保存左臂 `left_q_eepose_xarm` 和右臂 `right_eepose_xarm`，名称保留上述写法。
两者均为 `(N,4,4)` 基座到 TCP 的齐次变换矩阵：左上角 `[:3,:3]` 为旋转矩阵，
`[:3,3]` 为平移，单位米。位姿直接来自现有 xArm 状态，不增加 SDK 通信请求。
单臂运行时，另一侧必需字段保存 NaN，标记 `placeholder=true` 并引用全零有效性字段；
`arm_names` / `recorded_arm_names` 只列出实际采集的臂，`schema_arm_names` 列出左右两侧。
外力矩反馈关闭时，范数字段同样保留为 NaN 并标记无效。
下表用右臂示例；除单独列出的末端位姿名称外，左臂将 `right_` 换成 `left_`。
所有字段均位于 `teleop` 组。

| 数据 | H5 字段 |
| --- | --- |
| xArm 实测关节角、速度、加速度 | `right_q_xarm`、`right_dq_xarm`、`right_ddq_xarm` |
| xArm 原始加速度 | `right_ddq_raw_xarm` |
| GELLO 映射角、原始角 | `right_q_gello`、`right_q_raw_gello` |
| xArm 实测力矩、电流 | `right_tau_xarm`、`right_current_xarm` |
| xArm 末端位姿（左、右） | `left_q_eepose_xarm`、`right_eepose_xarm` |
| xArm 末端位姿有效性（左、右） | `left_eepose_valid_xarm`、`right_eepose_valid_xarm` |
| xArm 关节目标、跟踪误差 | `right_q_cmd_xarm`、`right_delta_q_xarm` |
| GELLO 反馈电流、总目标电流 | `right_current_feedback_gello`、`right_current_cmd_gello` |
| 夹爪实测、目标、主手闭合比例 | `right_gripper_xarm`、`right_gripper_cmd_xarm`、`right_gripper_fraction_gello` |
| 关节有效性、反馈时间戳、数据年龄 | `right_q_valid_xarm`、`right_q_timestamp_us_xarm`、`right_q_age_us_xarm` |
| GELLO 有效性、采样序号 | `right_q_valid_gello`、`right_q_sequence_gello` |
| 外力矩反馈基准、同源实测力矩 | `right_tau_urdf_xarm`、`right_tau_feedback_measured_xarm` |
| 外力矩七轴 L1 范数 | `right_tau_ext_l1_xarm` |

各臂有效性、数据年龄、序号、命令时间和夹爪字段为 `(N,1)`；
设备名始终放在最后，便于按侧别、量和设备筛选。
关节目标统一写作 `left_q_cmd_xarm` / `right_q_cmd_xarm`。
末端位姿缺失、非有限数值或来源状态过期时，对应 `eepose_valid_xarm` 为 0；
未启用的一侧保留 NaN 位姿。`required_eepose_datasets` 列出两个位姿字段，
`source_timestamp_path` 指向已保存的状态采集时间戳。读取端兼容旧 `ee_pose_xarm` 名称。
公共采样时间保存在 `teleop/timestamp_us`，公共采样延迟和模型调度字段不加臂前缀。
相机保存在
`cameras/<name>/frames` 和独立 `timestamp_us`，不会复制到 100 Hz。GELLO 过期、
控制异常或两侧任一命令失败时，两个从臂停止接收新目标。
配置 `gripper.enabled: true` 后会由各自状态线程读取夹爪宽度；未启用或 SDK 不提供
夹爪反馈时保存 NaN 和 `<side>_gripper_valid_xarm=0`，不会用零值冒充状态。

旧 H5 的 `q_follower` / `q_leader_mapped` 或带侧别的旧字段仍可读取。
已有文件可用迁移脚本统一名称，先预览更名计划，再执行原地转换：

```bash
python scripts/migrate_h5_device_schema.py runs/earae_board_1003
python scripts/migrate_h5_device_schema.py runs/earae_board_1003 --in-place
```

转换只更正文件内已有字段的名称及引用，例如旧右臂范数
`right_tau_ext_l1` 更正为 `right_tau_ext_l1_xarm`，保留数值、形状、压缩设置、时间戳及相机图像。
历史单右臂文件继续只保留实际右臂字段。读取端兼容旧范数名称。
迁移记录保存在文件的 `metadata` 中，可用同一脚本的 `--in-place --undo` 恢复旧名称。

### URDF 外力矩反馈与滑动均值

`config/xarm7_gello_dual_dataset.yaml` 当前使用 `force_feedback.source: urdf`，
以 xArm 的实测状态和 `models/xarm7_dynamics.urdf` 计算反馈。
保持原有采集命令和 `r/s/f/t/o/q` 操作；`enabled: false` 可关闭外力矩反馈。
URDF 模式使用 Pinocchio，不加载神经网络权重或使用 CUDA，关闭力矩窗口也能反馈。

```text
tau_m = lowpass(measured torque)
tau_urdf = RNEA(q, dq, lowpass(ddq))
tau_ext_cal = tau_m - tau_urdf
tau_ext = moving_mean(tau_ext_cal, tau_ext_mean_window)
tau_ext_l1 = sum(abs(tau_ext[J1..J7]))  # H5 外力矩结果只保存这一标量；可视化同时显示七轴 tau_ext
feedback = gain_raw_per_nm * tau_ext * sign   if abs(tau_ext) > threshold_nm
feedback = 0                                otherwise
GELLO 电流 = 限幅(阻尼电流 + joint_signs * 限幅(feedback))
```

`q`、`dq`、`ddq` 来自 xArm 实测状态，单位为 rad、rad/s、rad/s²；
RNEA 包含惯性、科氏/离心与重力项，结果为 Nm。
实测力矩和加速度各使用独立的一阶低通，当前截止频率均为 3 Hz。
模型在机械臂使能前加载，计算通过有界队列和独立线程完成，不阻塞控制或串口线程。
`payload.source: controller` 复用 SDK 缓存中的 Studio 负载质量和法兰系质心，
不增加硬件查询；负载缓存尚不可用时释放反馈。也可设置静态 `mass_kg`、`com_m`
及可选 `inertia_kg_m2`，或用空 `payload` 使用无附加负载的 URDF。

滑动窗口按不同时间戳的有效 xArm 状态计数，不重复计入 100 Hz 采样线程重用的状态。
将 `force_feedback.tau_ext_mean_window` 改成正整数即可调整，
`1` 表示直接使用当前残差。收集满窗口后才启用反馈。
窗口越大，曲线越平滑，同时会增加反馈延迟。
暂停、无效/过期数据、采样间断、时间倒退或负载变化时清空窗口，
恢复后重新收集满窗口；低于阈值时立即释放反馈，保留正常阻尼。

| 配置量 | 作用 |
| --- | --- |
| `source` | `urdf` 使用刚体动力学；`checkpoint` 保留原有网络模式 |
| `urdf_path` / `gravity_m_s2` | 七轴 URDF 与基座系重力向量；相对路径以配置文件目录为准 |
| `measured_torque_cutoff_hz` / `acceleration_cutoff_hz` | 实测力矩和 RNEA 输入加速度的截止频率 |
| `tau_ext_mean_window` | 残差滑动均值窗口，正整数，按不同有效状态计数 |
| `payload` | 控制器缓存负载或静态法兰系负载 |
| `threshold_nm` | 对平均后的残差取绝对值判断，超过后反馈完整残差 |
| `gain_raw_per_nm` / `sign` | 反馈强度与方向，另行应用 GELLO 标定 `joint_signs` |
| `current_limit_percent` | XL330 最大原始电流 1750 的百分比，范围 0～100 |
| `ramp_s` / `rate_limit_raw_s` | 有效反馈恢复时的渐入和电流上升变化率 |
| `maximum_age_s` / `maximum_sample_gap_s` | 源状态有效期及清空窗口的状态间隔上限 |

J1..J7 数组可逐轴设置；在 `force_feedback.sides.left` / `right` 中可覆盖模型、
负载、滤波、窗口、阈值及电流参数。`source` 为全局选择。
电流反馈与阻尼相加后仍受 `gello_damping.damping_current_limit` 限制。
旧配置的整数 `current_limit_raw` 继续兼容，同一配置层不可同时填写两种上限。

H5 的外力矩结果只保存平均后残差的七轴 L1 范数，字段固定为
`teleop/left_tau_ext_l1_xarm` 和 `teleop/right_tau_ext_l1_xarm`，均为 `(N,1)`。
新文件固定包含这两个范数字段，未采集的一侧保存 NaN 并标记无效。
不写入 `tau_ext`、`tau_ext_cal` 或 `tau_ext_raw` 七轴外力矩数组。
有效零力矩保存 0，窗口未满、暂停或结果失效时保存 NaN；有效性和源时间戳
分别引用同侧 `<side>_tau_free_valid_xarm`、`<side>_tau_free_source_timestamp_us_xarm`。
原有 `<side>_tau_urdf_xarm`、同源 `<side>_tau_feedback_measured_xarm`、
关节状态、相机和 GELLO 电流字段继续保存。
`<side>_tau_free_pred_xarm` 是所选基准力矩的兼容字段，在 URDF 模式等于
`<side>_tau_urdf_xarm`；`tau_free_*` 时间戳和有效性也按相同格式保存。
元数据记录 URDF 来源、范数阶数、七轴数量、滤波和均值窗口。
计算顺序是逐轴滑动均值后取 L1 范数；先取范数再求均值会得到不同结果。
可视化使用同一份残差和源状态时间戳，只展示每臂一条范数曲线。

如需原网络模式，设置 `source: checkpoint`、`checkpoint_path`、`device`，
以及 `measured_tau_filter: raw` 或 `checkpoint`；网络模式仍按权重恢复
q/dq/delta_q 输入、归一化、滤波和 50 帧历史。URDF 模式不等待网络历史。
URDF 残差也包含未建模摩擦、模型和力矩偏差，应结合现有阈值与实机状态调参。

### GELLO 阻尼

采集配置 `xarm7_gello_dual_dataset.yaml` 的 `gello_damping` 集中设置主手阻尼，
其参数覆盖 `gello_config` 内同名手感设置，省略整个配置块时沿用标定文件。
`enabled: false` 关闭阻尼；`sides.left` / `sides.right` 可分别覆盖公共设置。
重新标定不会改动采集配置。标定时记录原点前主手被动，记录后启用纯阻尼；
采集时在接管阶段启用阻尼。标定阶段强制关闭弱保持，使用编码器坐标，
不依赖尚未确认的关节方向。两侧标定命令保持不变，加 `--no-damping` 可临时关闭。

当前配置启用纯电流阻尼，七轴低强度初始增益 8、附加制动增益 8、电流上限 15，
速度阈值 0.5 rad/s。所有电流参数均为舵机原始单位，不是 Nm；尚未实机调参。
主手读写设为 15 Hz，从臂控制仍为 100 Hz。电流指令按标定符号转换回编码器方向，
使阻力反向于主手运动。加大 `damping_gain` 增加常规阻力，
`damping_brake_gain` 增加超过阈值后的阻力；阻尼不是硬限速。
`damping_current_limit` 控制每轴电流上限，不能据此推断输出力矩。

`weak_hold_enabled` 默认关闭：纯阻尼在零速度时不抗重力，不能保证松手不塌。
弱保持需要另外配置正的 `weak_hold_gain` 并调试，不能替代重力补偿。
接管时检查电机型号，只允许已知控制表启用电流模式，目标电流先清零再使能。
主手读写失败时结束跟随并尝试立即释放阻尼；暂停/退出也关闭主手力矩。
总线 watchdog 为 500 ms，用于通信中断；首轮请支撑主手确认各轴阻力方向。
控制表参考：[ROBOTIS XL330-M077 手册](https://emanual.robotis.com/docs/en/dxl/x/xl330-m077/)。

## 独立 xArm 关节电流测试

该工具不依赖 GELLO 或相机：

```bash
python -m xarm_stack.test_joint_current \
  -c xarm_stack/config/xarm_joint_current_test.yaml \
  --arm both --duration 30 --output ./current_test.h5
```

默认只连接和读取，不复位、不运动、不改变位置环；需要固定当前位置时明确加
`--hold`（同时把配置中的 `execution_enabled` 改为 `true`）。运行中按 `b`、`p`、`l`
标记 baseline、press、release，`q` 退出。工具调用 SDK 的
`set_report_tau_or_i(1)` 后只把 `get_joint_states()` 的 `effort` 字段记录为电流；
退出时按 `restore_feedback_signal` 恢复报告选择。电流不可用时保存 NaN 和有效性标志，
不会用零值冒充测量，也不会把同一字段同时标成 torque。输出包含 q、dq、current、
torque、时间戳、阶段和错误状态，并打印相对 baseline 的有符号变化和噪声。

### 双臂采集启动对齐

使用 `xarm7_gello_dual_dataset.yaml` 时，设备检查通过后 xArm 自动低速返回
保存的 `reset_q`。随后 GELLO 使用 mode 5，持续设置 `Goal Current=80` 做位置保持，
目标位置按配置次数插值从当前姿态到标定位置。`alignment.gello_hold_current_raw`
控制 Goal Current，`gello_reset_steps` 控制插值次数，`gello_reset_interval_s`
控制每步等待时间（默认 0.2 秒）。80 是原始电流值，不是百分比或固定实测力矩；
位置环按误差决定实际电流。插值结束后立即解除位置保持并恢复主手阻尼，自动采样核对误差。
每次采样都必须满足误差阈值；读取失败或任一次超限立即报错，不等待人手微调。
全部采样通过后接管 xArm，此时不再重新切换主手电机模式。
终端底行实时显示左/右 J1..J7 相对标定原点的误差（度），各轴须满足
`alignment.alignment_tolerance_rad`，自动核对 `alignment_samples` 次，任一次失败即报错；
全部通过后接管，`q` 退出。`t` 重新执行 GELLO 对齐到 xArm 当前保持姿态的流程。
