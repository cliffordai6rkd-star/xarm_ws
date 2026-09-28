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

只有关节顺序或方向尚未确定时，才需要逐轴标定：

```bash
python gello_teleop/calibrate_joint_directions.py \
  --config gello_teleop/config/xarm7_gello_teleop_dual.yaml \
  --output gello_teleop/config/xarm7_gello_teleop_dual_calibrated.yaml
```

脚本会让两台 xArm 回到各自 `RobotConfig.reset_q`，记录 GELLO 参考位置，
逐轴让 xArm 与 GELLO 正向试动；先观察是否为同一物理关节，再输入 `y`（同向）
或 `n`（反向）。若不是同一关节，输入 `x` 并调整 `joint_ids` 顺序后重试。
脚本把 `joint_signs`、`joint_offsets`、`leader_reference_q` 保存到新的配置文件，
原始文件不会被覆盖。

也可以直接使用保存的配置启动双臂遥操：

```bash
python gello_teleop/uf_robot_gello_teleop_dual.py \
  --config gello_teleop/config/xarm7_gello_teleop_dual_calibrated.yaml
```

启动时两台 xArm 会先复位到 `reset_q`，两个 GELLO 会自动回到保存的
`leader_reference_q` 并保持力矩；开始遥操前请托住主手。按 Enter 后程序释放
主手映射关节和夹爪的力矩，再进入双臂遥操；只有 `torque_joint_ids` 指定的
非映射固定关节继续保持。`hold_pwm_by_joint` 仅用于复位/对齐保持阶段。
以后每次开机继续传原始配置，程序会自动加载同目录的 `_calibrated.yaml`，
无需再运行标定脚本。传入保存的文件本身也可用，但那样其它参数应直接修改
保存的文件。可不连接硬件查看当前保存的映射：

```bash
python gello_teleop/calibrate_joint_directions.py \
  --config gello_teleop/config/xarm7_gello_teleop_dual_calibrated.yaml --show-mapping
```

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
`models/xarm7_dynamics.urdf` 从 UFACTORY `xarm_ros2` 官方 xArm7 Xacro 生成，
只保留关节、连杆和惯量，不包含已安装的夹爪或其他末端负载。因此它可用于
软件验证，真机使用前应补齐实际负载的动力学参数。`dynamics_joint_names`
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
