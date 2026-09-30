# 单侧选轴标定与双臂全链路采集

新版标定按 `--left` / `--right` 独立连接设备，先记录 GELLO 原点，再在终端输入
轴号选择方向测试。唯一结果文件是 `config/xarm7_gello_calibration.yaml`；数采
配置 `config/teleop/xarm7_gello_dual_dataset.yaml` 直接复用它。命令从仓库根目录执行。

## 1. 标定左臂与右臂

先停止其它遥操/标定进程，用 xArm 控制界面把要标定的从臂摆到新的参考姿态
并停止。托住该侧 GELLO。

```bash
sg dialout -c '/home/eid/miniforge3/envs/gello/bin/python gello_teleop/calibrate_joint_directions.py --left'
sg dialout -c '/home/eid/miniforge3/envs/gello/bin/python gello_teleop/calibrate_joint_directions.py --right'
```

两条命令分别运行；不会连接未选择的另一侧。也可显式添加
`-c gello_teleop/config/teleop/xarm7_gello_dual_dataset.yaml`，默认就是该配置。

1. 按提示托住 GELLO，按 Enter 关闭主手力矩。将 GELLO 手动摆到与 xArm
   当前参考姿态相对应的位置，并打开主手夹爪，保持静止后按 Enter 记录原点。
2. 程序记录当前 xArm 角度作为 `reset_q` 和主手原始角度作为
   `leader_reference_q`。这只改变遥操映射，不写机器人机械零位/Homing Offset。
3. 输入 `1`～`7` 选择轴，例如输入 `1` 选择 J1。将全部主手关节返回记录的
   原点，静止后按 Enter。程序只让 xArm 选中轴以 2°/s 正向运动 15°，核对到位。
4. 观察实际运动，把 GELLO **同一物理关节按相同物理方向**手动转动约 15°，
   其他关节不动；支撑静止后按 Enter。主手只施加配置的阻尼，不会自动试转。
5. 程序用实际编码器增减推断 `joint_signs`，并同步更新 `joint_offsets`。
   GELLO 位移必须约 15°（允许 10～20°），其他主手轴变化不得超过 2°。
   物理方向由操作者提供，程序不能仅从编码器证明你观察的物理方向正确。
6. 每轴成功后立即覆盖保存。xArm 返回试动前位置，GELLO 由人返回原点。
   可按任意顺序或重复选择轴。全部七轴完成后输入 `q` 退出，再标定另一侧。

`q`/Ctrl+C 停止；此前通过的结果已保存。启动和记录原点只更新内存；首个轴成功后才覆盖该侧结果，
保存本次新原点和方向确认进度，另一侧结果保留。尚无成功轴时退出或失败不会修改旧文件。每个轴更新时采用文件锁和原子替换写入。
原点或方向未全部完成时，`CalibrationStatus` 标记未完成，数采入口在连接硬件前
拒绝启动。两侧都完成后可只读显示映射：

```bash
python gello_teleop/calibrate_joint_directions.py --show-mapping
python -m gello_teleop.dual_gello_collect -c gello_teleop/config/teleop/xarm7_gello_dual_dataset.yaml --check-config
```

主手夹爪的打开参考随原点保存，关闭参考暂按配置 `gripper_travel_deg: -42`
计算。它不属于本次七轴方向测量，开始数采前需检查夹爪开合映射。

标定脚本默认读取采集 YAML 的 `gello_damping`，记录 GELLO 原点后启用纯电流阻尼，
在选择轴号和等待 Enter 时持续运行。未写该配置块时沿用模板内的阻尼设置。
标定阶段强制禁用弱保持，按编码器速度计算阻力，不依赖待确认的方向。
模式切换后若原始角度变化超过 1° 则停止并释放；读写失败也停止标定。
`q`、Ctrl+C、异常和正常退出都会尝试关闭主手力矩。纯阻尼不能保证松手不塌。
本次不需要阻尼时，在任一标定命令末尾加 `--no-damping`；不会修改采集配置。

## 2. 双臂、双相机、G1 夹爪采集

```bash
sg dialout -c '/home/eid/miniforge3/envs/gello/bin/python -m gello_teleop.dual_gello_collect -c gello_teleop/config/teleop/xarm7_gello_dual_dataset.yaml'
```

入口读取统一标定结果。设备/相机检查通过后，两台 xArm 自动低速返回 `reset_q`，
无需 Enter 确认复位；启动前留出复位运动空间。复位速度配置在
`alignment.reset_speed_rad_s`，单位 rad/s。xArm 复位使用 mode 0 轨迹接口，
接管时恢复 mode 1 位置流；终端显示每侧使能、下发阶段及实时复位误差/状态。随后两条 GELLO 在电流位置模式（mode 5）下持续设置 `Goal Current=80`，
位置目标由当前姿态到标定参考位置按 `gello_reset_steps` 做线性插值。每步默认等待 0.2 秒；
最后一个目标为标定位置，随后立即解除位置保持、恢复遥操阻尼，
自动采样核对误差；全部通过后接管 xArm，不等待人手微调。
`alignment.gello_hold_current_raw: 80` 是位置模式的 Goal Current，非百分比，
也不是把 EEPROM Current Limit 设置成 80。实际电流由位置误差决定。
默认单轴复位行程不超过 90°，每步跟踪偏差不超过 10°，超过则停止。
复位期间可按 `q` 停止。启动前支撑主手、留出运动空间，并打开主手夹爪。
终端显示两侧 J1..J7 相对对齐目标的误差（度）及自动采样进度。
每轴误差须不超过 `alignment.alignment_tolerance_rad`，
按 `alignment.alignment_samples: 5` 自动采样核对，任一次超限或读取失败立即报错停止接管，
不会重试等待人工调整。所有采样通过后自动接管。
接管前重新检查主从实际对齐；通过后沿用已启动的主手电流模式进行遥操。
接管时先等待左右主手各自连续产生至少两个新样本，才切换 xArm 到 mode 1。
等待 SDK 报告模式切换完成，再检查当前对齐并发送当前位置目标、启动跟随线程。
`control.leader_startup_timeout_s: 2.0` 只控制启动等待时间；运行中的过期保护仍由
`leader_max_age_s: 0.15` 控制。主手启动失败或过期时，报错会显示每侧数据年龄、
读取和整轮 I/O 耗时，以及最后一次通信错误；等待期间每 0.5 秒显示一次状态。
采样启动失败时 xArm 仍保持复位后的规划模式，不会下发跟随目标。
GELLO 阻尼电流使用七轴同步写入，再同步读回各轴 Goal Current 核对，
减少 57600 波特率下逐轴事务占用的时间；写入失败或读回不符会停止跟随。
关节参考限速使用 `control.max_velocity_rad_s`；夹爪单独 20 Hz I/O。
`gripper.control_mode: absolute_position` 将主手开合度直接映射到配置的绝对开口，
使用标定文件的 `gripper_open_deg/gripper_close_deg`；相同开合度对应相同目标开口。
改为 `rate_limited_position` 才会使用 `max_speed_m_s` 逐步靠近目标。
G1 硬件速度分别设置在 `arms.left.gripper_speed` / `arms.right.gripper_speed`，
示例为 5000 r/min，初始化时下发；`-1` 保留原速度。夹爪 I/O 频率设置在
`gripper.sample_rate_hz`（当前 20 Hz，支持不超过 30 Hz）。修改后重启数采。
端点或跟随问题可在第二终端运行
`python gello_teleop/inspect_gello_grippers.py -c gello_teleop/config/teleop/xarm7_gello_dual_dataset.yaml --left --watch`，
只读查看主手 ID8 原始角、闭合比例、目标/已发/实际开口。
初始关闭角由打开角减 42° 推算，须根据实测松开与按到底读数修正对应侧
`TeleoperatorConfig.gripper_open_deg/gripper_close_deg`；已有七轴标定保留。

- `r` 开始录制。
- 空格停止录制并自动保存，遥操状态保持不变。
- `F/f` 让 xArm 与夹爪保持当前位置，GELLO 当前阻尼继续运行。
- `o/O` 让两台 xArm 返回 `reset_q`，GELLO 保持原来的阻尼；必须按 `t` 才能再次接管。
- `t/T` 使用电流位置插值将 GELLO 对齐到 xArm 当前保持姿态，恢复阻尼，自动核对采样通过后接管；失败直接报错。
- `q` / Ctrl+C：停止遥操，双 xArm 返回 `reset_q` 并保留位置保持，再退出。

退出复位沿用 `alignment.reset_speed_rad_s`、`reset_acc_rad_s2`、`reset_timeout_s`、
`reset_tolerance_rad` 和 `reset_samples`，目标为本次启动读取的参考。
到位后保留控制器模式 0 和电机使能，仅关闭 SDK 连接；GELLO 按原有退出逻辑关闭力矩。
复位失败会报错并尝试原地保持，不清除设备故障；再次 Ctrl+C 可取消退出复位。
连接未完成或仅检查配置时不执行退出复位；运行故障退出时直接尝试原地保持。

只有 `r` 和空格启停录制。`F/t/o` 期间独立的 100 Hz 采样线程持续记录，
相机图像也继续写入同一 episode；`teleop_events` 记录操作区间。
切换主手模式时，旧角度样本保留原来的时间戳并标记无效，不伪造新反馈。
对齐或复位期间仍可按 `r` 开始录制、空格停止保存。
所有 `visualize: true` 的相机在一个 `Nero cameras` 窗口显示，视角以配置的 `name` 标注。
`left wrist` 对应 `419122271564`，`right wrist` 对应 `419122270270`；
HDF5 的相机分组使用这两个名称，设备连接仍根据 `serial_number`。

输出为 `episodes/full_pipeline/dual_gello_full_*.h5`。先录制约 10 秒，分别小幅
移动两侧每个关节，并分别开合夹爪。记录应包含 14 轴主从角、成功下发的关节
与夹爪目标/时间、`delta_q`、反馈有效性、两路 RGB/深度及各自时间线。

设备：左 xArm `192.168.1.203` / GELLO `FTBXAQE9`；右 xArm
`192.168.1.196` / GELLO `FTBX14YZ`。两台 D405 按 SDK 序列号绑定和命名：
`419122271564`、`419122270270`，RGB/深度 640×480@30 Hz。
USB by-id 名称可能显示不同编号，不能替代 SDK 序列号。

## 3. 从当前遥操姿态更新双臂参考

使用数采入口的参考位姿模式，在同一个进程中遥操和保存：

```bash
python -m gello_teleop.dual_gello_collect \
  -c gello_teleop/config/teleop/xarm7_gello_dual_dataset.yaml --reset-q
```

程序照常执行复位、对齐、遥操 pipeline。遥操到希望的位置后直接按 `s/S`，
取四条臂当前最新实际反馈，后台原子覆盖统一标定中的左右 `reset_q`、
`start_joints`、`leader_reference_q` 和 `joint_offsets`，保留已确认方向与夹爪端点。
无需 `F`、Enter、退出或另开脚本，也不要求连续静止采样。
保存完成打印四组 J1..J7 的 `q_rad` 与目标文件。可继续遥操、采集和重复按 `s` 覆盖；
`s` 不改变录制状态，录制时保存事件记入 `teleop_events`。
实际复位目标来自 `gello_config` 指向的标定文件，无需在采集 YAML 中另加 `reset_q`。
反馈过期、设备不符、对齐/复位中或文件锁冲突时直接报错且不覆盖。

保存后当前遥操继续沿用本次启动的映射；下一次启动数采时，两台 xArm 与
GELLO 将使用最新保存的成对姿态复位并自动对齐，无需重新测关节方向。
独立 `capture_reference_pose.py` 仍可用于多次静止采样或 `--dry-run` 只读检查，
在线保存不再需要它。

## 软件检查与结果清理

旧 `_calibrated`、`zero_auto*`、已测局部范围及方向复核/范围规划结果共 13 个
已删除。硬件配置模板、URDF/STL、测试夹具和原始只读/数采数据保留。
程序不会从已删除文件恢复旧映射。尚未产生新的真机标定文件，需要按上述步骤采样。

无机械臂动作的软件端到端检查会在临时目录生成明确标记的合成映射，不依赖
任何真机标定结果：

```bash
python scripts/smoke_dual_gello_pipeline.py --duration 2 --output /tmp/dual_gello_mock.h5
python scripts/smoke_dual_gello_pipeline.py --real-cameras --duration 3 --output /tmp/dual_gello_real_cameras.h5
```

第二条只使用真实相机，机械臂、GELLO 和夹爪仍是模拟设备。记录标记
`synthetic_robots: true`，不等于真机运动/夹爪验收。数采 100 Hz 是调度目标，
必须根据真机文件的时间戳和过期/溢出统计评估实际吞吐。

## 单臂遥操与夹爪按下打印

`dual_gello_collect` 支持顶层 `active_arms: [left]` 或 `[right]`，只连接、复位、
对齐、控制和记录所选侧。未填写时默认 `[left, right]`；双臂记录始终按 left、right
排列。当前 `xarm7_gello_dual_dataset.yaml` 已设置为 `[left]`。
未启用侧的 `arms` 和标定条目可以保留或移除；保存参考姿态只更新所选侧。
单臂 HDF5 关节数据为 7 列，夹爪为 1 列，侧别写在 `arm_names` 中。
相机独立按 `cameras` 配置启用；不需要右腕相机时在该相机条目设置 `enabled: false`。

遥操完成接管后，主手夹爪每次按下自动打印一行原始角度、闭合比例、映射开口、
已发开口、实际开口及跟随状态；无需另开查看进程。启动时请松开主手夹爪，
首次反馈作为松开参考。原始角度偏离参考至少 2° 触发一次，回到参考附近 1° 内
重新允许触发；持续按住不会刷屏，按 F 保持时仍可触发。
检测直接使用原始角度，可用于排查错误端点；ID8 原始读数不变化时无法检测按下。
设 `gripper.print_on_press: false` 可关闭。需要连续观察或实测全闭端点时仍可使用
`inspect_gello_grippers --watch`，默认只显示当前启用的侧。

## 夹爪 trigger / follow 模式

在采集配置的 `gripper` 下设置 `mode: follow`（默认）或 `mode: trigger`，重启生效。
`follow` 使用已标定开合度映射到从臂开口；`control_mode` 可选绝对位置或限速位置。
`trigger` 不在接管时改变夹爪位置，首次按到底发送全闭，松开后第二次按到底发送全开，
此后交替切换；每次切换只发送一次指令。目标取所选侧的 `gripper_min_width_m` /
`gripper_max_width_m`，运动速度使用 `gripper_speed`，不使用 `max_speed_m_s`。
触发依据标定后的闭合比例：达到 95% 视为按到底，回到 20% 以下允许再次触发。
松开只解除触发锁定，绝不下发张开指令；持续按住也不重复切换。
启动时需先松开主手，避免在接管时立即执行；trigger 模式的状态打印也只在按到底时触发。
保持期间的按下不会改变切换状态，恢复后需松开再按下；已有状态打印继续生效。
夹爪模式写入 episode 元数据的 `gripper_mode`。
