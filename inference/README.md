# xArm / Nero 模型推理

在线推理统一通过 `python -m inference.cli --config <配置文件> --run` 启动。
入口在 `inference/interactive.py` 中统一管理键盘和按键含义，DP、ACT、π0、π0-WM 和 Nero
运行时共用：默认等待 `s` 开始连续推理，`i` 停止推理并复位，复位后继续等待 `s`。
`p` 暂停，`c` 开始连续推理，`q` / Ctrl-C 退出；复位和退出的硬件动作由各运行时执行。
`--enable-command` 开启机械臂命令；只读模式下 `i` 仅清空推理状态，不移动机械臂。
`--auto-start` 可跳过启动等待，`--single-step` 则让每次 `s` 只执行一段动作后暂停。

## DP → latent WM：独立异步、只读观测

新配置 `inference/configs/dp_wm_xarm.yaml` 使用削黄瓜 DP 权重和 50000 步 latent WM。
DP 目标 25 Hz 输出当前＋未来 8 步计划，WM 根据 `execute_steps` 规划更新周期，
仅保存 q/tau/接触预测，不接运动控制。`--check` 只读取 CPU 元数据；
`--run --mock --auto-start --duration 5` 可验证调度而不连接硬件或加载模型。
真实运行用 `--run`，按 s 开始；该分支不接受 `--enable-command`，i 只暂停。
输入契约、控制器参考角与训练 q_cmd 的区别、依赖和输出文件见
[DP-WM 文档](dp_wm/README.md)。

## xArm：远端 π0 输出关节 q

本地通过 `pi0.action_type` 选择动作适配：`joint` 直接将七轴绝对关节角交给
现有执行器，`eepose` 先用本地 Pinocchio IK 转为七轴关节目标。两者共用公共入口、
相机、观测对齐、动作分段和 `s/i/p/q` 按键。WM 保持关闭。

当前关节模型使用 `inference/configs/pi0_wm_xarm.yaml`，其中：

```yaml
pi0:
  action_type: joint
  interface:
    action_semantic: action.joint
    coordinate_frame: null
    representation: joint_positions
    units: rad
    joint_names: [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
```

EE-pose 模型的完整示例见 `inference/configs/pi0_eepose_xarm.yaml`。它接收基座坐标系下
`xyz [m] + quaternion` 的七维绝对位姿，支持 `xyzw` 和 `wxyz`，使用同一双路图像及零状态输入：

```yaml
pi0:
  action_type: eepose
  interface:
    action_semantic: action.ee_pose
    coordinate_frame: link7
    reference_frame: base
    representation: xyz_quaternion
    quaternion_order: xyzw
    units: m
ik:
  urdf_path: ../../gello_teleop/models/xarm7_dynamics.urdf
  frame_name: link7
  max_iterations: 100
  position_tolerance_m: 0.0001
  rotation_tolerance_rad: 0.001
```

EE-pose 示例的 `training_config` 须换成实际训练 EE-pose 的配置名称，两端完整的
`pi0.interface` 必须一致。`action_type` 只选择本地解释和执行分支，不会改变模型的训练目标。
当前已审计的 `pi0_xarm_lora_erase_board` 应使用 `joint`。

首个 IK 目标以这次对齐观测中的实测 q 为种子，后续目标以刚解出的 q 为种子。
求解器约束 URDF/执行配置的关节限位，校验单位四元数，并拒绝不可达或不收敛的整段动作；
成功后再由现有关节执行器处理步长、速度和加速度。joint 分支不构建 IK 模型，也不处理四元数。

若动作描述 TCP，但 URDF 只包含 `link_eef`，显式填写 URDF 帧到 TCP 的固定偏移，
数值必须来自实际工具标定：

```yaml
# pi0.interface.coordinate_frame: tcp
ik:
  frame_name: link_eef
  frame_offset_xyz_rpy: [0.0, 0.0, 0.08, 0.0, 0.0, 0.0]  # 示例：m / rad；替换成实际值
```

目标先按 `T_base_URDF = T_base_TCP @ inv(T_URDF_TCP)` 转换再求解。
模型帧与动作帧不同而未提供偏移时会拒绝启动。

无硬件验证 EE-pose 分支（服务端、相机、机械臂均 mock；IK 使用真实 URDF）：

```bash
python -m inference.cli --config inference/configs/pi0_eepose_xarm.yaml --check
python -m inference.cli --config inference/configs/pi0_eepose_xarm.yaml \
  --backend mock --mock-policy --enable-command --auto-start --duration 2
```

`inference/configs/pi0_wm_xarm.yaml` 已改为 `policy.type: pi0`，接入 DP/ACT 共用的
关节执行器。文件名保留，但当前是 π0 → 绝对关节角 → xArm SDK，`wm.enabled: false`。
它复用 `hardware_config` 中的机械臂 IP、双路相机、关节限位和标定复位位置，
不需要 ZMQ bridge、WM checkpoint 或 GELLO 主手连接；当前控制右臂。

训练配置 `pi0_xarm_lora_erase_board` 使用 `ZeroObservationState` + `LiberoInputs`。
客户端发送下面的**字面量字段名**（`/` 不表示嵌套字典），不重复归一化：

| 数据/硬件输入 | WebSocket 字段 | LiberoInputs 内部位置 |
| --- | --- | --- |
| left_wrist，224×224 RGB uint8 | observation/image | base_0_rgb |
| right_wrist，224×224 RGB uint8 | observation/wrist_image | left_wrist_0_rgb |
| 7 维 float32 零向量 | observation/state | state；服务端仍保留 ZeroObservationState |
| 任务文本 | prompt | prompt |

服务端使用实际训练配置和 `create_trained_policy()` 恢复 checkpoint 的 normalization 和
输入/输出变换；它默认接收 repack **之后**的输入，所以客户端直接发送上表字段。
`LiberoOutputs` 截取前 7 维，客户端要求收到有限 `[50,7]` 物理动作，消费前 20 步后滚动更新。
动作时间步是 1/30 秒，100 Hz 控制循环按经过的时间选取动作，不把 100/30 取整。
输出不做四元数归一化、IK 或额外反归一化。

**当前训练动作映射已核实**：服务器对 `pi0_xarm_lora_erase_board` 实际训练数据的审计
确认 29,605 行 `action` 与 `action.joint` 完全一致，最大差为零，无 `action.ee_pose` 列。
结合原始 xArm 数据及已提供的 `ZeroObservationState`、`LiberoInputs`/`LiberoOutputs`
变换，当前输出按 J1～J7 的绝对关节角（rad）解释。一次录制双路图像的真实远程调用
返回有限 `[50,7]`，预热后该次服务端推理约 90 ms。此结论只对应本次审计的训练数据和模型。
服务端 YAML 须使用与本地相同的完整 `pi0.interface`：`state_semantic: zero`、
`representation: joint_positions`、`coordinate_frame: null`、`units: rad` 和七轴名称。
删除旧的 `quaternion_order`；更新文件后重启服务端，metadata 才会更新。
仅修改部署 YAML 不会改变模型训练的动作语义。

使用 `scripts/check_pi0_action_mapping.py` 在实际服务器环境核对注册的 TrainConfig、
LeRobot 缓存路径、RepackTransform 和全部训练 Parquet 行。脚本逐行比较训练动作列与
`action.joint`、`action.ee_pose`，只在唯一候选全部完全一致时报告 `joint` 或 `eepose`；
缺少来源列或存在歧义时报告 `unverified`。它同时列出输入/输出变换，不连接硬件。

```bash
# 本地上传审计脚本，SSH 端口/主机使用实际 AutoDL 登录信息
scp -P 47338 scripts/check_pi0_action_mapping.py \
  root@connect.weste.seetacloud.com:/root/autodl-tmp/openpi/scripts/
# 在服务器 openpi 项目根目录运行
uv run scripts/check_pi0_action_mapping.py --train-config pi0_xarm_lora_erase_board
```

本地原始数据也可独立核对，但不能代替实际服务器转换后数据的检查：

```bash
python scripts/check_pi0_action_mapping.py \
  --dataset-root runs/erase_board_lerobotv3 --action-key action.joint
```

首次远程推理可能触发 JAX 编译和 cuDNN 算法选择。客户端关闭自动 ping 超时机制，
由 `pi0.request_timeout_s` 限制连接和每次接收的等待时间，避免慢编译被误判为心跳掉线。
当前请求等待上限为 180 秒，启动等待上限为 240 秒；首次预热输出会被丢弃，之后
使用新观测重新预测。部署前也可先在服务器预热。

`pi0_wm_xarm.yaml` 默认设置 `pi0.local: false`，让 gello 通过 localhost:8000
连接独立的 openpi 推理进程。两个终端均在 xarm_ws 目录运行：

```bash
# 终端 1：openpi 环境加载模型并提供真实推理服务（需要空闲显存）
/home/eid/code/openpi/.venv/bin/python scripts/serve_pi0_wm.py \
  --config inference/configs/pi0_wm_xarm.yaml
# 终端 2：gello 环境负责相机和机械臂；服务就绪后启动
conda activate gello
python -m inference.cli --config inference/configs/pi0_wm_xarm.yaml --enable-command
```

服务端读取同一份 YAML 的权重路径和接口信息，不连接相机或机械臂。
给服务端命令加 `--check` 只检查配置及文件，不加载模型或监听端口。

可选设置 `pi0.local: true` 在客户端进程内直接读取
`pi0.local_checkpoint` 指定的 checkpoint 步数目录（包含 `params/` 和 `assets/`），
通过 `pi0.local_source_path` 使用本地 openpi 源码。需要使用 openpi 训练环境的 Python，
并安装机器人端所需依赖；模型仅在实际预测时加载，普通 `--check` 不加载权重。
本地模式同样需要推理显存，不会降低模型的显存需求；`--check-server` 会触发模型加载。
当前配置对应 peel_cucumber 的三路相机、25 Hz 和 20000 步权重。

设 `pi0.local: false` 可切回 `host` / `port` 的 WebSocket 服务（包括下面的 mock server）；
远程服务器需要使用相同的训练权重和 `pi0.interface`。
远程模式机器人端依赖只需 websocket/msgpack，官方 NumPy codec 通过 `policy.client_source_path`
读取本地 openpi-client 源码，不安装整个训练环境，不改 NumPy 版本：

```bash
pip install -r requirements-pi0-joint.txt
# 仅配置检查：不连接服务端、相机或机械臂
python -m inference.cli --config inference/configs/pi0_wm_xarm.yaml --check
# 完全离线：模型、机械臂和相机全部 mock，默认等 s；i 停止并复位
python -m inference.cli --config inference/configs/pi0_wm_xarm.yaml \
  --backend mock --mock-policy --enable-command
```

实际 WebSocket 协议可用本地 mock server 验证（两个终端）：

```bash
python scripts/mock_pi0_joint_server.py --config inference/configs/pi0_wm_xarm.yaml
python -m inference.cli --config inference/configs/pi0_wm_xarm.yaml \
  --backend mock --enable-command
```

Mock server 在 metadata 标记 `mock_policy: true`，客户端拒绝将它用于真机 backend。
离线自动回归可在客户端命令后加 `--auto-start --duration 2`。
实际服务端恢复后，设置 `pi0.host/port`，并把同一份 `pi0.interface` 同步到服务端 YAML。
先用以下命令只核对 metadata，再用默认只读模式观察；动作执行增加 `--enable-command`。
`s/i/p/q` 全部沿用公共入口，命令开启时 `i` 返回硬件配置中的右臂标定 `reset_q`。

```bash
python -m inference.cli --config inference/configs/pi0_wm_xarm.yaml --check --check-server
python -m inference.cli --config inference/configs/pi0_wm_xarm.yaml
```

## π0 EE-pose → WM

EE-pose/WM 流程仍使用 `inference/configs/pi0_wm.yaml`，由公共入口根据独立 `pi0` 配置分派。
相机、服务端和 WM 权重仍需填入真实部署值；`--check` 检查配置、文件和依赖，不连接硬件。
模拟验证可使用：

```bash
python -m inference.cli --config inference/configs/pi0_wm_mock.yaml --mock --headless --auto-start --steps 260
```

π0-WM 默认等待 `s`，开始后填充新观测历史并进行测速标定；测速期间也可按 `i/p/q`。
`i` 清空旧计划，按 `hardware.reset_speed_rad_s` 返回 `hardware.reset_q`，到位后继续等待 `s`。
`reset_q: null` 时优先使用 endpoint 的 `rest_q`，否则返回本次启动的实测姿态。
只读模式不使能或复位真机。`--single-step` 下每次 `s` 仅执行一个有效 WM 控制周期。
旧命令 `scripts/run_pi0_wm.py` 也复用公共键盘循环，兼容 `--enable-commands` 参数。

## xArm：Diffusion Policy、ACT 输出关节 q

现在 `inference.cli` 支持两个 xArm 配置：
[`configs/dp_xarm.yaml`](configs/dp_xarm.yaml)、
[`configs/act_xarm.yaml`](configs/act_xarm.yaml)。二者复用
`gello_teleop/config/xarm7_gello_dual_dataset.yaml` 的机械臂 IP、`active_arms`、
相机 serial 和有界位置控制参数。推理不需要 GELLO 串口、Contact WM 或 IK。

Native DP 从 `.ckpt` 中恢复训练配置、EMA 权重、图像 shape、相机 key、
normalizer、`n_obs_steps`、`n_action_steps`。ACT 从同一目录的
`policy_config.pkl`、`dataset_stats.pkl` 恢复相机顺序、模型结构及 q/action 归一化。
ACT 旧配置没有图像尺寸和 FPS 时，`policy.dataset_path` 指向原训练 LeRobot 数据，
读取 `meta/info.json`；也可显式配置 `policy.image_shape: [3, H, W]` 和
`execution.action_rate_hz`。相机输入使用 RGB；尺寸由模型适配器按训练尺寸调整。
新 xArm DP 入口面向本机 `diffusion_policy` 训练的 Hydra checkpoint；旧 LeRobot
DP / Nero 配置继续使用下面的原有入口。

先在有模型依赖的环境中检查，不连接机械臂或相机：

```bash
conda activate robodiff
# 真机相机/SDK，以及 ACT 原生代码依赖；已有 PyTorch 无需重装。
python -m pip install xArm-Python-SDK pyrealsense2 IPython
python -m inference.cli --config inference/configs/dp_xarm.yaml --active-arms right --check
python -m inference.cli --config inference/configs/act_xarm.yaml --checkpoint /path/to/act/policy_best.ckpt --active-arms right --check
```

也可复用 `gello` 环境。当前环境已有 `torch==2.8.0+cu128` 时，安装对应的
`torchvision==0.23.0+cu128` 以及 native DP 依赖：

```bash
conda activate gello
# PyTorch 2.8.0 + CUDA 12.8 的对应 torchvision；已有 torch、numpy、Pillow。
python -m pip install torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --no-deps
python -m pip install -r requirements-inference-native.txt
python -m inference.cli --config inference/configs/dp_xarm.yaml --active-arms right --check
```

版本配套依据见 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/#v280)。
`requirements-inference-native.txt` 固定了当前 checkpoint 使用的 Diffusers 和
Hugging Face Hub 配套版本，并补齐 OmegaConf、Hydra、Dill、Einops、Zarr 等依赖。
完整依赖见 `requirements-inference.txt`。ACT 默认从相邻的 `../act` 源码目录构建模型，
可用 `policy.source_path` 修改；仅加载 checkpoint 权重，不创建训练优化器或下载预训练 backbone。
DP 默认配置指向当前找到的 joint checkpoint；路径相对于 YAML。
ACT 默认使用 `../act/outputs/erase_board_joint/policy_last.ckpt`，
读取权重所在目录的配置与归一化统计；可通过 `policy.checkpoint` 修改路径。
`--checkpoint`、`--device` 可以在命令行覆盖；ACT 输出目录优先选 `policy_best.ckpt`，
否则在启动时选最新 epoch，运行中不更换权重。

模拟相机和机械臂跑通模型到 q 执行链路：

```bash
python -m inference.cli --config inference/configs/dp_xarm.yaml --active-arms right --device cpu --backend mock --enable-command --auto-start --duration 5
```

真机读取观测并推理（默认不使能、不发送 q）：

```bash
python -m inference.cli --config inference/configs/dp_xarm.yaml --active-arms right --run
```

执行模型输出的绝对关节角，单位 rad：

```bash
python -m inference.cli --config inference/configs/dp_xarm.yaml --active-arms right --run --enable-command
python -m inference.cli --config inference/configs/act_xarm.yaml --checkpoint /path/to/joint_act/policy_best.ckpt --active-arms right --run --enable-command
```

上述 `inference.cli` 命令用于 DP/ACT 推理；`gello_teleop.dual_gello_collect` 是遥操采集入口。
DP/ACT 启动后加载模型并持续采集相机与关节状态，默认等待按 `s` 开始连续推理。
按 `i` 停止当前推理、清空旧预测和历史，并低速返回所选臂的 `reset_q`；
到位后继续等待 `s`。复位目标取自 `hardware_config` 引用的 GELLO 标定文件，
速度、加速度、超时及到位确认复用硬件配置中的 `alignment.reset_*`。
不带 `--enable-command` 时，`i` 仅暂停并清空缓存，不使能或移动机械臂。

`c` 也可开始连续推理，`p` 只暂停并停止运动，`q` / Ctrl-C 停止并退出。
使用 `--single-step` 时，`s` 执行一个完整 action chunk 后暂停，`c` 仍恢复连续运行。
无人值守或没有交互键盘时可加 `--auto-start`，启动后直接开始推理。
开始执行时从实际 q 初始化位置参考；退出时停止位置伺服并保持电机供电，不清除控制器故障。

模型在独立 worker 中推理，相机持续采集，控制按 `execution.control_hz` 下发 q。
开始推理后默认先做 `execution.warmup_steps: 1` 次预热，不使能机械臂、不执行预热输出；
预热完成后清空模型历史，用新的对齐观测重新预测，避免首次 CUDA 初始化耗时使首个动作过期。
设为 `0` 可关闭预热。预热和首个有效预测分别受 `startup_timeout_s` 限制；
默认执行模式下，实际动作受 `maximum_prediction_age_s` 限制，不执行过期的动作。
如果尚未开始执行，预热后的结果仍超过这个时效，会丢弃该结果、等待新的观测后重试，
日志显示 `Discarding expired startup prediction`；重试不会重置启动超时。
机械臂模式切换后等待首个可执行结果时也沿用此行为，保持实测 q。
一旦开始执行，过期仍触发停止；未来时间戳、动作维度和关节范围异常也不会作为启动重试忽略。
启用命令时先检查一份有效模型输出，再完成 xArm 使能与模式切换，保持实测 q；
切换前的预测不执行。相机和关节状态都更新到切换之后，再重新预测并开始消费动作。
这样模式切换耗时不会消耗首段执行动作的时效；日志会显示 `enable_ms`。
状态日志中的 `status=warming_policy` / `inferencing`、`in_flight_ms` 和
`observation_age_ms` 分别显示等待阶段、当前预测耗时及最近对齐观测的年龄；
`prefetch_ms` 显示下一次预测相对于当前动作截止时间的提前量。
过期报错会显示动作的 `age_ms`、`maximum_age_ms`、`infer_ms` 及 `prediction_status`；
后者包含下一次预测是否在等待、等待时间、观测对齐原因、相机和关节状态的年龄。
相机和机械臂状态按原始时间戳因果对齐；DP 的历史窗口按训练采样周期取帧，首次不足时重复最早帧。
相机映射时间戳若比主机时钟提前不超过 5 ms，会等待主机时间追上后再对齐，
不会改写帧时间戳或提交未来观测；更大的提前量仍报错。
action chunk 的 q 按训练 FPS 顺序消费；下一次预测根据 chunk 剩余时长、
动作剩余有效时间及最近 8 次请求的最大耗时预取（不含预热）。
提前量还包含 `execution.prediction_prefetch_margin_s`，公共默认值为 0.1 秒，
`pi0_wm_xarm.yaml` 使用 0.15 秒，以容纳网络和推理耗时波动。
例如请求耗时 283 ms、控制周期 10 ms 时，该 pi0 配置会提前约 443 ms 请求下一段。
该参数只改变请求时机，不增加 `maximum_prediction_age_s`；提前返回的结果也会检查源观测年龄。
通常在完整 chunk 结束后切换；
如果尾部会超过观测时效，则在到期前切换到已经准备好的新预测。
到期时仍没有有效新结果，会停止所有正在执行的臂。
q 会经过现有的 `SecondOrderPositionFollower` 做速度、加速度、单次步长及跟踪误差限制。
关节范围读取 GELLO 标定中指定的 URDF；可通过 `execution.joint_limits` 按左右臂显式覆盖。
所需相机过期、状态过期、推理结果过期、模型异常或 SDK 发送异常会停止所有正在执行的臂。

`pi0_wm_xarm.yaml` 已启用 `execution.open_loop: true`：允许慢推理返回结果后执行。
每段从第一个动作开始，按训练频率依次执行 `consume_steps` 个动作；当前为 20 步、30 Hz，约 0.67 秒。
执行完再用新观测请求下一段，等待期间保持最后一个 q 目标，相机和状态读取继续运行。
开环模式不使用预测年龄截止时间或预取提前量；首段等待受 `startup_timeout_s` 限制，
单次远程请求仍受 `pi0.request_timeout_s` 限制。
动作形状、关节范围、速度/加速度限制、未来时间戳检查、相机和反馈时效检查及 `s/i/p/q` 保持有效。
要恢复提前预测和动作过期停止，将 `execution.open_loop` 设为 `false`。

该 pi0 配置同时启用 `execution.reset_on_start: true`。带 `--enable-command` 启动时，
先按标定文件的 `reset_q` 和硬件配置的 `alignment.reset_*` 参数低速复位选中的臂；
复位期间不请求模型，完成后默认等待 `s`，使用 `--auto-start` 则复位后自动推理。
`q` 可中断复位；只读运行和 `--check` 不移动机械臂。
`execution.quiet: true` 关闭配置大段打印、相机常规日志、SDK 版本/IP 横幅和每秒状态日志，保留操作提示及警告/错误。
`execution.log_timing: true` 在预热、使能前检查和每段执行结束时打印耗时；
`observe_ms` 是本地读取与对齐，`prepare_ms` 是输入准备，`connect_ms` 是连接和接口验证，
`remote_ms` 是发送到收到响应（包含网络、服务端排队与处理），`server_ms` 是服务端返回的处理耗时，
`postprocess_ms` 是输出验证/IK，`setup_ms` 是使能及伺服模式切换，
`request_ms` 是本地完整模型请求，`execute_ms` 是动作段执行，
`total_ms` 是本轮取样开始到动作段结束。所有数值单位为 ms，未返回的服务端耗时显示 `unknown`。
`status=interrupted` 表示该段被暂停、退出或异常中断；复位单独打印 `reset_ms`。

`active_arms` 沿用采集配置，也可在部署 YAML 或 `--active-arms right` 中覆盖。
单臂模型必须输出 7 维；双臂模型必须输出 14 维，按 `active_arms` 列表顺序拆分。
上述示例使用当前七维 checkpoint，只控制右臂；即使采集配置启用双臂，也通过
`--active-arms right` 为该模型选择右臂。使用十四维双臂权重时改为
`--active-arms left right`。相机数量独立于执行臂数量：当前 DP 即使只控制右臂，
仍需 `left_wrist`、`right_wrist` 两个视角。
`observation.camera_map` 可将 checkpoint key 映射到不同的硬件相机名称。

ACT 可在 `policy.brightness_jitter` 配置推理输入的随机亮度扰动，范围 `[0,1]`，
`0` 关闭。当前 `act_xarm.yaml` 设置为 `0.2`：每次预测对左右相机同时使用一个
`[0.8,1.2]` 内均匀随机的亮度倍率，在 ImageNet 归一化之前将 RGB 裁剪到 `[0,1]`。
`policy.brightness_seed: 0` 固定随机序列，设为 `null` 则每次启动重新随机。
扰动作用于模型输入；相机预览显示原始图像。修改配置后重启推理程序生效。

关节 ACT 使用 `--state_key observation.joint --action_key action.joint --image_only`
训练，从左右腕图像预测七轴绝对关节角，单位 rad；模型使用一个 action query。
关节目标不设置 `--quaternion_indices`。启动脚本先核对原始 `action.joint`、
样本标准差及归一化往返还原，再保存到 `../act/outputs/erase_board_joint`；
已有 checkpoint 时会拒绝覆盖。脚本使用 ACT 的 `.venv-act` 环境，
可通过 `ACT_PYTHON=/path/to/python` 指定其他训练环境。

```bash
# 只检查数据，不训练、不连接硬件。
bash scripts/train_act_joint.sh --check
# 训练关节目标；参数为 epoch 数。
bash scripts/train_act_joint.sh 5000
# 训练完成后检查默认关节权重，不连接硬件。
python -m inference.cli --config inference/configs/act_xarm.yaml --active-arms right --check
```

## 原有 Nero 推理流程

> 开发模块、公共契约和新增 DP/PI0 action expert 的指南见
> [`README_DEVELOPMENT.md`](README_DEVELOPMENT.md)。本文主要记录运行时配置、真机启动、H5 回放和 MuJoCo 使用方式。

## Modular inference components

在线运行时正在收敛到以下稳定边界：

```text
NeroObservationSampler -> ObservationProcessor -> HighLevelPolicy
                       -> ActionScheduler -> WorldModel
                       -> ActionResolver/SafetyGuard -> RobotController
```

公共契约位于 `inference/core/`。`InferenceBase` 只负责生命周期和阶段编排，
模型适配器位于 `inference/policies/`，WM 位于 `inference/world_models/`，控制器
适配器位于 `inference/control/`。现有 `NeroInferenceRuntime` 已经使用独立的
`NeroObservationSampler` 和 `NeroPipelineOutputController`，因此 DP、Contact WM
和 direct-IK 的旧入口仍保持兼容，后续可以逐步替换 `pipeline.py` 内部阶段。

当前迁移已将 DP 的观测历史/时间戳对齐放入 `inference/stages/DPObservationBuffer`，
action chunk 的时间戳推进放入 `ActionPlanExecutor`，DP forward 通过
`policies/dp.DiffusionPolicyAdapter` 调用。WM 未确定时继续使用 `NullWorldModel`；
动作解析和通用限幅可分别注入 `DirectActionResolver`/`BasicSafetyGuard`。

`architecture` 配置块用于选择新的编排层，默认关闭以保持旧 YAML 行为：

```yaml
architecture:
  enabled: false
  policy_type: legacy_pipeline
  world_model_type: none
```

WM 尚未确定时使用 `NullWorldModel`，它不会在基类中引入额外的分支。

Contact WM 的在线部署使用 timestamp 双异步路径。将配置收束为：

```yaml
architecture:
  enabled: true
  policy_type: lerobotdp
  world_model_type: contact_wm
predictor:
  enabled: true
  mode: contact_world_model
  inference_mode: asynchronous
```

该组合由 `NeroInferenceRuntime` 直接构造 Contact WM 管线和独立 worker，不经过
`InferenceBase/ModularInferenceRunner` 的同步 action scheduler。DP worker 在新图像到达后
执行一次 LeRobot DP，并把返回的 8 个 `[7]` 绝对关节动作以
`start_time + k*0.04 s` 写入 `ActionPlanBuffer`。`predictor.enabled=true` 时，WM worker
按 16 Hz 左右读取 `StateHistoryBuffer.query(t-0.5, t)` 和
`ActionPlanBuffer.query(t, t+0.32)`，把 25 Hz 动作以 ZOH 重采样到 100 Hz 条件后预测 32 个
`[q_ref, tau_ref]`；推理延迟对应的过期前缀会被丢弃，剩余轨迹按绝对时间写入
`WMTargetBuffer`，相邻预测的前 40 ms 做重叠融合。控制 worker 不等待模型，以 100 Hz
查询 `WMTargetBuffer`，并通过 MTC 计算和发送当前时刻的力矩。

当 `predictor.enabled=false` 时不加载 Contact WM，也不启动 WM worker；控制 worker 仍以
100 Hz 运行，并直接从 `ActionPlanBuffer` 消费 DP action chunk（每个 25 Hz token 由 ZOH
保持四个控制 tick），通过 `command_joint_positions()` 下发绝对关节位置。因此两种模式
共享同一套 DP/timestamp pipeline，只有 WM 阶段按开关启停。

相机的 `output_size` 只影响模型输入；开启 `visualize` 后预览默认使用裁剪后的
采集分辨率。需要单独缩放预览时配置 `preview_output_size`：

```yaml
cameras:
  - name: wrist
    width: 640
    height: 480
    output_size: [256, 192]
    preview_output_size: null
    visualize: true
```

在线推理会从 DP checkpoint 的 `input_features` 自动读取
`observation.images.<name>`，并将这些名称与 `runtime.collection_config` 中启用的
相机匹配。数采可以配置额外相机（例如 checkpoint 只声明 `side`、`wrist` 时仍可启用
`side_2`）；额外画面仍可采集和预览，但不会进入 DP/WM observation。checkpoint 声明的
任一路相机若未在数采配置中启用，runtime 会在启动时直接报错，而不是等待永远不会到达的
帧。`runtime.camera` 只选择时间对齐用的 anchor，不能扩大 checkpoint 的输入相机集合。

推理时的 `tau_ext` 图复用数采的 `RealtimeJointPlotter`，数据由
`inference/diagnostics/tau_ext.py` 适配，不会阻塞状态采样和控制线程。
`realtime_plot.norm` 默认为 `l1` 以保持历史曲线；需要数学上的欧氏范数时设置为 `l2`。

`predictor.enabled` 是执行链路开关：

- `true`：加载最新版 PINN CARS-WM v3，并按 `execution.mode` 选择 MTC、q 或 tau；
- `false`：不加载 `contactworldmodel`，DP action -> IK -> `q`，runtime 通过
  `command_joint_positions()` 下发关节位置。

顶层 `action` 声明 DP checkpoint 的 7 维输出语义：

- `action: eepose`（默认）：`[x,y,z,qx,qy,qz,qw]`，纯 DP 分支通过 IK 执行；
- `action: joint`：绝对七轴 `q`，纯 DP 分支直接执行，不调用 IK。启用 Contact WM 时，
  每个 action token 先通过机器人 FK 转成绝对 `ee_pose=[x,y,z,qx,qy,qz,qw]`，再传入 WM。

关闭 predictor 时的数据流为：

```text
image + wrench history -> configured DP execution -> future action chunk
future action chunk -> first/mean/all or linear-target pose plan -> frame transform -> damped IK
IK q -> joint-limit + per-cycle joint-step safety -> follower arm
```

`predictor.action_chunk_mode` 同时作用于纯 IK 与 predictor 链路：`first` 使用
chunk 第一帧；`mean` 对位置求均值，并在统一四元数半球后对姿态求均值；`all` 按时间
顺序执行完整 chunk。PINN condition 是否同步裁剪由 `action_condition_fill` 决定。

`predictor.inference_mode: open_loop` 使用严格的“采样 -> 同步推理 -> 执行”状态机：先采满
checkpoint 声明的 `n_obs_steps` 个 image/wrench 观测，冻结该窗口并同步执行一次 DP，然后
完整执行所选 action 计划。执行期间到达的图像不会写入下一轮观测；计划结束后清空旧窗口，
重新采满一批才再次推理。相机 freshness 只在采样阶段作为准入条件，动作执行期间不因图像
超过 `maximum_state_age_s` 而中断，但机械臂控制状态检查与控制超时始终有效。`asynchronous` 保留旧 worker
模式；当它与 `all` 组合时，后台结果只缓存为下一计划，不会中断正在执行的 chunk。
在线纯开环批次以实际相机时间戳为锚点，将 CAN 派生的 wrench 历史按时间戳最近邻对齐；
相机先到时等待 CAN 时间线追上，CAN 先到时等待相机。任一匹配的绝对误差超过
`runtime.maximum_observation_alignment_gap_s`（默认 30 ms）时整批不会送入模型，而会继续采样。
每张图像必须匹配 checkpoint 的 `obs_encoder.wrench_history_steps` 个互不重复、时间递增的
CAN 派生样本；例如 checkpoint 声明为 8 时，模型输入严格为每张图像 8 帧 CAN，不会用
重复最近邻样本补成 8 帧。完整批次的 CAN 数量为 `n_obs_steps * wrench_history_steps`。
`predictor.action_step_s: null` 默认读取 checkpoint 的 `task.dataset.timestamp_step_sec`
（当前 checkpoint 为 0.1 秒）；显式正数可以覆盖它。该间隔只控制高层 action waypoint
推进，低层 IK 或 PINN/control transport 仍在每个有效机械臂状态周期运行。

`predictor.action_execution_mode: minimum_jerk_target` 提供单目标平滑轨迹接口。目标仍由
`action_chunk_mode` 决定：`first` 取 chunk 第一帧，`mean` 取聚合位姿，`last` 取最后一帧；
`all` 在该模式下非法。pipeline 从安装 action 时的实际末端位姿出发，用
`s(u)=10u^3-15u^4+6u^5` 做五次时间缩放：位置沿直线推进，xyzw 四元数沿最短半球做
SLERP。该曲线在起点和终点的速度、加速度均为零。轨迹在
`action_interpolation_duration_s` 内分成 `action_interpolation_steps` 段；持续时间为
`null` 时沿用 `action_step_s`，后者也为 `null` 时沿用 checkpoint 观测步长。轨迹执行完成
前不会被异步 DP 结果中断；旧 `linear_target` 配置会自动归一化到 minimum-jerk 实现，
`action_execution_mode: chunk` 完全保持原有行为。

```yaml
predictor:
  action_chunk_mode: first  # 或 mean/last
  action_execution_mode: minimum_jerk_target
  action_interpolation_duration_s: 2.0
  action_interpolation_steps: 200
```

`predictor.action_condition_fill` 控制高频 PINN 如何消费低频 DP action：

- `auto`：CARS-WM v3 使用 `chunk`；
- `hold`：DP 新结果安装时锁存当前安全 action，并重复填满整个 PINN future horizon；
- `chunk`：输入尚未执行的 DP action chunk，长度不足时保持最后一帧补齐。

CARS-WM v3 的 action condition 始终取当前尚未执行的 DP chunk，长度不足时重复最后一帧
补齐到 checkpoint 声明的 8 个 token；这与训练时的 direct action padding 语义一致。

`ik` 配置收敛容差、阻尼、迭代步长和关节限位 margin；IK 不收敛时拒绝下发。
`safety.maximum_joint_position_step_rad` 再限制每个控制周期相对当前 `q` 的最大变化。

DP 采样器由部署配置覆盖，不需要修改 checkpoint：

```yaml
dp_sampling:
  method: ddim             # ddim 或 ddpm
  num_inference_steps: 8
```

切换采样器只替换 inference scheduler；训练 checkpoint 中的 `num_train_timesteps`、
beta schedule、prediction type、模型权重和 normalizer 保持不变。DDIM 通常用于少步快速
推理；DDPM 通常应配置更多 inference steps，且不能超过 checkpoint 的训练扩散步数。

## H5 离线推理与 MuJoCo 播放

在线 runtime 默认等待 `s` 开始连续推理。使用 `--single-step` 时，`s` 请求并执行
一个完整的有效推理/控制周期，然后回到等待状态；`c` 恢复连续运行，`i` 停止并复位，
复位后继续等待 `s`，`q` 退出：

```bash
uv run python -m inference.cli --config inference/configs/nero_contact_wm.yaml --single-step
```

单步请求会保留到下一张有效相机帧和 CAN 状态都可用，因此一次 `s` 不会因为传感器暂时
没有新数据而消耗掉；它只对应一次 `pipeline.step()` 和一次硬件命令。

`scripts/infer_h5_direct_ik.py` 从 `runs` episode 读取与在线推理相同的观测契约，按最近
时间戳将相机帧对齐到 `teleop` 状态，然后逐相机帧同步执行 DP -> action chunk -> IK。
同步入口保证每个 H5 帧只进入一次 observation history，结果不会随离线读取速度变化；
配置 `open_loop + all` 时也会按照 H5 timestamp 推进完整 chunk，而不会每帧重新推理。

MuJoCo viewer 会同步显示三类诊断信息：机器人使用当帧实际 `q_command`；绿色小方块
显示 DP checkpoint 原始输出的完整 action chunk（未经过 safety clip 或
`first/mean/all` 选择）；橙色大方块显示该 `q_command` 经 MuJoCo FK 得到的实际末端
位置。与该次推理对齐的 RGB 相机帧显示在 viewer 右上角。这样可直接区分“policy 的
整段预测在跳”与“选取/IK/关节步长后下发结果在跳”。

```bash
python scripts/infer_h5_direct_ik.py \
  runs/insert_usb --episode 1 \
  --config inference/configs/nero_direct_ik.yaml \
  --playback-speed 1.0
```

无图形桌面时可只执行 MuJoCo forward/safety 检查：

```bash
python scripts/infer_h5_direct_ik.py \
  runs/insert_usb/episode_0001_20260728_151517.h5 \
  --max-frames 20 --no-viewer
```

默认在 H5 旁保存 `_direct_ik.npz`。v2 格式新增 `dp_action_chunk [N,H,7]`，保存每帧
最新的完整原始 DP chunk；`action [N,7]` 是执行模式选取并经 action safety 处理后的目标；
`q_command [N,7]` 是关节步长限制后实际送入播放/真机接口的命令。此外还包含原始
`q_observed`、完整 IK 解 `q_ik`、时间戳对齐索引和 IK 残差，同时保存生成的
`.scene.xml`。默认配置为 `configs/nero_direct_ik.yaml`，其中不包含
`contactworldmodel`，并通过 `dp_checkpoint.dino_model_path` 使用
`/mnt/code/lcx/model/dinov3-vitb16-pretrain-lvd1689m` 的本地 backbone，不访问网络。

当前唯一的 WM 推理契约是最新版 PINN 的 `ContactWorldModel (carswm_v3)`。配置支持的规范模式为
`contact_world_model` 和 `contact_world_model_opd`；历史 `swm`、`torque_world_model`、
`world_model_v3/v4/v5` 字符串只在配置解析时兼容映射到这两个模式，不再实例化旧模型。

`inference/configs/nero_contact_wm.yaml` 默认加载父目录 `PINN/outputs/contact_world_model_final/checkpoints/latest.pt`，并使用 `model/dp/pretrained_model-20260901T082955Z-1-001/pretrained_model`
作为 LeRobot DP。该 Contact WM checkpoint
必须包含与模型实现一致的 `model_version: carswm_v3`、`carswm_contract`、`joint_dim=7` 和 `action_dim=7`；
`model.inputs` 必须是 checkpoint contract 声明的四路 `[q,dq,delta_q,tau]`。运行时输入为
50 个 100 Hz 的状态历史与 8 个 25 Hz、7 维绝对 ee_pose action token；输出长度由 checkpoint 的
`future_horizon` 声明（当前模型为 32 步）
`q/tau` 及可选的 `dq/delta_q/contact_state_pred`。所有低维输入和输出均按 checkpoint
中的 `normalizer` 处理，`delta_q` 始终由实际 command history 的 `q_cmd - q` 构造。

使用配置时，通过 `execution.mode` 选择三种执行方式：

- `mtc`：计算 `tau_pd = Kp(q_cmd-q)-Kd*dq`，再把 `g(q)` 和 WM 的
  `tau_pred` 残差合成 `t_ff = g(q) + tau_pred`；`mtc_alpha` 只缩放 residual。
  `mtc_q_cmd_source: wm_state` 使用 `q_hat`；`wm_delta` 使用 `q_hat + delta_q_hat`。
  固件继续使用配置的 `mit_kp/mit_kd`、零速度目标，`torque_target` 发送的是
  `t_ff`，不会再整体低通总力矩。异步 worker 使用同一语义，`q_ref` 按 100 Hz 直接消费。
- `q`：对预测 q 做关节限位/单周期步长保护后调用 `command_joint_positions()`。
- `tau`：下发限幅/滤波后的 `g(q) + WM tau_pred`。硬件使用 MIT 报文作为传输
  envelope，但 `kp/kd` 设为零，因此不会额外叠加 firmware 的位置/速度反馈。

`execution.mode: mtc` 的 `torque_target` 是 feed-forward 残差加重力；
`tau_command` 是按当前采样的 q/dq 计算出的总力矩（触发发送限幅时记录限幅后的值）。

最小配置形态如下：

```yaml
dp_checkpoint:
  path: ../../model/dp/pretrained_model-20260901T082955Z-1-001/pretrained_model
  device: cuda:0
  use_ema: true

contactworldmodel:
  path: ../../model/carswm/latest.pt
  device: cuda:0
  use_ema: true

action: joint
predictor:
  enabled: true
  mode: contact_world_model
  inference_mode: asynchronous
  action_chunk_mode: all
  action_execution_mode: chunk
  action_step_s: 0.04

execution:
  mode: mtc  # q / tau
```

Ubuntu 20.04 上使用 `uv` 配置推理环境：

```bash
sudo apt update
sudo apt install -y build-essential pkg-config git curl libgl1 libglib2.0-0 \
  libglfw3 can-utils v4l-utils
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"
cd /mnt/code/lcx/nero_ws
uv python install 3.10
bash setup_env.sh
```

启动在线推理：

```bash
uv run python -m inference.cli --config inference/configs/nero_contact_wm.yaml
```

`uv run` 会自动使用项目的 `.venv`，无需手动激活环境。若需要直接使用
`python` 命令，也可以先运行 `source .venv/bin/activate`。

## H5 -> MuJoCo 独立仿真分支

仿真分支不会初始化 CAN、相机或真机 runtime。它读取一条 H5 episode，在状态时钟上运行
同一个 DP/contact-WM pipeline，并把输出交给带七个 torque motor 的 MuJoCo 模型动态积分。
`q` 是软件 PD 位置伺服，`mtc` 是固定固件增益下的 gravity+residual MIT 控制，`tau`
直接使用 `tau_command`；三种模式都不会直接写 `data.qpos`。

默认每 10 ms 调一次策略、每 1 ms 做 MuJoCo 子步。CARS-WM v3 接收 50 个 100 Hz 的
q/dq/delta_q/tau 历史和 DP 的 8 个 7 维绝对关节 action token。默认 `recorded` 观测模式用 H5 的 q/dq/ddq/tau
作为策略输入，同时记录仿真状态；要检查闭环漂移可使用 `hybrid_closed_loop`，此时图像和
wrench 仍来自 H5，而 q/dq/tau 改用仿真上一周期状态。

先用无窗口模式验证模型和 scene：

```bash
python scripts/infer_h5_mujoco.py \
  runs/wipe_board/episode_0024_20260815_153252.h5 \
  --config inference/configs/nero_contact_wm.yaml \
  --simulation-config calibration/config.yaml \
  --dp-checkpoint /path/to/dp.ckpt \
  --contactworldmodel /path/to/contact_wm_student.pt \
  --mode mtc --max-steps 200 --output /tmp/nero_mujoco_mtc.npz
```

`--dp-checkpoint` and `--contactworldmodel` are optional overrides (`--pinn-checkpoint`
remains a CLI compatibility alias). The checked-in Contact WM YAML uses paths relative
to the repository config directory (`model/carswm/latest.pt` and the LeRobot DP
directory);
replace only the DP path when
using a different policy checkpoint.  The CLI validates both checkpoint files
and the local DINO directory before allocating the models, so a stale path fails
with a short actionable message.  For a pure DP -> IK replay,
use `inference/configs/nero_direct_ik.yaml`; the runner automatically selects
the `q` MuJoCo servo when `predictor.enabled: false`.

其余模式只需替换 `--mode` 为 `q` 或 `tau`。需要图形窗口时加 `--viewer`；在
无显示服务器的机器上保持默认的 headless 模式。输出 NPZ 包含 recorded/simulated q、dq、
ddq、command/applied torque、q/dq/tau target、DP/PINN 更新标记和 MuJoCo 接触计数，生成的
执行器 MJCF 可用 `--scene-output` 保存。

H5 loader 要求 `teleop/timestamp_us`，GELLO/xArm 数据按所选臂读取
`<side>_q_xarm`、`<side>_dq_xarm`、`<side>_tau_xarm`，外力使用
`<side>_wrench_ext_xarm`。Nero 和旧录音继续兼容 `q_follower`、`dq_follower`、
`tau_follower`、`wrench_ext` 及 `wrench_cal`/`wrench_pred`，也兼容带侧别的旧字段。
此外要求所选 camera 的 `frames/timestamp_us`。如果录音没有
`<side>_ddq_xarm` 或旧字段 `ddq_follower`，默认用 timestamp-aware 的因果
`dq` 后向差分补出 ddq；严格复现实验可通过 `derive_ddq_if_missing=False` 禁止该回退。arm
名称同时支持 `teleop` 属性和旧格式的 `metadata/arm_names_json`。相机对状态是因果对齐的
（只使用当前 tick 之前的最新帧），direct-IK 的 wrench 窗口按每个历史图像重新构造为
`[n_obs_steps, wrench_history_steps, 6]`。当前环境若出现 h5py/NumPy ABI 错误，需要先修复
Python 环境，代码会明确报告该依赖错误。

离线 runner 要求 `predictor.inference_mode: open_loop`，这样 DP 在新图像窗口到达时同步
更新，结果不依赖主机线程调度；只有明确接受非确定性时才传 `--allow-asynchronous`。

CARS-WM v3 运行时数据流为：

```text
image + wrench history -> DP checkpoint -> 8-token action (joint or ee_pose)
joint action -> FK -> 8-token absolute ee_pose
q/dq/delta_q/tau history + ee_pose action/action_mask -> ContactWorldModel (carswm_v3)
  -> future q/dq/delta_q/tau/contact_state
future q -> q command (mode=q) or MTC q_cmd
future tau -> causal torque filter (mode=tau or MTC feed-forward)
```

历史 `configs/nero_pipeline.yaml` 已随旧控制链删除。当前配置仍通过
`runtime.collection_config` 指向 `configs/master_slave_can.yaml`，复用其中的：

- 从臂 CAN、固件和 rest pose；
- checkpoint 声明且数采启用的相机（当前 DP 为 `side`、`wrist`；额外配置的相机仍可采集和预览）；
- `tau_other` checkpoint；
- Pinocchio 逆动力学；
- `tau_ext -> wrench_ext` 阻尼 Jacobian 映射。

`predictor.enabled: false` 时 `contactworldmodel` 可从 YAML 中完全省略。开启 predictor 时，
Contact WM 的网络参数、输入维度、horizon、action condition 和归一化器必须保存在
checkpoint 的 `config`/`normalizer` 中，推理配置不会重复声明它们。加载器会在恢复权重前
校验 `carswm_v3` 版本、`carswm_contract`、四路输入以及 7 维关节/action；不匹配的旧权重会明确拒绝。

完整在线观测链为：

```text
follower pyAgxArm latest SDK cache
  -> 由 teleop.command.sample_rate_hz 驱动的固定 tick
  -> 同一 tick 的 q/dq/ddq/tau
  -> tau-ext estimate_aligned(timestamp,q,dq,tau,q_cmd)
  -> tau_g + tau_other_pred - tau
  -> tau_ext
  -> damped J(q)^T inverse
  -> measured wrench_ext
```

实机推理使用独立的连续状态流，DP 前向和动作执行不会暂停 CAN 派生量计算：

```text
isolated hardware process
  pyagx getters     -> fixed-rate latest SDK state stream
inference process  <- ordered state history -> causal KF + dual fixed windows
                   -> tau_ext_cal_raw -> one tau_ext filter -> wrench_cal
                   -> bounded wrench history (4096 canonical samples)
main runtime       <- latest state for control
                   <- drain all accumulated wrench samples for DP history
```

每个状态样本的 `q/dq/ddq/tau/tau_other/wrench` 都使用同一个固定 tick
`timestamp_us`；`acquired_timestamp_us` 只用于控制侧检查状态是否过期。PyAgx SDK、CAN parser、
状态采样和机械臂指令只存在于独立硬件子进程。主循环忙于 DP 推理或 minimum-jerk 动作执行
时，固定频率状态线程仍持续调用 SDK getter，进程内有界历史保留中间状态；主循环随后按时间顺序消费这些状态，
推进 tau-other/tau-next 两个 50 帧滑动窗口，并把期间产生的每一帧 wrench 补回 open-loop 的带时间戳状态历史。相机帧仍由
`CameraManager` 连续采集，DP 在形成完整窗口时再按相机时间戳与状态历史对齐；因此
不会重复消费同一个 tick 来填充 checkpoint 所需的历史（例如一张图像对应 8 个互不重复的状态
样本）。

实机 backend 默认启用该流；`mock` backend 默认保留同步读取，仅用于测试和离线回放。
episode reset、startup recovery 和 shutdown 会先停止并清空连续流，再重置两个 torque 窗口、因果 Kalman
与 DP 历史，避免复位运动污染下一段窗口输入。状态流异常或 tau-ext/wrench
计算异常会记录为 stream fault，主循环随后 fail-closed，而不是继续使用旧状态。

相机预览与状态流完全隔离：每台真实 V4L2 相机由独立 `spawn` 子进程持续采集，并将配置中
`visualize: true` 的最新 RGB 帧直接投递给 GUI 子进程。GUI 子进程拥有唯一的 OpenCV 窗口
和事件循环，即使主推理进程正在执行模型，预览仍可刷新。数据与预览队列都有界且非阻塞，
消费者跟不上时只丢弃旧帧；相机帧不会被预览进程重新用于 DP 对齐，
因此不会阻塞或改变 `q/dq/ddq_kf_causal/tau_ext/wrench` 的计算时间线。数采和推理都复用同一个
`configs/master_slave_can.yaml` 相机配置；当前 `wrist` 预览默认开启。

在线 runtime 读取 `runtime.collection_config` 的 `teleop.command.sample_rate_hz`，与数采
`MasterSlaveTeleop` 使用同一固定频率最新缓存语义。外力矩 RNEA 不使用普通状态字段 `ddq_follower`，
而是由与 PINN offline forward pass 一致的因果 Kalman 从同一 tick 的 `q/dq` 估计 `ddq_kf_causal`。
每个底层状态帧先复制，并由共享的 `source_butterworth_filter` 对 `q/dq/tau/q_cmd` 做因果滤波；
随后 `tau_other` 和 `tau_next` 分别按
`teleop.command.sample_rate_hz / observation_sample_rate_hz` 的整数 stride 取第 0、N、2N…帧。
两路不再按时间戳执行最近邻或 Unix epoch 固定相位重采样；无法得到整数 stride 的配置会在启动时报错。
episode 复位会重启固定频率状态流，并清空两个固定窗口、Kalman 状态和两条 tau_ext 滤波状态，等待新的有限
`q/dq/tau/q_cmd` 有效后才恢复推理。collection runtime 要求 `tau_ext_inference.enabled: true`
并至少配置实际使用的 checkpoint。主从遥操作通过
`tau_ext_inference.feedback_source: tau_other | tau_free` 选择反馈分支：`tau_other` 对应
`tau_ext_cal`，`tau_free` 对应内部历史名 `tau_next` 的 `tau_ext_pred`。两路同时配置时仍会同时推理和记录，
选择项只影响主臂反馈；每路独立执行 history-ready 和 prediction-age 判定，所选分支未就绪时反馈为零。
`tau_other` checkpoint 的输入严格为 `q/dq/delta_q`，不接收测量力矩、`ddq` 或 `tau_id`。
其目标契约为 `tau_other=tau_measured-tau_g`，其中 `tau_measured=observation.torque`、
`tau_g=RNEA(q,0,0)`；在线反馈使用 `tau_g + tau_other_pred - tau_follower`。`tau_id` 仍单独由
因果 Kalman 得到 `ddq` 后计算完整的 `RNEA(q,dq,ddq)`，只用于记录和其他需要动力学输入的分支。

DP 的 observation 也按 checkpoint 时间契约取样：只把新相机 timestamp 加入图像时间线，
以训练配置的 `timestamp_step_sec=0.1` 选择 10 Hz image anchors；每个 anchor 从带时间戳的
wrench ring buffer 中按 `0.1 / wrench_history_steps` 选择对应高频历史（每个 tick 取不晚于
anchor 的最新状态）。控制循环重复使用
同一相机帧时不会再重复写入 DP observation history。多相机 checkpoint 以
`runtime.camera`（当前为 `wrist`）作为 master anchor；其余相机严格选择 timestamp
不晚于该 anchor 的最新帧，复现 wipe-board H5 转换中的 `resample: previous`，不会把
未来 side 帧错配给 wrist。

启用 `wrench_visualization` 后，独立窗口显示四张在线曲线：第一行是 raw tau_ext 经 Jacobian
反解后的 wrench 合力和六个分量，第二行是一次 tau_ext 滤波后映射、再经过 DP
checkpoint 接触门控后的物理量合力和六个分量。启用 `tau_ext_filter` 时，
`observation_protection` 只负责启动预热，不再重复低通 wrench；关闭 `tau_ext_filter` 后才启用其后备 wrench 滤波。接触门控参数（阈值、force dims、历史
reducer）从 DP checkpoint 恢复；当前 checkpoint 使用 0.6 N、前三个线性力分量和 mean
历史 reducer。DP 模型内部仍在归一化之后执行同一个门控，避免把物理零值提前到
normalizer 之前。

推理循环不配置固定 Hz，也不会 `sleep`：

- `open_loop` 下 DP 仅在 action 计划完成后同步推理一次；`asynchronous` 下才使用独立
  worker，并且仅在出现新的训练频率 image anchor 时提交，避免对同一观测反复采样。
- CARS-WM v3 独立维护 checkpoint 指定长度的 `q/dq/delta_q/tau` 物理量窗口；
  episode 开头按训练数据的边界规则复制首帧左填充，并按 checkpoint 的 `high_fps` 做固定
  时间网格插值。`delta_q` 在插值时由保持的 `q_cmd` 与插值后的 `q` 重新计算，避免把旧测量
  状态嵌入 command delta。归一化输入和输出严格使用 checkpoint 的 normalizer。
- 每次 PINN 完成后立即更新 world-model 参考；q、MTC 或 tau 控制链直接消费最新参考。
- DP 尚未完成新推理时，控制链持续执行上一次 action；不会等待 DP。

Contact WM 的 `predictor.inference_mode: asynchronous` 使用绝对 monotonic timestamp
连接三个独立 worker：DP worker 将 25 Hz、8-token chunk 写入 `ActionPlanBuffer`，WM
worker 在 `StateHistoryBuffer` 和 action plan 上做 snapshot，并把推理结果中已过期的
前缀丢弃后写入带时间戳的 `WMTargetBuffer`。100 Hz control worker 只查询
`WMTargetBuffer`，不会等待任一模型；新旧 WM trajectory 在 40 ms overlap 内插值融合。
因此 WM 的推理频率和 trajectory 的执行频率完全解耦，约 60 ms 的推理延迟不会导致每
次从 prediction[0] 回退执行。

因此实际控制更新速度由 PINN 推理与传输耗时自然决定，不由额外的 QP 离散步长决定。
力矩滤波器使用相邻 `InferenceInput.timestamp_s` 的实测周期更新。

`timing.enabled: true` 时使用 `time.perf_counter()` 统计并按
`timing.report_interval_s` 汇总打印实际控制频率、完整周期最新耗时、PINN/DP
平均推理耗时。汇总打印不会在每个控制周期执行，减少终端 I/O 对实时性的影响。

`torque_filter` 只对 WM residual 执行；MTC 不会在 q/v 与 gravity 融合前滤波：

```text
MTC: WM tau_ref -> residual clip
  -> q/v + gravity
  -> residual filter/limit
  -> causal median spike rejection
  -> first-order low-pass
  -> hard per-axis slew-rate limit
  -> final torque clip
  -> MIT t_ff residual

tau mode: WM tau_pred
  -> residual clip
  -> causal median spike rejection
  -> first-order low-pass
  -> hard per-axis slew-rate limit
  -> final torque clip
  -> MIT t_ff = g(q) + residual
```

`median_window` 必须为正奇数；窗口越大会抑制更长尖峰，但也增加相位延迟。
`lowpass_cutoff_hz: null` 可关闭低通，`rate_limit_nm_s: null` 可关闭硬变化率限制。
滤波器以当前实测关节力矩作为首周期状态，并在每次 `reset()` 时清空历史。
`InferenceOutput.tau_unfiltered` 保留滤波前的限幅值（MTC 为融合总力矩，`tau`
模式为 WM 力矩），`InferenceOutput.tau_command` 是实际准备下发的滤波后输出。

DP checkpoint 需遵循 diffusion-policy workspace 格式，并提供
`policy.predict_action(obs)["action"]` future chunk 和 `action_target`。CARS-WM v3
最终接收绝对 `ee_pose=[x,y,z,qx,qy,qz,qw]` action；`action: joint` 时由运行时对每个
token 做 FK，`action: eepose` 时直接传入。FK 使用 `robot.action_frame_name`；Nero
采集数据的 `teleop/ee_pose_follower` 标记为 `tcp`，对应当前 URDF 的 `link7`，所以
Contact WM 配置应显式设置 `action_frame_name: link7`。`robot.frame_name` 仍可保持
`gripper_tcp` 作为控制 frame，运行时会用固定的 link7 到 gripper_tcp 变换下发控制。
chunk 短于 8 个 token 时尾部重复最后一帧补齐，长于 8 则截断。

PINN checkpoint 支持 `/mnt/code/lcx/PINN` 原生格式
`checkpoint["config"] + checkpoint["model"] + checkpoint["normalizer"]`。恢复出的
`ContactWorldModel` 必须提供 `predict(batch, steps, solver)`，返回 flat 的
`q_pred/dq_pred/delta_q_pred/tau_pred`（以及可选 contact 输出）。
可视化采样复用同一模型的 `encode_conditions` + `integrate_flow`，一次编码后并行积分 N
个高斯源；不会在 Python 循环中重复完整条件编码。

CARS-WM v3 的输入和输出均带 batch 维。加载器会在恢复 state dict 前验证版本、carswm_contract 和维度；
不符合契约的旧权重不会静默回退到其它模型。

推荐 Contact WM 在线配置：

```yaml
predictor:
  enabled: true
  mode: contact_world_model
  action_chunk_mode: first
  action_condition_fill: chunk
```

### real-time kinematic visualization of sampled futures

在 `mujoco_visualization` 中设置 `enabled: true` 可启动独立 MuJoCo 进程。该进程只把
真机关节 q 写入主 `MjData` 并调用 `mj_forward`，预测 q 则使用独立 scratch `MjData`
做 FK；它不是 MuJoCo physics rollout，也不会向控制器回传数据。队列容量为 1，发布端
始终 `put_nowait` 覆盖旧 packet，因此渲染、FK 或窗口异常不会阻塞控制线程。

```yaml
mujoco_visualization:
  enabled: true
  num_future_samples: 8
  mujoco_model_path: ../../calibration/data/excitation_validation.scene.xml
  ee_body_name: gripper_base
  robot_joint_names: [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
  prediction_visualization_hz: 8
  render_fps: 30
  observed_q_update_hz: 30
  latest_only: true
  use_fixed_noise_bank: true
  headless: false
```

离线验证不依赖真机链路，使用 LeRobot v3 episode 的 `action.ee_pose` 标签作为 action
condition：

```bash
PYTHONPATH=. python scripts/visualize_wm_lerobotv3.py \
  --config inference/configs/nero_contact_wm.yaml \
  --episode /path/to/lerobot_v3_episode \
  --action-key action.ee_pose --max-steps 300
```

命令只运行 CARS-WM 异步采样和未来轨迹 FK；MuJoCo 中的机械臂运动学姿态播放
`q_samples[0,0,:]`（第一条 WM 样本的第一个 future step），录制的 `observation.joint`
只用于 WM history 和诊断字段，不再驱动显示姿态。N 条完整 WM future 仍作为末端轨迹
小球显示。输出 JSON 会报告 `[N,H_f,7]` 形状、`display_q_source` 以及批量采样耗时。
`H_f` 始终从 checkpoint 读取，不在脚本中硬编码。该入口只调用 `mj_forward`，不是
MuJoCo physics rollout。

Mock 端到端运行（仍会真实加载三个 checkpoint）：

```bash
python scripts/nero_inference.py \
  --config inference/configs/nero_contact_wm.yaml \
  --run --backend mock --duration 10
```

真机只读运行会连接并使能从臂、启动相机和执行完整推理，但不发送控制命令：

```bash
python scripts/nero_inference.py \
  --config inference/configs/nero_contact_wm.yaml \
  --run
```

确认急停、关节范围、wrench 符号和控制限制后，必须显式添加下列参数才会下发命令。
predictor 开启时按 `execution.mode` 下发（`tau` 模式使用零增益 MIT 传输），关闭时通过位置接口下发 IK `q`：

```bash
python scripts/nero_inference.py \
  --config inference/configs/nero_contact_wm.yaml \
  --run --enable-command
```

启用命令后，runtime 在第一个推理 episode 前使用
`runtime.collection_config` 中对应 `arm_pair` 的 `follower.rest_q` 复位从臂；复位的插值速度、
超时、误差阈值和平均采样数继续复用该 collection config 的 `teleop.command.reset_*`
配置。`runtime.maximum_inference_steps` 是单个 episode 的最大控制步数：达到上限后结束
当前 episode、复位、清空 DP/PINN/滤波历史，再自动开始下一次推理；设为 `null` 可关闭
按步数切分。

运行期间按键：

- `s`：开始连续推理；添加 `--single-step` 时只执行一个控制周期；
- `i`：结束当前 episode，清空旧预测和历史，复位到 `follower.rest_q`，等待 `s`；
- `p`：暂停推理并保持当前关节位置；`c` 恢复连续推理；
- `q` 或 `Ctrl-C`：结束当前 episode，复位到 `follower.rest_q`，保持 follower enabled，
  然后断开进程并退出；
- `--duration` 到期执行与退出相同的最终复位流程。

推理、PINN 或控制传输抛出 exception 时，runtime 会优先停止控制循环并立即执行同一套
`follower.rest_q` 复位和使能确认，再清理模型 worker 并退出。只有硬件复位本身失败时，
会尽最大努力切回 follower、读取当前关节位置并发送当前位置保持、再次确认 enable；
异常路径不会主动调用 `disable()`，原始 exception 仍会继续抛出以保留报错堆栈。

不带 `--enable-command` 时保持只读语义，不会为了 episode 边界移动或复位机械臂。
`--check` 永远不会连接机械臂。

### pi0 + Contact WM（xArm）

`inference/configs/pi0_wm_xarm.yaml` 使用 `pipeline: pi0_contact_wm`，保留
pi0 的 50 步、25 Hz 绝对关节动作计划。Contact WM 使用
`model/wm/peel_cucumber/contact_wm/checkpoints/step_00250000.pt`，从 50 步
100 Hz 状态历史的时间戳向后取 8 个动作，经训练对应的 FK 转换为末端位姿，
预测 32 步关节状态、力矩和接触概率。当前配置启用 WM，设置
`pi0.open_loop: true`、`consume_steps: 50`：pi0 每段生成完整 50 个动作，
按 25 Hz 固定推进 2 秒，已提交的动作不会被后续推理替换。
WM 仍使用实时状态历史，控制器仍检查反馈和预测年龄。

先在 openpi 环境运行 `python scripts/serve_pi0_wm.py --config inference/configs/pi0_wm_xarm.yaml`。
在推理环境检查并启动：

```bash
python -m inference.cli --config inference/configs/pi0_wm_xarm.yaml --check
python -m inference.cli --config inference/configs/pi0_wm_xarm.yaml --run --enable-command
```

`s` 开始，`p` 暂停，`i` 重置并暂停，`q` 退出。当前控制器为 `active`，
需要 `--enable-command` 授权输出和复位；只读评估请设置
`controller.execution: shadow`，并省略 `--enable-command`。
WM 的 `delta_q` 使用 SDK 控制器参考值减实测关节角，
需在实际部署中核对其与训练时 host `q_cmd` 的一致性。

更新的单任务 Contact WM 位于 `model/wm/xarm_single_checkpoint/`。它训练时接收
`link7` 的记录位姿 `action.ee_pose`，而当前 DP/pi0 只输出关节目标；
`dp_contact_wm_xarm_single_shadow.yaml` 和 `pi0_wm_xarm_single_shadow.yaml`
显式使用 `joint_fk_link7_approx` 将关节目标做 FK，再送入新权重。该近似在本地
示教样本中与记录位姿存在毫米级偏差，必须先只读比较预测再考虑用于真机输出。
可以用推理环境的 Python 执行无硬件检查：

```bash
/home/eid/miniforge3/envs/gello/bin/python -m inference.cli --config inference/configs/dp_contact_wm_xarm_single_shadow.yaml --check
/home/eid/miniforge3/envs/gello/bin/python -m inference.cli --config inference/configs/pi0_wm_xarm_single_shadow.yaml --check
```

`model/wm/latent_wm_v2/` 的 checkpoint 声明 `latent_carswm_lstm_v2`，现已使用更新的
`../PINN_LCP/model/pinn_model/latent_contact_world_model.py` 加载。v2 使用可学习的
时间位置编码和依赖当前状态历史的条件解码；推理与 CUDA Graph 回放均传入这份历史，
并把 DP 动作插值到状态历史 anchor 后的 25 Hz 网格。原有 v1 配置仍保留其时间编码
和动作调度方式。DP 与原 π0 v2 配置使用只读 shadow；单独命名的
`pi0_wm_v2_xarm.yaml` 当前设置为 active，启动前需核对输出配置。
这些 v2 配置显式指定本地训练 URDF；`--check` 会比较训练和部署的 `link_eef` FK 链：

```bash
/home/eid/miniforge3/envs/gello/bin/python -m inference.cli --config inference/configs/dp_latent_wm_v2_xarm_shadow.yaml --check
/home/eid/miniforge3/envs/gello/bin/python -m inference.cli --config inference/configs/pi0_latent_wm_v2_xarm_shadow.yaml --check
/home/eid/miniforge3/envs/gello/bin/python -m inference.cli --config inference/configs/pi0_wm_v2_xarm.yaml --check
```

DP 的 active 与 shadow WM 配置设置 `policy.refresh_steps: 8` 和
`wait_for_window_exhaustion: true`：以输入观测的
时间戳为起点，8 个 25 Hz 动作占 320 ms，窗口结束后才提交下一次 DP 观测请求。
WM 仍要求每次有 8 个未来动作，因此段末可能没有新的 WM 预测。控制器的
`gap_hold` 在这段空档限时向最后一条已下发位置减速；超过 200 ms、关节跟踪
误差超限或无法满足位置/速度/加速度/jerk 限制时，仍停止下发并由看门狗停机。
启动前可用同名 shadow 配置检查预测和保持事件；该 320 ms 时序尚未经过真机验证。

无硬件、无模型的完整分支检查（先将配置改为 `controller.execution: shadow`）：

```bash
python -m inference.cli --config inference/configs/pi0_wm_xarm.yaml --run --mock --auto-start --duration 2
```

WM 参考年龄从状态历史的 anchor 计算，包含推理耗时。此配置的 WM 更新周期为
80 ms（12.5 Hz）；实机日志中 WM P95 约 57 ms，叠加最多 30 ms 的历史年龄及
3 ms 调度余量，估计预算约 170 ms，因此 `maximum_forecast_age_ms` 设为
180 ms，启动延迟估计设为 60 ms。`wm_reference_age_budget_ms` 显示当前估计预算，
控制器日志的 `forecast_age_ms` 显示实际参考年龄。180 ms 是部署预算，不是预测
精度验证；超过年龄上限或两步参考超出 320 ms 预测范围时仍会停止输出。

`pi0.action_horizon` 必须与训练/服务端输出长度相同（当前模型为 50），不能通过
把 YAML 改成 8 来缩短模型输出。开环接续模式要求 `consume_steps` 等于
`action_horizon`。首段准备好后建立独立的动作执行时钟，后续段的起点固定为
前段起点 + 2 秒；日志保留原始观测 `anchor_us`，另记 `execution_start_us`，
不把旧观测伪装成新观测。预取提前量包含实测 pi0 P95、WM 的未来 8 动作窗口、
一个动作周期的边界余量、30 ms 历史余量、调度余量及
`pi0.prefetch_margin_ms`（默认 100 ms）。同时只保留当前段和一段待执行计划，
WM 的八动作窗口可跨段插值；预取期间仍使用当前段，没有中途换计划。
开环指段内动作固定，下一段仍使用预取时的相机观测；不做跨段动作融合或 RTC。
接续保证时间连续，不能保证不同 pi0 段的关节值连续，现有公共限幅仍然生效。

下一段缺失时不填充最后一个动作，也不外推；超过既有 WM 预测年龄或发送间隔
限制时停止输出。超过预定接续起点才到达的段被拒绝，不移动边界或跳过动作。
暂停、重置及重新启动清空已提交的段，并丢弃之前正在推理的结果。
状态日志增加 `policy_open_loop`、`policy_chunk_ms`、`policy_prefetch_lead_ms`
和 `buffered_policy_chunks`。设 `pi0.open_loop: false` 可使用原来的计划替换模式，
此时 `consume_steps: 8` 约每 320 ms 请求一次，WM 保留完整 50 步未来窗口。

当同一配置设置 `wm.enabled: false` 时，公共入口和 `scripts/run_pi0_wm.py`
自动使用 pi0 直接执行分支，不加载 WM/WM 控制器。保持 `action_horizon: 50`，
`consume_steps: 8` 表示按 25 Hz 执行前 8 步，再采集新观测请求下一段；等待
推理时保持最后一个关节目标。`wm.enabled: true` 可切回联合推理分支。
