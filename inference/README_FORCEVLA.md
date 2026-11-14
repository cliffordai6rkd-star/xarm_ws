# xArm → ForceVLA 远程推理

本仓库运行 client：持续读取一个 xArm7 的实测 `q/dq/tau`，按训练数据的因果 RNEA
处理流程计算七维 `tau_ext`，与 RGB 图像、任务文本发送给 Tailscale 内的 GPU server，
再执行返回的绝对关节角。
入口为 [`scripts/run_forcevla.py`](../scripts/run_forcevla.py)，配置为
[`configs/forcevla_xarm.yaml`](configs/forcevla_xarm.yaml)。复用现有机械臂和相机适配器，
不连接 GELLO，也不控制夹爪。

## 远端现状：最新版本已完成前向和路由对齐

审查的远端仓库为
[`forcvla_tauext`](https://github.com/cliffordai6rkd-star/forcvla_tauext)，
最新审查 commit 为 `5335007c975b34a51152c6cd36b13ce320001c12`。现有
`scripts/serve_policy.py` 提供 WebSocket server，并监听 `0.0.0.0:8000`。
本地客户端可按现有协议接入该版本，无需为本入口继续修改远端仓库；本次工作未修改远端文件。

该版本已修复此前发现的运行和数值对齐问题：

1. [`pi0_force.py:318`](https://github.com/cliffordai6rkd-star/forcvla_tauext/blob/5335007c975b34a51152c6cd36b13ce320001c12/src/openpi/models/pi0_force.py#L318)
   在每个去噪步骤联合计算 prefix/suffix，使用与训练 `compute_loss` 相同的 mask、位置和力融合。
   这保留缺失相机、文本 padding 的原有训练语义；图像和文本的初始编码仍只做一次。
2. [`limoe_simple.py:288`](https://github.com/cliffordai6rkd-star/forcvla_tauext/blob/5335007c975b34a51152c6cd36b13ce320001c12/src/openpi/models/limoe_simple.py#L288)
   单样本的 817 个路由 token 补至 820 个，推理固定每组一个 token，返回前裁回 817 个。
   路由与原训练 batch=4 的独立逐 token 分组一致，不再因小 batch 竞争专家容量而丢弃 token。
   参数名称和形状保持兼容已有 checkpoint。

本地使用独立的 Python 3.12、CPU JAX 0.5.3 / Flax 0.10.2 环境运行远端模型回归测试，
30 项全部通过。额外独立复现保留生产输入的 817-token 长度、缺失左腕和文本 padding，
两步采样与显式训练式联合前向参考的 RMS 差为 `0.001277`，通过上游数值容差。
相同参数的 MoE 构造输入在单样本和训练 batch=4 下均无 token 丢弃，输出最大差为 `0`。
前一项来自随机小模型内部数值，不能解释为真实关节角误差。

[上游验证记录](https://github.com/cliffordai6rkd-star/forcvla_tauext/blob/5335007c975b34a51152c6cd36b13ce320001c12/docs/lerobot_v3_training.md#L115)
另报告：黄瓜、擦板两个真实 `*_tau_ext_b4/49999` checkpoint 各取一个真实样本，
使用相同初始 noise，比较两步单样本推理与 batch=4 联合前向参考，还原后的七关节动作最大差
分别为 `0.00078` / `0.00152 rad`，低于其 `0.005 rad` 容差，两份 `Policy.infer`
输出均为有限的 `(50,7)`。这属于上游报告，本次没有在本机加载真实 checkpoint 或连接 GPU server。

当前检查范围内未发现需要继续修改服务器的前向/路由缺陷。部署时需根据 checkpoint
选择对应的训练 `tau_ext` 处理记录、核对相机预处理，并实测默认 10 个去噪步骤的服务器耗时。
联合前向比旧缓存路径更慢；
上述两步离线结果不能替代实际 Tailscale 请求延迟和实机任务验证。WebSocket 连接成功或
`/healthz` 返回 OK 也不代表模型验证通过；下面的 `--check-server` 会真正调用当前服务器模型。

## 安装与配置

在本仓库的 Python 环境安装轻量客户端及本地硬件/动力学依赖：

```bash
python -m pip install -r requirements-forcevla.txt
```

本地无需安装 JAX、PyTorch 或下载 ForceVLA checkpoint。服务器沿用原 ForceVLA 环境、
checkpoint 和 checkpoint 内的 normalization assets。

默认文件包含 `YOUR_SERVER_TAILSCALE_IP`、`YOUR_SERVER_USER` 和示例服务器路径，
需要替换后才能部署。先编辑 `inference/configs/forcevla_xarm.yaml`：填入服务器 Tailscale IP 或 MagicDNS、
训练配置/checkpoint、机械臂侧别/IP、实际相机 serial，以及训练数据的 `tau_ext` 处理记录。
`hardware_config`、`tau_ext.processing_record`、`tau_ext.urdf_path` 相对于 YAML；
`server.repo_path`、`server.checkpoint` 是服务器路径，
SSH 私钥使用绝对路径或 `~`。

| 配置 | 含义 |
| --- | --- |
| `server.host` / `port` | 推理地址；填 Tailscale IP 或 MagicDNS，默认端口 8000，不含 `ws://`。 |
| `server.connect_timeout_s` | WebSocket 建连及初始 metadata 接收超时。 |
| `server.request_timeout_s` | 单次请求从编码、上传到接收结果的总超时，包含首次 JAX 编译；超时中断连接。 |
| `server.startup_timeout_s` | 经 SSH 启动服务器时等待其就绪的总时限。 |
| `server.autostart` | `true` 时运行脚本自动通过 SSH 启动现有 server；默认 `false`。 |
| `server.repo_path` / `checkpoint` | 远端现有代码目录与具体 checkpoint step 目录，不能只填实验父目录。 |
| `server.python_command` | 远端启动命令的数组，如 `[uv, run]` 或 `[/path/to/env/bin/python]`。 |
| `server.ssh.user` / `host` / `port` | SSH 用户、主机和端口；SSH host 为空时复用 `server.host`，也可填 SSH config 别名。推理连接仍使用 `server.host`。 |
| `server.ssh.identity_file` | 可选本地 SSH 私钥路径；为空时使用现有 key/agent 或已授权的 Tailscale SSH。 |
| `policy.training_config` | `pi0_force_lora_board` 或 `pi0_force_lora_huanggua`，必须与 server/checkpoint 一致。 |
| `policy.prompt` | 每次请求发送的任务文本，例如 `Wipe the board.` 或 `Peel the cucumber.`。 |
| `policy.action_rate_hz` | 训练数据 FPS，现有两个配置为 25 Hz。它不同于本地控制频率。 |
| `policy.execute_steps` | 丢弃因网络/等待已过期的动作后，最多执行 N 帧，默认 10 帧，即最多 0.4 秒；剩余尾部较短时执行可用帧。 |
| `policy.image_size` | 发送前图像大小 `[height, width]`，默认 `[224, 224]`。 |
| `hardware_config` | 复用本仓库数采配置中的机械臂/位置控制参数；只读取对应配置，不启动遥操。 |
| `arm` | 单次策略控制的侧别 `left` 或 `right`，默认右臂。 |
| `arm_options` | 覆盖 `robot_ip`、`sdk_timeout_s` 等适配器设置；入口强制七关节、实测速度、Nm 力矩反馈。 |

server 当前返回空 metadata，不能自动识别训练配置、FPS 或动作含义；连接已有 server 时，
`policy.training_config` 是本地契约声明，须与远端实际启动参数一致。
旧 `forcevla_lora` 的输入/动作是 xyz、rpy、gripper 语义，本入口不接受该配置。

`cameras` 覆盖 `hardware_config.cameras`；未填写时才继承硬件配置。示例序列号来自本仓库
已有设备配置，部署时应按现场设备修改。SSH 私钥建议填写绝对路径或 `~/.ssh/...`。

| 相机/观测配置 | 含义 |
| --- | --- |
| `cameras[].name` / `backend` | 本地相机名与后端，示例使用 `realsense`。 |
| `serial_number`、`width`、`height`、`fps` | 设备序列号、采集分辨率和采集 FPS。 |
| `output_size` | 本地采集后的缩放 `[width, height]`；`null` 保留原比例，发送前再补边。 |
| `crop` | 可选裁剪 `[y0, y1, x0, x1]`；裁剪和缩放应与训练数据保持一致。 |
| `depth` / `visualize` | 是否采深度/显示预览；策略只发送 RGB，示例均关闭。 |
| `observation.camera_map` | server 字段到本地相机名的映射：`image: front`、`right_wrist_image: right_wrist`。 |
| `maximum_camera_age_s` | 最新相机帧的最大年龄。 |
| `maximum_camera_skew_s` | 对齐到共同图像时刻后，各相机允许的时间差。 |
| `maximum_state_age_s` | 最新机械臂状态的最大年龄。 |
| `maximum_alignment_gap_s` | 与图像锚点匹配的历史状态允许落后的最长时间。 |

这两个训练配置实际使用 front 和 right_wrist 两路。默认省略 `wrist_image`，服务器会将
左腕图像补零并 mask。只有训练确实使用左腕时才增加该映射及对应相机。
默认继续使用相应任务 prompt 和 25 Hz 动作契约，不将缺失左腕改成有效相机。
`cameras[].output_size` 与 `policy.image_size` 的宽高顺序不同；前者可能直接改变长宽比，
后者使用 OpenPI 的 PIL bilinear 等比缩放和补黑边，不能在部署时随意互换预处理方式。
RealSense 的 RGB、depth 和 preview 均先使用配置的 crop，再分别缩放；默认 crop 覆盖完整画面。

| `tau_ext` 配置 | 含义 |
| --- | --- |
| `tau_ext.processing_record` | 训练数据的处理记录 JSON，默认 `tau_ext_dp_hirol_clean.json`；也可直接指定实际数据集的 `meta/tau_ext_processing.json`。 |
| `tau_ext.urdf_path` | 动力学 URDF，默认 `../../gello_teleop/models/xarm7_dynamics.urdf`；启动时核对训练记录中的 SHA256。 |
| `tau_ext.max_gap_s` | 允许的原始机械臂反馈最大间隔，默认 0.15 秒。 |

处理记录决定 FPS、滤波、残差符号、滑动均值、接触门限和动力学负载。两套已核对的
`dp_hirol_main` 数据处理版本如下，必须选择生成该 checkpoint 训练输入的版本：

| 训练数据版本 | 本地处理记录 | 七关节 L1 门限 | 因果均值窗口 |
| --- | --- | --- | --- |
| `*_clean` | `tau_ext_dp_hirol_clean.json` | 14 Nm | 1 帧 |
| `*_clean_ma5_l1_12p5` | `tau_ext_dp_hirol_ma5_l1_12p5.json` | 12.5 Nm | 5 帧 |

当前远端 `pi0_force_lora_board` 配置指向 `*_clean`，因此示例选择第一套。若实际 checkpoint
由第二套数据训练，须改 `tau_ext.processing_record`；以训练时保存的数据和参数为准。
两套均使用 25 Hz、3 Hz 二阶因果 Butterworth、`tool-on-environment` 符号、
基座重力 `[0, 0, -9.81] m/s²`，并在 URDF 的 `link_eef` 加入固定负载：
质量 `0.79 kg`、质心 `[0, 0, 0.05] m`、零附加转动惯量和零力矩偏置。
本入口使用这份训练负载记录，不从 Studio 负载设置自动改写特征模型。
两套记录使用同一 URDF，SHA256 为
`2272e49a41f25e67e949b57425db47c298eafbe579518ef2953ca068e682a652`。

| 执行配置 | 含义 |
| --- | --- |
| `execution.enable_command` | 默认 `false`，只读取观测和推理；设为 `true` 后普通启动即下发关节位置。 |
| `control_hz` | 本地原始状态读取和位置控制频率，默认 100 Hz；`tau_ext` 按训练记录独立以 25 Hz 更新。 |
| `startup_timeout_s` | 等待相机、有效对齐状态和首个预测的最长时间。 |
| `maximum_prediction_age_s` | 从本地图像观测时刻计算的预测最大年龄；包含网络和推理延迟。 |
| `position_reference` | 可覆盖硬件 `control` 中的 `position_mode`、`kp`、`kd`、`max_velocity_rad_s`、`max_acceleration_rad_s2`、`max_step_rad`、`max_tracking_error_rad`。 |

## 启动方式

以下命令在本仓库根目录运行。可用 `--config /path/to/deployment.yaml` 选择另一份配置。

```bash
# 仅检查本地配置，不联网、不连接硬件
python scripts/run_forcevla.py --check

# 打印按照当前配置生成的远端启动命令，不执行
python scripts/run_forcevla.py --print-server-command
```

两端先配置好 Tailscale，使本机可以访问 server 的 tailnet 地址和 TCP 8000。
无需公网 IP、路由器端口映射或反向隧道；本客户端禁用 WebSocket 代理。
服务器原脚本监听所有网卡，本入口通过配置的 Tailscale 地址连接。

手动在服务器启动现有 server，例如：

```bash
cd /absolute/path/to/forcvla_tauext
/absolute/path/to/forcevla/env/bin/python scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config=pi0_force_lora_board \
  --policy.dir=/absolute/path/to/checkpoints/pi0_force_lora_board/experiment/step
```

也可将本地 `--print-server-command` 输出的命令直接用于服务器终端。
这里启动的是远端原有服务器脚本；服务器 checkout 应使用上述已修复的版本。

服务器可推理后，在本地先用一次合成观测验证模型并预热 JAX，再连接真机：

```bash
# 真实请求一次服务器，不连接机械臂或相机；成功应返回 actions_shape: [50, 7]
python scripts/run_forcevla.py --check-server

# 使用合成机械臂反馈和图像、真实标签处理算法，仍调用真实服务器
python scripts/run_forcevla.py --mock --duration 5

# 真机读取 q/dq/tau 和相机、计算 tau_ext 并推理，不发送运动指令
python scripts/run_forcevla.py --dry-run --duration 10

# 执行服务器返回的关节位置
python scripts/run_forcevla.py --enable-command
```

首次 JAX 编译可能长于 `maximum_prediction_age_s`。尚未开始运动时，过期的首个结果会被
丢弃，并使用新观测重试，整个过程仍受 `execution.startup_timeout_s` 限制；期间 `tau_ext` 持续更新。
先运行 `--check-server` 预热可以缩短等待。增加请求超时不会放宽动作年龄限制。
动作时间从本地观测锚点计算，接收到结果时不会把动作第 0 帧重新当作起点。
网络、服务器推理和预取等待期间已经过期的前缀会被跳过；整段 horizon 已过期时，
尚未开始运动则丢弃并重新观察，执行期间则报错停止。
`--mock` 使用合成反馈和图像，只用于连通流程，不验证真实任务效果。

要实现配置后只运行一个脚本，将 `server.autostart: true`，并填好服务器路径、
Python 启动命令及 SSH 配置；需要执行动作时再设 `execution.enable_command: true`：

```bash
python scripts/run_forcevla.py
```

SSH 使用 `BatchMode=yes`，需事先完成 key/agent 或 Tailscale SSH 授权和已知主机初始化。
脚本会启动已有远端 server、等待连接、运行本地 client；退出时终止该次 SSH 启动进程。
手动独立运行的服务器不由本脚本管理。
`--start-server` 可临时开启自动启动；已有独立运行的 server 时保持 `autostart: false`。
`--dry-run` 始终覆盖配置为只读，`--enable-command` 临时开启执行，`--mock` 始终不执行真机。

运行中 `c` 连续推理，`p` 暂停并停止运动，`s` 执行一个配置长度的 chunk 后暂停，
`q` 或 Ctrl-C 停止退出。`--single-step` 以暂停状态启动，`--duration N` 限制运行秒数。
启动时由实际关节位置初始化参考，不自动回零。

## 训练标签、时间对齐与线上契约

本地实现 [`xarm_stack/causal_tau_ext.py`](../xarm_stack/causal_tau_ext.py) 复用
`dp_hirol_main/diffusion_policy/common/xarm_tau_ext.py:CausalTauExt` 的训练标签算法。
SDK 输入为同一份反馈中的 `q`（rad）、实测 `dq`（rad/s）、实测 `tau`（Nm）；
`q` 保持原值，不由位置差分替代实测速度。每个 25 Hz 样本执行：

1. 对 `dq` 和 `tau` 分别做二阶、3 Hz 的因果 Butterworth 低通，得到 `dq_f`、`tau_f`。
2. 用相邻滤波速度后向差分计算 `ddq = (dq_f[k] - dq_f[k-1]) / 0.04`。
3. 使用训练 URDF、重力和固定负载，计算 `tau_free = RNEA(q, dq_f, ddq)`。
4. 计算带符号残差 `r = tau_f - torque_bias - tau_free`，按记录对最近 1 或 5 帧做均值。
5. 若该七维均值的 `sum(abs(r))` 达到记录门限，保留整个带符号向量；否则七维全部置零。

这里发送的是训练字段 `observation.tau_ext` 对应的门控结果。门限作用于整个向量的 L1，
不会对单关节分别扣门限或取绝对值。`tool-on-environment` 在这份训练实现中对应上述
`tau_f - torque_bias - tau_free` 符号；线上不再额外反号、归一化或叠加另一层滤波。

首帧低通状态由首个输入初始化，`ddq=0`；均值窗口不足时使用已有帧。
启动不增加额外预热帧，首帧行为与标签生成一致；新的 rollout 清空滤波器、差分和均值历史。
两套训练处理均按 episode 独立重置，导出的 `q/dq/tau`、`tau_ext` 和图像保持原数据行对应关系。

原始反馈读取与 100 Hz 位置控制独立于 25 Hz 特征时钟。特征线程按固定 0.04 秒时刻，
选取不晚于该时刻的最新一份原始 `q/dq/tau`，以零阶保持采样；不用未来反馈插值，
网络等待也不会暂停特征更新。原始反馈使用 SDK 返回完成的本机时间，属于主机收包时间，
不代表硬件曝光同步。

每个请求以各相机最新帧中的最早时间作为共同锚点，为其他相机和已计算的机械臂特征
选择不晚于锚点的最新历史样本；发送的 `q` 与 `tau_ext` 取自同一个特征样本。
只发送当前观测，不构造时间堆叠；预测年龄依据本地观测时间，不要求两端时钟同步。
反馈、相机失效或执行期间的预测超龄会报错并停止执行。

| 请求字段 | 格式 |
| --- | --- |
| `state` | float32 `[7]`，J1～J7，rad |
| `tau_ext` | float32 `[7]`，J1～J7，原始 Nm |
| `image` | front，RGB uint8 `[224,224,3]` |
| `right_wrist_image` | 右腕，同格式 |
| `prompt` | 任务文本 |

协议使用 WebSocket 二进制 msgpack 和与 OpenPI 相同的 ndarray 字节编码。
client 不做归一化；server 拼接 `[q, tau_ext, padding]` 并使用 checkpoint 的统计。
返回 `actions[50,7]` 已在 server 反归一化、恢复为七个绝对关节角，client 不再次叠加 q。
没有夹爪通道。服务器返回的完整 50 帧保留至安装动作计划时，以
`floor((本机当前时间-观测锚点)/0.04)` 跳过已流逝的前缀，再取最多 `execute_steps`
帧。控制时保留不足 0.04 秒的相位余数，使动作仍沿原 25 Hz 时间线执行；预测年龄始终
使用原观测时间。例如结果晚到 215 ms，会跳过前 5 帧，并保留 15 ms 相位，而不是从第 0 帧重播。
首次使能机械臂如需等待 SDK 报告，等待结束后也会重新选择当前时刻对应的目标帧。
100 Hz 位置控制沿用本仓库的关节范围、限速、加速度、单步和跟踪误差约束，并预取下一个 chunk。

标签回放使用擦板、黄瓜两任务的两版真实 parquet 数据，共 346 帧；
`tau_ext`、门控前残差、补偿力矩、L1 与接触标记均与存储值逐元素一致。
小型回归样本及来源记录位于 `tests/fixtures/causal_tau_ext_*`，客户端及相关回归共 132 项通过。
本地验证覆盖训练特征处理、协议、假 WebSocket server、模拟执行链路，以及远端模型的 CPU 小模型
测试和数值对比；没有连接真实 xArm、相机、Tailscale GPU 服务器或真实 ForceVLA checkpoint。
上游文档中的真实 checkpoint 验证
与本地客户端验证分别陈述；实机部署仍需核对训练数据的观测处理，并验证实际服务器和任务效果。
