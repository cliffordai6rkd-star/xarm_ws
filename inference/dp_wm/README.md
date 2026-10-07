# DP → latent WM（实时只读）

### 公共 WM 可视化与训推检查

当前 single 配置使用 `wm.scheduling: continuous` 和
`controller.forecast_transition: {enabled: true, duration_ms: 20.0}`。
WM 完成后立即用最新有效历史重推；控制层对新旧 q 参考做同一时刻交接。
single 配置的 `controller.command_hz: 50` 每 20 ms 下发 +20 ms 的第二候选，
两个 10 ms 物理步和 WM 100 Hz 预测网格保持不变，命令 q/v/a 历史推进到实际选择的第二步。
continuous 模式的 `execute_steps: 5` 只选取日志中的未来点，不会每执行五步停下来等下一块。
交接在有效重叠时间内保持位置与局部斜率，随后回到新预测；不外推或放宽超时限制。
新旧预测原值和交接修正均记录在 controller NPZ 中。训练模型及 π0 2 秒动作承诺保持原有契约。

`visualization` 支持所有 DP/π0 → Contact/latent WM 分支。
`pi0_latent_wm_v2_xarm_shadow.yaml` 已启用 MuJoCo 和相机接触状态叠加。
在推理环境安装 `python -m pip install -r requirements-wm-visualization.txt`。
MuJoCo 可直接加载 `wm.urdf_path`；无需另外制作 MJCF。显示实测 q 和 WM 末端预测轨迹，
不运行物理仿真，不参与命令发送。

每关节沿其世界坐标关节轴显示一根柱子，正力矩为橙色，负力矩为蓝色。
柱子默认侧移 0.09 m 避免被机械臂网格遮住，用细线连接真实关节中心；可调 `torque_offset_m`。
柱长为 `torque_base_length_m * torque_gain * abs(tau) / max(abs(initial_tau), torque_baseline_floor_nm)`。
`initial_tau` 是本次启动/恢复后首个有效 WM 预测的首步力矩，逐关节固定；
默认单位长度 0.08 m、统一倍率 1、零值分母下限 0.1 Nm。
标签显示 Nm 和相对初值倍率。后续画的是当前时间对应的未来预测步，而非一直画首步；
预测超过最后时间戳后隐藏轨迹和柱子。按 p/i/s 清理旧预测并重新建立初值。

相机窗口下方显示当前预测时刻的 `free / pre-contact / contact` 概率、预测解除时刻，
以及完整未来 320 ms 三色时间带。这三个类别来自训练标签，不能解释为 stick/slip。
过期预测显示 expired。叠加仅在预览进程绘制，模型输入和采集图像保持原像素。

`visualization.tau_ext.enabled: true` 复用硬件配置中 `force_feedback.source: urdf` 的
实时动量观测器或 RNEA 残差，计算外力矩而不启用 GELLO 力反馈。
继承该配置的 URDF、摩擦、负载缓存、状态滤波和滑动均值；即使
`force_feedback.enabled: false`，显示仍可启用。估算在线程中运行，不额外读取硬件。
这条实时估计与 WM 的总力矩 `tau` 分开显示，也不等同于训练时离线 BiLSTM 生成的外力矩标签。
MuJoCo 中 `WM Jn` 为预测总力矩，`EXT Jn` 为实测状态的实时外力矩估计，两根柱子位于关节两侧，
各自使用首次有效值作固定长度基准；无效/过期外力矩隐藏。
外力矩正值为紫红、负值为青绿；WM 总力矩仍为橙/蓝。
启用外力矩时默认只显示 EXT 文字标签，避免两组文字重叠，
可用 `visualization.mujoco.torque_label_source: wm | tau_ext | both | none` 切换。
相机预览显示 J1..J7 的 `tau_ext`（Nm）与 `sum(abs(tau_ext))`，暂停 WM 时实时估计仍可显示。
`visualization.tau_ext.plot: true` 另外打开 L1 和七关节的时序窗口。
当前硬件配置的动量观测器为 10 Hz、后续 20 帧均值，需要预热后才出现有效外力矩。
修改文件后需重启原推理进程；不会自动打断已有实机运行。

配置示例：

```yaml
visualization:
  mujoco:
    enabled: true
    torque_base_length_m: 0.08
    torque_gain: 1.0
    torque_baseline_floor_nm: 0.1
    render_fps: 30
    prediction_visualization_hz: 25
  camera_contact: true
  tau_ext:
    enabled: true
    plot: true
```

`--check` 输出 checkpoint 训练滤波、动作语义、采样参数和训推差异。
下面的检查只读 checkpoint 和已有日志，不连接硬件：

```bash
python scripts/audit_wm_deployment.py \
  --config inference/configs/pi0_latent_wm_v2_xarm_shadow.yaml \
  --run-dir runs/pi0_latent_wm_v2_shadow/1791275516791570171 \
  --output runs/wm_alignment_audit.json
```

这份 latent v2 checkpoint 的动作语义为 `va_measured_joint_snapshot_fk`，部署条件为 π0 目标 FK；
两者存在训练分布差异，不能因维度/FK 一致而视为控制一致。
`controller.mode: mit_position` 跟踪 WM 预测 q，经外环积分与命令限制后发送位置，
不是直接执行 π0 q；文件名中的 shadow 不代表只读，执行行为由 `controller.execution` 决定。
已对齐 host_command、15 Hz 力矩逆滤波、16 步 Heun 和 bfloat16。
已有日志的 WM/π0 同时刻差值不是模型监督误差；日志没有未来真值，不能据此断言模型预测准确。

## Contact WM 入口

`inference/configs/dp_contact_wm_xarm.yaml` 使用 `pipeline: dp_contact_wm`，加载
`model/wm/peel_cucumber/contact_wm/checkpoints/step_00250000.pt` 的 `carswm_v9` EMA 权重。
复用 DP 的完整 15 步计划、预热、实时采集和控制器日志。

```bash
python -m inference.cli --config inference/configs/dp_contact_wm_xarm.yaml --check
python -m inference.cli --config inference/configs/dp_contact_wm_xarm.yaml --run
```

按 `s` 开始。`controller.execution: active` 时必须加 `--enable-command`；
shadow 仅计算参考，适合检查模型输出，`--enable-command` 仍允许复位。
Contact WM 使用原生采样接口，`cuda_graph: true` 捕获整条采样路径；
`cache_condition_kv: true` 对每个新请求重建各 Flow block 的条件 K/V，
只在该次积分的多次注意力计算中复用，不缓存跨窗口的历史或动作。
`cuda_graph: false` 可回到原生 eager 路径；CPU 自动使用 eager。
CUDA 图需要 checkpoint 的 `runtime_checks: false`，适配器仍检查输入形状、
历史时效、输出形状与有限性；全部图在连接硬件及并发 worker 启动前预热捕获。

当前优化配置使用 FP16、16 步 Heun、1 个样本，`execute_steps: 4` 为 40 ms / 25 Hz。
WM 的 `precision` 支持 float32/bfloat16/float16；归一化与物理输出保持 FP32。
DP 也支持 `policy.precision`，本机的小 batch 测量中 AMP 更慢，默认保留 float32。
`policy.cache_observation_features: true` 对相邻观测窗口重复出现的同一个只读图像
快照复用视觉特征；新对象、新图像、不同精度或设备会重新编码，训练模式及可写图像
自动走原始路径。状态报告中的 `dp_feature_cache` 显示命中和重新编码数量。
历史窗口只在新的 100 Hz 状态到达时重建，WM 延迟 P95 只在新的结果到达时更新。
低精度和采样步数变化都需要独立质量验证；推理速度不能证明闭环轨迹正确。
预测年龄、状态年龄和输出停机阈值继续生效。

```bash
python scripts/benchmark_dp_contact_wm.py --iterations 30 --duration 5
```

该脚本不连接硬件，比较 eager、KV cache、CUDA 图和 FP32/BF16/FP16，
使用固定噪声验证图与原生采样的等价性，并记录低精度相对 FP32 的物理单位误差。
同样使用固定 DP 噪声验证视觉缓存，并在测速中使用持续更新的观测窗口。
同时运行真实 DP/WM、合成观测、两步 shadow 控制器和真实文件写入，
结果保存到 `runs/dp_contact_wm_benchmark.json`。实机表现仍需看实时日志。
`logging.async_write: true` 将 NPZ/JSON 写入独立线程，队列有界，错误或积压会明确报错；
退出时先停止输出再等待记录写完。每秒状态新增 `timing_p95_ms` 和 `logging`，
分别显示采集 poll、调度、控制计算、日志提交、主循环及写盘耗时。
每秒状态继续写入 `events.jsonl`；默认 `logging.console_status: false` 不在终端打印
整段 JSON。设为 true 可恢复。启动、交互和停机原因仍打印；停机行同时包含策略输入
新鲜度/时间对齐原因以及 WM 等待原因。π0 单帧观测失效时会报告相机名称、年龄或
状态对齐问题，不再用 DP 的双帧间隔提示代替。

Contact WM 使用固定序号位置编码：桥接将 DP 关节计划线性插值到
`history_anchor + [40,80,...,320] ms`，再做 FK；不外推尾部、不修改原计划时间戳。
完整 15 步计划允许该窗口的锚点最多延后 240 ms（含边界），与 latent 的显式时间网格不同。
输出为物理单位 q/tau `[samples,32,7]` 及接触概率 `[samples,32,3]`。
输入沿用硬件 dq 与 checkpoint 的因果滤波；`controller_reference` 与训练主机 q_cmd
的等价性仍未验证，接入完成不代表闭环控制效果已验证。

入口是 `python -m inference.cli --config inference/configs/dp_wm_xarm.yaml`。
该分支独立于原来的关节伺服运行时，默认 DP/WM 输出用于 shadow 评估。
显式传入 `--enable-command` 时允许复位；同时设置 `controller.execution: active`
时允许发送 WM 控制器生成的位置目标。
DP 使用 `model/dp/peel_cucumber/.../checkpoints/latest.ckpt`，WM 使用
`PINN_LCP/outputs/latent_carswm_lstm_grid/xarm_peel_cucumber_25hz_3cam_20261005/checkpoints/step_00050000.pt`。
模型只在显式 `--run` 且没有 `--mock` 时加载。检查及运行命令均从 xarm_ws 执行。

```bash
# CPU 读取 checkpoint 元数据；不构造网络、不访问 CUDA、不连接硬件
python -m inference.cli --config inference/configs/dp_wm_xarm.yaml --check

# 合成观测及假模型，只验证异步调度（不能用于判断真实模型速度/质量）
python -m inference.cli --config inference/configs/dp_wm_xarm.yaml \
  --run --mock --auto-start --duration 5

# 真实模型和实时只读硬件；默认等待 s
python -m inference.cli --config inference/configs/dp_wm_xarm.yaml --run
```

`s` 开始预测，`p` 暂停并中止正在进行的复位，`q` / Ctrl-C 退出。
`i` 暂停并清除计划；有 `--enable-command` 时同时复位所选的 `observation.arm`。
复位目标取 `execution.initial_q.<arm>`，未指定时取硬件标定中的 `RobotConfig.reset_q`。
`execution.reset_on_start: true` 配合 `--enable-command` 会在启动时复位；
速度、加速度、超时和到位判定沿用硬件配置的 `alignment.reset_*`。
复位期间不提交推理，完成后重新等待新观测；按 `i` 复位后需按 `s` 开始。
`--auto-start` 可在启动复位完成后自动开始推理（active 模式也会开始位置输出）。

```bash
python -m inference.cli --config inference/configs/dp_wm_xarm.yaml --run --enable-command
```

### 实机位置输出

设置 `controller.enabled: true` 和 `controller.execution: active`，再使用上述命令。
按 `s` 后切到 mode 1，将控制器经过关节、速度、加速度、单步和跟踪误差限制的
`first_q_command` 通过 `set_servo_angle_j` 发送。`joint_q_tau` 仍然是位置输出，
力矩仅参与目标计算，不直接发送力矩。
`mit_position` 是第四种方法，当前两个 DP→WM 示例配置选用此模式：
WM q/差分速度与实测反馈计算 MIT 力矩，WM tau 作前馈，
通过完整质量矩阵求解加速度，按 10 ms 积分出增量 q，再形成绝对 q 并公共限幅。
`controller.mit` 显式配置参考来源、积分基准和主机 kp/kd。
`integration_base: measured` 使用本周期实测 q/dq；`previous_command`
使用上一条成功发送的位置及命令速度，首周期回退到实测值。
MIT 始终从实测反馈计算第一步力矩，只将第二步候选视为预测状态；仅发送第一步 q。
新日志保存 `tau_mit`、`a_plan`、`delta_q_raw` 和 `delta_q_command`。
`q_only` 直接使用 WM 按当前状态时间对齐到 +10/+20 ms 的 q 预测，
经公共位置限幅后按 `controller.command_hz` 选择第一步（100 Hz）或第二步（50 Hz），不运行两步最小二乘求解。
`smooth_weight`、`previous_weight` 等目标权重不影响 q_only 的原始目标；
关节范围、速度、加速度、单步变化及跟踪误差限制仍在公共限幅器中执行。

发送前重新检查状态年龄和第一步目标时间；只有发送成功才推进命令历史。
开始发送后，无效结果或超过 30 ms 没有新指令会停止输出并等待 `s` 重新启动。
控制轮询遇到重复或过期的实测状态时等待新帧，不消耗下一控制时隙、
不推进已发送指令历史，也不发送或重新发送旧目标。新帧到达后立即重试，
仍要求状态年龄小于 10 ms。`stale_state_skips` 记录跳过的不同过期样本；
等待期间不会刷新 30 ms watchdog，持续缺少新指令仍会停机。
独立 watchdog 检测主循环阻塞；`p`、`i`、`q`、异常及退出也会停止位置输出。
现有 `maximum_forecast_age_ms` 和时效检查继续生效，不会为使机械臂运动而放宽。
`--mock` 不接受 active；测试用假机械臂验证发送和停机路径。

active π0 配置使用 `host_command` 记录成功发送的主机目标；shadow 配置的
`controller_reference` 与训练主机 `q_cmd` 的等价性仍未验证。
active 接口完成不代表控制增益、动力学误差或闭环表现已完成实机验证。

需要同时安装原有 DP 环境、PINN 模型依赖、
Pinocchio、RealSense 和 xArm SDK。当前配置默认 DP 在 CUDA、WM 在 CPU；
执行前需留出 DP 显存。没有自动安装、升级环境依赖。

## 输入与时间约定

- DP checkpoint 是 joint 动作，尽管训练名含 eepose。保留 `action_pred[:,1:16]`，
  15 步是观测锚点对应的当前动作和未来 14 步；旧入口仍返回原来的 8 步前缀。
- WM 每次按状态锚点选取严格位于其后的连续 8 个动作，窗口可以重叠。
  计划年龄 0–不足 40 ms 选 a1..a8，40–不足 80 ms 选 a2..a9，
  240–不足 280 ms 选 a7..a14；到 280 ms 时覆盖不足，拒绝使用。
  不补 ZOH，不重置动作时间戳；WM 预测年龄和实机输出检查保持原有限制。
- 每步间隔 40 ms；计划保留原观测时间和版本号。结果迟到不能重新起算。
- WM 按历史末帧时间跳过当前保持的动作，要求后面完整 8 步。
  在同一动作周期的两个 WM 请求可以使用相同动作值，但相对时间不同。
- FK 使用训练的 `xarm7_dynamics.urdf`，joint1..joint7 → link_base 下 link_eef 的
  xyz + xyzw，四元数 qw 非负。WM 归一化及反归一化使用 checkpoint 自带统计。
- 状态历史是 50 行、100 Hz 的 q/dq/delta_q/tau；dq、tau 来自硬件。
  原始在线数据按训练配置应用一次因果滤波；不把缺失状态补零，不填充启动历史。
  超过训练允许的状态周期抖动会清空并重新积累历史。
- 图像按实际采集时间对齐，两个 DP 观测目标间隔 40 ms，帧复制后交给 worker。

## 命令参考角的来源

`pi0_wm_xarm.yaml`、single 和 latent v2 的 active 配置使用
`observation.command_source: host_command`。本进程记录每次成功的 servo/reset 目标及
SDK 返回时间，并串行化状态采集与发送；每个状态行只使用采集开始前最后成功发送的
`q_cmd`，然后计算 `delta_q = q_cmd - measured_q`。保持不变的命令不会因年龄而失效，
实测状态仍必须新鲜。失败或运行中的发送不进入历史，不读取 SDK reference 代替 host 目标。
V3 历史只保留真实的状态行，超过允许周期的缺帧清空历史重建 50 行，不补造中点；
真实原始行按 checkpoint 配置因果滤波一次再归一化。

每次 `s` 后先实际发送并持续保持捕获的当前位置。全部 50 行状态历史必须来自这次
hold 之后；第一段策略激活点固定在同一 25 Hz 动作时钟上，预留 500 ms 历史和 WM
预算。hold token 与策略 token 共用提交流，预测窗口可以跨越该边界。
切换前持续每 10 ms 发送同一 hold 目标；到激活点没有可用的首份预测则停机，
不延后提交时间轴。成功切换后沿用 hold 的发送历史，避免把测量漂移作为新命令起点。
暂停、复位和重新开始清除提交流与旧推理 generation，重新建立真实 hold 历史。

`observation.command_source: controller_reference` 使用 SDK
`get_servo_angle(is_real=False)` 的控制器报告角，实测 q 来自 `get_joint_states()`。
**尚未实机验证这个控制器参考角与训练采集程序记录的 host q_cmd 等价。**
模型能运行不代表该输入语义已验证；输出文件保留来源标签。
`last_used_angles` 是 SDK 客户端缓存，且 servo-j 不更新它，本框架不使用该字段。
SDK 另有底层电机指令位置寄存器，因单位和零位没有验证，本框架不把它直接当弧度。

可改 `command_source: external_file`，从 `command_file` 读取发送端真正记录的目标：

```json
{"timestamp_us": 1791170000000000, "q_cmd": [0, 0, 0, 1.57, 0, 1.57, 0]}
```

写入方应原子替换文件，持续发布当前命令参考及对应时间。外部命令不得晚于实测状态；
两者最大时间差由 `maximum_pair_skew_ms` 限制。控制器参考则记录前后两次读取的偏差。
WM 使用 `delta_q = reference - measured_q`，不会用 DP 计划冒充已发送的命令。
本进程只查询并验证力矩报告模式；若控制器报告的是电流，会退出，绝不当作 tau 使用。

## 调度与输出

两个独立 worker，各自最多一个请求，无积压队列。DP 以观测时间对齐 25 Hz 请求；
默认 `wm.scheduling: periodic`，WM 目标周期由 `execute_steps / 100 Hz` 自动确定，并用最近 100 次 P95 延迟加余量
规划下一轮提前请求时间。`execute_steps: 2` 目标 50 Hz，不保证真实计算能达到。
缺少有效输入时在同一截止时间内重试；错过窗口记录计数，不集中补跑旧请求。
新 DP 计划整体替换旧计划；已运行的 WM 保留自己的不可变快照。
若输出完成时计划版本已经更新，记录 `plan_superseded`，不悄悄当作新计划结果。

标准、single 和 latent v2 的 active π0 配置使用 `wm.scheduling: continuous`：收到 WM 结果后，
同一轮用最新有效的实测历史发起下一次请求，不等固定时隙，不排队，也不沿用旧请求输入。
`wm.inference_budget_ms: 100` 只定义每次请求的截止时间和预算统计，不额外等待，
也不会取消运行中的 GPU 推理。此时 `execute_steps` 只限制日志中选中的未来点数，
`target_wm_hz` 为 null；实际 WM 频率由完成速度决定。控制器独立按 100 Hz 运行，
推理期间使用上一份预测的有效后半段，结果时间戳始终相对历史锚点，不按返回时间重排。
过期到不能覆盖控制器两个 10 ms 目标的结果不发布；年龄、预测覆盖和输出 watchdog 检查仍然有效。
连续模式的参考年龄预算估计为两次推理延迟加 30 ms 历史年龄和余量，不是单次推理耗时。

默认 `wm.action_sampling: checkpoint` 从训练元数据选择动作取样方式；这几份
V3 checkpoint 均解析为 `recorded_tokens`，不因 FK 或 learned position encoding 改变。
当前动作索引 +1 后连续八个原始 token，保留提交动作的 25 Hz 相位，不插值动作值。
显式 `anchor_grid` 对 V3 indexed checkpoint 会报错。训练元数据不符合 V3、offset=1、
direct 和零 action delay 时拒绝启用 `recorded_tokens`。历史仍按 checkpoint 过滤一次；
此 single checkpoint 的力矩逆滤波截止频率为 15 Hz。
single 配置将最大预测年龄设为 240 ms，并启用 `q_only_smoothing: jerk_tracking`，
使预测切换通过位置、速度、加速度和 jerk 限幅，active 历史仅由成功发送的指令推进。
这些修正对齐 WM 的状态行、动作索引和因果 host 命令时间契约，仍不证明
`joint_fk_link7_approx` 与训练记录的末端动作数值等价，或预测目标与训练未来实测快照等价。
π0 的图像观测时间保留在 ActionChunk 中，提交动作使用独立执行时间轴；
这不是将 π0 的观测到动作延迟训练语义改为一致，WM 预测本身不重置锚点。

latent 示例保留 16 步 Heun、FP32。`wm.cuda_graph: true`
在 CUDA 上重放相同的 Flow 积分与解码，减少 Python/kernel launch 开销；CPU 仍走原始路径。
WM 使用独立 CUDA stream，避免和 DP 的默认 stream 不必要地串行排队。
可显式修改采样步数、
solver、precision（float32/bfloat16/float16），但加速后的预测质量需要另行验证。
报告会给出实测 P95、目标频率、实际频率、忙碌跳过、动作断供和建议 execute_steps。
`execution.warmup_steps: 3` 控制模型加载后的预热轮数以及实际工作线程中的预热轮数
（必须为正整数）。预热在连接硬件前完成；合成输出全部丢弃，不进入计划或运行时统计。
`--check` 不加载模型、不预热。采样图捕获在并发工作线程启动前完成。

无硬件对比测速与固定噪声等价性检查：

```bash
python scripts/benchmark_dp_wm.py --config inference/configs/dp_wm_xarm.yaml
```

结果写入 `runs/dp_wm_benchmark.json`，包含原始/优化实现的独立和并发 P50/P95、
逐次耗时及模型输出误差；这些合成输入测试不代表完整实机循环性能。

每次运行新建 `runs/dp_latent_wm/<run_id>/`：

- `events.jsonl`：DP/WM 锚点、计划版本、延迟、截止时间与运行状态。
- `prediction_*.npz`：q/tau `[samples,32,7]`（rad/Nm）、contact_probability
  `[samples,32,3]`、32 个绝对预测时间戳、来源及 mock 标志。
- `selected_indices`：完成时尚未过期的下几个预测点，仅用于检查时间对齐，**没有执行**。
- `summary.json`：计数与延迟摘要。

shadow 模式的真实反馈轨迹是机器人当前实际运动，DP 计划并没有执行，因此这些预测是
以候选计划为条件的预测，不能直接把后续实际反馈当成该计划的预测准确率标签。

## 模块边界

`core.py` 只接收两帧 policy 观测及 `HistorySnapshot`，不依赖硬件。
`models.py` 实现 DP 元数据验证和 latent WM 的 FK/归一化/采样。
`live.py` 采集实时观测并按授权配置机械臂接口，`reset.py` 管理复位。
`output.py` 管理显式 active 位置输出和指令中断 watchdog。
`cli.py` 负责生命周期、控制器调用、发送确认及结果文件。


## pi0 → latent WM

`inference/configs/pi0_latent_wm_xarm.yaml` 使用 `pipeline: pi0_latent_wm`，
复用 pi0 的完整 50 步、25 Hz 顺序动作流、下一段预取及公共 WM 控制器下发。
`pi0.open_loop: true` 时，下一段追加到固定执行边界，不替换未消费动作；WM 的
8 个未来动作窗口可跨 chunk 边界。latent WM 读取
`model/wm/peel_cucumber/latent_wm/checkpoints/step_00050000.pt`，使用 FK 后的
末端位姿及实际时间戳构造模型时间网格，预测 32 步 q、tau 和接触概率。

```bash
python -m inference.cli --config inference/configs/pi0_latent_wm_xarm.yaml --check
python -m inference.cli --config inference/configs/pi0_latent_wm_xarm.yaml --run --enable-command
```

pi0 服务可继续使用相同的 pi0 权重和 interface；也可用新配置启动
`python scripts/serve_pi0_wm.py --config inference/configs/pi0_latent_wm_xarm.yaml`。
按 `s` 开始、`p` 暂停、`i` 复位、`q` 退出。新配置沿用当前 pi0 的
`controller.execution: active`，必须传入 `--enable-command`；改为 `shadow`
可只计算和记录。设置 `wm.enabled: false` 自动切回 pi0 直接执行分支。
暂停/复位丢弃当前动作流及在途结果；动作不足不会外推，实测状态年龄和预测
年龄检查仍生效。新配置的延迟值是启动估计，应以实机 P95 核对。
