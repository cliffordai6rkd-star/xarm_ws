# 公共 WM 两步位置参考控制层

`inference.core.WmControlLayer` 与 DP/ACT/pi0 无关：输入 WM 物理单位的
q/tau 预测及当前反馈，输出两步 q_command 和 first_q_command。
目前接入 DP-WM CLI；原纯 DP/ACT/pi0 的执行链不变。该模块没有机器人句柄、
send、enable 或模式切换接口；active 的发送由 DP-WM CLI 单独处理，只有发送成功才提交历史。

配置 `inference/configs/dp_wm_xarm.yaml` 的 `controller`：

```yaml
controller:
  enabled: true
  mode: mit_position  # q_only / tau_only / joint_q_tau / mit_position
  execution: shadow
  horizon_steps: 2
  dt: 0.01
  mit:
    reference_source: wm_q
    velocity_source: zero
    feedforward_source: nominal_wm_blend
    wm_tau_weight: 0.05
    nominal_acceleration_source: zero
    wm_residual_acceleration_limit_rad_s2: 0.5
    feedback_space: acceleration
    integration_base: previous_command
    kp: 8.0
    kd: 6.0
  limits:
    max_velocity_rad_s: 0.5
    max_acceleration_rad_s2: 1.0
    max_jerk_rad_s3: 5.0
    max_step_rad: 0.01
    max_tracking_error_rad: 0.1
```

WM 更新频率由 wm.execute_steps 决定。控制参考在新反馈可用时按 100 Hz 目标更新，
使用旧 WM chunk 仍有效的时间段，不把一次预测到达时间当作新 anchor。
计算过慢、反馈超时、轨迹不足或共同运动边界不可满足时，输出 valid=false。
本实现没有 contact QP、没有引入 OSQP，也没有保证真实闭环达到任何力矩目标。

## 转换定义

以当前 q0/v0 为起点，h=0.01，两个加速度 a1/a2：

- q1 = q0 + h*v0 + 0.5*h²*a1
- q2 = q0 + 2*h*v0 + 1.5*h²*a1 + 0.5*h²*a2
- tau_i_model = M0*a_i + n0 + b0；n0 = RNEA(q0,v0,0)

q_only 直接把按时间对齐的 WM q 作为原始位置目标，经过公共限幅后输出第一步；
不进行两步最小二乘求解，不使用 tau、动力学或连续性目标权重。

`controller.command_hz: 50 | 100` 将位置下发频率与 WM 的物理预测步长分开。
默认 100 Hz 选择 +10 ms 的第一候选；50 Hz 每 20 ms 选择 +20 ms 的第二候选。
`controller.dt: 0.01`、WM 100 Hz/32 步预测和状态历史保持原训练网格，不修改成 0.02。
两个 10 ms 物理步都经过位置、速度、加速度和 jerk 限幅；50 Hz 的成功发送确认
推进到第二步 q/v/a，shadow/offline 也推进到同一候选，不会简单丢弃一半计算后仍积分 10 ms。
日志中的 `command_index`（0/1）、`command_target_timestamp_us` 和 `command_hz` 标明实际下发候选。
为了兼容现有日志，`first_q_command` 字段表示本次实际准备下发的位置，50 Hz 时等于 `q_command[1]`。
状态仍要求在原 10 ms 预算内，反馈锁获取后再次核对所选目标期限；30 ms watchdog 保持原值。
single 的 `minimum_state_remaining_ms: 5.0` 在计算开始前预留 4 ms 计算和 1 ms 发送时间。
50 Hz 主循环与 100 Hz 反馈相位漂移时，9.7 ms 老的状态虽未过期，却不足以计算并发送；
此时等待下一帧，不消耗控制时隙、不推进已发送轨迹。启动 hold 同样等待足够新鲜的状态。
其他配置缺省为 0，保留原行为。新增日志的 `send_state_age_ms`、`feedback_lock_wait_ms`、
`remaining_state_budget_ms` 和 `send_phase` 标明发送阶段的真实时效，停机时也打印年龄和锁等待。
single 配置已选择 50 Hz，其他配置缺省仍为 100 Hz。降频可改变下发节拍，不能自动消除
π0 两秒块的反向起点、预测噪声或位置跟踪器的速度衰减。

```yaml
controller:
  dt: 0.01
  command_hz: 50
```
默认 `limits.q_only_smoothing: legacy` 保留原来的位置投影；仅配置 `max_jerk_rad_s3`
不会自动替换 q_only 的运动逻辑。`jerk_tracking` 是显式可选的参考生成器，
使用命令状态的位置/速度阻尼、加速度/jerk 限制和提前减速；其默认参考增益为
16 1/s² 和 8 1/s，可由 `q_only_position_gain_s2`、`q_only_velocity_gain_s` 配置。
`dp_contact_wm_xarm.yaml` 显式启用该生成器，使用 100 1/s²、20 1/s、
20 rad/s³；这组数值只通过三段记录回放和延迟伺服模拟，尚需实机验证。
日志中的 `solver_status: direct_wm_q`、`q_raw == q_reference` 标识该路径；
原始 v_plan/a_plan 是位置差分诊断，实际限幅后的值为 command_v_plan/command_a_plan。

`q_only` 可选开启按实际时间衔接 WM 新旧预测：

```yaml
controller:
  forecast_transition:
    enabled: true
    duration_ms: 20.0
```

交接从新预测到达后第一个实际下发目标时刻开始（100 Hz 为 +10 ms，50 Hz 为 +20 ms）。
50 Hz 的较早 +10 ms 内部候选在安装时沿用旧有效参考。
先分别取新旧预测在**同一个绝对时间戳**
上的位置和局部插值斜率，再使用五次多项式将位置/速度差在指定时间内衰减到零。
第一点保持原有效参考的位置和斜率；交接完成后回到新预测原值。
连续多次换预测时，从已修正的有效轨迹重新衔接，始终只保存一条最新原始轨迹和一组修正系数。
时长限制在新旧预测的有效重叠范围及年龄预算内；旧预测过期或重叠不足时直接用新预测，
不外推、不改 anchor、不给过期预测续命。下游速度、加速度、jerk、跟踪误差和发送时效限制照常执行。
这个可选项目前仅支持 `q_only`，其他控制方法要保持位置/速度/力矩参考的一致性，不能直接混用。
默认关闭；`pi0_wm_xarm_single.yaml` 显式开启 20 ms，并采用 `wm.scheduling: continuous`。
两段原时戳日志回放中，20 ms 的同刻交接差降到数值误差，速度/加速度/jerk 仍满足原有限幅，
相比 50 ms 修正幅度和附加迟滞更小。原始预测仍是分段线性插值；这里保持的是交接点的同一局部斜率，
不表示预测在所有采样结点都具有连续导数。回放固定使用原记录反馈，不等同于实机闭环或成功率验证。

日志分别保留 `q_reference_raw`、修正后的 `q_reference`、`q_reference_correction` 和
`forecast_transition_*`。`forecast_switch_jump_rad` 记录同一时刻新原始参考与旧有效参考之差。
该处理减少目标切换跳变，并不取消 π0 的 50 步/2 秒动作承诺，也不保证实机任务成功率提高。

使用已有日志比较关闭/开启交接，整个过程不连接硬件：

```bash
python scripts/replay_wm_transitions.py \
  --run-dir runs/pi0_contact_wm_shadow/1791279183124946002 \
  --duration-ms 20 --output runs/wm_transition_replay/compare_20ms.json
```

回放只为隔离分析机器的调度抖动临时提高副本中的计算期限；实机配置仍为原 4 ms，
状态年龄、预测年龄、覆盖范围和命令限制均照常验证。
`tau_only` 复用下述 MIT 力矩前馈、逐步正向动力学和位置积分路径，
不再使用加权最小二乘。`joint_q_tau` 联合最小化位置和力矩误差，使用固定 14 维加权最小二乘，
可加入 a2-a1 与 a1-a_prev 的连续性正则。不显式求质量矩阵逆，不做逐轴 tau/Mjj。

第四种方法 `mit_position` 在主机端依次计算：

```text
dq_d = 0（默认）；wm_difference 用原始差分，filtered_wm_difference 用九点线性斜率并限幅
ddq_d = 0（默认）；可选 wm_difference 求 WM 轨迹二阶导数并按加速度上限限幅
tau_nominal = M(q_feedback) * ddq_d + n(q_feedback,dq_feedback) + b
a_wm_raw = solve(M(q_feedback), tau_wm - tau_nominal)
tau_wm_effective = tau_nominal + M(q_feedback) * clip(a_wm_raw, -0.5, 0.5)
tau_ff = (1 - beta) * tau_nominal + beta * tau_wm_effective
tau_feedback = M(q_feedback) * (kp * (q_d - q_feedback) + kd * (dq_d - dq_feedback))
tau_mit = tau_feedback + tau_ff
ddq_raw = solve(M(q_feedback), tau_mit - n(q_feedback,dq_feedback) - b)
ddq = 限制 ddq_raw 的加速度和相对上一条已发送命令的 jerk
delta_q = dt * v_base + 0.5 * dt^2 * ddq
q_next = q_base + delta_q
v_next = v_base + dt * ddq
```

`q_d` 取同一 WM q 轨迹的时间对齐位置，WM 总力矩沿用 `torque_reference`
指定的滤波逆变换。`nominal_wm_blend` 使用名义逆动力学与 WM 总力矩融合，
`mit.wm_tau_weight` 是 beta，可为标量或七轴数组，范围 [0,1]；0 表示名义前馈，
1 表示 WM 前馈。二者都包括重力，融合后不额外加重力。
设置 `wm_residual_acceleration_limit_rad_s2` 时，WM 相对名义前馈的贡献先转换到
加速度空间限幅，再用完整质量矩阵转换回力矩；原始与有效 WM 力矩分别记录。
旧 `feedforward_source: wm_tau` 保留为 beta=1 的兼容路径。
`tau_only` 固定 `feedforward_source: wm_tau`、`wm_tau_weight: 1`，
配置中的 `mit.wm_tau_weight` 只对 `mit_position` 生效。
`tau_only` 继承 `controller.mit` 的 `wm_residual_acceleration_limit_rad_s2`、
`integration_base` 和 `feedback_state`，
使用相同的全耦合动力学、加速度/jerk 整形、公共位置投影和已发送命令历史。
WM q/dq/ddq 完全不参与计算；kp/kd、参考速度和名义参考加速度均固定为零。
因此 `tau_feedback=0`、`tau_nominal=n+b`，最终力矩为限幅后的 WM 前馈。
可直接把现有配置中的 `mode` 改为 `tau_only`。未配置 `mit` 时兼容默认
`feedforward_source: wm_tau`、`integration_base: measured`、`feedback_state: measured`。
旧 `weights` 不再影响 tau_only 的数值；`torque_scale` 必须为 1。
例如 MIT 的 beta=0.01、残差加速度上限=0.5 时，MIT 的 WM 修正最多为
0.005 rad/s²；同一配置下 tau_only 的原始驱动加速度最多为 0.5 rad/s²，
仍经过加速度/jerk 和公共命令限制，不含位置跟踪项。
两种模式最终都经公共发送器下发关节位置，未连接电机力矩下发接口。
`feedback_space: torque` 保留旧 MIT 对角力矩 PD，kp 单位为 Nm/rad，kd 为 Nm/(rad/s)。
`feedback_space: acceleration` 使用完整质量矩阵把跟踪加速度转换成力矩，kp 为 1/s²，
kd 为 1/s；两种模式的数值不能直接互换。增益可为标量或七轴数组，必须显式配置。

`pi0_wm_xarm.yaml` 使用 `mit_position`、kp=8.0、kd=6.0、
`integration_base: previous_command`、`feedback_state: command`，并将滤波后的
WM 参考速度限制在 0.03 rad/s。其他配置继续按各自 YAML 配置运行。
名义前馈为 n+b，WM 总力矩相对名义前馈的修正先限制为 ±0.5 rad/s²，
再乘 beta=0.05，因此对原始跟踪加速度的扰动不超过 ±0.025 rad/s²。
这使 WM 位置误差主导运动，避免小惯量关节和模型偏置让原始 WM 力矩压过位置反馈。
这些仍是初始外环参数，仿真/回放不能证明实机任务成功或真实电机力矩跟踪。
局部运动上限为 0.5 rad/s、1 rad/s²、5 rad/s³，不改公共遥操配置。
在 100 Hz 下，每周期命令加速度变化不超过 0.05 rad/s²；力矩突变或 WM chunk 切换
也经过这条约束。正向动力学的原始加速度在积分前整形，避免第二个候选从极端预测状态计算。
最终投影先从 solver 的 `q_raw` 计算目标加速度，保证 measured 与 previous_command
两种积分基准实际产生不同的位置目标；不能直接复用 a_plan 并始终从命令历史积分。
投影沿用 `q_next=q+h*v+0.5*h²*a` 和 `v_next=v+h*a`，提前保留到速度上限前
将加速度降到零所需的空间。`command_v_plan` 是周期末速度，不是 delta_q/h。
位置、速度、加速度与 jerk 边界没有交集时返回 invalid，由 active 输出停止；
不会为了平滑放宽位置或跟踪误差边界。暂停、重启、错误会清空命令历史。

`feedback_state: measured` 的第一步 MIT 误差和动力学使用最新实测 q/dq；
`feedback_state: command` 使用上一条成功发送的命令状态，以免位置伺服滞后时
虚拟积分持续加速。第二步候选使用第一步预测状态，
重新计算质量矩阵及非线性项。`integration_base: measured` 用实测 q/dq 积分；
`previous_command` 用上一条成功发送并限幅的位置及命令速度积分，首次回退到实测状态。
shadow/offline 中该选项使用上一条虚拟限幅参考；`integration_history_kind` 区分实测、
已发送和虚拟积分历史。
增量加到该基准后，经公共限幅，仅发送第一步绝对位置。下一周期重新读反馈并计算。
旧三种模式的 `weights` 不参与 MIT 力矩公式；第四种模式使用 `mit.kp/kd`。
该模式仍通过 xArm 的位置伺服接口发送 q；主机计算的 tau_mit 不直接下发电机。

日志增加 `dq_reference`、`acceleration_reference`、`tau_nominal`、`wm_tau_weight`、
`tau_feedback`、`tau_feedforward`、`tau_mit`、`acceleration_raw`、`a_plan`、
`tau_wm_effective`、`wm_acceleration_raw`、`delta_q_raw`、`integration_q0/v0`、
`delta_q_command`，以及每个候选的 M/n/b。
`solver_status: mit_forward_dynamics` 标识 MIT 和 tau_only 共用的转换方法，
`actual_mode` 区分两者，`wm_tracking_enabled` 标识是否使用 WM 位置反馈。
`acceleration_raw` 是未限幅的动力学解，`a_plan` 是积分前整形的加速度，
`command_a_plan`、`command_v_plan`、`command_jerk_plan` 是最终投影的命令状态。
`dynamics_residual` 仅检查原始动力学解；`command_dynamics_residual` 单独记录
限幅后命令与 tau_mit 的差异，不能把近零原始残差当作实机力矩跟踪证明。

Wq=1/eps_q²；Wtau=rho/(eps_tau²+sigma_model²)。配置中的初始尺度只是
shadow 试验值，未做实机增益／模型误差辨识。position_scale=0 或 torque_scale=0
可验证联合最小二乘模式退化；tau_only 的力矩参考缺失或非有限时直接拒绝。

四个模式共用位置、命令速度／加速度、步长和相对实测 q 的领先量限制。
这些是加权参考生成后的共同投影，不宣称求出了带硬约束的联合 QP 最优解。
原始 q_raw 与投影后的 q_command 分别记录，防止把限幅结果当成原动力学解。
shadow 内部历史仅是虚拟参考，不是实际已发送命令；100 Hz 推进第一个候选，50 Hz 推进第二个候选。
暂停、错误或重新开始会清空该虚拟历史。

## 时间、力矩与模型

输入预测必须是同一样本的 q/tau `[samples,time,7]`，配套绝对时间戳。
WmReferenceAligner 按最新反馈时刻后的 h、2h 插值，跳过过期前缀，不外推。
反馈源时间、计算时间、预测 anchor、样本 ID、版本与选点索引均记录。
同一反馈时间戳不被重复当成新控制状态。raw hardware dq 用于动力学，
与进入 WM 历史的过滤 dq 分开，保留其真实时间。

本 WM 的 torque 标签经过 20 Hz 因果一阶滤波。默认 causal_inverse 根据同一样本
在 t0/t0+h/t0+2h 的预测，以及必要的原始 anchor filtered tau，重建等效区间力矩：
`tau_ref_i = (tau_filtered_i - r*tau_filtered_(i-1))/(1-r)`，
`r=exp(-2*pi*20*h)`。这是一种离散参考重建，插值／模型误差会影响它，
不代表真实原始电机力矩被精确恢复。未经滤波的其他 WM 可选择 raw。

动力学使用带 G2 的 xarm7_dynamics_g2.urdf，夹爪冻结闭合端，仍是七轴。
每个控制层有独立 Pinocchio Data，CRBA 的上三角显式镜像后检查正定性。
b_mode=zero 明确忽略摩擦、接触和模型偏置；configured 可提供静态 b0_nm，
不能把它称为已校准外力。没有接入实时接触估计；接触下不能保证 tau 跟踪。

## 日志与运行

pi0 Contact WM 示例每 40 ms 更新 WM，控制计算预算为 4 ms；反馈年龄上限仍是
10 ms，WM 参考年龄上限仍是 180 ms，发送前仍检查首个目标是否过期。
`ActiveOutput` 对无效结果或超过 30 ms 的发送间隔执行锁存停止，需要按 s 重新启动。
一次停止不表示策略判断削皮完成。i 表示复位并暂停，p 表示暂停。

每次运行保存 `config.json`，记录实际加载的参数。`events.jsonl` 中的 `interaction`
记录 s/p/i/q 按键；`output_stop` 立即记录自动停止原因、参考年龄和计算耗时，
`summary.json.last_output_stop` 保留最后一次自动停止原因，即使随后按 p/i 或退出。
`policy_*.npz` 保存完整策略 q 与实际使用的时间戳（pi0 开环使用执行时钟），
`prediction_*.npz` 的 `policy_condition_q`、`policy_condition_timestamps_us` 保存送入
WM 的实际八个动作，`wm_history_q_anchor`、`wm_history_delta_q_anchor` 保存历史锚点。
这些记录用于区分策略没有下行动作、WM 预测不响应动作、控制跟踪偏差和软件保护停止。
DP 图像输入以真实采集时间选择最接近 40 ms 间隔的两帧，支持 30 Hz 采集、
25 Hz 筛选产生的正常 33/67 ms 间隔；仍拒绝过旧数据和复位前数据。
WM 的 100 Hz 状态历史遇到单次 14–24 ms 的采样间隔时，只在两个真实反馈之间
插入一个线性插值样本；超过 24 ms 仍清空并重建历史。`source_status` 记录插值和
重建次数，便于核对参考过期是否由状态采样中断引起。
`policy_input_observation_timestamps_us` 记录实际输入图像观测时刻，
`policy_observation_status` 记录配对状态，避免把历史状态已满误认为图像观测也已就绪。

```bash
# 当前 checkpoint 元数据及配置检查；没有神经模型加载或硬件连接
python -m inference.cli --config inference/configs/dp_wm_xarm.yaml --check
# 假模型/假反馈联调，没有硬件
python -m inference.cli --config inference/configs/dp_wm_xarm.yaml \
  --run --mock --auto-start --duration 3
# 真实模型与只读反馈，按 s 开始；仍没有运动发送
python -m inference.cli --config inference/configs/dp_wm_xarm.yaml --run
```

controller_*.npz 保存 q_raw/q_command/first_q_command、原参考、q0/v0/tau_measured、
a/v 计划、M/n/b 和残差；events.jsonl 保存状态、原因及时间。
`logging.record_controller: false` 关闭逐帧控制参考记录，转换仍执行。
sent=false 表示没有任何实际发送。已有 WM prediction 文件照常保存。

已有预测与同时间的 H5 可离线回放：

```bash
python scripts/replay_wm_controller.py --forecast /path/prediction_000001.npz \
  --states /path/recorded.h5 --mode joint_q_tau --output /path/new-output-dir
```

时间必须重叠；工具不会挪动时间戳拼接不相关记录。不加载神经网络、不连接硬件。
录制数据只能用于参考生成与耗时比较，不能证明这些未发送的位置会产生录制的运动。
