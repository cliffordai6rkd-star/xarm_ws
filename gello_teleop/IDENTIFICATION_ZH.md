# xArm7 GELLO 主手 URDF 与动力学辨识

> 2026-09-29：旧方向/零点/局部范围结果已清理。当前主从遥操标定采用
> [单侧选轴标定流程](RECALIBRATION_AND_PIPELINE_ZH.md)，统一结果为
> `config/xarm7_gello_calibration.yaml`。下文机械几何/动力学流程若需已测范围，
> 必须重新采集，不能将历史说明中的已测文件视为仍然存在。

本流程建模的是 GELLO 主手。`models/xarm7_dynamics.urdf` 是从臂模型。
目前已经提供 STL 预览、几何测量模板、激励轨迹规划、主手采集及离线辨识。
现在无需 CAD 即可查看从 STL 安装孔重建的七轴装配草稿；实物装配角、编码器零位
仍需对照确认。绝对动力学参数需要标定后的轴端力矩。

## 当前产物和验证

| 文件 | 用途和状态 |
| --- | --- |
| `models/gello_xarm7_preview.urdf` | 九个 STL 的固定展开预览，零可动关节，未装配、无惯量 |
| `models/gello_xarm7_mesh_bounds.json` | 原始包围盒与网格 SHA256；数值尺度符合毫米导出，单位仍需确认 |
| `models/gello_xarm7_geometry.yaml` | 已填入 STL 推导的轴心、轴向及装配位姿；几何待确认，运动范围和惯量未知 |
| `models/gello_xarm7_draft.urdf` | 七轴可动装配草稿；continuous 关节仅用于显示，不表示无限实物行程 |
| `models/gello_xarm7_viewer.html` | 自包含浏览器三维预览；无需 CAD、ROS、网络或启动服务器 |
| `models/gello_xarm7_stl_features.json` | 圆柱面拟合的孔心、轴线、半径及拟合残差 |
| `reconstruct_gello_geometry.py` | 根据打印件安装基准与电机尺寸重建草稿，可调整虚拟装配角 |
| `measure_gello_ranges.py` | 人工逐轴摆动、只读记录舒适工作区间，不写电机寄存器 |
| `config/gello_identification_left.yaml` / `right.yaml` | 从已有 v4 标定复制串口、ID、符号和偏置；机械范围仍需测量 |
| `build_gello_urdf.py` | 生成预览或从确认过的测量生成七轴 URDF |
| `gello_identification.py` | `init`、`inspect`、`plan`、`collect`、`fit` 命令 |
| `gello_identification_hardware.py` | XL330 独立总线访问；不连接 xArm |

已经在两侧真实串口完成各 10 秒只读采集，每侧 106 帧，完整采样实测速率约
10.4 Hz，最大一次总线读取耗时约 96 ms，最早与最晚电机的接收时刻相差约
48 ms。七个臂关节均为型号 1200；采集期间力矩关闭、电流为零、硬件错误为零。
初始激励配置使用 8 Hz，给 57600 波特率下的读取和指令留余量。

这些数据在 `identification_runs/initial_left_readonly.npz`、
`initial_right_readonly.npz`，摘要为 `initial_readonly_summary.json`。
它们用于验证通信和日志，尚未激励动力学。

`tests/test_gello_identification.py` 使用合成几何、模拟硬件和 Pinocchio RNEA
生成的力矩验证软件，不是硬件参数辨识结果。

## 1. 没有 CAD：检查已重建的装配草稿

先用浏览器打开 `models/gello_xarm7_viewer.html`。页面可以旋转视角、缩放，
逐轴拖动七个滑块，并查看虚拟零位和弯曲示例。它不访问串口。
滑块 ±180° 是显示用范围，不能当作实物限位。

重建依据：STL 中识别出的外壳固定孔间距 16×30 mm、输出盘孔圆直径 12 mm，
与 [ROBOTIS X330 尺寸图](https://www.robotis.com/service/download.php?no=1986) 一致。
尺寸图标注外壳深度 23 mm、输出盘突出 3 mm、轴心距外壳短边 9.5 mm，
即相对固定孔矩形中心偏移 7.5 mm。因此模型使用毫米网格单位和 26 mm
的后壳安装面到输出盘配合面距离。拟合残差反映网格圆柱面的数值一致性，
不能代替实物尺寸精度。

每个打印件采用“输入输出盘固定在本 link、下一电机外壳固定在本 link”的串联
分配。七个主手关节的局部轴均为 `[0,0,1]`；原始 STL 轴孔方向已通过 visual
变换转换到这些关节坐标系。深灰电机盒仅表示 20×34×23 mm 名义外壳尺寸。
后壳安装面的选择、输出轴位于孔矩形哪一端、输出盘的装配角仍是需要实物确认的
假设，尤其需对照 L3/L4 倾斜支架和手柄方向。

草稿的 q=0 是本次定义的虚拟装配基准，**不是当前遥操作配置的编码器零位**。
不要直接复用 v4 偏置并设置 `joint_coordinates_verified: true`。
`geometry_verified: false` 保留，`build` 与动力学拟合仍拒绝未确认模型。
草稿不含惯量、碰撞模型、橡皮筋、线缆或活动扳机。

重新导出显示草稿：

```bash
/home/eid/miniforge3/envs/gello/bin/python gello_teleop/build_gello_urdf.py draft \
  --geometry gello_teleop/models/gello_xarm7_geometry.yaml \
  --output /tmp/gello_xarm7_draft.urdf
```

发现某个打印件绕输入轴装配角不同，可用 `reconstruct_gello_geometry.py`
的 `--clock-degrees` 指定七个角度，写到新的几何和 URDF 文件，然后重新导出预览。
这只调整模型装配基准，不能自动标定编码器：

```bash
/home/eid/miniforge3/envs/gello/bin/python gello_teleop/reconstruct_gello_geometry.py \
  --clock-degrees 0 0 0 0 0 0 0 \
  --output-geometry /tmp/gello_geometry.yaml --output-urdf /tmp/gello_draft.urdf
/home/eid/miniforge3/envs/gello/bin/python gello_teleop/preview_gello_geometry.py \
  --geometry /tmp/gello_geometry.yaml --output /tmp/gello_viewer.html
```

## 1.1 无 CAD 测量实际工作范围

停止使用同一串口的其他程序。向导要求电机已处于位置模式、零 Homing Offset、
力矩关闭，且只读寄存器，不自动关闭力矩或修改模式。
支撑主手，选择能避开线缆、橡皮筋、干涉的小段舒适工作区间。
不要寻找或顶住机械硬限位。

```bash
sg dialout -c '/home/eid/miniforge3/envs/gello/bin/python gello_teleop/measure_gello_ranges.py \
  --config gello_teleop/config/gello_identification_left.yaml \
  --output gello_teleop/config/gello_identification_left_measured.yaml'
```

按提示先记录舒适中位，再每次只摆动一个关节，记录中位两侧端点 A/B，
然后返回中位。每次取十帧静止样本；检测到力矩开启、硬件错误或跨圈会拒绝。
端点跨度达到 180° 会要求补充连续扫动日志，避免仅凭端点产生路径歧义。
默认从两端各内缩 2°，生成新的 `lower_rad/upper_rad`、`center_rad` 和保守幅值，
并保存 `.ranges.json` 原始读数与完整姿态。初始文件不覆盖。

范围暂按原来的编码器坐标表达，`joint_coordinates_verified` 仍为 false。
它们是人工选择的局部工作区间，不能证明整个多关节轨迹无碰撞。
确认或重新标定主手 URDF 零位/方向后，需要在相同新坐标下重新测量或转换范围，
再填入几何文件的 `lower/upper`。软件无法远程代替人工摆动确认实物行程。

## 1.2 确认几何并生成可用于辨识的模型

实物照片对照结果见 [PHOTO_REVIEW_ZH.md](models/PHOTO_REVIEW_ZH.md)。最新草稿
已调整虚拟装配基准，参考姿态前臂水平、手柄输出朝下；此前草稿已备份。
这些调整没有修改已有编码器标定或范围，几何和坐标确认标志仍为 false。
用户随后指出配合端面错误。v2 已修正安装侧、L5 输出盘配合面，并补入名义电机
外壳和输出盘；先检查 `models/gello_xarm7_mount_review_v2.html`，再进行方向核对。
后续重建默认虚拟装配角为 `0 0 -90 71.5 0 180 180` 度；`--clock-degrees`
可以覆盖它。上方全零示例会重建此前采用的虚拟基准。

先用更新后的网页核对关节正方向。点击“返回中位”，当前模型中位为
`[90, 0, -90, 90, 0, 90, -90]` 度，将实物摆成同一姿态。旧配置的 `center_rad`
属于此前的编码器坐标，不能直接当作当前模型角度。
再每次将一个关节增加 3～5°、观察运动方向，人工同向转动实物。只读向导记录
现有符号是否与观察一致，自动运动保持禁用：

```bash
sg dialout -c '/home/eid/miniforge3/envs/gello/bin/python gello_teleop/verify_gello_directions.py \
  --config gello_teleop/config/gello_identification_left_measured.yaml \
  --output gello_teleop/identification_runs/left_direction_review.json'
```

右侧替换为 right 的配置和输出文件。需要支撑其他关节；工具拒绝同时移动多个
关节、目标关节变化过小或过大以及跨圈。若网页姿态与实物明显不同，请停止核对。
方向一致只能确认相对运动符号，不能确认绝对零位；不要据此开启两个确认标志。

也可以使用主动核对模式：先由人摆到上述模型中位，按 Enter 采样实际编码器
位置，再使能七个电机保持该位置。程序不会用模型角度或旧偏置驱动主手去找中位。
每次只运动一个关节 15°，默认去程 8 秒；等待终端 y/n 时保持目标并持续监测，
回答后用 8 秒返回该关节初始位置，再核对下一轴。

```bash
sg dialout -c '/home/eid/miniforge3/envs/gello/bin/python gello_teleop/verify_gello_directions.py \
  --config gello_teleop/config/gello_identification_left_measured.yaml \
  --execute --step-deg 15 \
  --output gello_teleop/identification_runs/left_direction_active_01.json'
```

打开网页并点击“返回中位”；每次把网页当前关节从中位增加 15°以对照。
实物方向与网页正向相同输入 `y`，相反输入 `n`，均按 Enter 提交。
`n` 记录建议反转符号，然后返回原位，不立即尝试反向。若运动的关节编号不对，
输入 `q` 或 Ctrl+C 停止。主动模式需要交互终端；未加 `--execute` 仍为只读模式。
其保持为受 PWM/电流限制的位置闭环，不能当作机械锁死；全程保留主手支撑。
每次等待默认最多 120 秒，超时、故障或中断都会尝试关闭力矩。
全部回到初始位置后，程序提示托住主手并按 Enter，再关闭力矩保存。
输出 JSON 核对结果和同名 NPZ 遥测；中断也保存已采集的数据及故障记录。

所有七轴去程路径在使能前一次性检查已测原始编码器区间及单圈边界；任何轴
15° 超出区间则拒绝使能。可按报错选择更小的 `--step-deg`，或重新测量范围。
主动核对使用独立的编码器坐标路径；不会修改配置的零位、方向或一般自动运动
许可。供电电流仍不等于轴端力矩，核对日志不能直接确认动力学参数。

如果报 `J7: initial pose is outside the measured interval`，说明现在摆好的模型
中位不在此前采集的 J7 局部区间。缩小运动步长不能解决初始位置越界。
用下列只读操作围绕当前网页中位重新采集 J7，两侧端点应自然可达、线缆有余量；
其余六轴沿用原始端点，不改零位/符号。默认从端点向内保留 2°，若要运动 15°，
测试方向的端点应比中位至少多 17°且留有余量，不要寻找硬限位。

```bash
sg dialout -c '/home/eid/miniforge3/envs/gello/bin/python gello_teleop/measure_gello_ranges.py \
  --config gello_teleop/config/gello_identification_left_measured.yaml --joints 7 \
  --output gello_teleop/config/gello_identification_left_measured_v2.yaml'

sg dialout -c '/home/eid/miniforge3/envs/gello/bin/python gello_teleop/verify_gello_directions.py \
  --config gello_teleop/config/gello_identification_left_measured_v2.yaml \
  --execute --step-deg 15 \
  --output gello_teleop/identification_runs/left_direction_active_02.json'
```

如果其他轴也报初始位置越界，可在 `--joints` 后添加其编号一起重测。
失败的主动核对日志现在也保存已采集的参考原始角度/ticks，报错会显示允许区间；
未进入使能流程时明确标记未写入电机。已有旧版失败日志未记录实际位置，不能
用其中的 Goal Position 寄存器值当作本次实际位置，也不能事后恢复漏记的读数。

右臂装配已由用户确认，记录在 `models/gello_xarm7_geometry_right.yaml`；
共同草稿的确认状态不自动扩展到左臂。右臂下一步运行：

```bash
sg dialout -c '/home/eid/miniforge3/envs/gello/bin/python gello_teleop/verify_gello_directions.py \
  --config gello_teleop/config/gello_identification_right_measured.yaml \
  --geometry gello_teleop/models/gello_xarm7_geometry_right.yaml \
  --output gello_teleop/identification_runs/right_direction_review_v3.json'
```

该日志分别保存模型参考角度、参考姿态的原始编码器读数和逐轴正方向观察。
右臂使用主动模式时，同样添加 `--execute --step-deg 15` 并选择新的输出文件名。
参考姿态与符号确认后，可按 `offset = raw_reference - sign * q_reference`
计算新偏置，并把已有原始端点转换到同一模型坐标。暂不执行自动激励。

在 `models/gello_xarm7_geometry.yaml` 填写：

- 确认打印文件的单位后设置 `mesh_units_verified: true` 和 `mesh_scale`。
- 确认七个刚体的零件归属。初始分配为 base、L1～L6、handle，需要按实际装配确认。
  电机外壳、输出盘及打印件要归属其实际固定的刚体，可为同一 link 添加多个 visual。
- 各关节 `xyz`、`rpy`、`axis`、实际 `lower/upper`、电机 `effort/velocity`。
- 各 STL 相对所属 link 的 `visuals[].xyz/rpy`。
- 如需物理惯量先验，填写整个刚体的 `mass_kg`、`com_m` 和质心惯量
  `inertia_kg_m2`，单位分别为 kg、m、kg·m²，惯量矩阵按 link 的轴方向表达。
  打印填充率、电机、线缆、支架和负载均影响这些值。
- 七轴模型把扳机作为固定负载处理。采集时固定其位置，并计入末端惯量。
  如扳机需要运动，应扩展成八轴模型及对应采集，而不是将它当作固定惯量。

所有位移使用米、角度使用弧度。关节 origin 是零位下父 link 到关节坐标系的
变换；visual origin 是所属 link 到原始 STL 坐标系的变换。包围盒中心不能确定
关节轴心。当前草稿用轴孔和电机尺寸约束恢复轴心；STL 本身不包含装配约束。

确认装配和零位后设置 `geometry_verified: true`。生成模型：

```bash
/home/eid/miniforge3/envs/gello/bin/python gello_teleop/build_gello_urdf.py build \
  --geometry gello_teleop/models/gello_xarm7_geometry.yaml \
  --output gello_teleop/models/gello_xarm7.urdf
```

需要完整惯量先验时追加 `--require-inertia`。生成器检查正质量、惯量正定和
主惯量三角不等式。它不自动填充未知测量，不覆盖已有输出。
生成 URDF 使用本机 `file://` 网格路径；搬迁工作区后重新生成。
可在 RViz 逐轴检查旋转中心、方向、零位和连杆运动。

原项目装配与 CAD 信息：
[GELLO Mechanical](https://github.com/wuphilipp/gello_mechanical)。

## 2. 填写轨迹采集配置

选一侧 `config/gello_identification_*.yaml`：

- `encoder_offsets_rad` / `encoder_signs` / `joint_ids` 必须和主手 URDF 的
  **同一个 q=0 和关节顺序**对应，确认后设置 `joint_coordinates_verified: true`。
  从遥操作复制的偏置定义的是从臂关节坐标，只有主手模型采用同样零位和正方向
  时才能直接复用。
- 填写实测 `lower_rad/upper_rad`，包括线缆、弹簧和橡皮筋允许的范围。
  EEPROM 的 0～4095 编码器范围不能代替这些机械限制。
- 选定 `center_rad` 与激励幅值。初始 3°、60 s 是试验起点，需要按实际范围调整。
- 记录 `elastic_elements` 与 `payload_description`。无弹性件填 `none`，
  有橡皮筋/弹簧填其安装描述；末端负载和扳机位置在两次试验中保持一致。
- 检查电流、PWM、电压、温度和跟踪误差阈值。

规划只做计算并保存轨迹：

```bash
/home/eid/miniforge3/envs/gello/bin/python gello_teleop/gello_identification.py plan \
  --config gello_teleop/config/gello_identification_left.yaml \
  --output gello_teleop/identification_runs/left_plan.npz
```

轨迹采用 `sin^4` 窗口的三频正弦激励，起止位置为 center，起止速度和加速度为零。
程序使用解析上界检查完整轨迹的角度、速度和加速度，并拒绝跨越单圈编码器边界。
每关节限位检查不能证明整个运动过程没有自碰撞或外部障碍。

## 3. 采集真实数据

先完成机械安装、运动区域检查并安排现场可断电的人。停止占用同一串口的程序。
只读采集默认不改变力矩、目标、运行模式或 EEPROM：

```bash
/home/eid/miniforge3/envs/gello/bin/python gello_teleop/gello_identification.py collect \
  --config gello_teleop/config/gello_identification_left.yaml \
  --duration 10 --output gello_teleop/identification_runs/left_readonly.npz
```

当前进程若未取得 dialout 组权限，可在新登录会话运行；本机 eid 已列入 dialout，
本次验证通过 `sg dialout -c '上述完整命令'` 使用已有组成员权限。

自动激励通过 `--execute` 明确选择：

```bash
/home/eid/miniforge3/envs/gello/bin/python gello_teleop/gello_identification.py collect \
  --config gello_teleop/config/gello_identification_left.yaml --execute \
  --output gello_teleop/identification_runs/left_excitation_01.npz
```

启动前主手需要手动放到 center 附近并保持静止、力矩关闭。程序不做自动接近运动。
它要求位置模式 3、速度型 Profile、零 Homing Offset、关闭 Torque On by Goal Update，
并检查全轨迹在电机和机械范围内。它只写 RAM 中的 Goal Position、Goal PWM、
Profile 和 Bus Watchdog，不修改 EEPROM 或运行模式。

执行时监测实际位置/速度、跟踪误差、电流、电压、温度、硬件错误和读取间隔。
SIGINT/SIGTERM、超时及失败会退出并尝试关闭所选关节力矩，恢复原 PWM、Profile
和 watchdog。退出后主手需要有机械支撑；断线时软件不能保证关闭力矩，电机的
Bus Watchdog提供通信中断后的停机机制。扳机电机和另一侧主手不参与运动控制。

NPZ 包含原始/映射后位置、速度反馈、电流、PWM、电压、温度、硬件状态、
各电机接收时间、servo tick、实际读耗时、命令和实际样本率。采样用单调时钟，
不把规划轨迹作为测得的运动。失败数据标记 `session_status: aborted`，fit 会拒绝。
输出文件不覆盖，`identification_runs/` 已加入 Git 忽略列表。

## 4. 标定轴端力矩，再拟合动力学

ROBOTIS 明确说明 XL330 测的是输入电源侧电流，而非直流电机快速变化的相电流：
[XL330-M288-T 官方手册](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/)。
不能把 `0.52 Nm / 1.47 A` 当作通用输出力矩常数，也不能把 PWM 当作力矩。
需要输出轴扭矩传感器，或在已知负载下标定并验证电机力矩模型。
砝码与已知力臂可用于静态标定；该结果是否适用于运动，需要额外验证。

拟合使用：

```text
tau_output = Y(q, dq, ddq) * inertial_parameters
           + viscous_friction + coulomb_friction + bias
           + reflected_rotor_inertia + elastic_terms
```

需要提供和主手日志同步的 `torque.npz`：

- `time_s[N]`：和日志的 `time_s` 完全相同；用实际同步的传感器测量对齐。
- `tau_nm[N,7]`：标定后的轴端力矩，正方向及列顺序和 URDF 一致。
- 可选 `sample_time_s[N,7]`：每列力矩实际采样时刻，采用同一主机时基。
- `metadata_json`：JSON 字符串，至少有
  `calibrated_output_torque: true`、`joint_names: [joint1,...,joint7]` 和
  `source`，说明传感器/标定方式。这个字段记录实验事实，不能代替实际标定。

辨识命令：

```bash
/home/eid/miniforge3/envs/gello/bin/python gello_teleop/gello_identification.py fit \
  --urdf gello_teleop/models/gello_xarm7.urdf \
  --data gello_teleop/identification_runs/left_excitation_01.npz \
  --torque-data gello_teleop/identification_runs/left_torque_01.npz \
  --output gello_teleop/identification_runs/left_fit_01.npz
```

有弹性件时追加 `--fit-elastic`；当前基函数是线性、sin、cos，未覆盖橡皮筋迟滞。
更可靠的刚体惯量试验应先移除弹性补偿件，或独立测量它们的力矩曲线。

程序按各列时间戳插值到公共时基，以实测位置做局部三阶多项式平滑和微分，再用
[Pinocchio 力矩回归矩阵](https://docs.ros.org/en/jazzy/p/pinocchio/generated/function_namespacepinocchio_1af97c3d2d695ef4636bf010c2ff6031e8.html)
和按列归一化的 SVD 拟合可观察子空间。电机接收时刻近似采样时刻，尚未标定
传输延迟；精确动态试验仍需处理传感器时钟及延迟。

结果报告训练/最后 30% 连续留出数据的误差、秩、零空间维度和条件数。
`observable_basis_scaled` 与 `observable_coefficients` 给出可辨识组合。
`minimum_norm_representative_NOT_link_parameters` 只是一个数学代表解。
由于质量/质心/惯量的结构欠定、摩擦、转子惯量和弹性力矩耦合，程序不将这个解
写入 URDF，也不声称每个刚体参数已经确认。

要将结果用于控制，还需要在另一条轨迹上验证，结合 CAD/称重的惯量先验构造
物理可行的参数代表，检查质量、质心、惯量及已知负载的重力力矩，最后重新验证
残差和稳定性。当前工具完成数据和辨识管线，尚未替代这些实物实验。

## 软件验证

```bash
/home/eid/miniforge3/envs/gello/bin/python -m unittest discover -s tests -p 'test_gello*.py'
```

所有测试不连接硬件。采集依赖现有 numpy、pyyaml 和 Dynamixel SDK，拟合依赖
`pin` 提供的 Pinocchio，微分与 SVD 不需要额外安装 SciPy。
