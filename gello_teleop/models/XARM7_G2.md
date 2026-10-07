# xArm7 + G2（闭合端、七轴模型）

`xarm7_dynamics_g2.urdf` 包含机械臂和 G2 的视觉/碰撞网格及惯性参数。
官方夹爪六个机构关节冻结于 drive_joint=0.85 rad（闭合端），控制维度仍是
joint1..joint7，nq=nv=7。此角度不是 SDK 的 0～840 开度。

`link_eef` 保留原训练的法兰坐标系；`link_tcp` 是官方宏的名义夹爪 TCP。
改变夹爪开合位置或加装工具时需重新构建/配置负载，不能假设当前惯性仍精确。

来源为 UFACTORY xarm_ros2 提交
`62936f7ea1846a85f7350de2c4c18f39e6d19715`：

- [夹爪宏](https://github.com/xArm-Developer/xarm_ros2/blob/62936f7ea1846a85f7350de2c4c18f39e6d19715/xarm_description/urdf/gripper/xarm_gripper.urdf.xacro)
- [机械臂宏](https://github.com/xArm-Developer/xarm_ros2/blob/62936f7ea1846a85f7350de2c4c18f39e6d19715/xarm_description/urdf/xarm7/xarm7.urdf.xacro)

G2 网格使用 `_g2.dae`。该官方宏为 G1/G2 共用一组惯性默认值，总夹爪质量
0.7514295781 kg；这些不是针对实际 G2 的辨识结果。没有伪造或按外形猜测惯性。
上游许可保留于 `LICENSE.xarm_ros2`；网格来源及 SHA256 存于
`../meshes/xarm7_g2/SOURCE.json`，原宏保存于 `sources/`。

重新生成：

```bash
python scripts/build_xarm_g2_urdf.py --gripper-angle 0.85
```

裸臂 `xarm7_dynamics.urdf` 保留作为原训练 FK 和旧力矩可视化负载模型的基准。
DP-WM 配置已改用 G2 模型；FK 兼容检查比较 base→link_eef 的运动学链，
允许该坐标系之后添加固定夹爪，但拒绝改变机械臂关节变换。

已经包含 G2 惯性的模型不能再重复添加包含同一 G2 的完整 Studio TCP payload。
本次没有改动旧力矩可视化的裸臂＋payload 组合。

Pinocchio 读取几何时，使用 URDF 所在目录作为相对资源搜索目录：

```python
pin.buildGeomFromUrdf(model, str(urdf), pin.GeometryType.VISUAL,
                      package_dirs=[str(urdf.parent)])
```
