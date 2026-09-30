# Three-service xArm collection

The runtime is split into three independent hardware processes and one
collection process. Each hardware process owns its SDK connection; the
collector only uses ZeroMQ RPC and the latest state publishers.

Before starting, install the xArm Python SDK, the Gello driver package,
`pyrealsense2`, `pyyaml`, `numpy`, and a NumPy-compatible `h5py`. Replace the
robot IP, Gello serial path, camera serial numbers, and rest poses in
`configs/xarm_three_server.yaml`.

Start the services in separate terminals:

```bash
python -m xarm_stack.xarm_server --config configs/xarm_three_server.yaml
python -m xarm_stack.gello_server --config configs/xarm_three_server.yaml
python -m xarm_stack.gripper_server --config configs/xarm_three_server.yaml
```

Then start collection:

```bash
python -m xarm_stack.collect --config configs/xarm_three_server.yaml
```

The xArm and gripper services are read-only by default. The collector's startup sequence
moves xArm and Gello along interpolated trajectories, resets the gripper, and
requires five consecutive joint alignment checks below the configured error;
run it only after a manual safety check and explicitly setting both
`xarm.execution_enabled: true` and `gripper.execution_enabled: true`. The
collector then switches xArm to follower mode and enables Gello. In the
terminal, `t` starts zero-force teleoperation
without recording, `r` starts a recorded episode, `space` stops it, and `q`
exits.

RGB frames are stored at `cameras/wrist/frames` and `cameras/side/frames`.
Setting a camera's `depth` to `true` adds the matching
`cameras/<name>/depth` dataset. Camera identity is selected only by the
configured RealSense `serial_number`.

Force feedback is disabled in the example until joint signs and torque bias
are calibrated. Enable it only after verifying that the xArm state stream
reports measured torque or motor current; the collector refuses to enter the
closed loop when required feedback is unavailable.
When using a current report, set `force_feedback.source: current` and provide
the measured per-joint `current_to_torque_nm_per_a` conversion explicitly.
The feedback path applies calibration, filtering and limits to a Gello torque
request. The xArm follower receives joint position targets only. Its public
SDK adapter rejects Nero-style MIT/torque commands by default, so this is not
Nero MIT control. No physical force-feedback loop has been validated here.

## Dual GELLO dataset pipeline and current diagnostic

Collection configs enable a separate measured / URDF theoretical torque window
with seven rows and two columns per active arm, in Nm. The model directly calls
RNEA with q, dq and low-pass-filtered ddq. Acceleration is derived from velocity
feedback and filtered at `torque_visualization.acceleration_cutoff_hz` before
RNEA; there is no additional low-pass on the theoretical torque. Measured torque
is independently filtered at `torque_visualization.measured_torque_cutoff_hz`.
Both cutoffs default to 3 Hz and can be overridden per arm. Missing motion
components leave the model curve blank. The H5 keeps original measured torque.
The default URDF uses the official
`xarm7_type7_HT_BR2` inertial parameters (10.4315 kg moving links). Tool mass
and flange-frame CoM are read from the SDK TCP load report; without a supplied
rotational tensor the tool is represented as a point mass. Use `--left`, `--right`, or
`--both` to select arms and their wrist cameras. Install
`matplotlib>=3.7` and `pin>=3,<4`; use `--no-torque-plot` for headless collection.
See [configuration and model limitations](../gello_teleop/README_ZH.md#双臂遥操数采入口).

The dedicated dual-arm entry owns independent GELLO reader threads and a
100 Hz xArm position-control thread:

```bash
python -m gello_teleop.dual_gello_collect \
  -c gello_teleop/config/xarm7_gello_dual_dataset.yaml
```

It checks both connections, moves both xArms slowly to `reset_q`, waits for
manual GELLO placement and per-joint alignment confirmation, then takes over.
Press `r` to start an episode, Space to stop and hold, `t` to recheck the
current pose before taking over again, and `q` to exit. `q_cmd` is the most
recent target that the SDK call actually accepted at each state sample.

The independent current diagnostic does not use GELLO or cameras:

```bash
python -m xarm_stack.test_joint_current \
  -c xarm_stack/config/xarm_joint_current_test.yaml \
  --arm both --duration 30 --output current_test.h5
```

After selecting `set_report_tau_or_i(1)`, it records the SDK
`get_joint_states()` effort field as current. Use `b`, `p`, and `l` for
baseline, press, and release. `--hold` is required to enable a position hold;
otherwise the tool does not move or reset the arm. Feedback channels are saved
separately with validity flags and the configured selector is restored on exit.
