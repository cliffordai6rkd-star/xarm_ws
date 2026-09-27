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
