"""Independent real-time kinematic visualization of sampled CARS-WM futures."""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import struct
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


log = logging.getLogger(__name__)


@dataclass
class VisualizationPacket:
    timestamp: float
    observed_q: np.ndarray
    predicted_q: np.ndarray | None = None
    predicted_ee_position: np.ndarray | None = None
    prediction_id: int | None = None
    # Optional q to put on the displayed robot.  Online packets leave this
    # unset, so the display follows the latest measured ``observed_q``.  The
    # offline LeRobot replay sets it to the first WM future q, allowing the
    # kinematic model to play the same q that q-mode execution would consume
    # without changing the required observed_q diagnostic field.
    playback_q: np.ndarray | None = None
    predicted_tau: np.ndarray | None = None
    future_timestamps: np.ndarray | None = None
    initial_tau: np.ndarray | None = None
    episode_id: int = 0
    observed_tau_ext: np.ndarray | None = None
    tau_ext_timestamp: float | None = None
    initial_tau_ext: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.timestamp = float(self.timestamp)
        if not np.isfinite(self.timestamp):
            raise ValueError("visualization packet timestamp must be finite")
        self.observed_q = _joint_vector(self.observed_q, "observed_q")
        if self.predicted_q is not None:
            values = np.asarray(self.predicted_q, dtype=np.float64)
            if values.ndim != 3 or values.shape[-1] != self.observed_q.size or not np.isfinite(values).all():
                raise ValueError("predicted_q must have shape [N,H,J] and be finite")
            self.predicted_q = values.copy()
        if self.predicted_ee_position is not None:
            values = np.asarray(self.predicted_ee_position, dtype=np.float64)
            if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
                raise ValueError("predicted_ee_position must have shape [N,H,3] and be finite")
            if self.predicted_q is not None and values.shape[:2] != self.predicted_q.shape[:2]:
                raise ValueError("predicted_ee_position and predicted_q must share [N,H]")
            self.predicted_ee_position = values.copy()
        if self.playback_q is not None:
            values = _joint_vector(self.playback_q, "playback_q")
            if values.size != self.observed_q.size:
                raise ValueError(
                    "playback_q must have the same joint dimension as observed_q"
                )
            self.playback_q = values
        if self.predicted_tau is not None:
            values = np.asarray(self.predicted_tau, dtype=float)
            if self.predicted_q is None or values.shape != self.predicted_q.shape or not np.isfinite(values).all():
                raise ValueError('predicted_tau must match predicted_q [N,H,J] and be finite')
            self.predicted_tau = values.copy()
        if self.future_timestamps is not None:
            values = np.asarray(self.future_timestamps, dtype=float)
            if (self.predicted_q is None or values.shape != (self.predicted_q.shape[1],)
                    or not np.isfinite(values).all() or np.any(np.diff(values) <= 0)):
                raise ValueError('future_timestamps must be finite increasing [H] seconds')
            self.future_timestamps = values.copy()
        if self.initial_tau is not None:
            self.initial_tau = _joint_vector(self.initial_tau, 'initial_tau')
            if self.initial_tau.shape != self.observed_q.shape:
                raise ValueError('initial_tau must match the joint dimension')
        for name in ('observed_tau_ext', 'initial_tau_ext'):
            values = getattr(self, name)
            if values is not None:
                values = _joint_vector(values, name)
                if values.shape != self.observed_q.shape:
                    raise ValueError(f'{name} must match the joint dimension')
                setattr(self, name, values)
        if self.observed_tau_ext is not None and (self.tau_ext_timestamp is None
                or not np.isfinite(self.tau_ext_timestamp)):
            raise ValueError('observed_tau_ext requires a finite source timestamp')


def _joint_vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.ndim != 1 or result.size < 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite joint vector")
    return result.copy()


def _put_latest(sample_queue: Any, value: Any) -> None:
    """Non-blocking bounded queue update; an old frame is disposable."""
    try:
        sample_queue.put_nowait(value)
        return
    except (queue.Full, OSError, ValueError):
        pass
    try:
        sample_queue.get_nowait()
    except (queue.Empty, OSError, ValueError):
        return
    try:
        sample_queue.put_nowait(value)
    except (queue.Full, OSError, ValueError):
        pass


class MujocoKinematicVisualizer:
    """Process-isolated latest-only visualizer.

    MuJoCo never steps dynamics here.  It only writes qpos and calls
    ``mj_forward`` for the displayed pose and scratch FK data.  Online
    packets display measured ``observed_q``; offline replay may set
    ``VisualizationPacket.playback_q`` to display a WM command instead.
    """

    def __init__(self, config: Any) -> None:
        if not bool(config.enabled):
            self.enabled = False
            return
        self.enabled = True
        self.config = config
        self._ctx = mp.get_context("spawn")
        self._queue = self._ctx.Queue(maxsize=1)
        self._process: mp.Process | None = None
        self._latest_observed = np.zeros(len(config.robot_joint_names), dtype=np.float64)
        self._latest_observed_timestamp = -np.inf
        self._latest_prediction: VisualizationPacket | None = None
        self._prediction_id = -1
        self._last_prediction_publish_s = -np.inf
        self._initial_tau = None
        self._episode_id = 0
        self._latest_tau_ext = None
        self._tau_ext_timestamp = None
        self._initial_tau_ext = None
        self._queue.cancel_join_thread()

    def start(self) -> None:
        if not self.enabled or self._process is not None:
            return
        self._process = self._ctx.Process(
            target=_visualizer_process,
            args=(self.config, self._queue),
            name="nero-mujoco-kinematic-visualizer",
            daemon=True,
        )
        self._process.start()

    def publish_observed(self, timestamp: float, observed_q: Any, *,
                         tau_ext: Any | None = None, tau_ext_timestamp: float | None = None) -> None:
        if not self.enabled:
            return
        try:
            timestamp = float(timestamp)
            observed = _joint_vector(observed_q, "observed_q")
            if not np.isfinite(timestamp):
                raise ValueError("observed visualization timestamp must be finite")
            # Online callers use the monotonic runtime clock.  A delayed state
            # callback must not move the displayed robot back in time.
            if timestamp < self._latest_observed_timestamp:
                return
            self._latest_observed = observed
            self._latest_observed_timestamp = timestamp
            previous = self._latest_prediction
            packet = VisualizationPacket(
                timestamp,
                self._latest_observed,
                None if previous is None else previous.predicted_q,
                None if previous is None else previous.predicted_ee_position,
                None if previous is None else previous.prediction_id,
                None if previous is None else previous.playback_q,
                None if previous is None else previous.predicted_tau,
                None if previous is None else previous.future_timestamps,
                self._initial_tau,
                self._episode_id,
                tau_ext,
                tau_ext_timestamp,
                self._initial_tau_ext,
            )
            self._latest_tau_ext = packet.observed_tau_ext
            self._tau_ext_timestamp = packet.tau_ext_timestamp
            if self._initial_tau_ext is None and packet.observed_tau_ext is not None:
                self._initial_tau_ext = packet.observed_tau_ext.copy()
                packet.initial_tau_ext = self._initial_tau_ext.copy()
            _put_latest(self._queue, packet)
        except Exception:
            log.debug("dropping invalid observed visualization packet", exc_info=True)

    def publish_prediction(
        self,
        timestamp: float,
        observed_q: Any,
        predicted_q: Any,
        *,
        prediction_id: int | None = None,
        predicted_ee_position: Any | None = None,
        playback_q: Any | None = None,
        predicted_tau: Any | None = None,
        future_timestamps: Any | None = None,
    ) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        period = 1.0 / float(self.config.prediction_visualization_hz)
        if prediction_id is not None and prediction_id == self._prediction_id:
            return
        if now - self._last_prediction_publish_s < period:
            return
        try:
            timestamp = float(timestamp)
            if not np.isfinite(timestamp):
                raise ValueError("prediction visualization timestamp must be finite")
            predicted = np.asarray(predicted_q, dtype=np.float64)
            if self._latest_observed_timestamp == -np.inf:
                # This is only needed for offline callers that publish a
                # prediction before their first observed packet.
                self._latest_observed = _joint_vector(observed_q, "observed_q")
                self._latest_observed_timestamp = timestamp
            packet = VisualizationPacket(
                max(timestamp, self._latest_observed_timestamp),
                self._latest_observed.copy(),
                predicted,
                None if predicted_ee_position is None else np.asarray(predicted_ee_position, dtype=np.float64),
                prediction_id,
                None if playback_q is None else np.asarray(playback_q, dtype=np.float64),
                predicted_tau,
                future_timestamps,
                self._initial_tau,
                self._episode_id,
                self._latest_tau_ext,
                self._tau_ext_timestamp,
                self._initial_tau_ext,
            )
        except Exception:
            log.debug("dropping invalid prediction visualization packet", exc_info=True)
            return
        self._latest_prediction = packet
        if packet.predicted_tau is not None and self._initial_tau is None:
            self._initial_tau = packet.predicted_tau[getattr(self.config, 'sample_id', 0), 0].copy()
            packet.initial_tau = self._initial_tau.copy()
        self._prediction_id = -1 if prediction_id is None else int(prediction_id)
        self._last_prediction_publish_s = now
        _put_latest(self._queue, packet)

    def reset(self) -> None:
        if not self.enabled:
            return
        self._latest_prediction = None
        self._prediction_id = -1
        self._last_prediction_publish_s = -np.inf
        self._initial_tau = None
        self._episode_id += 1
        self._latest_tau_ext = self._initial_tau_ext = None
        self._tau_ext_timestamp = None
        _put_latest(self._queue, 'reset')

    def close(self) -> None:
        if not self.enabled:
            return
        _put_latest(self._queue, None)
        process = self._process
        self._process = None
        if process is not None:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        try:
            self._queue.close()
        except Exception:
            pass

    publish = publish_prediction
    publish_observation = publish_observed


class MujocoKinematicFK:
    """Synchronous FK helper used by tests and offline validation.

    ``display_data`` and ``scratch_data`` are intentionally separate; callers
    can update the observed pose without clobbering a prediction rollout.
    """

    def __init__(self, config: Any) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = load_kinematic_model(mujoco, config.mujoco_model_path)
        self.display_data = mujoco.MjData(self.model)
        self.scratch_data = mujoco.MjData(self.model)
        self.addresses, self.ee_id, self.ee_kind = _resolve_ids(mujoco, self.model, config)

    def set_observed_q(self, q: Any) -> np.ndarray:
        return _forward_position(self.mujoco, self.model, self.display_data, self.addresses, _joint_vector(q, "observed_q"), self.ee_id, self.ee_kind)

    def predicted_ee_positions(self, predicted_q: Any) -> np.ndarray:
        values = np.asarray(predicted_q, dtype=np.float64)
        if values.ndim != 3 or values.shape[-1] != self.addresses.size or not np.isfinite(values).all():
            raise ValueError(f"predicted_q must have shape [N,H,{self.addresses.size}]")
        return np.stack(
            [[_forward_position(self.mujoco, self.model, self.scratch_data, self.addresses, q, self.ee_id, self.ee_kind) for q in trajectory] for trajectory in values],
            axis=0,
        )


def load_kinematic_model(mujoco: Any, path: Any) -> Any:
    """Accept MJCF or the deployment URDF, retaining fixed flange/TCP frames."""
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != '.urdf':
        return mujoco.MjModel.from_xml_path(str(path))
    root = ET.parse(path).getroot()
    assets, converted = {}, {}
    for mesh in root.iter('mesh'):
        filename = mesh.get('filename')
        if not filename:
            continue
        source = (path.parent / filename).resolve()
        if source.suffix.lower() == '.dae':
            # The G2 gripper uses COLLADA; MuJoCo accepts STL/OBJ. Convert in
            # memory with Pinocchio's mesh loader, leaving repository assets intact.
            if source not in converted:
                try:
                    from coal import MeshLoader
                except ImportError:
                    from hppfcl import MeshLoader
                geometry = MeshLoader().load(str(source))
                vertices = np.asarray([geometry.vertex(i) for i in range(geometry.num_vertices)])
                indices = np.asarray([[geometry.tri_indices(i)[j] for j in range(3)]
                                      for i in range(geometry.num_tris)])
                records = np.zeros(geometry.num_tris, dtype=[('normal', '<f4', (3,)),
                                      ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
                records['vertices'] = vertices[indices]
                asset_name = f'wm_gripper_{len(converted)}.stl'
                assets[asset_name] = bytes(80) + struct.pack('<I', geometry.num_tris) + records.tobytes()
                converted[source] = asset_name
            mesh.set('filename', converted[source])
        else:
            mesh.set('filename', str(source))
    extension = root.find('mujoco')
    if extension is None:
        extension = ET.SubElement(root, 'mujoco')
    compiler = extension.find('compiler')
    if compiler is None:
        compiler = ET.SubElement(extension, 'compiler')
    compiler.attrib.update(fusestatic='false', discardvisual='false', balanceinertia='true')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'), assets=assets)


class TorqueBars:
    """Fixed per-joint initial magnitudes; one common length multiplier."""
    def __init__(self, config):
        self.config = config
        self.baseline = None

    def lengths(self, tau):
        tau = _joint_vector(tau, 'torque')
        if self.baseline is None:
            self.baseline = np.maximum(np.abs(tau), self.config.torque_baseline_floor_nm)
        return self.config.torque_base_length_m * self.config.torque_gain * np.abs(tau) / self.baseline

    def draw(self, viewer, mj, model, data, tau, *, label_prefix='', offset_sign=1., show_label=True):
        lengths = self.lengths(tau)
        with viewer.lock():
            scene = viewer.user_scn
            for name, value, length, baseline in zip(self.config.robot_joint_names, tau, lengths, self.baseline):
                if scene.ngeom >= len(scene.geoms):
                    break
                joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
                anchor = data.xanchor[joint].copy()
                axis = data.xaxis[joint]
                # Move the bar clear of the robot mesh, while a thin stem
                # marks the exact joint origin. Its direction still follows
                # the joint's world axis at the current measured pose.
                local_axis = model.jnt_axis[joint]
                basis = np.eye(3)[int(np.argmin(np.abs(local_axis)))]
                perpendicular = (data.xmat[model.jnt_bodyid[joint]].reshape(3, 3)
                                 @ np.cross(local_axis, basis))
                perpendicular /= np.linalg.norm(perpendicular)
                start = anchor + offset_sign * perpendicular * getattr(self.config, 'torque_offset_m', .09)
                end = start + axis * length
                geom = scene.geoms[scene.ngeom]
                if offset_sign < 0:
                    color = (.95, .2, .8, .9) if value >= 0 else (.1, .9, .7, .9)
                else:
                    color = (1., .35, .1, .85) if value >= 0 else (.1, .55, 1., .85)
                mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_CYLINDER, np.zeros(3), start,
                               np.eye(3).reshape(-1), np.asarray(color, dtype=np.float32))
                # Connector size is a diameter; zero torque has a zero-length bar.
                mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CYLINDER, .012, start, end)
                geom.emission = .5
                if show_label:
                    geom.label = f'{label_prefix}J{name.removeprefix("joint")}: {value:+.2f} Nm ({abs(value)/baseline:.2f}x)'
                scene.ngeom += 1
                if scene.ngeom < len(scene.geoms):
                    stem = scene.geoms[scene.ngeom]
                    mj.mjv_initGeom(stem, mj.mjtGeom.mjGEOM_CYLINDER, np.zeros(3), anchor,
                                   np.eye(3).reshape(-1), np.asarray((.7, .7, .7, .6), dtype=np.float32))
                    mj.mjv_connector(stem, mj.mjtGeom.mjGEOM_CYLINDER, .002, anchor, start)
                    scene.ngeom += 1


def _resolve_ids(mujoco: Any, model: Any, config: Any) -> tuple[np.ndarray, int, int]:
    addresses = []
    for name in config.robot_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if joint_id < 0:
            raise ValueError(f"MuJoCo model has no configured robot joint {name!r}")
        if int(model.jnt_type[joint_id]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError(f"visualization joint {name!r} must be hinge or slide")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    if len(set(addresses)) != len(addresses):
        raise ValueError("configured MuJoCo robot joints must map to unique qpos addresses")
    if config.ee_site_name:
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(config.ee_site_name))
        if ee_id < 0:
            raise ValueError(f"MuJoCo model has no configured ee site {config.ee_site_name!r}")
        return np.asarray(addresses, dtype=np.int64), ee_id, int(mujoco.mjtObj.mjOBJ_SITE)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(config.ee_body_name))
    if ee_id < 0:
        raise ValueError(f"MuJoCo model has no configured ee body {config.ee_body_name!r}")
    return np.asarray(addresses, dtype=np.int64), ee_id, int(mujoco.mjtObj.mjOBJ_BODY)


def _forward_position(mujoco: Any, model: Any, data: Any, addresses: np.ndarray, q: np.ndarray, ee_id: int, ee_kind: int) -> np.ndarray:
    if q.shape != addresses.shape:
        raise ValueError(f"q shape {q.shape} does not match configured joint mapping {addresses.shape}")
    data.qpos[addresses] = q
    mujoco.mj_forward(model, data)
    if ee_kind == int(mujoco.mjtObj.mjOBJ_SITE):
        return np.asarray(data.site_xpos[ee_id], dtype=np.float64).copy()
    return np.asarray(data.xpos[ee_id], dtype=np.float64).copy()


def _draw(viewer: Any, mujoco: Any, observed_position: np.ndarray, trajectories: np.ndarray, config: Any) -> None:
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        capacity = len(scene.geoms)
        identity = np.eye(3, dtype=np.float64).reshape(-1)
        # Use a highly transparent light green for the displayed q prediction
        # samples so the cloud remains visible without obscuring the robot.
        trajectory_color = np.asarray((0.55, 0.5, 0.65, 1.0), dtype=np.float32)
        for trajectory in trajectories:
            for step_index, position in enumerate(trajectory):
                if scene.ngeom >= capacity:
                    return
                rgba = trajectory_color
                size = np.full(3, float(config.point_size) * (1.0 - 0.35 * step_index / max(1, trajectory.shape[0] - 1)))
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, size, position, identity, rgba)
                scene.ngeom += 1
        if scene.ngeom < capacity:
            # Mark the current displayed TCP pose separately from the sampled
            # future points: a translucent red box remains easy to distinguish
            # while still showing the trajectory behind it.
            # Keep the current-observation marker visible when trajectory
            # points are configured very small.  ``point_size`` controls only
            # the predicted samples; the marker has a practical lower bound.
            marker_edge = max(float(config.point_size) * 2.5, 0.012)
            marker_size = np.full(3, marker_edge)
            marker_rgba = np.asarray((1.0, 0.0, 0.0, 0.28), dtype=np.float32)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_BOX,
                marker_size,
                observed_position,
                identity,
                marker_rgba,
            )
            scene.ngeom += 1


def _visualizer_process(config: Any, sample_queue: Any) -> None:
    try:
        import mujoco

        model = load_kinematic_model(mujoco, config.mujoco_model_path)
        addresses, ee_id, ee_kind = _resolve_ids(mujoco, model, config)
        log.info("CARS-WM joint order -> MuJoCo qpos addresses: %s", dict(zip(config.robot_joint_names, addresses.tolist())))
        display_data = mujoco.MjData(model)
        scratch_data = mujoco.MjData(model)
        bars = TorqueBars(config) if hasattr(config, 'torque_base_length_m') else None
        ext_bars = TorqueBars(config) if bars is not None else None
        viewer = None
        if not bool(config.headless):
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(model, display_data)
            if bars is not None:
                viewer.opt.label = mujoco.mjtLabel.mjLABEL_GEOM
        latest: VisualizationPacket | None = None
        latest_trajectory = np.empty((0, 0, 3), dtype=np.float64)
        computed_prediction_id: int | None = None
        latest_packet_timestamp = -np.inf
        current_episode = None
        timing_reported = False
        fk_timing_reported = False
        next_render = time.monotonic()
        while True:
            try:
                item = sample_queue.get(timeout=0.05)
                if item is None:
                    break
                if isinstance(item, str) and item == 'reset':
                    latest = None
                    latest_trajectory = np.empty((0, 0, 3))
                    computed_prediction_id = None
                    if bars is not None:
                        bars.baseline = None
                        ext_bars.baseline = None
                    if viewer is not None:
                        with viewer.lock():
                            viewer.user_scn.ngeom = 0
                        viewer.sync()
                    continue
                if float(item.timestamp) < latest_packet_timestamp:
                    continue
                latest = item
                # Drain stale packets so rendering is always latest-only.
                while True:
                    try:
                        item = sample_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        return
                    if isinstance(item, str) and item == 'reset':
                        # Process reset before any subsequent episode packet.
                        latest_trajectory = np.empty((0, 0, 3))
                        computed_prediction_id = None
                        if bars is not None:
                            bars.baseline = None
                            ext_bars.baseline = None
                        latest.predicted_q = latest.predicted_tau = latest.observed_tau_ext = None
                        continue
                    if float(item.timestamp) < latest_packet_timestamp:
                        continue
                    latest = item
            except queue.Empty:
                pass
            if latest is None:
                continue
            latest_packet_timestamp = float(latest.timestamp)
            if latest.episode_id != current_episode:
                current_episode = latest.episode_id
                latest_trajectory = np.empty((0, 0, 3))
                computed_prediction_id = None
                if bars is not None:
                    bars.baseline = None
                    ext_bars.baseline = None
            if bars is not None and latest.initial_tau is not None:
                bars.baseline = np.maximum(np.abs(latest.initial_tau), config.torque_baseline_floor_nm)
            if ext_bars is not None and latest.initial_tau_ext is not None:
                ext_bars.baseline = np.maximum(np.abs(latest.initial_tau_ext), config.torque_baseline_floor_nm)
            # ``playback_q`` is used only by offline simulation/replay.  The
            # real-time hardware path never sets it and therefore continues
            # to show the measured observed_q pose.
            display_q = latest.playback_q if latest.playback_q is not None else latest.observed_q
            _forward_position(mujoco, model, display_data, addresses, display_q, ee_id, ee_kind)
            if (
                latest.predicted_q is not None
                and latest.prediction_id is not None
                and latest.prediction_id != computed_prediction_id
            ):
                predicted = latest.predicted_q
                fk_started = time.perf_counter()
                # Always execute FK in this process from predicted q.  A packet
                # may carry a precomputed position for auditing, but it is not
                # used as a rendering shortcut: every q goes through the
                # independent scratch MjData and mj_forward contract.
                latest_trajectory = np.stack(
                    [[_forward_position(mujoco, model, scratch_data, addresses, q, ee_id, ee_kind) for q in trajectory] for trajectory in predicted],
                    axis=0,
                )
                latest.predicted_ee_position = latest_trajectory
                computed_prediction_id = latest.prediction_id
                fk_elapsed_ms = (time.perf_counter() - fk_started) * 1e3
                if not fk_timing_reported:
                    log.info("sampled futures FK: N=%s H_f=%s shape=%s elapsed_ms=%.3f", predicted.shape[0], predicted.shape[1], latest_trajectory.shape, fk_elapsed_ms)
                    fk_timing_reported = True
            if viewer is not None and time.monotonic() >= next_render:
                render_started = time.perf_counter()
                display_q = latest.playback_q if latest.playback_q is not None else latest.observed_q
                # Keep the reference marker tied to the measured/recorded
                # observation.  Offline replay may display a predicted
                # playback_q on the robot, but comparing the WM trajectory to
                # that moving pose would hide its offset from the true anchor.
                observed_position = _forward_position(
                    mujoco, model, scratch_data, addresses,
                    latest.observed_q, ee_id, ee_kind,
                )
                expired = latest.future_timestamps is not None and time.monotonic() > latest.future_timestamps[-1]
                _draw(viewer, mujoco, observed_position,
                      np.empty((0, 0, 3)) if expired else latest_trajectory, config)
                if bars is not None and latest.predicted_tau is not None:
                    times = latest.future_timestamps
                    wall = time.monotonic()
                    if times is None or wall <= times[-1]:
                        index = 0 if times is None else min(int(np.searchsorted(times, wall)), len(times)-1)
                        # Anchor the scale to the first forecast, independent of render latency.
                        if bars.baseline is None:
                            bars.lengths(latest.predicted_tau[config.sample_id, 0])
                        bars.draw(viewer, mujoco, model, display_data,
                                  latest.predicted_tau[config.sample_id, index], label_prefix='WM ',
                                  show_label=getattr(config, 'torque_label_source', 'both') in ('wm', 'both'))
                if (ext_bars is not None and latest.observed_tau_ext is not None
                        and 0 <= time.monotonic()-latest.tau_ext_timestamp
                        <= getattr(config, 'tau_ext_stale_s', .15)):
                    ext_bars.draw(viewer, mujoco, model, display_data,
                                  latest.observed_tau_ext, label_prefix='EXT ', offset_sign=-1.,
                                  show_label=getattr(config, 'torque_label_source', 'both') in ('tau_ext', 'both'))
                viewer.sync()
                render_elapsed_ms = (time.perf_counter() - render_started) * 1e3
                if not timing_reported:
                    log.info("MuJoCo render elapsed_ms=%.3f", render_elapsed_ms)
                    timing_reported = True
                next_render = time.monotonic() + 1.0 / float(config.render_fps)
                if not viewer.is_running():
                    break
    except Exception:
        log.exception("real-time kinematic visualization stopped")
    finally:
        if 'viewer' in locals() and viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass


MujocoRealtimeVisualizer = MujocoKinematicVisualizer

__all__ = ["MujocoKinematicFK", "MujocoKinematicVisualizer", "MujocoRealtimeVisualizer", "VisualizationPacket"]
