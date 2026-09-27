from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from xarm_stack.protocol import RpcServer, StatePublisher

log = logging.getLogger(__name__)


class GelloService:
    def __init__(self, config: dict[str, Any], state_endpoint: str, rate_hz: float = 100.0) -> None:
        self.config = config
        self.ids = tuple(int(x) for x in config.get("joint_ids", range(1, 8)))
        self.port = str(config["port"])
        self.baudrate = int(config.get("baudrate", 4_000_000))
        self.servo_types = tuple(config.get("servo_types", ("XM430_W210_T",) * len(self.ids)))
        if len(self.servo_types) != len(self.ids):
            raise ValueError("gello servo_types must match joint_ids")
        try:
            from gello.dynamixel.driver import DynamixelDriver
        except ImportError as exc:
            raise RuntimeError("gello_software is required by the Gello service") from exc
        try:
            self.driver = DynamixelDriver(self.ids, port=self.port, baudrate=self.baudrate)
        except TypeError:
            self.driver = DynamixelDriver(self.ids, self.servo_types, port=self.port, baudrate=self.baudrate)
        self.publisher = StatePublisher(state_endpoint)
        self.period_s = 1.0 / float(rate_hz)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.state_thread: threading.Thread | None = None
        self.enabled = False
        self._target: np.ndarray | None = None
        self._target_kp = float(config.get("reset_kp", 3.0))
        self._target_kd = float(config.get("reset_kd", 0.08))
        self._last_q: np.ndarray | None = None
        self._last_t: float | None = None
        self._target: np.ndarray | None = None

    def start(self) -> None:
        self.state_thread = threading.Thread(target=self._publish_loop, name="gello-state", daemon=True)
        self.state_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.state_thread is not None:
            self.state_thread.join(timeout=2.0)
        with self.lock:
            try:
                self._set_torque(np.zeros(len(self.ids)))
                self.driver.set_torque_mode(False)
            finally:
                self.driver.close()
        self.publisher.close()

    def methods(self) -> dict[str, Any]:
        return {
            "ping": lambda: {"service": "gello", "enabled": self.enabled},
            "state": self.state,
            "enable": self.enable,
            "disable": self.disable,
            "set_torque": self.set_torque,
            "move_interpolated": self.move_interpolated,
            "hold": self.hold,
            "stop": self.stop_command,
        }

    def state(self) -> dict[str, Any]:
        with self.lock:
            q, dq = self._read()
        return {"timestamp_us": time.time_ns() // 1000, "q": q, "dq": dq, "enabled": self.enabled}

    def enable(self) -> None:
        with self.lock:
            set_mode = getattr(self.driver, "set_operating_mode", None)
            if callable(set_mode):
                set_mode(int(self.config.get("current_mode", 0)))
            self.driver.set_torque_mode(True)
            self.enabled = True

    def disable(self) -> None:
        with self.lock:
            self._set_torque(np.zeros(len(self.ids)))
            self.driver.set_torque_mode(False)
            self.enabled = False
            self._target = None

    def set_torque(self, torque: list[float]) -> None:
        with self.lock:
            if not self.enabled:
                raise RuntimeError("Gello must be enabled before torque commands")
            self._target = None
            self._set_torque(np.asarray(torque, dtype=np.float64))

    def move_interpolated(self, q: list[float], duration_s: float = 3.0, rate_hz: float = 100.0) -> None:
        target = np.asarray(q, dtype=np.float64).reshape(len(self.ids))
        with self.lock:
            self.enable()
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            alpha = min(1.0, elapsed / max(float(duration_s), 1.0e-3))
            with self.lock:
                current, velocity = self._read()
                torque = self._target_kp * (target - current) - self._target_kd * velocity
                self._set_torque(torque)
            if alpha >= 1.0 and np.max(np.abs(target - current)) < float(self.config.get("reset_error_rad", 0.03)):
                break
            time.sleep(1.0 / max(float(rate_hz), 1.0))
        with self.lock:
            self._target = target.copy()

    def hold(self) -> None:
        with self.lock:
            q, _ = self._read()
            self._target = q.copy()

    def stop_command(self) -> None:
        self.hold()

    def _read(self) -> tuple[np.ndarray, np.ndarray]:
        result = self.driver.get_positions_and_velocities() if hasattr(self.driver, "get_positions_and_velocities") else None
        if result is None:
            q = np.asarray(self.driver.get_joints(), dtype=np.float64)
            now = time.monotonic()
            dt = max(now - self._last_t, 1.0e-6) if self._last_t is not None else 1.0e-2
            dq = np.zeros_like(q) if self._last_q is None else (q - self._last_q) / dt
            self._last_q, self._last_t = q.copy(), now
            return q, dq
        q = np.asarray(result[0], dtype=np.float64).reshape(len(self.ids))
        dq = np.asarray(result[1], dtype=np.float64).reshape(len(self.ids))
        return q, dq

    def _set_torque(self, torque: np.ndarray) -> None:
        torque = np.asarray(torque, dtype=np.float64).reshape(len(self.ids))
        torque = np.clip(torque, -float(self.config.get("max_torque", 0.5)), float(self.config.get("max_torque", 0.5)))
        setter = getattr(self.driver, "set_torque", None)
        if callable(setter):
            setter(torque.tolist())
            return
        setter = getattr(self.driver, "set_current", None)
        if callable(setter):
            scalar = np.asarray(self.config.get("torque_to_current", [1.0] * len(self.ids)), dtype=np.float64)
            setter((torque * scalar).tolist())
            return
        raise RuntimeError("Gello driver does not expose set_torque() or set_current()")

    def _publish_loop(self) -> None:
        next_t = time.monotonic()
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    if self.enabled and self._target is not None:
                        q, velocity = self._read()
                        self._set_torque(self._target_kp * (self._target - q) - self._target_kd * velocity)
                self.publisher.publish(self.state())
            except Exception:
                log.exception("Gello state publish failed")
            next_t += self.period_s
            time.sleep(max(0.0, next_t - time.monotonic()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gello hardware service")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rpc", default=None)
    parser.add_argument("--state", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    data = yaml.safe_load(Path(args.config).expanduser().read_text())
    services = data.get("services", {})
    rpc = args.rpc or services.get("gello_rpc", "tcp://127.0.0.1:5561")
    state = args.state or services.get("gello_state", "tcp://127.0.0.1:5562")
    service = GelloService(data["gello"], state, float(services.get("gello_state_hz", 100.0)))
    service.start()
    server = RpcServer(rpc, service.methods())
    log.info("Gello service ready rpc=%s state=%s", rpc, state)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.close()
        service.stop()


if __name__ == "__main__":
    raise SystemExit(main())
