from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from xarm_stack.protocol import RpcServer, StatePublisher

log = logging.getLogger(__name__)


class GripperService:
    def __init__(self, config: dict[str, Any], state_endpoint: str, rate_hz: float = 25.0) -> None:
        self.config = config
        self.backend = str(config.get("backend", "xarm")).lower()
        self.execution_enabled = bool(config.get("execution_enabled", False))
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.publisher = StatePublisher(state_endpoint)
        self.period_s = 1.0 / float(rate_hz)
        self._device: Any = None
        self._value = float(config.get("start_value", 1.0))
        self._force = 0.0
        self._mode = "width"
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.backend == "xarm":
            try:
                from xarm.wrapper import XArmAPI
            except ImportError as exc:
                raise RuntimeError("xArm SDK is required by the xArm gripper service") from exc
            ip = self.config.get("robot_ip", self.config.get("ip"))
            if not ip:
                raise ValueError("gripper.robot_ip is required for backend=xarm")
            self._device = XArmAPI(str(ip))
            # Constructing the SDK object is read-only.  Enabling the arm or
            # gripper is reserved for an explicit execution-enabled endpoint.
            if self.execution_enabled:
                self._call("motion_enable", True, required=True)
                self._call("set_gripper_enable", True, required=True)
                self._call("set_gripper_mode", 0, required=True)
        elif self.backend == "none":
            self._device = None
        else:
            raise ValueError(f"unsupported gripper backend {self.backend!r}")
        self.thread = threading.Thread(target=self._publish_loop, name="gripper-state", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self._device is not None:
            if self.execution_enabled:
                self._call("set_gripper_enable", False, required=False)
            for name in ("disconnect", "close"):
                fn = getattr(self._device, name, None)
                if callable(fn):
                    fn()
                    break
        self.publisher.close()

    def methods(self) -> dict[str, Any]:
        return {
            "ping": lambda: {"service": "gripper", "backend": self.backend},
            "state": self.state,
            "enable": self.enable,
            "disable": self.disable,
            "reset": self.reset,
            "move": self.move,
            "move_interpolated": self.move_interpolated,
            "stop": self.stop_command,
        }

    def state(self) -> dict[str, Any]:
        with self.lock:
            value = self._read_value()
            return {"timestamp_us": time.time_ns() // 1000, "value": value, "force": self._force, "mode": self._mode}

    def enable(self) -> None:
        self._require_execution("enable")
        self._call("set_gripper_enable", True, required=False)

    def disable(self) -> None:
        self._require_execution("disable")
        self._call("set_gripper_enable", False, required=False)

    def reset(self) -> None:
        self.move(float(self.config.get("reset_value", 0.0)), float(self.config.get("force_n", 3.0)), "width")

    def move(self, value: float, force_n: float = 0.0, mode: str = "width") -> None:
        with self.lock:
            self._require_execution("move")
            self._value = float(value)
            self._force = float(force_n)
            self._mode = str(mode)
            if self._device is None:
                return
            if mode == "width":
                position = self._width_to_position(self._value)
            else:
                position = int(round(self._value))
            self._call("set_gripper_position", position, wait=False, required=True)

    def move_interpolated(self, value: float, duration_s: float = 1.0, rate_hz: float = 25.0) -> None:
        with self.lock:
            start = self._read_value()
        steps = max(1, int(round(float(duration_s) * float(rate_hz))))
        for index in range(1, steps + 1):
            self.move(start + (float(value) - start) * index / steps, self._force, self._mode)
            time.sleep(1.0 / max(float(rate_hz), 1.0))

    def stop_command(self) -> None:
        self.move(self._read_value(), self._force, self._mode)

    def _read_value(self) -> float:
        result = self._call("get_gripper_position", required=False)
        if isinstance(result, tuple) and len(result) == 2:
            result = result[1]
        try:
            return float(result)
        except (TypeError, ValueError):
            return self._value

    def _width_to_position(self, width_m: float) -> int:
        minimum = float(self.config.get("min_width_m", 0.0))
        maximum = float(self.config.get("max_width_m", 0.085))
        normalized = 0.0 if maximum <= minimum else (width_m - minimum) / (maximum - minimum)
        normalized = min(1.0, max(0.0, normalized))
        open_position = float(self.config.get("open_position", 800))
        close_position = float(self.config.get("close_position", 0))
        return int(round(open_position + normalized * (close_position - open_position)))

    def _call(self, name: str, *args: Any, required: bool = False, **kwargs: Any) -> Any:
        if self._device is None:
            if required:
                raise RuntimeError("gripper device is not connected")
            return None
        fn = getattr(self._device, name, None)
        if not callable(fn):
            if required:
                raise RuntimeError(f"gripper backend does not expose {name}()")
            return None
        return fn(*args, **kwargs)

    def _require_execution(self, operation: str) -> None:
        if not self.execution_enabled:
            raise RuntimeError(
                f"gripper execution is disabled; set execution_enabled=true explicitly before {operation}"
            )

    def _publish_loop(self) -> None:
        next_t = time.monotonic()
        while not self.stop_event.is_set():
            try:
                self.publisher.publish(self.state())
            except Exception:
                log.exception("gripper state publish failed")
            next_t += self.period_s
            time.sleep(max(0.0, next_t - time.monotonic()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="gripper hardware service")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rpc", default=None)
    parser.add_argument("--state", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    data = yaml.safe_load(Path(args.config).expanduser().read_text())
    services = data.get("services", {})
    rpc = args.rpc or services.get("gripper_rpc", "tcp://127.0.0.1:5571")
    state = args.state or services.get("gripper_state", "tcp://127.0.0.1:5572")
    service = GripperService(data["gripper"], state, float(services.get("gripper_state_hz", 25.0)))
    service.start()
    server = RpcServer(rpc, service.methods())
    log.info("gripper service ready rpc=%s state=%s", rpc, state)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.close()
        service.stop()


if __name__ == "__main__":
    raise SystemExit(main())
