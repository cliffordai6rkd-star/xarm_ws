from __future__ import annotations

from nero_collection.arms.base import ArmInterface
from nero_collection.arms.mock import MockArm
from nero_collection.arms.process import IsolatedArmProcess
from nero_collection.config import ArmEndpointConfig


def build_arm(config: ArmEndpointConfig, backend: str) -> ArmInterface:
    normalized = backend.lower().replace("-", "_")
    if normalized in {"mock", "sim", "simulation"}:
        return MockArm(config)
    if normalized in {"pyagxarm", "py_agx_arm", "agx"}:
        return IsolatedArmProcess(config, backend=normalized)
    if normalized in {"xarm", "x_arm", "ufactory_xarm"}:
        return IsolatedArmProcess(config, backend="xarm")
    raise ValueError(f"Unsupported arm backend {backend!r}")
