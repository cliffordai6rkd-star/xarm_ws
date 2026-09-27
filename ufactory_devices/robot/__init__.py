try:
    from .uf_robot import UFRobotConfig, UFRobot
except ModuleNotFoundError as exc:
    if exc.name != "xarm":
        raise
    # Keep lightweight adapters importable on machines that only run the
    # orchestration or simulation side of the stack.
    UFRobotConfig = None
    UFRobot = None

from .xarm_adapter import XArmAdapter

__all__ = ["UFRobotConfig", "UFRobot", "XArmAdapter"]
