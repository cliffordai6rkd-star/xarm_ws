"""Core contracts and orchestration for modular inference."""

from inference.core.base import (
    ActionChunkScheduler,
    InferenceBase,
    NullActionResolver,
    NullObservationProcessor,
    NullSafetyGuard,
    NullWorldModel,
)
from inference.core.contracts import ActionChunk, ControlTarget, InferenceCycle, Observation
__all__ = [
    "ActionChunk",
    "ActionChunkScheduler",
    "ControlTarget",
    "InferenceBase",
    "InferenceCycle",
    "NullActionResolver",
    "NullObservationProcessor",
    "NullSafetyGuard",
    "NullWorldModel",
    "NeroPipelineRunner",
    "ModularInferenceRunner",
    "Observation",
    "WmControlLayer",
    "WmReferenceAligner",
]


def __getattr__(name):
    if name in {"WmControlLayer", "WmReferenceAligner"}:
        from inference.core.wm_control import WmControlLayer, WmReferenceAligner

        return {"WmControlLayer": WmControlLayer, "WmReferenceAligner": WmReferenceAligner}[name]
    # ``inference.pipeline`` imports contracts through this package.  Importing
    # the compatibility runner eagerly would import the pipeline again and
    # create a circular import during the real DP CLI startup.
    if name in {"NeroPipelineRunner", "ModularInferenceRunner"}:
        from inference.core.legacy_runner import ModularInferenceRunner, NeroPipelineRunner

        return {"NeroPipelineRunner": NeroPipelineRunner, "ModularInferenceRunner": ModularInferenceRunner}[name]
    raise AttributeError(name)
