"""Inference package facade with optional heavy backends loaded on demand."""

from __future__ import annotations

import importlib


_EXPORTS = {
    "ActionChunk": ("inference.core", "ActionChunk"),
    "ActionChunkScheduler": ("inference.core", "ActionChunkScheduler"),
    "ControlTarget": ("inference.core", "ControlTarget"),
    "InferenceBase": ("inference.core", "InferenceBase"),
    "InferenceCycle": ("inference.core", "InferenceCycle"),
    "ModularInferenceRunner": ("inference.core", "ModularInferenceRunner"),
    "NeroPipelineRunner": ("inference.core", "NeroPipelineRunner"),
    "Observation": ("inference.core", "Observation"),
    "InferenceConfig": ("inference.config", "InferenceConfig"),
    "MujocoVisualizationConfig": ("inference.config", "MujocoVisualizationConfig"),
    "ArchitectureConfig": ("inference.config", "ArchitectureConfig"),
    "ExecutionConfig": ("inference.config", "ExecutionConfig"),
    "load_inference_config": ("inference.config", "load_inference_config"),
    "MTCController": ("inference.control.mtc", "MTCController"),
    "MTCResult": ("inference.control.mtc", "MTCResult"),
    "InferenceInput": ("inference.pipeline", "InferenceInput"),
    "InferenceOutput": ("inference.pipeline", "InferenceOutput"),
    "IKResult": ("inference.pipeline", "IKResult"),
    "NeroInferencePipeline": ("inference.pipeline", "NeroInferencePipeline"),
    "ContactWMInferencePipeline": ("inference.contact_wm_pipeline", "ContactWMInferencePipeline"),
    "ContactWorldModelInferencePipeline": ("inference.contact_wm_pipeline", "ContactWorldModelInferencePipeline"),
    "ContactWMPipeline": ("inference.contact_wm_pipeline", "ContactWMPipeline"),
    "ContactInferencePipeline": ("inference.contact_wm_pipeline", "ContactInferencePipeline"),
    "SWMInferencePipeline": ("inference.swm_pipeline", "SWMInferencePipeline"),
    "SWMPipeline": ("inference.swm_pipeline", "SWMPipeline"),
    "TorqueWorldModelInferencePipeline": ("inference.swm_pipeline", "TorqueWorldModelInferencePipeline"),
    "MujocoBackendConfig": ("inference.mujoco_backend", "MujocoBackendConfig"),
    "MujocoBackend": ("inference.mujoco_backend", "MujocoBackend"),
    "MujocoCommand": ("inference.mujoco_backend", "MujocoCommand"),
    "MujocoDynamicsBackend": ("inference.mujoco_backend", "MujocoDynamicsBackend"),
    "MujocoSimulationBackend": ("inference.mujoco_backend", "MujocoSimulationBackend"),
    "MujocoState": ("inference.mujoco_backend", "MujocoState"),
    "H5ObservationEpisode": ("inference.h5_observation_stream", "H5ObservationEpisode"),
    "H5ObservationStream": ("inference.h5_observation_stream", "H5ObservationStream"),
    "H5ObservationTick": ("inference.h5_observation_stream", "H5ObservationTick"),
    "load_h5_observation_stream": ("inference.h5_observation_stream", "load_h5_observation_stream"),
    "NeroInferenceRuntime": ("inference.runtime", "NeroInferenceRuntime"),
    "JointInferenceRuntime": ("inference.joint_runtime", "JointInferenceRuntime"),
    "JointDeployment": ("inference.joint_config", "JointDeployment"),
    "MujocoKinematicFK": ("inference.mujoco_visualization", "MujocoKinematicFK"),
    "MujocoKinematicVisualizer": ("inference.mujoco_visualization", "MujocoKinematicVisualizer"),
    "MujocoRealtimeVisualizer": ("inference.mujoco_visualization", "MujocoRealtimeVisualizer"),
    "VisualizationPacket": ("inference.mujoco_visualization", "VisualizationPacket"),
    "DiffusionPolicyInference": ("inference.model_inference", "DiffusionPolicyInference"),
    "VLAInference": ("inference.model_inference", "VLAInference"),
    "TAVLAInference": ("inference.model_inference", "TAVLAInference"),
    "TAVLAInferencePipeline": ("inference.model_inference", "TAVLAInferencePipeline"),
    "TAVLAPipeline": ("inference.model_inference", "TAVLAPipeline"),
    "DiffusionPolicy": ("inference.policies", "DiffusionPolicy"),
    "DPPolicy": ("inference.policies", "DPPolicy"),
    "ACTPolicy": ("inference.policies", "ACTPolicy"),
    "LeRobotDP": ("inference.policies", "LeRobotDP"),
    "LeRobotDiffusionPolicy": ("inference.policies", "LeRobotDiffusionPolicy"),
    "is_lerobot_checkpoint": ("inference.policies", "is_lerobot_checkpoint"),
    "TAVLA": ("inference.policies", "TAVLA"),
    "TAVLAAdapter": ("inference.policies", "TAVLAAdapter"),
    "TAVLAInferencePolicy": ("inference.policies", "TAVLAInferencePolicy"),
    "TAVLAPolicy": ("inference.policies", "TAVLAPolicy"),
    "TAVLAObservationBuilder": ("inference.policies", "TAVLAObservationBuilder"),
    "BasicSafetyGuard": ("inference.control", "BasicSafetyGuard"),
    "CallableActionResolver": ("inference.control", "CallableActionResolver"),
    "DirectActionResolver": ("inference.control", "DirectActionResolver"),
    "DPObservationBuffer": ("inference.stages", "DPObservationBuffer"),
    "ActionPlanExecutor": ("inference.stages", "ActionPlanExecutor"),
    "ComponentRegistry": ("inference.factory", "ComponentRegistry"),
    "POLICY_REGISTRY": ("inference.factory", "POLICY_REGISTRY"),
    "WORLD_MODEL_REGISTRY": ("inference.factory", "WORLD_MODEL_REGISTRY"),
    "CONTROLLER_REGISTRY": ("inference.factory", "CONTROLLER_REGISTRY"),
    "ActionPlan": ("inference.async_fast_slow", "ActionPlan"),
    "ActionPlanBuffer": ("inference.async_fast_slow", "ActionPlanBuffer"),
    "ActionTrajectory": ("inference.async_fast_slow", "ActionTrajectory"),
    "ControlWorker": ("inference.async_fast_slow", "ControlWorker"),
    "DPWorker": ("inference.async_fast_slow", "DPWorker"),
    "StateHistoryBuffer": ("inference.async_fast_slow", "StateHistoryBuffer"),
    "StateHistorySnapshot": ("inference.async_fast_slow", "StateHistorySnapshot"),
    "StateSample": ("inference.async_fast_slow", "StateSample"),
    "TimestampFastSlowRuntime": ("inference.async_fast_slow", "TimestampFastSlowRuntime"),
    "WMTarget": ("inference.async_fast_slow", "WMTarget"),
    "WMTargetBuffer": ("inference.async_fast_slow", "WMTargetBuffer"),
    "WMWorker": ("inference.async_fast_slow", "WMWorker"),
    "SimulationRunResult": ("inference.simulation_runner", "SimulationRunResult"),
    "SimulationRunnerConfig": ("inference.simulation_runner", "SimulationRunnerConfig"),
    "build_pipeline": ("inference.simulation_runner", "build_pipeline"),
    "run_h5_simulation": ("inference.simulation_runner", "run_h5_simulation"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value
