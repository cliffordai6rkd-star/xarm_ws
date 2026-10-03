"""Public GELLO/xArm H5 signal names and legacy reader aliases.

Collectors keep their internal leader/follower signal names. Only persistence
uses ``{arm}_{quantity}_{device}``, so an arm remains identifiable in single-arm
episodes as well as dual-arm episodes.
"""
from __future__ import annotations

import numpy as np

DEVICE_SCHEMA_VERSION = "gello_xarm/side_quantity_device/v3"
REQUIRED_XARM_ARMS = ("left", "right")
_EEPOSE_SIGNALS = frozenset({
    "ee_pose_follower", "ee_pose_xarm", "q_eepose_xarm", "eepose_xarm",
})
# Internal names remain shared with the collector; persistence gives every
# episode a stable ten-channel xArm layout, including single-arm episodes.
REQUIRED_XARM_SIGNALS = {
    "q_follower": 7,
    "dq_follower": 7,
    "tau_follower": 7,
    "q_cmd": 7,
    "tau_ext_l1": 1,
}
SHARED_DATASETS = frozenset({
    "timestamp_us", "sample_lateness_us", "model_observation_updated",
    "model_observation_timestamp_us", "model_prediction_age_us",
})

_SPECIAL_QUANTITIES = {
    "q_leader": "q_gello",
    "q_leader_mapped": "q_gello",
    "q_leader_raw": "q_raw_gello",
    "gello_feedback_current": "current_feedback_gello",
    "gello_current_cmd": "current_cmd_gello",
    "gello_feedback_valid": "feedback_valid_gello",
    "tau_ext_l1": "tau_ext_l1_xarm",
}


def device_quantity_name(dataset_name: str) -> str:
    """Map an internal signal to a quantity with the device suffix last."""
    if dataset_name in SHARED_DATASETS:
        return dataset_name
    if dataset_name in _SPECIAL_QUANTITIES:
        return _SPECIAL_QUANTITIES[dataset_name]
    tokens = dataset_name.split("_")
    for role, device in (("follower", "xarm"), ("leader", "gello")):
        if role in tokens:
            return "_".join([token for token in tokens if token != role] + [device])
    if dataset_name.endswith(("_xarm", "_gello")):
        return dataset_name
    # Commands and calculated torque channels describe the xArm signal.
    return f"{dataset_name}_xarm"


def public_dataset_name(arm_name: str, dataset_name: str) -> str:
    """Return a new-schema dataset basename (shared clocks stay unprefixed)."""
    if dataset_name in SHARED_DATASETS:
        return dataset_name
    if dataset_name in _EEPOSE_SIGNALS:
        # Preserve the two end-effector field names requested for this schema.
        quantity = "q_eepose_xarm" if arm_name == "left" else "eepose_xarm"
        return f"{arm_name}_{quantity}"
    return f"{arm_name}_{device_quantity_name(dataset_name)}"


def dataset_candidates(arm_name: str, dataset_name: str) -> tuple[str, ...]:
    """Reader candidates: new names, old arm names, then old single-arm names.

    Values are basenames relative to ``teleop``. Legacy unprefixed vectors may
    contain multiple arms; readers must still select by ``arm_names`` and width.
    The two calibrated leader-position spellings resolve to the same new name.
    """
    if dataset_name in SHARED_DATASETS:
        return (dataset_name,)
    aliases = (dataset_name,)
    if dataset_name in {"q_leader", "q_leader_mapped", "q_gello"}:
        aliases = tuple(dict.fromkeys((dataset_name, "q_leader_mapped", "q_leader")))
    if dataset_name in {"tau_ext_l1", "tau_ext_l1_xarm"}:
        aliases = tuple(dict.fromkeys((dataset_name, "tau_ext_l1", "tau_ext_l1_xarm")))
    if dataset_name in _EEPOSE_SIGNALS:
        aliases = tuple(dict.fromkeys((dataset_name, "ee_pose_follower", "ee_pose_xarm",
                                       "q_eepose_xarm", "eepose_xarm")))
    return tuple(dict.fromkeys([
        public_dataset_name(arm_name, dataset_name),
        *(f"{arm_name}_{name}" for name in aliases),
        *aliases,
    ]))


def required_xarm_dataset_names() -> tuple[str, ...]:
    """The required public channels, with each arm kept separate."""
    return tuple(public_dataset_name(side, signal)
                 for side in REQUIRED_XARM_ARMS for signal in REQUIRED_XARM_SIGNALS)


def ensure_required_xarm_datasets(
    teleop,
    recorded_arm_names: tuple[str, ...] | list[str] | None = None,
    missing_context: str = "not_recorded",
) -> tuple[str, ...]:
    """Complete an xArm H5 layout without inventing unrecorded measurements.

    Existing signal arrays and recorded ``arm_names`` are preserved. Missing
    fields contain NaN and explicitly false validity channels. Return all new
    dataset basenames so a migration can journal and undo the additions.
    """
    import h5py

    if "timestamp_us" not in teleop:
        raise ValueError("required xArm layout needs teleop/timestamp_us")
    count = teleop["timestamp_us"].shape[0]
    recorded = tuple(_decode_text(side) for side in (
        teleop.attrs.get("arm_names", ()) if recorded_arm_names is None
        else recorded_arm_names))
    string_dtype = h5py.string_dtype(encoding="utf-8")
    teleop.attrs["recorded_arm_names"] = np.asarray(recorded, dtype=string_dtype)
    teleop.attrs["schema_arm_names"] = np.asarray(REQUIRED_XARM_ARMS, dtype=string_dtype)
    teleop.attrs["required_xarm_datasets"] = np.asarray(required_xarm_dataset_names(), dtype=string_dtype)
    teleop.attrs["missing_signal_policy"] = "NaN with false validity; no measurements synthesized"
    created: list[str] = []
    validity_signals = {
        "q_follower": "q_follower_valid",
        "dq_follower": "dq_valid_follower",
        "tau_follower": "torque_valid_follower",
        "q_cmd": "q_cmd_send_ok",
        "tau_ext_l1": "tau_free_valid",
    }
    units = {"q_follower": "rad", "dq_follower": "rad/s", "tau_follower": "Nm",
             "q_cmd": "rad", "tau_ext_l1": "Nm"}
    states = {"q_follower": "q", "dq_follower": "dq", "tau_follower": "torque",
              "q_cmd": "q", "tau_ext_l1": "torque_norm"}
    source_clocks = {
        "q_follower": "q_follower_timestamp_us",
        "dq_follower": "q_follower_timestamp_us",
        "tau_follower": "q_follower_timestamp_us",
        "q_cmd": "q_cmd_timestamp_us",
        "tau_ext_l1": "tau_free_source_timestamp_us",
    }

    for side in REQUIRED_XARM_ARMS:
        for signal, width in REQUIRED_XARM_SIGNALS.items():
            name = public_dataset_name(side, signal)
            placeholder = name not in teleop
            if placeholder:
                data = np.full((count, width), np.nan, dtype=np.float64)
                dataset = teleop.create_dataset(name, data=data, compression="gzip" if data.size else None)
                created.append(name)
                dataset.attrs["placeholder"] = True
                dataset.attrs["source"] = "not_recorded"
                dataset.attrs["missing_reason"] = (
                    "arm_not_recorded" if side not in recorded else f"{signal}_not_recorded")
                dataset.attrs["missing_context"] = missing_context
                dataset.attrs["state_name"] = states[signal]
                dataset.attrs["lowpass"] = False
                dataset.attrs["median_window"] = 1
            else:
                dataset = teleop[name]
                if dataset.shape != (count, width):
                    raise ValueError(f"required xArm dataset {name} has shape {dataset.shape}; "
                                     f"expected {(count, width)}")
                placeholder = bool(dataset.attrs.get("placeholder", False))

            dataset.attrs["arm_name"] = side
            dataset.attrs["device_name"] = "xarm"
            dataset.attrs["source_signal_name"] = signal
            dataset.attrs.setdefault("unit", units[signal])
            dataset.attrs.setdefault("timestamp_path", "teleop/timestamp_us")
            dataset.attrs.setdefault("joint_layout", "J1..J7 per arm")
            source_clock = public_dataset_name(side, source_clocks[signal])
            if source_clock in teleop:
                dataset.attrs.setdefault("source_timestamp_path", f"teleop/{source_clock}")
            validity_name = public_dataset_name(side, validity_signals[signal])
            if placeholder:
                # The model-valid flag may describe other recorded model
                # outputs; an absent norm must not overwrite that information.
                if signal == "tau_ext_l1":
                    validity_name = public_dataset_name(side, "tau_ext_l1_valid")
                elif validity_name in teleop and np.any(teleop[validity_name][:]):
                    quantity = device_quantity_name(signal).removesuffix("_xarm")
                    validity_name = public_dataset_name(side, f"{quantity}_available")
                if validity_name not in teleop:
                    validity = teleop.create_dataset(validity_name, data=np.zeros((count, 1), np.uint8))
                    created.append(validity_name)
                    validity.attrs["unit"] = "boolean"
                    validity.attrs["state_name"] = "validity"
                    validity.attrs["source"] = "not_recorded"
                    validity.attrs["definition"] = "0 because the corresponding signal was not recorded"
                    validity.attrs["arm_name"] = side
                    validity.attrs["device_name"] = "xarm"
                    validity.attrs["timestamp_path"] = "teleop/timestamp_us"
                dataset.attrs["validity_path"] = f"teleop/{validity_name}"
            elif validity_name in teleop:
                dataset.attrs.setdefault("validity_path", f"teleop/{validity_name}")

            if signal == "tau_ext_l1":
                dataset.attrs.setdefault("norm_order", 1)
                dataset.attrs.setdefault("joint_count", 7)
                dataset.attrs.setdefault("invalid_value", "NaN")
                dataset.attrs["joint_layout"] = "scalar L1 reduction over J1..J7 per arm"
                dataset.attrs.setdefault("input_signal", "tau_ext")
                dataset.attrs.setdefault("definition", "sum(abs(tau_ext[j])) over J1..J7")
                dataset.attrs["moving_average_window_recorded"] = (
                    "moving_average_window" in dataset.attrs and not placeholder)
                # An absent source clock/window cannot be inferred from today's
                # YAML. Keep only paths to objects actually present in this file.
                for key in ("source_timestamp_path", "model_sample_timestamp_path", "validity_path"):
                    target = dataset.attrs.get(key)
                    if target is not None and _decode_text(target) not in teleop.file:
                        del dataset.attrs[key]

    commands = tuple(_decode_text(name) for name in teleop.attrs.get("command_datasets", ()))
    teleop.attrs["command_datasets"] = np.asarray(tuple(dict.fromkeys([
        *commands, *(public_dataset_name(side, "q_cmd") for side in REQUIRED_XARM_ARMS)
    ])), dtype=string_dtype)
    return tuple(created)


def _decode_text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def ensure_xarm_eepose_datasets(teleop, recorded_arm_names, frame_settings=None) -> tuple[str, ...]:
    """Save one base-to-TCP transform per arm, with explicit missing validity."""
    import h5py

    count = teleop["timestamp_us"].shape[0]
    recorded = tuple(_decode_text(side) for side in recorded_arm_names)
    frame_settings = frame_settings or {}
    string_dtype = h5py.string_dtype(encoding="utf-8")
    teleop.attrs["required_eepose_datasets"] = np.asarray([
        public_dataset_name(side, "ee_pose_follower") for side in REQUIRED_XARM_ARMS
    ], dtype=string_dtype)
    created = []
    for side in REQUIRED_XARM_ARMS:
        name = public_dataset_name(side, "ee_pose_follower")
        missing = name not in teleop
        if missing:
            data = np.full((count, 4, 4), np.nan, dtype=np.float64)
            pose = teleop.create_dataset(name, data=data, compression="gzip" if data.size else None)
            created.append(name)
            pose.attrs["placeholder"] = True
            pose.attrs["source"] = "not_recorded"
            pose.attrs["missing_reason"] = (
                "arm_not_recorded" if side not in recorded else "ee_pose_follower_not_recorded")
        else:
            pose = teleop[name]
            if pose.shape != (count, 4, 4):
                raise ValueError(f"xArm end-effector dataset {name} has shape {pose.shape}; "
                                 f"expected {(count, 4, 4)}")
            pose.attrs.setdefault("source", "xarm_state_feedback")
        frames = frame_settings.get(side, {})
        pose.attrs["arm_name"] = side
        pose.attrs["device_name"] = "xarm"
        pose.attrs["source_signal_name"] = "ee_pose_follower"
        pose.attrs["state_name"] = "ee_pose"
        pose.attrs["representation"] = "homogeneous_transform"
        pose.attrs["definition"] = "base_to_tcp homogeneous transform; rotation in [:3,:3], translation in [:3,3]"
        pose.attrs["translation_unit"] = "m"
        pose.attrs["rotation_representation"] = "rotation_matrix"
        pose.attrs["frame_type"] = "end_effector"
        pose.attrs.setdefault("unit", "m")
        pose.attrs.setdefault("reference_frame", str(frames.get("base_frame", "base")))
        pose.attrs.setdefault("frame_name", str(frames.get("tcp_frame", "tcp")))
        pose.attrs["timestamp_path"] = "teleop/timestamp_us"
        pose.attrs["pose_layout"] = "one base_to_tcp transform per sample: (N,4,4)"
        for clock in ("q_follower_acquired_timestamp_us", "q_follower_timestamp_us"):
            source_clock = public_dataset_name(side, clock)
            if source_clock in teleop:
                pose.attrs["source_timestamp_path"] = f"teleop/{source_clock}"
                break

        valid_name = public_dataset_name(side, "eepose_valid")
        valid_values = np.isfinite(pose[:]).all(axis=(1, 2))
        if missing or pose.attrs.get("placeholder", False):
            valid_values[:] = False
        state_valid = public_dataset_name(side, "q_follower_valid")
        if state_valid in teleop and teleop[state_valid].attrs.get("source") != "not_recorded":
            flags = np.asarray(teleop[state_valid][:])
            if flags.shape not in {(count,), (count, 1)}:
                raise ValueError(f"xArm state validity {state_valid} must have one flag per sample")
            valid_values &= flags.reshape(count).astype(bool)
        if valid_name not in teleop:
            validity = teleop.create_dataset(valid_name, data=valid_values.astype(np.uint8)[:, None])
            created.append(valid_name)
        else:
            validity = teleop[valid_name]
            if validity.shape != (count, 1):
                raise ValueError(f"xArm pose validity {valid_name} must have shape {(count, 1)}")
            validity[:] = valid_values.astype(np.uint8)[:, None]
        validity.attrs["unit"] = "boolean"
        validity.attrs["arm_name"] = side
        validity.attrs["device_name"] = "xarm"
        validity.attrs["state_name"] = "validity"
        validity.attrs["definition"] = "finite recorded end-effector matrix and valid source state; 0 for missing or stale data"
        validity.attrs["timestamp_path"] = "teleop/timestamp_us"
        pose.attrs["validity_path"] = f"teleop/{valid_name}"
    return tuple(created)
