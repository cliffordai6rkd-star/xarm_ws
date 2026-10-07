from copy import deepcopy
from pathlib import Path
import threading

import numpy as np
import pytest
import yaml

from inference.joint_config import JointDeployment, load_joint_policy
from inference.joint_runtime import JointObservation
from inference.policies.errors import JointActionContractError
from inference.policies.pi0_joint import MockPiConnection, Pi0JointPolicy
from test_joint_policy_inference import deployment_file, runtime_fixture


def pi_config():
    return {"host": "127.0.0.1", "port": 8000, "prompt": "erase board",
            "action_hz": 30, "action_horizon": 50, "consume_steps": 20,
            "output_key": "actions", "request_timeout_s": 1.,
            "interface": {"training_config": "pi0_xarm_lora_erase_board",
                          "state_key": "observation/state", "prompt_key": "prompt",
                          "images": {"left_wrist": "observation/image",
                                     "right_wrist": "observation/wrist_image"},
                          "state_semantic": "zero", "action_semantic": "action.joint",
                          "coordinate_frame": None, "representation": "joint_positions",
                          "units": "rad", "joint_names": [f"joint{i}" for i in range(1, 8)],
                          "image_format": "uint8_hwc_rgb", "output_space": "physical_absolute",
                          "action_hz": 30}}


def observation(stamp=100):
    images = {"observation/image": np.full((224, 224, 3), 17, np.uint8),
              "observation/wrist_image": np.full((224, 224, 3), 83, np.uint8)}
    return JointObservation(stamp, np.arange(7), images,
                            {key: stamp for key in images})


class Client:
    def __init__(self, config):
        self.config = config
        # Last four entries deliberately cannot be a unit quaternion.
        self.actions = np.tile(np.arange(7, dtype=np.float32) / 10, (50, 1))
        self.closed = False
        self.requests = []

    def get_server_metadata(self):
        return {"pi0_wm": deepcopy(self.config["interface"])}

    def infer(self, request):
        self.requests.append(request)
        return {"actions": self.actions}

    def close(self):
        self.closed = True


def test_training_keys_zero_state_and_joint_output_are_preserved():
    client = Client(pi_config())
    policy = Pi0JointPolicy(pi_config(), client_factory=lambda _: client)
    obs = observation()
    chunk = policy.predict_observations((obs,))
    request = client.requests[0]
    assert set(request) == {"observation/state", "observation/image", "observation/wrist_image", "prompt"}
    np.testing.assert_array_equal(request["observation/state"], np.zeros(7))
    assert request["observation/state"].dtype == np.float32
    np.testing.assert_array_equal(request["observation/image"], obs.images["observation/image"])
    np.testing.assert_array_equal(request["observation/wrist_image"], obs.images["observation/wrist_image"])
    assert request["prompt"] == "erase board"
    assert chunk.semantic == "joint" and chunk.frame_name is None
    assert chunk.timestamp_us == 100 and chunk.step_s == 1/30
    np.testing.assert_array_equal(chunk.values, client.actions[:20])
    assert policy.server_verified


def test_pi0_stage_timing_includes_server_time_without_changing_actions():
    client = Client(pi_config())
    infer = client.infer
    def timed_response(request):
        return {**infer(request), "server_timing": {"infer_ms": 72.5}}
    client.infer = timed_response
    policy = Pi0JointPolicy(pi_config(), client_factory=lambda _: client)
    chunk = policy.predict_observations((observation(),))
    assert chunk.metadata['timing']['server_ms'] == 72.5
    assert all(chunk.metadata['timing'][name] >= 0.
               for name in ('prepare_ms', 'connect_ms', 'remote_ms', 'postprocess_ms'))
    np.testing.assert_array_equal(chunk.values, client.actions[:20])


@pytest.mark.parametrize("key,value", [("action_semantic", "action.ee_pose"),
                                     ("state_semantic", "observation.ee_pose"),
                                     ("representation", "xyz_quaternion"),
                                     ("coordinate_frame", "link7"), ("units", "deg"),
                                     ("joint_names", [f"joint{i}" for i in range(7, 0, -1)]),
                                     ("output_space", "normalized")])
def test_incompatible_joint_contract_is_rejected(key, value):
    config = pi_config()
    config["interface"][key] = value
    with pytest.raises(JointActionContractError):
        Pi0JointPolicy(config)


def test_metadata_mismatch_closes_connection_before_inference():
    client = Client(pi_config())
    client.config["interface"]["training_config"] = "different_checkpoint"
    policy = Pi0JointPolicy(pi_config(), client_factory=lambda _: client)
    with pytest.raises(JointActionContractError, match="interface mismatch"):
        policy.predict_observations((observation(),))
    assert client.closed and not client.requests and not policy.server_verified


@pytest.mark.parametrize("values", [np.zeros((50, 32)), np.zeros((49, 7)),
                                   np.full((50, 7), np.nan), np.zeros((1, 50, 7))])
def test_padded_invalid_or_wrong_horizon_actions_are_rejected(values):
    client = Client(pi_config())
    client.actions = values
    policy = Pi0JointPolicy(pi_config(), client_factory=lambda _: client)
    with pytest.raises(JointActionContractError, match="post-transform"):
        policy.predict_observations((observation(),))


def test_cameras_require_training_rgb_size_and_dtype_before_network():
    policy = Pi0JointPolicy(pi_config(), client_factory=lambda _: pytest.fail("network reached"))
    obs = observation()
    obs.images["observation/image"] = np.zeros((224, 224, 3), np.float32)
    with pytest.raises(ValueError, match="uint8 RGB"):
        policy.predict_observations((obs,))


def test_mock_server_is_rejected_for_real_execution():
    policy = Pi0JointPolicy(pi_config(), client_factory=MockPiConnection)
    with pytest.raises(JointActionContractError, match="--backend mock"):
        policy.verify_server()
    policy.allow_mock_server = True
    policy.verify_server()


def pi_deployment(tmp_path):
    path = deployment_file(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["policy"] = {"type": "pi0"}
    document["pi0"] = pi_config()
    document["wm"] = {"enabled": False}
    document["execution"] = {"action_rate_hz": 30}
    path.write_text(yaml.safe_dump(document))
    return path


def test_joint_deployment_maps_two_training_cameras_without_checkpoint_or_connection(tmp_path):
    path = pi_deployment(tmp_path)
    config = JointDeployment.load(path)
    policy, mapping = load_joint_policy(config)
    assert config.policy["checkpoint"] is None
    assert mapping == {"observation/image": "left_wrist", "observation/wrist_image": "right_wrist"}
    assert policy.client is None
    config.execution["action_rate_hz"] = 25
    with pytest.raises(ValueError, match="differs"):
        load_joint_policy(config)


def test_public_entry_check_does_not_connect_to_models_or_hardware(monkeypatch, tmp_path):
    from inference.cli import main

    monkeypatch.setattr("inference.policies.pi0_joint.OpenPiConnection",
                        lambda _: pytest.fail("check connected"))
    report = main(["--config", str(pi_deployment(tmp_path)), "--check"])
    assert report["policy"] == "pi0" and not report["server_contract_verified"]


def test_offline_policy_cannot_be_selected_with_real_hardware(tmp_path):
    from inference.cli import main

    with pytest.raises(SystemExit):
        main(["--config", str(pi_deployment(tmp_path)), "--mock-policy"])


def test_pi0_uses_shared_s_i_loop_and_cancels_inflight_result():
    import time

    runtime, arms, _, cameras = runtime_fixture()
    client = Client(pi_config())
    client.actions[:] = .01
    policy = Pi0JointPolicy(pi_config(), client_factory=lambda _: client)
    # Keep fixture camera alignment/executor; reshape observations at the adapter boundary.
    predict = policy.predict_observations
    policy.predict_observations = lambda window: predict((observation(window[-1].timestamp_us),))
    runtime.policy = runtime.worker.policy = policy
    runtime.config.policy["type"] = "pi0"
    runtime.config.execution["warmup_steps"] = 0
    start = time.monotonic()
    pressed = set()

    def read_key(_):
        elapsed = time.monotonic() - start
        for key, deadline in (("s", .05), ("i", .22), ("q", .45)):
            if elapsed >= deadline and key not in pressed:
                if key == "s":
                    assert not client.requests and not arms["right"].commands
                pressed.add(key)
                return key

    runtime.run(.8, read_key=read_key, start_paused=True)
    assert pressed == {"s", "i", "q"}
    assert client.requests and runtime.predictions > 0
    assert client.closed and cameras.stopped and arms["right"].stops >= 1
    assert runtime.plan is None


def test_openpi_websocket_numpy_protocol_round_trip(monkeypatch):
    # Tested with the actual openpi client codec; optional client dependency in CI.
    websocket = pytest.importorskip("websockets.sync.server")
    source = Path(__file__).resolve().parents[2] / "openpi/packages/openpi-client/src"
    if not source.is_dir():
        pytest.skip("openpi client source unavailable")
    monkeypatch.syspath_prepend(str(source))
    codec = pytest.importorskip("openpi_client.msgpack_numpy")
    config = pi_config()
    mock = Client(config)

    def handle(socket):
        socket.send(codec.packb(mock.get_server_metadata()))
        request = codec.unpackb(socket.recv())
        socket.send(codec.packb(mock.infer(request)))

    with websocket.serve(handle, "127.0.0.1", 0, compression=None, max_size=None) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        config["port"] = server.socket.getsockname()[1]
        policy = Pi0JointPolicy(config)
        try:
            chunk = policy.predict_observations((observation(),))
            np.testing.assert_array_equal(chunk.values, mock.actions[:20])
            assert policy.server_verified
            assert mock.requests[0]["observation/wrist_image"].dtype == np.uint8
        finally:
            policy.close()
            server.shutdown()
            thread.join(timeout=2.)


def test_compilation_wait_disables_ping_deadline_but_keeps_bounded_receives(monkeypatch):
    transport = pytest.importorskip('websockets.sync.client')
    source = Path(__file__).resolve().parents[2] / 'openpi/packages/openpi-client/src'
    if not source.is_dir():
        pytest.skip('openpi client source unavailable')
    monkeypatch.syspath_prepend(str(source))
    codec = pytest.importorskip('openpi_client.msgpack_numpy')
    from inference.policies.pi0_joint import OpenPiConnection
    from unittest.mock import Mock

    socket = Mock()
    socket.recv.side_effect = [codec.packb({'pi0_wm': pi_config()['interface']}), TimeoutError('slow model')]
    connect = Mock(return_value=socket)
    monkeypatch.setattr(transport, 'connect', connect)
    config = pi_config()
    config['request_timeout_s'] = 90.
    client = OpenPiConnection(config)
    assert connect.call_args.kwargs['ping_interval'] is None
    assert connect.call_args.kwargs['open_timeout'] == 90.
    with pytest.raises(TimeoutError, match='slow model'):
        client.infer({'prompt': 'test'})
    assert all(call.kwargs['timeout'] == 90. for call in socket.recv.call_args_list)
    client.close()
    socket.close.assert_called_once()


def test_worker_preserves_server_contract_error_and_names_changed_fields():
    from inference.joint_runtime import PolicyWorker

    client = Client(pi_config())
    client.config['interface']['representation'] = 'xyz_quaternion'
    policy = Pi0JointPolicy(pi_config(), client_factory=lambda _: client)
    worker = PolicyWorker(policy)
    worker.thread.start()
    try:
        worker.submit((observation(),))
        with worker.condition:
            assert worker.condition.wait_for(lambda: worker.error is not None, timeout=.3)
        with pytest.raises(JointActionContractError, match="representation: server='xyz_quaternion', local='joint_positions'") as error:
            worker.take_result()
        assert 'restart serve_pi0_wm.py' in str(error.value)
        assert 'images:' not in str(error.value)
        assert client.closed
    finally:
        worker.close()


def test_runtime_contract_failure_is_reported_by_public_cli_without_wrapper_traceback(tmp_path, monkeypatch, capsys):
    from inference import joint_cli
    from types import SimpleNamespace

    monkeypatch.setattr('inference.joint_runtime.JointInferenceRuntime',
                        lambda *args, **kwargs: SimpleNamespace(predictions=0, commands=0))

    def fail(runtime, args):
        raise JointActionContractError('pi0 server interface mismatch: stale server YAML')

    with pytest.raises(SystemExit) as error:
        joint_cli.main(['--config', str(pi_deployment(tmp_path))], run_loop=fail)
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert 'stale server YAML' in stderr
    assert 'Traceback' not in stderr
