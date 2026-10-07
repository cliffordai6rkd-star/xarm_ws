import builtins
import logging

from inference.logging_utils import quiet_hardware_output


def test_quiet_output_hides_known_banners_but_retains_errors_and_restores_print(capsys, caplog):
    original = builtins.print
    with caplog.at_level(logging.INFO):
        with quiet_hardware_output(True):
            print("SDK_VERSION: 1.18.5")
            print("ROBOT_IP: offline")
            print("change protocol identifier to 3")
            print("s starts; i resets")
            print("xArm controller fault C31")
            logging.getLogger("nero_collection.cameras").info("starting camera")
            logging.getLogger("nero_collection.cameras").error("camera failed")
            logging.getLogger("origin.print").info("routine SDK message")
            logging.getLogger("origin.print").warning("controller warning")
    assert builtins.print is original
    output = capsys.readouterr().out
    assert "SDK_VERSION" not in output and "ROBOT_IP" not in output
    assert "protocol identifier" not in output
    assert "s starts; i resets" in output and "controller fault" in output
    assert "starting camera" not in caplog.text and "routine SDK message" not in caplog.text
    assert "camera failed" in caplog.text and "controller warning" in caplog.text


def test_verbose_mode_keeps_sdk_banner(capsys):
    with quiet_hardware_output(False):
        print("SDK_VERSION: offline")
    assert "SDK_VERSION: offline" in capsys.readouterr().out
