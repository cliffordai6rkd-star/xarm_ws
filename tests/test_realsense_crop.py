"""RealSense crop/resize behavior using synthetic aligned SDK frames."""
from dataclasses import replace
from types import SimpleNamespace
import sys

import cv2
import numpy as np
import pytest

from nero_collection.cameras import CameraUnavailable, RealSenseCamera
from nero_collection.config import CameraConfig


@pytest.fixture
def sdk_camera(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrealsense2", SimpleNamespace())
    rows, columns = np.indices((6, 8))
    # Separate RGB channels detect channel swaps, wrong crop and stale full-frame resize.
    rgb = np.stack([rows*20, columns*25, rows*8+columns], axis=-1).astype(np.uint8)
    depth = (rows*1000+columns*100).astype(np.uint16)
    frames = SimpleNamespace(
        get_timestamp=lambda: 33.333,
        get_color_frame=lambda: SimpleNamespace(get_data=lambda: rgb),
        get_depth_frame=lambda: SimpleNamespace(get_data=lambda: depth),
    )
    config = CameraConfig("wrist", backend="realsense", serial_number="synthetic",
                          width=8, height=6, depth=True, visualize=True)

    def camera_with(**settings):
        camera = RealSenseCamera(replace(config, **settings))
        camera._pipeline = SimpleNamespace(poll_for_frames=lambda: frames)
        camera._align = SimpleNamespace(process=lambda incoming: incoming)
        return camera

    return camera_with, rgb, depth


def test_realsense_crop_selects_same_region_for_rgb_depth_and_preview(sdk_camera):
    camera_with, rgb, depth = sdk_camera
    camera = camera_with(crop=(1, 5, 2, 7))
    frame = camera.poll()
    np.testing.assert_array_equal(frame.frame, rgb[1:5, 2:7])
    np.testing.assert_array_equal(frame.depth, depth[1:5, 2:7])
    np.testing.assert_array_equal(frame.preview_frame, rgb[1:5, 2:7])
    assert frame.frame.shape == (4, 5, 3) and frame.depth.shape == (4, 5)
    assert frame.frame.dtype == np.uint8 and frame.depth.dtype == np.uint16
    assert frame.frame.flags.c_contiguous and frame.depth.flags.c_contiguous
    # Crop/resize must not mutate SDK buffers, or a repeated acquisition would drift.
    assert rgb.shape == (6, 8, 3) and depth.shape == (6, 8)
    assert camera.poll() is None


def test_realsense_crops_before_independent_policy_and_preview_resize(sdk_camera):
    camera_with, rgb, depth = sdk_camera
    camera = camera_with(crop=(1, 5, 2, 7), output_size=(3, 2), preview_output_size=(10, 8))
    frame = camera.poll()
    expected_rgb = cv2.resize(rgb[1:5, 2:7], (3, 2), interpolation=cv2.INTER_AREA)
    expected_depth = cv2.resize(depth[1:5, 2:7], (3, 2), interpolation=cv2.INTER_NEAREST)
    expected_preview = cv2.resize(rgb[1:5, 2:7], (10, 8), interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(frame.frame, expected_rgb)
    np.testing.assert_array_equal(frame.depth, expected_depth)
    np.testing.assert_array_equal(frame.preview_frame, expected_preview)
    assert frame.frame.shape == (2, 3, 3) and frame.depth.shape == (2, 3)
    assert frame.preview_frame.shape == (8, 10, 3)
    assert np.isin(frame.depth, depth[1:5, 2:7]).all()


@pytest.mark.parametrize("crop", [(0, None, 0, None), (None, None, None, None)])
def test_realsense_default_crop_preserves_full_frame(sdk_camera, crop):
    camera_with, rgb, depth = sdk_camera
    frame = camera_with(crop=crop).poll()
    np.testing.assert_array_equal(frame.frame, rgb)
    np.testing.assert_array_equal(frame.depth, depth)
    np.testing.assert_array_equal(frame.preview_frame, rgb)
    expected_rgb, expected_depth = rgb.copy(), depth.copy()
    # RealSense owns and may reuse these buffers after poll returns.
    rgb.fill(0)
    depth.fill(0)
    np.testing.assert_array_equal(frame.frame, expected_rgb)
    np.testing.assert_array_equal(frame.preview_frame, expected_rgb)
    np.testing.assert_array_equal(frame.depth, expected_depth)


@pytest.mark.parametrize("crop", [(-1, 5, 0, 8), (0, 7, 0, 8), (0, 6, 2, 2), (0, 6, 0, 9)])
def test_realsense_invalid_crop_fails_instead_of_silently_slicing(sdk_camera, crop):
    camera_with, _, _ = sdk_camera
    camera = camera_with(crop=crop)
    with pytest.raises(CameraUnavailable, match=r"crop .* is outside frame 8x6"):
        camera.poll()
