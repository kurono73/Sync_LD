"""Projection state persistence and Camera RNA application."""

from __future__ import annotations

from .coordinates import PerspectiveState


def commit_perspective_state(data, state: PerspectiveState):
    data.undistorted_lens = state.lens
    data.undistorted_sensor_width = state.sensor_width
    data.undistorted_sensor_height = state.sensor_height
    data.undistorted_sensor_fit = state.sensor_fit
    data.undistorted_shift_x = state.shift_x
    data.undistorted_shift_y = state.shift_y


def commit_fit_candidate(data, candidate, sensor_fit: str):
    data.distorted_fisheye_fov = candidate.fisheye_fov
    data.distorted_k0 = candidate.k0
    data.distorted_k1 = candidate.k1
    data.distorted_k2 = candidate.k2
    data.distorted_k3 = candidate.k3
    data.distorted_k4 = candidate.k4
    data.distorted_sensor_width = candidate.sensor_width
    data.distorted_sensor_height = candidate.sensor_height
    data.distorted_sensor_fit = sensor_fit
    data.distorted_shift_x = candidate.shift_x
    data.distorted_shift_y = candidate.shift_y
    data.rms_error = candidate.rms_error
    data.mean_error = candidate.mean_error
    data.max_error = candidate.max_error
    data.corner_error = candidate.corner_error
    data.mean_x_offset = candidate.mean_x_offset
    data.mean_y_offset = candidate.mean_y_offset
    data.monotonic = candidate.monotonic
    data.sample_count = candidate.sample_count
    data.center_mismatch = candidate.center_mismatch
    data.refinement_iterations = candidate.refinement_iterations


def apply_undistorted(camera, data):
    camera.type = "PERSP"
    camera.lens = data.undistorted_lens
    camera.sensor_width = data.undistorted_sensor_width
    camera.sensor_height = data.undistorted_sensor_height
    camera.sensor_fit = data.undistorted_sensor_fit
    camera.shift_x = data.undistorted_shift_x
    camera.shift_y = data.undistorted_shift_y
    data.projection_mode = "UNDISTORTED"


def apply_distorted(camera, data):
    camera.type = "PANO"
    camera.panorama_type = "FISHEYE_LENS_POLYNOMIAL"
    camera.fisheye_fov = data.distorted_fisheye_fov
    camera.fisheye_polynomial_k0 = data.distorted_k0
    camera.fisheye_polynomial_k1 = data.distorted_k1
    camera.fisheye_polynomial_k2 = data.distorted_k2
    camera.fisheye_polynomial_k3 = data.distorted_k3
    camera.fisheye_polynomial_k4 = data.distorted_k4
    camera.sensor_width = data.distorted_sensor_width
    camera.sensor_height = data.distorted_sensor_height
    camera.sensor_fit = data.distorted_sensor_fit
    camera.shift_x = data.distorted_shift_x
    camera.shift_y = data.distorted_shift_y
    data.projection_mode = "DISTORTED"


def apply_projection(camera, data, mode: str):
    if mode == "UNDISTORTED":
        apply_undistorted(camera, data)
    elif mode == "DISTORTED":
        apply_distorted(camera, data)
    else:
        raise ValueError(f"Unknown projection mode: {mode}")
