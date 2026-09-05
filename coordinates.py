"""Blender/Cycles camera coordinate conversions used by Sync LD."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerspectiveState:
    lens: float
    sensor_width: float
    sensor_height: float
    sensor_fit: str
    shift_x: float
    shift_y: float


@dataclass(frozen=True)
class SensorGeometry:
    width: float
    height: float
    horizontal_fit: bool


def _horizontal_sensor_fit(
    state: PerspectiveState,
    image_width: int,
    image_height: int,
    pixel_aspect: float,
) -> bool:
    fit_x = float(image_width) * float(pixel_aspect)
    fit_y = float(image_height)
    if fit_x <= 0.0 or fit_y <= 0.0:
        raise ValueError("Image dimensions and pixel aspect must be positive.")
    if state.sensor_fit == "AUTO":
        return fit_x > fit_y
    if state.sensor_fit == "HORIZONTAL":
        return True
    if state.sensor_fit == "VERTICAL":
        return False
    raise ValueError(f"Unsupported sensor fit: {state.sensor_fit}")


def sensor_geometry(
    state: PerspectiveState,
    image_width: int,
    image_height: int,
    pixel_aspect: float = 1.0,
) -> SensorGeometry:
    """Mirror Cycles' sensor dimensions for polynomial panorama cameras."""
    fit_x = float(image_width) * float(pixel_aspect)
    fit_y = float(image_height)
    horizontal_fit = _horizontal_sensor_fit(
        state, image_width, image_height, pixel_aspect
    )
    sensor_size = state.sensor_width if horizontal_fit else state.sensor_height

    if horizontal_fit:
        return SensorGeometry(sensor_size, sensor_size * fit_y / fit_x, True)
    return SensorGeometry(sensor_size * fit_x / fit_y, sensor_size, False)


def perspective_projection_center(
    state: PerspectiveState,
    image_width: int,
    image_height: int,
    pixel_aspect: float = 1.0,
) -> tuple[float, float]:
    """Return the Perspective optical axis in normalized raster coordinates."""
    fit_x = float(image_width) * float(pixel_aspect)
    fit_y = float(image_height)
    horizontal_fit = _horizontal_sensor_fit(
        state, image_width, image_height, pixel_aspect
    )
    if horizontal_fit:
        aspect_ratio = fit_x / fit_y
        shift_u = state.shift_x
        shift_v = aspect_ratio * state.shift_y
    else:
        aspect_ratio = fit_y / fit_x
        shift_u = aspect_ratio * state.shift_x
        shift_v = state.shift_y
    return 0.5 - shift_u, 0.5 - shift_v


def panorama_shift_for_perspective(
    state: PerspectiveState,
    image_width: int,
    image_height: int,
    pixel_aspect: float = 1.0,
) -> tuple[float, float]:
    """Convert Perspective shift to Cycles panorama viewplane shift."""
    center_u, center_v = perspective_projection_center(
        state, image_width, image_height, pixel_aspect
    )
    return 0.5 - center_u, 0.5 - center_v


def state_from_calibration(calibration: dict) -> PerspectiveState:
    """Build the solved Perspective intrinsics stored by a MovieClip."""
    width = int(calibration["width"])
    height = int(calibration["height"])
    pixel_aspect = float(calibration["pixel_aspect"])
    sensor_width = float(calibration["sensor_width"])
    if width < 2 or height < 2 or pixel_aspect <= 0.0 or sensor_width <= 0.0:
        raise ValueError("Source camera intrinsics are invalid.")

    focal_length = float(
        calibration.get(
            "focal_length",
            float(calibration["focal_length_pixels"]) * sensor_width / width,
        )
    )
    principal_x, principal_y = (
        float(value) for value in calibration["principal_point"]
    )
    fit_x = width * pixel_aspect
    fit_y = height
    return PerspectiveState(
        lens=focal_length,
        sensor_width=sensor_width,
        sensor_height=sensor_width * fit_y / fit_x,
        sensor_fit="HORIZONTAL",
        shift_x=-0.5 * principal_x,
        shift_y=-0.5 * principal_y * fit_y / fit_x,
    )


def state_from_properties(data) -> PerspectiveState:
    return PerspectiveState(
        lens=float(data.undistorted_lens),
        sensor_width=float(data.undistorted_sensor_width),
        sensor_height=float(data.undistorted_sensor_height),
        sensor_fit=str(data.undistorted_sensor_fit),
        shift_x=float(data.undistorted_shift_x),
        shift_y=float(data.undistorted_shift_y),
    )
