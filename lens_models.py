"""MovieClip lens model evaluators matching Blender 5.2 libmv."""

from __future__ import annotations

import json

import numpy as np


MODEL_COEFFICIENTS = {
    "POLYNOMIAL": ("k1", "k2", "k3"),
    "DIVISION": ("division_k1", "division_k2"),
    "NUKE": ("nuke_k1", "nuke_k2", "nuke_p1", "nuke_p2"),
    "BROWN": (
        "brown_k1",
        "brown_k2",
        "brown_k3",
        "brown_k4",
        "brown_p1",
        "brown_p2",
    ),
}

SUPPORTED_MODELS = frozenset(MODEL_COEFFICIENTS)


def clip_size(clip) -> tuple[int, int]:
    width, height = clip.size
    return int(width), int(height)


def calibration_snapshot(clip) -> dict:
    """Return the calibration fields that define Blender's lens mapping."""
    camera = clip.tracking.camera
    width, height = clip_size(clip)
    model = camera.distortion_model
    coefficients = {
        name: float(getattr(camera, name, 0.0))
        for name in MODEL_COEFFICIENTS.get(model, ())
    }
    return {
        "clip_name": clip.name_full,
        "clip_filepath": clip.filepath,
        "width": width,
        "height": height,
        "distortion_model": model,
        "focal_length": float(camera.focal_length),
        "focal_length_pixels": float(camera.focal_length_pixels),
        "sensor_width": float(camera.sensor_width),
        "pixel_aspect": float(camera.pixel_aspect),
        "principal_point": [float(value) for value in camera.principal_point],
        "coefficients": coefficients,
    }


def calibration_signature(clip) -> str:
    return json.dumps(
        calibration_snapshot(clip),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def distort_polynomial(x, y, k1: float, k2: float, k3: float):
    """Apply libmv POLYNOMIAL distortion in normalized camera space."""
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    return x * radial, y * radial


def distort_division(x, y, k1: float, k2: float):
    """Apply libmv DIVISION distortion in normalized camera space."""
    r2 = x * x + y * y
    denominator = 1.0 + k1 * r2 + k2 * r2 * r2
    if np.any(np.abs(denominator) < 1.0e-14):
        raise ValueError("Division distortion is singular inside the image.")
    return x / denominator, y / denominator


def distort_brown(
    x,
    y,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    p1: float,
    p2: float,
):
    """Apply libmv BROWN radial and tangential distortion."""
    x2 = x * x
    y2 = y * y
    r2 = x2 + y2
    radial = 1.0 + (((k4 * r2 + k3) * r2 + k2) * r2 + k1) * r2
    xy2 = 2.0 * x * y
    tangential_x = p1 * (r2 + 2.0 * x2) + p2 * xy2
    tangential_y = p2 * (r2 + 2.0 * y2) + p1 * xy2
    return x * radial + tangential_x, y * radial + tangential_y


def _invert_nuke_scaled(x, y, k1: float, k2: float, p1: float, p2: float):
    """Evaluate NUKE's direct distorted-to-undistorted equation."""
    x2 = x * x
    y2 = y * y
    r2 = x2 + y2
    r4 = r2 * r2
    denominator_x = 1.0 + k1 * r2 + k2 * r4 + p1 * y2
    denominator_y = 1.0 + k1 * r2 + k2 * r4 + p2 * x2
    return x / denominator_x, y / denominator_y


def distort_nuke_pixels(
    normalized_x,
    normalized_y,
    focal: float,
    principal_x: float,
    principal_y: float,
    image_width: int,
    image_height: int,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
):
    """Solve libmv NUKE forward distortion with a vectorized Newton method."""
    half_size = 0.5 * max(int(image_width), int(image_height))
    if half_size <= 0.0:
        raise ValueError("NUKE distortion requires a non-empty image.")

    target_x = np.asarray(normalized_x, dtype=np.float64) * focal / half_size
    target_y = np.asarray(normalized_y, dtype=np.float64) * focal / half_size
    x = target_x.copy()
    y = target_y.copy()

    for _index in range(40):
        x2 = x * x
        y2 = y * y
        r2 = x2 + y2
        r4 = r2 * r2
        common_x = 2.0 * k1 * x + 4.0 * k2 * r2 * x
        common_y = 2.0 * k1 * y + 4.0 * k2 * r2 * y
        denominator_x = 1.0 + k1 * r2 + k2 * r4 + p1 * y2
        denominator_y = 1.0 + k1 * r2 + k2 * r4 + p2 * x2
        if np.any(np.abs(denominator_x) < 1.0e-14) or np.any(
            np.abs(denominator_y) < 1.0e-14
        ):
            raise ValueError("NUKE distortion is singular inside the image.")

        value_x = x / denominator_x - target_x
        value_y = y / denominator_y - target_y
        dxx = (denominator_x - x * common_x) / (denominator_x * denominator_x)
        dxy = -x * (common_y + 2.0 * p1 * y) / (denominator_x * denominator_x)
        dyx = -y * (common_x + 2.0 * p2 * x) / (denominator_y * denominator_y)
        dyy = (denominator_y - y * common_y) / (denominator_y * denominator_y)
        determinant = dxx * dyy - dxy * dyx
        if np.any(np.abs(determinant) < 1.0e-14):
            raise ValueError("NUKE distortion solve has a singular Jacobian.")

        step_x = (value_x * dyy - value_y * dxy) / determinant
        step_y = (dxx * value_y - dyx * value_x) / determinant
        step_length = np.hypot(step_x, step_y)
        scale = np.minimum(1.0, np.divide(0.5, step_length, out=np.ones_like(step_length), where=step_length > 0.5))
        step_x *= scale
        step_y *= scale
        x -= step_x
        y -= step_y
        if float(np.max(np.maximum(np.abs(step_x), np.abs(step_y)))) < 1.0e-12:
            break

    solved_x, solved_y = _invert_nuke_scaled(x, y, k1, k2, p1, p2)
    residual = np.maximum(np.abs(solved_x - target_x), np.abs(solved_y - target_y))
    if not np.all(np.isfinite(residual)) or float(np.max(residual)) > 1.0e-8:
        raise ValueError("NUKE forward distortion solve did not converge.")
    return principal_x + half_size * x, principal_y + half_size * y


def distort_pixels(calibration: dict, pixel_x, pixel_y):
    """Apply MovieClip distortion in the source image's raster coordinates."""
    width = int(calibration["width"])
    height = int(calibration["height"])
    focal = float(calibration["focal_length_pixels"])
    pixel_aspect = float(calibration["pixel_aspect"])
    if focal <= 0.0 or pixel_aspect <= 0.0:
        raise ValueError("Source focal length and pixel aspect must be positive.")

    principal = calibration["principal_point"]
    principal_x = 0.5 * width * (1.0 + float(principal[0]))
    principal_y_raster = 0.5 * height * (1.0 + float(principal[1]))
    principal_y = principal_y_raster / pixel_aspect
    normalized_x = (np.asarray(pixel_x, dtype=np.float64) - principal_x) / focal
    normalized_y = (
        np.asarray(pixel_y, dtype=np.float64) / pixel_aspect - principal_y
    ) / focal
    model = calibration["distortion_model"]
    coefficients = calibration["coefficients"]

    if model == "POLYNOMIAL":
        distorted_x, distorted_y = distort_polynomial(
            normalized_x,
            normalized_y,
            coefficients["k1"],
            coefficients["k2"],
            coefficients["k3"],
        )
    elif model == "DIVISION":
        distorted_x, distorted_y = distort_division(
            normalized_x,
            normalized_y,
            coefficients["division_k1"],
            coefficients["division_k2"],
        )
    elif model == "BROWN":
        distorted_x, distorted_y = distort_brown(
            normalized_x,
            normalized_y,
            coefficients["brown_k1"],
            coefficients["brown_k2"],
            coefficients["brown_k3"],
            coefficients["brown_k4"],
            coefficients["brown_p1"],
            coefficients["brown_p2"],
        )
    elif model == "NUKE":
        distorted_x, distorted_y = distort_nuke_pixels(
            normalized_x,
            normalized_y,
            focal,
            principal_x,
            principal_y,
            width,
            int(height / pixel_aspect),
            coefficients["nuke_k1"],
            coefficients["nuke_k2"],
            coefficients["nuke_p1"],
            coefficients["nuke_p2"],
        )
        return distorted_x, distorted_y * pixel_aspect
    else:
        raise ValueError(f"Unsupported MovieClip distortion model: {model}")

    return (
        focal * distorted_x + principal_x,
        (focal * distorted_y + principal_y) * pixel_aspect,
    )
