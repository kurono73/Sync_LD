"""Fit Cycles Fisheye Lens Polynomial projection from MovieClip rays."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .coordinates import (
    PerspectiveState,
    perspective_projection_center,
    sensor_geometry,
)
from .lens_models import distort_pixels


MIN_FOV_MARGIN_DEGREES = 2.0


@dataclass(frozen=True)
class FitCandidate:
    k0: float
    k1: float
    k2: float
    k3: float
    k4: float
    fisheye_fov: float
    sensor_width: float
    sensor_height: float
    shift_x: float
    shift_y: float
    rms_error: float
    mean_error: float
    max_error: float
    corner_error: float
    mean_x_offset: float
    mean_y_offset: float
    monotonic: bool
    sample_count: int
    center_mismatch: float
    refinement_iterations: int


@dataclass(frozen=True)
class ProjectionErrorSamples:
    reference_x: np.ndarray
    reference_y: np.ndarray
    projected_x: np.ndarray
    projected_y: np.ndarray
    error: np.ndarray


def _polynomial_radius_from_angle(alpha, coefficients):
    """Mirror Cycles' 20-step Newton inverse for direction-to-polynomial."""
    k0, k1, k2, k3, k4 = coefficients
    theta = -np.asarray(alpha, dtype=np.float64)
    if abs(k1) < 1.0e-15:
        raise ValueError("The fitted K1 coefficient is too close to zero.")

    radius = (theta - k0) / k1
    for _index in range(20):
        r2 = radius * radius
        value = k0 + k1 * radius + k2 * r2 + k3 * r2 * radius + k4 * r2 * r2
        derivative = k1 + 2.0 * k2 * radius + 3.0 * k3 * r2 + 4.0 * k4 * r2 * radius
        if np.any(np.abs(derivative) < 1.0e-14):
            raise ValueError("Fitted polynomial has a zero derivative.")
        step = (theta - value) / derivative
        radius = radius + step
        if float(np.max(np.abs(step))) < 1.0e-9:
            break
    return radius


def _sample_grid(width: int, height: int, grid_size: int):
    xs = np.linspace(0.5, width - 0.5, grid_size, dtype=np.float64)
    ys = np.linspace(0.5, height - 0.5, grid_size, dtype=np.float64)
    return np.meshgrid(xs, ys)


def _build_samples(
    calibration,
    perspective,
    grid_size,
    render_width,
    render_height,
    render_pixel_aspect,
):
    width = int(calibration["width"])
    height = int(calibration["height"])
    geometry = sensor_geometry(
        perspective, render_width, render_height, render_pixel_aspect
    )
    center_u, center_v = perspective_projection_center(
        perspective, render_width, render_height, render_pixel_aspect
    )
    pixel_x, pixel_y = _sample_grid(width, height, grid_size)
    sensor_x = (pixel_x / width - center_u) * geometry.width
    sensor_y = (pixel_y / height - center_v) * geometry.height
    ray_x = sensor_x / perspective.lens
    ray_y = sensor_y / perspective.lens
    ray_radius = np.hypot(ray_x, ray_y)
    alpha = np.arctan(ray_radius)
    unit_x = np.divide(
        ray_x, ray_radius, out=np.zeros_like(ray_x), where=ray_radius > 1.0e-15
    )
    unit_y = np.divide(
        ray_y, ray_radius, out=np.zeros_like(ray_y), where=ray_radius > 1.0e-15
    )
    focal = float(calibration["focal_length_pixels"])
    pixel_aspect = float(calibration["pixel_aspect"])
    principal_x, principal_y = calibration["principal_point"]
    principal_x = 0.5 * width * (1.0 + float(principal_x))
    principal_y = 0.5 * height * (1.0 + float(principal_y))
    source_x = focal * ray_x + principal_x
    source_y = (focal * ray_y + principal_y / pixel_aspect) * pixel_aspect
    reference_x, reference_y = distort_pixels(calibration, source_x, source_y)
    return (
        geometry,
        center_u,
        center_v,
        alpha,
        unit_x,
        unit_y,
        reference_x,
        reference_y,
    )


def _project_pixels(
    alpha,
    unit_x,
    unit_y,
    polynomial,
    center_u,
    center_v,
    geometry,
    width,
    height,
):
    test_radius = _polynomial_radius_from_angle(alpha, polynomial)
    if not np.all(np.isfinite(test_radius)) or np.any(test_radius < -1.0e-10):
        raise ValueError("Polynomial inverse produced an invalid sensor radius.")
    test_x = (test_radius * unit_x / geometry.width + center_u) * width
    test_y = (test_radius * unit_y / geometry.height + center_v) * height
    return test_x, test_y


def _errors(test_x, test_y, reference_x, reference_y):
    dx = test_x - reference_x
    dy = test_y - reference_y
    error = np.hypot(dx, dy)
    if not np.all(np.isfinite(error)):
        raise ValueError("Polynomial validation produced non-finite pixel errors.")
    return dx, dy, error


def _scaled_polynomial(polynomial, radius_scale):
    return np.asarray(
        [polynomial[index] * radius_scale**index for index in range(1, 5)],
        dtype=np.float64,
    )


def _unscaled_polynomial(parameters, radius_scale):
    return (
        0.0,
        *(float(parameters[index - 1]) / radius_scale**index for index in range(1, 5)),
    )


def _refine_pixel_fit(
    polynomial,
    center_u,
    center_v,
    alpha,
    unit_x,
    unit_y,
    reference_x,
    reference_y,
    geometry,
    width,
    height,
    radius_scale,
    max_iterations=8,
):
    parameters = np.concatenate(
        (
            _scaled_polynomial(polynomial, radius_scale),
            np.asarray((center_u * width, center_v * height), dtype=np.float64),
        )
    )
    initial_center = parameters[4:].copy()
    max_center_offset = 0.1 * min(width, height)

    def evaluate(values):
        candidate_polynomial = _unscaled_polynomial(values, radius_scale)
        test_x, test_y = _project_pixels(
            alpha,
            unit_x,
            unit_y,
            candidate_polynomial,
            values[4] / width,
            values[5] / height,
            geometry,
            width,
            height,
        )
        dx, dy, error = _errors(test_x, test_y, reference_x, reference_y)
        residual = np.stack((dx, dy), axis=-1).reshape(-1)
        return residual, float(np.sqrt(np.mean(error * error)))

    residual, rms = evaluate(parameters)
    iterations = 0
    damping = 1.0e-4
    for _index in range(max_iterations):
        jacobian = np.empty((residual.size, parameters.size), dtype=np.float64)
        for column in range(parameters.size):
            epsilon = (
                1.0e-5 * max(1.0, abs(parameters[column]))
                if column < 4
                else 0.05
            )
            perturbed = parameters.copy()
            perturbed[column] += epsilon
            perturbed_residual, _rms = evaluate(perturbed)
            jacobian[:, column] = (perturbed_residual - residual) / epsilon

        normal = jacobian.T @ jacobian
        diagonal = np.maximum(np.diag(normal), 1.0)
        normal += np.diag(damping * diagonal)
        gradient = jacobian.T @ residual
        try:
            step = np.linalg.solve(normal, -gradient)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break

        accepted = False
        previous_rms = rms
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = parameters + scale * step
            if np.linalg.norm(trial[4:] - initial_center) > max_center_offset:
                continue
            try:
                trial_residual, trial_rms = evaluate(trial)
            except ValueError:
                continue
            if trial_rms < rms:
                parameters = trial
                residual = trial_residual
                rms = trial_rms
                iterations += 1
                damping = max(damping * 0.25, 1.0e-8)
                accepted = True
                break
        if not accepted:
            damping *= 10.0
            if damping > 1.0e4:
                break
            continue
        if previous_rms - rms < max(1.0e-6, previous_rms * 1.0e-5):
            break

    return (
        _unscaled_polynomial(parameters, radius_scale),
        float(parameters[4] / width),
        float(parameters[5] / height),
        iterations,
    )


def projection_error_samples(
    calibration: dict,
    perspective: PerspectiveState,
    polynomial,
    panorama_shift,
    grid_size: int,
    render_width: int,
    render_height: int,
    render_pixel_aspect: float,
) -> ProjectionErrorSamples:
    width = int(calibration["width"])
    height = int(calibration["height"])
    (
        geometry,
        _perspective_center_u,
        _perspective_center_v,
        alpha,
        unit_x,
        unit_y,
        reference_x,
        reference_y,
    ) = _build_samples(
        calibration,
        perspective,
        grid_size,
        render_width,
        render_height,
        render_pixel_aspect,
    )
    center_u = 0.5 - float(panorama_shift[0])
    center_v = 0.5 - float(panorama_shift[1])
    projected_x, projected_y = _project_pixels(
        alpha,
        unit_x,
        unit_y,
        polynomial,
        center_u,
        center_v,
        geometry,
        width,
        height,
    )
    _dx, _dy, error = _errors(
        projected_x, projected_y, reference_x, reference_y
    )
    return ProjectionErrorSamples(
        reference_x=reference_x,
        reference_y=reference_y,
        projected_x=projected_x,
        projected_y=projected_y,
        error=error,
    )


def fit_phase1(
    calibration: dict,
    perspective: PerspectiveState,
    grid_size: int = 41,
    fov_margin_degrees: float = MIN_FOV_MARGIN_DEGREES,
    render_width: int | None = None,
    render_height: int | None = None,
    render_pixel_aspect: float = 1.0,
    refine_pixel: bool = True,
) -> FitCandidate:
    """Fit a supported MovieClip model to the current Perspective projection."""
    width = int(calibration["width"])
    height = int(calibration["height"])
    focal = float(calibration["focal_length_pixels"])
    if width < 2 or height < 2:
        raise ValueError("Source clip has no usable resolution.")
    if focal <= 0.0 or perspective.lens <= 0.0:
        raise ValueError("Focal length must be positive.")
    if grid_size < 5:
        raise ValueError("Sampling grid must be at least 5 by 5.")
    if render_width is None:
        render_width = width
    if render_height is None:
        render_height = height
    if render_width < 2 or render_height < 2 or render_pixel_aspect <= 0.0:
        raise ValueError("Render dimensions and pixel aspect must be positive.")

    (
        geometry,
        center_u,
        center_v,
        alpha,
        unit_x,
        unit_y,
        reference_x,
        reference_y,
    ) = _build_samples(
        calibration,
        perspective,
        grid_size,
        render_width,
        render_height,
        render_pixel_aspect,
    )
    reference_sensor_x = (reference_x / width - center_u) * geometry.width
    reference_sensor_y = (reference_y / height - center_v) * geometry.height
    reference_radius = np.hypot(reference_sensor_x, reference_sensor_y)

    fit_mask = np.isfinite(reference_radius) & np.isfinite(alpha) & (reference_radius > 1.0e-12)
    fit_radius = reference_radius[fit_mask]
    fit_target = -alpha[fit_mask]
    if fit_radius.size < 16:
        raise ValueError("Not enough finite lens samples were generated.")

    matrix = np.column_stack(
        (fit_radius, fit_radius**2, fit_radius**3, fit_radius**4)
    )
    fitted, _residuals, rank, _singular_values = np.linalg.lstsq(
        matrix, fit_target, rcond=None
    )
    if rank < 4 or not np.all(np.isfinite(fitted)):
        raise ValueError("Polynomial least-squares fit is rank deficient.")
    polynomial = (0.0, *(float(value) for value in fitted))

    test_x, test_y = _project_pixels(
        alpha,
        unit_x,
        unit_y,
        polynomial,
        center_u,
        center_v,
        geometry,
        width,
        height,
    )
    dx, dy, error = _errors(test_x, test_y, reference_x, reference_y)
    initial_rms = float(np.sqrt(np.mean(error * error)))

    max_initial_radius = max(
        math.hypot((u - center_u) * geometry.width, (v - center_v) * geometry.height)
        for u in (0.0, 1.0)
        for v in (0.0, 1.0)
    )
    refinement_iterations = 0
    if refine_pixel and initial_rms > 0.02:
        try:
            polynomial, center_u, center_v, refinement_iterations = _refine_pixel_fit(
                polynomial,
                center_u,
                center_v,
                alpha,
                unit_x,
                unit_y,
                reference_x,
                reference_y,
                geometry,
                width,
                height,
                max_initial_radius,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            refinement_iterations = 0
        test_x, test_y = _project_pixels(
            alpha,
            unit_x,
            unit_y,
            polynomial,
            center_u,
            center_v,
            geometry,
            width,
            height,
        )
        dx, dy, error = _errors(test_x, test_y, reference_x, reference_y)

    corner_indices = ((0, 0), (0, -1), (-1, 0), (-1, -1))
    corner_error = max(float(error[index]) for index in corner_indices)

    max_sensor_radius = max(
        math.hypot((u - center_u) * geometry.width, (v - center_v) * geometry.height)
        for u in (0.0, 1.0)
        for v in (0.0, 1.0)
    )
    radius_check = np.linspace(0.0, max_sensor_radius, 2049, dtype=np.float64)
    k0, k1, k2, k3, k4 = polynomial
    theta = -(k0 + k1 * radius_check + k2 * radius_check**2 + k3 * radius_check**3 + k4 * radius_check**4)
    theta_max = float(np.max(np.abs(theta)))
    fov_margin_degrees = max(float(fov_margin_degrees), MIN_FOV_MARGIN_DEGREES)
    fisheye_fov = 2.0 * theta_max + math.radians(fov_margin_degrees)
    if not math.isfinite(fisheye_fov) or fisheye_fov <= 0.0:
        raise ValueError("Required fisheye FOV is invalid.")

    derivative = -(k1 + 2.0 * k2 * radius_check + 3.0 * k3 * radius_check**2 + 4.0 * k4 * radius_check**3)
    monotonic = bool(np.all(derivative > 0.0))

    principal_x, principal_y = calibration["principal_point"]
    source_center_u = 0.5 * (1.0 + float(principal_x))
    source_center_v = 0.5 * (1.0 + float(principal_y))
    perspective_center_u, perspective_center_v = perspective_projection_center(
        perspective, render_width, render_height, render_pixel_aspect
    )
    center_mismatch = math.hypot(
        (perspective_center_u - source_center_u) * width,
        (perspective_center_v - source_center_v) * height,
    )

    return FitCandidate(
        k0=k0,
        k1=k1,
        k2=k2,
        k3=k3,
        k4=k4,
        fisheye_fov=fisheye_fov,
        sensor_width=geometry.width,
        sensor_height=geometry.height,
        shift_x=0.5 - center_u,
        shift_y=0.5 - center_v,
        rms_error=float(np.sqrt(np.mean(error * error))),
        mean_error=float(np.mean(error)),
        max_error=float(np.max(error)),
        corner_error=corner_error,
        mean_x_offset=float(np.mean(dx)),
        mean_y_offset=float(np.mean(dy)),
        monotonic=monotonic,
        sample_count=int(error.size),
        center_mismatch=center_mismatch,
        refinement_iterations=refinement_iterations,
    )


def quality_label(rms_error: float) -> str:
    if rms_error < 0.10:
        return "Excellent"
    if rms_error < 0.25:
        return "Very Good"
    if rms_error < 0.50:
        return "Good"
    if rms_error < 1.00:
        return "Usable"
    return "Poor"
