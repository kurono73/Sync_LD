"""Sync and projection switching operators."""

from __future__ import annotations

import bpy
from bpy.props import EnumProperty
from bpy.types import Operator

from .coordinates import state_from_calibration
from .fitting import fit_phase1
from .lens_models import SUPPORTED_MODELS
from .lens_models import calibration_signature, calibration_snapshot
from .projection import (
    apply_projection,
    commit_fit_candidate,
    commit_perspective_state,
)
from .properties import PROJECTION_MODE_ITEMS, source_render_signature


class SyncLDError(RuntimeError):
    pass


DATA_TRANSACTION_FIELDS = (
    "projection_mode",
    "sync_status",
    "has_projection",
    "error_message",
    "snapshot_json",
    "render_snapshot_json",
    "undistorted_lens",
    "undistorted_sensor_width",
    "undistorted_sensor_height",
    "undistorted_sensor_fit",
    "undistorted_shift_x",
    "undistorted_shift_y",
    "distorted_fisheye_fov",
    "distorted_k0",
    "distorted_k1",
    "distorted_k2",
    "distorted_k3",
    "distorted_k4",
    "distorted_sensor_width",
    "distorted_sensor_height",
    "distorted_sensor_fit",
    "distorted_shift_x",
    "distorted_shift_y",
    "rms_error",
    "mean_error",
    "max_error",
    "corner_error",
    "mean_x_offset",
    "mean_y_offset",
    "monotonic",
    "sample_count",
    "center_mismatch",
    "refinement_iterations",
)

CAMERA_TRANSACTION_FIELDS = (
    "type",
    "panorama_type",
    "lens",
    "sensor_width",
    "sensor_height",
    "sensor_fit",
    "shift_x",
    "shift_y",
    "fisheye_fov",
    "fisheye_polynomial_k0",
    "fisheye_polynomial_k1",
    "fisheye_polynomial_k2",
    "fisheye_polynomial_k3",
    "fisheye_polynomial_k4",
)

RENDER_TRANSACTION_FIELDS = (
    "resolution_x",
    "resolution_y",
    "pixel_aspect_x",
    "pixel_aspect_y",
)


def _rna_values(owner, field_names):
    return {name: getattr(owner, name) for name in field_names}


def _restore_rna_values(owner, values):
    for name, value in values.items():
        setattr(owner, name, value)


def camera_from_context(context):
    camera = getattr(context, "camera", None)
    if isinstance(camera, bpy.types.Camera):
        return camera
    obj = getattr(context, "object", None)
    if obj is not None and obj.type == "CAMERA":
        return obj.data
    scene_camera = getattr(context.scene, "camera", None)
    if scene_camera is not None and scene_camera.type == "CAMERA":
        return scene_camera.data
    return None


def _validate_phase1(calibration: dict):
    if calibration["distortion_model"] not in SUPPORTED_MODELS:
        raise SyncLDError("The source MovieClip distortion model is not supported.")
    if calibration["focal_length_pixels"] <= 0.0:
        raise SyncLDError("Source focal length is invalid.")
    if calibration["pixel_aspect"] <= 0.0:
        raise SyncLDError("Source pixel aspect is invalid.")


def _apply_source_render_settings(render, calibration):
    pixel_aspect = float(calibration["pixel_aspect"])
    render.resolution_x = int(calibration["width"])
    render.resolution_y = int(calibration["height"])
    if pixel_aspect >= 1.0:
        render.pixel_aspect_x = pixel_aspect
        render.pixel_aspect_y = 1.0
    else:
        render.pixel_aspect_x = 1.0
        render.pixel_aspect_y = 1.0 / pixel_aspect


def sync_camera(camera, clip, scene=None, sync_render=False):
    data = camera.sync_ld
    if clip is None:
        raise SyncLDError("Select a source Movie Clip.")

    calibration = calibration_snapshot(clip)
    perspective = state_from_calibration(calibration)
    _validate_phase1(calibration)

    render_width = int(calibration["width"])
    render_height = int(calibration["height"])
    render_pixel_aspect = float(calibration["pixel_aspect"])
    candidate = fit_phase1(
        calibration,
        perspective,
        grid_size=data.sample_grid,
        fov_margin_degrees=data.fov_margin_degrees,
        render_width=render_width,
        render_height=render_height,
        render_pixel_aspect=render_pixel_aspect,
    )

    previous_mode = data.projection_mode
    previous_data = _rna_values(data, DATA_TRANSACTION_FIELDS)
    previous_camera = _rna_values(camera, CAMERA_TRANSACTION_FIELDS)
    previous_render = (
        _rna_values(scene.render, RENDER_TRANSACTION_FIELDS)
        if scene is not None and sync_render
        else None
    )
    try:
        if scene is not None and sync_render:
            _apply_source_render_settings(scene.render, calibration)
        commit_perspective_state(data, perspective)
        commit_fit_candidate(data, candidate, perspective.sensor_fit)
        data.snapshot_json = calibration_signature(clip)
        data.render_snapshot_json = source_render_signature(calibration)
        data.has_projection = True
        data.error_message = ""
        data.sync_status = "SYNCED"
        apply_projection(camera, data, previous_mode)
    except Exception:
        _restore_rna_values(data, previous_data)
        _restore_rna_values(camera, previous_camera)
        if previous_render is not None:
            _restore_rna_values(scene.render, previous_render)
        raise
    return candidate


class SYNCLD_OT_sync_distortion(Operator):
    bl_idname = "sync_ld.sync_distortion"
    bl_label = "Sync Lens Distortion"
    bl_description = "Fit the selected MovieClip calibration to this camera"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return camera_from_context(context) is not None

    def execute(self, context):
        camera = camera_from_context(context)
        data = camera.sync_ld
        try:
            candidate = sync_camera(camera, data.source_clip, context.scene)
        except (RuntimeError, ValueError, TypeError, FloatingPointError) as exc:
            data.sync_status = "ERROR"
            data.error_message = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        quality = "monotonic" if candidate.monotonic else "non-monotonic"
        self.report(
            {"INFO"},
            f"Synced: RMS {candidate.rms_error:.3f} px, Max {candidate.max_error:.3f} px, {quality}",
        )
        return {"FINISHED"}


class SYNCLD_OT_set_projection(Operator):
    bl_idname = "sync_ld.set_projection"
    bl_label = "Set Projection"
    bl_description = "Switch between the stored projection states"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(items=PROJECTION_MODE_ITEMS)

    @classmethod
    def poll(cls, context):
        camera = camera_from_context(context)
        return camera is not None and camera.sync_ld.has_projection

    def execute(self, context):
        camera = camera_from_context(context)
        apply_projection(camera, camera.sync_ld, self.mode)
        return {"FINISHED"}


class SYNCLD_OT_sync_active_clip(Operator):
    bl_idname = "sync_ld.sync_active_clip"
    bl_label = "Sync Lens Distortion"
    bl_description = "Use the active clip and synchronize the scene camera"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        space = getattr(context, "space_data", None)
        return (
            space is not None
            and space.type == "CLIP_EDITOR"
            and space.clip is not None
            and context.scene.camera is not None
            and context.scene.camera.type == "CAMERA"
        )

    def execute(self, context):
        clip = context.space_data.clip
        camera = context.scene.camera.data
        camera.sync_ld.source_clip = clip
        try:
            candidate = sync_camera(camera, clip, context.scene, sync_render=True)
        except (RuntimeError, ValueError, TypeError, FloatingPointError) as exc:
            camera.sync_ld.sync_status = "ERROR"
            camera.sync_ld.error_message = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Synced: RMS {candidate.rms_error:.3f} px")
        return {"FINISHED"}


CLASSES = (
    SYNCLD_OT_sync_distortion,
    SYNCLD_OT_set_projection,
    SYNCLD_OT_sync_active_clip,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
