"""Persistent camera data for Sync LD."""

from __future__ import annotations

import json
import math

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup


SENSOR_FIT_ITEMS = (
    ("AUTO", "Auto", "Fit to the render aspect"),
    ("HORIZONTAL", "Horizontal", "Fit to sensor width"),
    ("VERTICAL", "Vertical", "Fit to sensor height"),
)

PROJECTION_MODE_ITEMS = (
    ("UNDISTORTED", "Undistorted", "Perspective projection"),
    ("DISTORTED", "Distorted", "Fisheye Lens Polynomial projection"),
)

SYNC_STATUS_ITEMS = (
    ("NOT_SYNCED", "Not Synced", "No fitted projection is available"),
    ("SYNCED", "Synced", "Source calibration matches the fitted projection"),
    ("OUT_OF_SYNC", "Out of Sync", "Source calibration has changed"),
    ("ERROR", "Error", "The latest sync attempt failed"),
)


def _source_clip_updated(data, _context):
    data.error_message = ""
    if data.source_clip is None:
        data.show_source_details = False
        data.show_camera_fit_details = False
        data.show_clip_fit_details = False
        data.sync_status = "NOT_SYNCED"
    else:
        data.sync_status = "OUT_OF_SYNC" if data.has_projection else "NOT_SYNCED"


class SYNCLD_PG_camera_data(PropertyGroup):
    source_clip: PointerProperty(
        name="Movie Clip",
        type=bpy.types.MovieClip,
        update=_source_clip_updated,
    )
    projection_mode: EnumProperty(
        name="Projection",
        items=PROJECTION_MODE_ITEMS,
        default="UNDISTORTED",
    )
    sync_status: EnumProperty(
        name="Status",
        items=SYNC_STATUS_ITEMS,
        default="NOT_SYNCED",
    )
    has_projection: BoolProperty(default=False, options={"HIDDEN"})
    error_message: StringProperty(name="Error", options={"HIDDEN"})
    snapshot_json: StringProperty(name="Source Snapshot", options={"HIDDEN"})
    render_snapshot_json: StringProperty(name="Render Snapshot", options={"HIDDEN"})

    sample_grid: IntProperty(
        name="Sampling Grid",
        default=41,
        min=5,
        max=101,
    )
    fov_margin_degrees: FloatProperty(
        name="FOV Margin",
        description="Angular clipping margin in degrees",
        default=2.0,
        min=0.0,
        max=10.0,
        precision=3,
    )
    show_source_details: BoolProperty(name="Source Details", default=False)
    show_camera_fit_details: BoolProperty(name="Camera Fit Details", default=False)
    show_clip_fit_details: BoolProperty(name="Clip Fit Details", default=False)
    show_error_heatmap: BoolProperty(name="Heat Map", default=False)
    show_error_vectors: BoolProperty(name="Error Vectors", default=False)
    error_vector_scale: FloatProperty(
        name="Vector Scale",
        default=20.0,
        min=1.0,
        max=200.0,
        soft_max=50.0,
    )

    undistorted_lens: FloatProperty(default=50.0, options={"HIDDEN"})
    undistorted_sensor_width: FloatProperty(default=36.0, options={"HIDDEN"})
    undistorted_sensor_height: FloatProperty(default=24.0, options={"HIDDEN"})
    undistorted_sensor_fit: EnumProperty(
        items=SENSOR_FIT_ITEMS,
        default="AUTO",
        options={"HIDDEN"},
    )
    undistorted_shift_x: FloatProperty(default=0.0, options={"HIDDEN"})
    undistorted_shift_y: FloatProperty(default=0.0, options={"HIDDEN"})

    distorted_fisheye_fov: FloatProperty(default=3.141592653589793, options={"HIDDEN"})
    distorted_k0: FloatProperty(default=0.0, options={"HIDDEN"})
    distorted_k1: FloatProperty(default=-0.02, options={"HIDDEN"})
    distorted_k2: FloatProperty(default=0.0, options={"HIDDEN"})
    distorted_k3: FloatProperty(default=0.0, options={"HIDDEN"})
    distorted_k4: FloatProperty(default=0.0, options={"HIDDEN"})
    distorted_sensor_width: FloatProperty(default=36.0, options={"HIDDEN"})
    distorted_sensor_height: FloatProperty(default=24.0, options={"HIDDEN"})
    distorted_sensor_fit: EnumProperty(
        items=SENSOR_FIT_ITEMS,
        default="AUTO",
        options={"HIDDEN"},
    )
    distorted_shift_x: FloatProperty(default=0.0, options={"HIDDEN"})
    distorted_shift_y: FloatProperty(default=0.0, options={"HIDDEN"})

    rms_error: FloatProperty(default=0.0, options={"HIDDEN"})
    mean_error: FloatProperty(default=0.0, options={"HIDDEN"})
    max_error: FloatProperty(default=0.0, options={"HIDDEN"})
    corner_error: FloatProperty(default=0.0, options={"HIDDEN"})
    mean_x_offset: FloatProperty(default=0.0, options={"HIDDEN"})
    mean_y_offset: FloatProperty(default=0.0, options={"HIDDEN"})
    monotonic: BoolProperty(default=True, options={"HIDDEN"})
    sample_count: IntProperty(default=0, options={"HIDDEN"})
    center_mismatch: FloatProperty(default=0.0, options={"HIDDEN"})
    refinement_iterations: IntProperty(default=0, options={"HIDDEN"})


CLASSES = (SYNCLD_PG_camera_data,)


def render_signature(scene) -> str:
    render = scene.render
    return json.dumps(
        {
            "resolution_x": int(render.resolution_x),
            "resolution_y": int(render.resolution_y),
            "pixel_aspect_x": float(render.pixel_aspect_x),
            "pixel_aspect_y": float(render.pixel_aspect_y),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def source_render_signature(calibration: dict) -> str:
    pixel_aspect = float(calibration["pixel_aspect"])
    if pixel_aspect >= 1.0:
        pixel_aspect_x = pixel_aspect
        pixel_aspect_y = 1.0
    else:
        pixel_aspect_x = 1.0
        pixel_aspect_y = 1.0 / pixel_aspect
    return json.dumps(
        {
            "resolution_x": int(calibration["width"]),
            "resolution_y": int(calibration["height"]),
            "pixel_aspect_x": pixel_aspect_x,
            "pixel_aspect_y": pixel_aspect_y,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _render_aspect_from_signature(signature: str) -> float:
    values = json.loads(signature)
    return (
        float(values["resolution_x"])
        * float(values["pixel_aspect_x"])
        / (float(values["resolution_y"]) * float(values["pixel_aspect_y"]))
    )


def render_aspect_warning_text(data, scene=None) -> str:
    if data.source_clip is None or scene is None:
        return ""
    try:
        from .lens_models import calibration_snapshot

        source_aspect = _render_aspect_from_signature(
            source_render_signature(calibration_snapshot(data.source_clip))
        )
        current_aspect = _render_aspect_from_signature(render_signature(scene))
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        json.JSONDecodeError,
    ):
        return ""
    if math.isclose(current_aspect, source_aspect, rel_tol=1.0e-9):
        return ""
    return "Render aspect differs."


def out_of_sync_reason(data, scene=None) -> str:
    if data.source_clip is None or not data.has_projection or not data.snapshot_json:
        return ""
    try:
        from .lens_models import calibration_signature, calibration_snapshot

        if calibration_signature(data.source_clip) != data.snapshot_json:
            return "Source lens calibration has changed."
        source_signature = source_render_signature(calibration_snapshot(data.source_clip))
        if data.render_snapshot_json and data.render_snapshot_json != source_signature:
            return "Stored fit uses an old aspect basis."
    except (AttributeError, KeyError, TypeError, ValueError):
        return "Source lens calibration has changed."
    return ""


def effective_status(data, scene=None) -> str:
    if data.sync_status == "ERROR":
        return "ERROR"
    if data.source_clip is None or not data.has_projection or not data.snapshot_json:
        return "NOT_SYNCED"
    return "OUT_OF_SYNC" if out_of_sync_reason(data, scene) else "SYNCED"


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Camera.sync_ld = PointerProperty(type=SYNCLD_PG_camera_data)


def unregister():
    del bpy.types.Camera.sync_ld
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
