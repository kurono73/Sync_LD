"""Camera Properties and Movie Clip Editor panels."""

from __future__ import annotations

import json
import math

import bpy
import numpy as np
from bpy.types import Panel

from .coordinates import state_from_properties
from .fitting import projection_error_samples, quality_label
from .lens_models import calibration_snapshot, MODEL_COEFFICIENTS
from .operators import camera_from_context
from .properties import effective_status, out_of_sync_reason, render_aspect_warning_text


STATUS_LABELS = {
    "NOT_SYNCED": ("Not Synced", "INFO"),
    "SYNCED": ("Synced", "CHECKMARK"),
    "OUT_OF_SYNC": ("Out of Sync", "ERROR"),
    "ERROR": ("Error", "CANCEL"),
}

SUPPORTED_EEVEE_VERSION = (5, 3, 0)
HEATMAP_POINT_SIZE = 9.0
VECTOR_LINE_WIDTH = 4.0
VECTOR_OUTLINE_WIDTH = 7.0

_ERROR_OVERLAY_HANDLE = None
_ERROR_OVERLAY_CACHE = {}


def _overlay_samples(scene, camera, data):
    render_snapshot = (
        json.loads(data.render_snapshot_json)
        if data.render_snapshot_json
        else {
            "resolution_x": scene.render.resolution_x,
            "resolution_y": scene.render.resolution_y,
            "pixel_aspect_x": scene.render.pixel_aspect_x,
            "pixel_aspect_y": scene.render.pixel_aspect_y,
        }
    )
    key = (
        camera.as_pointer(),
        data.snapshot_json,
        data.render_snapshot_json,
        data.distorted_k0,
        data.distorted_k1,
        data.distorted_k2,
        data.distorted_k3,
        data.distorted_k4,
        data.distorted_shift_x,
        data.distorted_shift_y,
        data.sample_grid,
    )
    cached = _ERROR_OVERLAY_CACHE.get(key)
    if cached is not None:
        return cached
    calibration = (
        json.loads(data.snapshot_json)
        if data.snapshot_json
        else calibration_snapshot(data.source_clip)
    )
    samples = projection_error_samples(
        calibration,
        state_from_properties(data),
        (
            data.distorted_k0,
            data.distorted_k1,
            data.distorted_k2,
            data.distorted_k3,
            data.distorted_k4,
        ),
        (data.distorted_shift_x, data.distorted_shift_y),
        min(data.sample_grid, 21),
        int(render_snapshot["resolution_x"]),
        int(render_snapshot["resolution_y"]),
        float(render_snapshot["pixel_aspect_x"])
        / float(render_snapshot["pixel_aspect_y"]),
    )
    if len(_ERROR_OVERLAY_CACHE) > 8:
        _ERROR_OVERLAY_CACHE.clear()
    _ERROR_OVERLAY_CACHE[key] = (samples, calibration)
    return samples, calibration


def _square_tris(center, size):
    x, y = center
    half = size * 0.5
    left = x - half
    right = x + half
    bottom = y - half
    top = y + half
    return (
        (left, bottom),
        (right, bottom),
        (right, top),
        (left, bottom),
        (right, top),
        (left, top),
    )


def _segment_tris(start, end, width):
    start_x, start_y = start
    end_x, end_y = end
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    length = math.hypot(delta_x, delta_y)
    if length <= 1.0e-5:
        return ()
    offset_x = -delta_y / length * width * 0.5
    offset_y = delta_x / length * width * 0.5
    p1 = (start_x + offset_x, start_y + offset_y)
    p2 = (start_x - offset_x, start_y - offset_y)
    p3 = (end_x - offset_x, end_y - offset_y)
    p4 = (end_x + offset_x, end_y + offset_y)
    return (p1, p2, p3, p1, p3, p4)


def _draw_error_overlay():
    context = bpy.context
    space = context.space_data
    region = context.region
    scene = context.scene
    if space is None or space.type != "CLIP_EDITOR" or region is None or space.clip is None:
        return
    camera_object = scene.camera
    if camera_object is None or camera_object.type != "CAMERA":
        return
    camera = camera_object.data
    data = camera.sync_ld
    if (
        not data.has_projection
        or data.source_clip != space.clip
        or not (data.show_error_heatmap or data.show_error_vectors)
    ):
        return

    try:
        samples, calibration = _overlay_samples(scene, camera, data)
        from gpu_extras.batch import batch_for_shader
        import gpu
    except (KeyError, TypeError, ValueError, AttributeError, RuntimeError):
        return

    width = float(calibration["width"])
    height = float(calibration["height"])
    view2d = region.view2d

    def region_position(x, y):
        return view2d.view_to_region(float(x) / width, float(y) / height, clip=False)

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    try:
        if data.show_error_heatmap:
            error = samples.error.reshape(-1)
            reference_x = samples.reference_x.reshape(-1)
            reference_y = samples.reference_y.reshape(-1)
            bands = (
                (error < 0.10, (0.10, 0.85, 0.25, 0.85)),
                ((error >= 0.10) & (error < 0.50), (0.95, 0.80, 0.10, 0.90)),
                ((error >= 0.50) & (error < 1.00), (1.00, 0.35, 0.05, 0.95)),
                (error >= 1.00, (1.00, 0.05, 0.05, 1.00)),
            )
            for mask, color in bands:
                indices = np.flatnonzero(mask)
                if indices.size == 0:
                    continue
                positions = []
                for index in indices:
                    positions.extend(
                        _square_tris(
                            region_position(reference_x[index], reference_y[index]),
                            HEATMAP_POINT_SIZE,
                        )
                    )
                batch = batch_for_shader(shader, "TRIS", {"pos": positions})
                shader.bind()
                shader.uniform_float("color", color)
                batch.draw(shader)

        if data.show_error_vectors:
            scale = float(data.error_vector_scale)
            positions = []
            for reference_x, reference_y, projected_x, projected_y in zip(
                samples.reference_x.reshape(-1),
                samples.reference_y.reshape(-1),
                samples.projected_x.reshape(-1),
                samples.projected_y.reshape(-1),
            ):
                start = region_position(reference_x, reference_y)
                end = region_position(
                    reference_x + (projected_x - reference_x) * scale,
                    reference_y + (projected_y - reference_y) * scale,
                )
                positions.extend(_segment_tris(start, end, VECTOR_OUTLINE_WIDTH))
            if positions:
                batch = batch_for_shader(shader, "TRIS", {"pos": positions})
                shader.bind()
                shader.uniform_float("color", (0.0, 0.0, 0.0, 0.65))
                batch.draw(shader)

            positions = []
            for reference_x, reference_y, projected_x, projected_y in zip(
                samples.reference_x.reshape(-1),
                samples.reference_y.reshape(-1),
                samples.projected_x.reshape(-1),
                samples.projected_y.reshape(-1),
            ):
                start = region_position(reference_x, reference_y)
                end = region_position(
                    reference_x + (projected_x - reference_x) * scale,
                    reference_y + (projected_y - reference_y) * scale,
                )
                positions.extend(_segment_tris(start, end, VECTOR_LINE_WIDTH))
            if positions:
                batch = batch_for_shader(shader, "TRIS", {"pos": positions})
                shader.bind()
                shader.uniform_float("color", (1.0, 0.15, 0.05, 0.95))
                batch.draw(shader)
    except (RuntimeError, ValueError, TypeError):
        pass
    finally:
        gpu.state.blend_set("NONE")


def _draw_projection(layout, data):
    row = layout.row(align=True)
    operator = row.operator(
        "sync_ld.set_projection",
        text="Undistorted",
        depress=data.projection_mode == "UNDISTORTED",
    )
    operator.mode = "UNDISTORTED"
    operator = row.operator(
        "sync_ld.set_projection",
        text="Distorted",
        depress=data.projection_mode == "DISTORTED",
    )
    operator.mode = "DISTORTED"
    row.enabled = data.has_projection


def _draw_status(layout, data, scene=None):
    status = effective_status(data, scene)
    label, icon = STATUS_LABELS[status]
    layout.label(text=label, icon=icon)
    if status == "OUT_OF_SYNC":
        layout.label(text=out_of_sync_reason(data, scene), icon="ERROR")
    elif status == "ERROR" and data.error_message:
        layout.label(text=data.error_message, icon="CANCEL")
    return status


def _render_engine_warning_text(scene) -> str:
    engine = scene.render.engine
    if engine == "CYCLES":
        return ""
    if (
        engine in {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}
        and bpy.app.version >= SUPPORTED_EEVEE_VERSION
    ):
        return ""
    return "Cycles for distorted render."


def _draw_render_engine_warning(layout, scene):
    warning = _render_engine_warning_text(scene)
    if warning:
        layout.label(text=warning, icon="INFO")


def _draw_render_aspect_warning(layout, data, scene):
    warning = render_aspect_warning_text(data, scene)
    if warning:
        layout.label(text=warning, icon="INFO")


def _draw_fit_quality(layout, data, details_property):
    show_details = getattr(data, details_property)
    row = layout.row(align=True)
    row.prop(
        data,
        details_property,
        text=f"Fit Quality: {quality_label(data.rms_error)}",
        emboss=False,
        icon="TRIA_DOWN" if show_details else "TRIA_RIGHT",
    )
    if not show_details:
        return
    split = layout.split(factor=0.55)
    labels = split.column(align=True)
    values = split.column(align=True)
    labels.label(text="RMS")
    labels.label(text="Mean")
    labels.label(text="Maximum")
    labels.label(text="Corner")
    values.label(text=f"{data.rms_error:.3f} px")
    values.label(text=f"{data.mean_error:.3f} px")
    values.label(text=f"{data.max_error:.3f} px")
    values.label(text=f"{data.corner_error:.3f} px")
    if not data.monotonic:
        layout.label(text="Polynomial may fold over near the frame edge.", icon="ERROR")
    if data.center_mismatch > 0.25:
        layout.label(
            text=f"Principal Point / Camera Shift differ by {data.center_mismatch:.2f} px.",
            icon="ERROR",
        )
    if data.refinement_iterations:
        layout.label(text=f"Pixel refinement: {data.refinement_iterations} iterations")


def _draw_source_details(layout, clip):
    camera = clip.tracking.camera
    width, height = (int(value) for value in clip.size)
    column = layout.column(align=True)
    column.label(text=f"Resolution: {width} x {height}")
    column.label(text=f"Focal Length: {camera.focal_length:.3f} mm")
    column.label(text=f"Sensor Width: {camera.sensor_width:.3f} mm")
    column.label(
        text=f"Principal Point: {camera.principal_point[0]:.6f}, {camera.principal_point[1]:.6f}"
    )
    column.label(text=f"Pixel Aspect: {camera.pixel_aspect:.6f}")
    for name in MODEL_COEFFICIENTS.get(camera.distortion_model, ()):
        if not hasattr(camera, name):
            continue
        label = (
            name.removeprefix("division_")
            .removeprefix("nuke_")
            .removeprefix("brown_")
            .upper()
        )
        column.label(text=f"{label}: {getattr(camera, name):.8g}")


class DATA_PT_sync_lens_distortion(Panel):
    bl_label = "Sync Lens Distortion"
    bl_idname = "DATA_PT_sync_lens_distortion"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        return camera_from_context(context) is not None

    def draw(self, context):
        camera = camera_from_context(context)
        data = camera.sync_ld
        layout = self.layout

        layout.label(text="Source")
        layout.prop(data, "source_clip")
        if data.source_clip is not None:
            model = data.source_clip.tracking.camera.distortion_model.replace("_", " ").title()
            layout.label(text=f"Distortion Model: {model}")
            source_camera = data.source_clip.tracking.camera
            if source_camera.distortion_model in {"NUKE", "BROWN"}:
                prefix = "nuke" if source_camera.distortion_model == "NUKE" else "brown"
                if abs(getattr(source_camera, f"{prefix}_p1", 0.0)) > 1.0e-12 or abs(
                    getattr(source_camera, f"{prefix}_p2", 0.0)
                ) > 1.0e-12:
                    layout.label(text="Tangential distortion is approximated.", icon="INFO")

        status = effective_status(data, context.scene)
        button_text = "Retry Sync" if status == "ERROR" else "Sync Lens Distortion"
        _draw_status(layout, data, context.scene)
        _draw_render_engine_warning(layout, context.scene)
        _draw_render_aspect_warning(layout, data, context.scene)
        layout.operator("sync_ld.sync_distortion", text=button_text, icon="FILE_REFRESH")

        _draw_projection(layout, data)
        if data.projection_mode == "UNDISTORTED":
            layout.label(text="Perspective", icon="CAMERA_DATA")
        else:
            layout.label(text="Fisheye Lens Polynomial", icon="CAMERA_DATA")

        if data.has_projection and data.source_clip is not None:
            layout.separator()
            _draw_fit_quality(layout, data, "show_camera_fit_details")

        if data.source_clip is not None:
            row = layout.row()
            row.prop(
                data,
                "show_source_details",
                text="Source Details",
                emboss=False,
                icon="TRIA_DOWN" if data.show_source_details else "TRIA_RIGHT",
            )
            if data.show_source_details:
                _draw_source_details(layout, data.source_clip)


class CLIP_PT_sync_lens_distortion(Panel):
    bl_label = "Sync Lens Distortion"
    bl_idname = "CLIP_PT_sync_lens_distortion"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "UI"
    bl_category = "Sync LD"

    @classmethod
    def poll(cls, context):
        return context.space_data.clip is not None

    def draw(self, context):
        layout = self.layout
        clip = context.space_data.clip
        layout.prop(context.scene, "camera", text="Camera")

        camera_object = context.scene.camera
        if camera_object is not None and camera_object.type == "CAMERA":
            data = camera_object.data.sync_ld
            layout.label(
                text=f"Distortion Model: {clip.tracking.camera.distortion_model.title()}"
            )
            if data.source_clip == clip:
                status = _draw_status(layout, data, context.scene)
            else:
                status = "NOT_SYNCED"
                layout.label(text="Not Synced", icon="INFO")
            _draw_render_engine_warning(layout, context.scene)
            if data.source_clip == clip:
                _draw_render_aspect_warning(layout, data, context.scene)
            button_text = "Retry Sync" if status == "ERROR" else "Sync Lens Distortion"
            layout.operator("sync_ld.sync_active_clip", text=button_text, icon="FILE_REFRESH")

            _draw_projection(layout, data)

            if data.has_projection and data.source_clip == clip:
                layout.separator()
                _draw_fit_quality(layout, data, "show_clip_fit_details")
                row = layout.row(align=True)
                row.prop(data, "show_error_heatmap", toggle=True, icon="IMAGE_DATA")
                row.prop(data, "show_error_vectors", toggle=True, icon="TRACKER")
                if data.show_error_vectors:
                    layout.prop(data, "error_vector_scale", slider=True)
        else:
            layout.label(text="Not Synced", icon="INFO")


CLASSES = (DATA_PT_sync_lens_distortion, CLIP_PT_sync_lens_distortion)


def register():
    global _ERROR_OVERLAY_HANDLE
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    _ERROR_OVERLAY_HANDLE = bpy.types.SpaceClipEditor.draw_handler_add(
        _draw_error_overlay, (), "WINDOW", "POST_PIXEL"
    )


def unregister():
    global _ERROR_OVERLAY_HANDLE
    if _ERROR_OVERLAY_HANDLE is not None:
        bpy.types.SpaceClipEditor.draw_handler_remove(_ERROR_OVERLAY_HANDLE, "WINDOW")
        _ERROR_OVERLAY_HANDLE = None
    _ERROR_OVERLAY_CACHE.clear()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
