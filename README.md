# Sync_LD
Render engine support:

- Blender 4.2 through 5.2: distorted rendering is supported with Cycles.
- Blender 5.3 and newer: distorted rendering is supported with both Eevee and
  Cycles.

## Overview

Sync LD applies the lens distortion parameters stored in Blender's Movie Clip
Editor to a render camera. The camera can switch between an undistorted
Perspective view and a distorted Fisheye Lens Polynomial view.

This can also be useful outside live-action matchmoving. Blender's Fisheye Lens
Polynomial camera settings are difficult to tune by feel, so Sync LD lets you
adjust or enter distortion in the Clip Editor first, then apply that visual
result to the render camera.

The source of truth is the Movie Clip's lens settings in the Clip Editor. Those
settings can come from Blender's camera solve workflow, another distortion
calibration add-on, or externally entered values. Sync LD itself does not
calibrate lenses; it uses the existing Movie Clip values and does not edit them.

## Supported Distortion Models

- Polynomial
- Division
- Nuke
- Brown

Brown tangential distortion is fitted as a radial Fisheye Lens Polynomial
approximation. This usually works well for render preview and checker
comparison, but it cannot be mathematically identical to the original tangential
model.

Blender 5.1 and later also supports Nuke's anamorphic coefficients `P1` and
`P2`. Sync LD uses them when they are available and set. Because those
coefficients are anamorphic and the render camera fit is radial, Nuke `P1` and
`P2` are not well suited for accurate matching and can only be a limited
approximation.

## UI

Sync LD appears in two places:

- Camera Data Properties: choose a source Movie Clip manually, sync it to the
  selected camera, and switch the camera between `Undistorted` and `Distorted`.
- Movie Clip Editor sidebar: use the active clip as the source, sync it to the
  scene camera, and inspect fit residuals with `Heat Map` or `Error Vectors`.

## Basic Workflow

1. Set or solve the lens distortion parameters on a Movie Clip.
2. Select or assign the scene camera you want to use for rendering.
3. Open `Sync Lens Distortion`.
4. Click `Sync Lens Distortion`.
5. Switch between `Undistorted` and `Distorted` on the same camera.

`Undistorted` uses Blender's normal Perspective camera. `Distorted` uses
Fisheye Lens Polynomial projection.

## Camera Panel Workflow

Use the Camera Data Properties panel when you want to choose a source Movie Clip
manually.

Camera-panel sync updates the camera projection and lens settings, but preserves
the scene render resolution, pixel aspect, and resolution percentage. The
distortion fit always uses the source Movie Clip aspect, even when the scene
render aspect is different.

## Clip Editor Workflow

Use the Movie Clip Editor panel when you are already working on the active clip.
The active clip becomes the source automatically.

Clip-editor sync updates the scene camera and aligns render resolution and pixel
aspect to the active clip. Resolution percentage is preserved.

The Clip Editor also includes optional fitting overlays:

- `Heat Map` shows residual error points on the clip.
- `Error Vectors` shows the error direction and can be magnified with
  `Vector Scale`.

These overlays are diagnostic tools. They do not change the camera or the clip.

## Status

- `Not Synced`: no valid synced projection is available for the current source.
- `Synced`: the source Movie Clip values still match the stored fit.
- `Out of Sync`: the source Movie Clip values changed.
- `Error`: the latest sync attempt failed and the previous valid camera state was
  kept.

Render aspect changes do not mark the result out of sync. Sync LD keeps the
source Movie Clip aspect for the distortion calculation and shows a warning when
the scene render aspect differs.

## Fit Quality

Fit Quality reports the pixel-space difference between the Movie Clip distortion
model and the fitted render camera projection.

- `Excellent`: RMS below 0.10 px
- `Very Good`: RMS below 0.25 px
- `Good`: RMS below 0.50 px
- `Usable`: RMS below 1.00 px
- `Poor`: RMS 1.00 px or higher

Open the Fit Quality section to see RMS, mean, maximum, corner error, and pixel
refinement information. Fit Quality is a guide only; confirm the rendered result
and decide whether the fit is acceptable for your use case or delivery
requirements.

## What Sync LD Changes

Sync LD may update these camera data settings:

- Camera type and panorama projection
- Focal length
- Sensor size and sensor fit
- Camera shift
- Fisheye field of view
- Fisheye polynomial coefficients

Sync LD does not edit the Movie Clip values.

## Known Limits

- The distorted projection uses Blender's Fisheye Lens Polynomial camera.
- Tangential Brown distortion is approximated because Fisheye Lens
  Polynomial is radial.
- Nuke `P1` and `P2` are anamorphic coefficients and are only a limited radial
  approximation.
- Very wide lenses can produce larger edge error because the fitted projection is
  limited to fourth order with `K0 = 0`.
