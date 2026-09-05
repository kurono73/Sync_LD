"""Sync LD Blender Extension."""

try:
    from . import operators, properties, ui
except ModuleNotFoundError as exc:
    if exc.name != "bpy":
        raise
    _MODULES = ()
else:
    _MODULES = (properties, operators, ui)


def register():
    if not _MODULES:
        raise RuntimeError("Sync LD can only be registered inside Blender.")
    for module in _MODULES:
        module.register()


def unregister():
    for module in reversed(_MODULES):
        module.unregister()

