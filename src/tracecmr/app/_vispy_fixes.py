"""Small speed fixes for the drawing library (vispy) under napari.

Every time a shapes layer changes (each mouse move while a contour dot is dragged), vispy's mesh builds a
brand-new, identical colour function, which makes the graphics card recompile and relink the shader program
twice per frame. On some laptop graphics drivers that alone takes tens of milliseconds and the dragged dot
visibly trails the mouse. Reusing the function keeps the compiled shader. Behaviour is otherwise unchanged:
only the plain (no colour map) case is reused.
"""

from __future__ import annotations

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    try:
        from vispy.visuals.mesh import MeshVisual
    except Exception:  # vispy missing or reorganised: nothing to fix
        return
    import weakref

    original = MeshVisual._build_color_transform
    cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # visuals are frozen: no new attributes

    def build_color_transform(self, colors):
        if getattr(colors, "ndim", 1) == 2 and colors.shape[1] == 1:  # colour-mapped values: as before
            return original(self, colors)
        try:
            fun = cache.get(self)
            if fun is None:
                fun = cache[self] = original(self, colors)
            return fun
        except TypeError:  # not weak-referenceable: as before
            return original(self, colors)

    MeshVisual._build_color_transform = build_color_transform
