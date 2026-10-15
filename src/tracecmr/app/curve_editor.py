"""Segment-style contour editing in napari.

Each contour (endocardium, epicardium, RV) on each slice of every frame is a
smooth closed curve through about 16-26 control points. Mouse, as in Segment's
interpolation-point tool (no modes to switch):

  drag a dot              move it
  click on a curve        add a dot there (keep holding to drag it)
  right-click a dot       delete it
  drag empty image        nothing (editing never moves the image); a drag that starts near a dot grabs
                          the nearest dot
  Pan button on           left-drag moves the image (no editing until it's switched off)
  right- or middle-drag   pan; scroll / pinch zooms
  while drawing           click to add points, drag a point to fix it, right-click to close
  clicks on empty image   start a new contour where this slice has none; right-click closes it
  [ / ]                   shrink / grow the selected contour by half a pixel all the way round
  Ctrl+Z / Undo           undo the last edit (drag, add, delete, smooth, snap, grow/shrink, clear)

Only the slice and frame being viewed are drawn, which keeps dragging smooth. Contours of the ED and ES frames
are built up front; those of any other frame the first time it is viewed (load_frame), so contours drawn or
auto-drawn on any frame show and can be edited.
When the user computes, edited contours are turned back into labels; untouched
slices keep their original masks exactly.
"""

from __future__ import annotations

import time

import numpy as np

from .. import labels as L
from ..contours import Curve, curve_to_mask, mask_to_curve, spline_through

START_POINTS = 18  # evenly spaced, plus a few at sharp tips, never bunched (docs/methods.md)
# endocardium deep red, epicardium light blue, right ventricle yellow
COLOURS = {"endo": (0.84, 0.10, 0.11, 1.0), "epi": (0.31, 0.76, 0.97, 1.0), "rv": (1.0, 0.84, 0.0, 1.0)}
LINE_WIDTH = 0.6  # contour line width (pixels of the image)
Key = tuple[int, int, str]  # (frame, slice, "endo"/"epi"/"rv")


def light_refresh(layer) -> None:
    """Redraw a layer whose data changed in place, skipping what a moving dot doesn't change: its thumbnail in
    napari's layer list (hidden while editing curves, and the costliest part), its extent and its highlight.
    Older napari without these options gets a plain refresh."""
    try:
        layer.refresh(thumbnail=False, extent=False, highlight=False)
    except TypeError:
        layer.refresh()


def curves_from_labels(label4d: np.ndarray, frames: list[int], n_points: int = START_POINTS) -> dict[Key, np.ndarray]:
    """Control points for every contoured slice of the given frames. label4d is (x, y, slice, time)."""
    out: dict[Key, np.ndarray] = {}
    for t in frames:
        for z in range(label4d.shape[2]):
            lab = label4d[:, :, z, t]
            blood = lab == L.LV_BLOODPOOL
            lv = blood | (lab == L.LV_MYOCARDIUM) | (lab == L.SCAR)
            rv = lab == L.RV_BLOODPOOL
            for name, mask in (("endo", blood), ("epi", lv), ("rv", rv)):
                # epicardium: smoother (anatomically smooth surface) and evenly spaced, no tip points;
                # RV: crescent-shaped, so evenly spaced with light smoothing
                if name == "endo":
                    c = mask_to_curve(mask, n_points)
                elif name == "epi":
                    c = mask_to_curve(mask, 16, None, smooth_px=3.0)
                else:
                    c = mask_to_curve(mask, 20, 1.5, smooth_px=2.0) if mask.sum() > 10 else None
                if c is not None:
                    out[(t, z, name)] = c.control
    return out


def labels_from_curves(endo: np.ndarray | None, epi: np.ndarray | None, shape: tuple[int, int],
                       rv: np.ndarray | None = None) -> np.ndarray:
    """One slice of standard labels from endo/epi (and RV) control points. The RV never overwrites the LV."""
    lab = np.zeros(shape, dtype=np.uint8)
    if rv is not None:
        lab[curve_to_mask(Curve(rv, spline_through(rv)), shape)] = L.RV_BLOODPOOL
    if epi is not None:
        lab[curve_to_mask(Curve(epi, spline_through(epi)), shape)] = L.LV_MYOCARDIUM
    if endo is not None:
        lab[curve_to_mask(Curve(endo, spline_through(endo)), shape)] = L.LV_BLOODPOOL
    return lab


def smooth_points(control: np.ndarray, which: np.ndarray | None = None, strength: float = 0.5,
                  passes: int = 2) -> np.ndarray:
    """Smooth a closed contour's control points (Taubin smoothing: no shrinkage).

    Each point moves toward the midpoint of its neighbours (strength lambda), then
    a slightly stronger step back (mu) cancels the shrinking that plain averaging
    causes. `which` (boolean mask) limits the change to selected points. When the
    whole contour is smoothed, it is rescaled to keep its exact area.
    """
    from ..contours import polygon_area

    pts = control.astype(float).copy()
    sel = np.ones(len(pts), bool) if which is None else np.asarray(which, bool)
    if len(pts) < 4 or not sel.any():
        return pts
    a0 = polygon_area(pts)
    lam, mu = strength, -(strength + 0.03)
    for _ in range(passes):
        for f in (lam, mu):
            mid = 0.5 * (np.roll(pts, 1, axis=0) + np.roll(pts, -1, axis=0))
            pts[sel] = pts[sel] + f * (mid[sel] - pts[sel])
    if which is None:
        a1 = polygon_area(pts)
        if a1 > 0:
            c = pts.mean(axis=0)
            pts = c + (pts - c) * np.sqrt(a0 / a1)
    return pts


def offset_points(control: np.ndarray, d: float) -> np.ndarray:
    """Move every control point d pixels along the outward normal (d > 0 grows the contour, d < 0 shrinks it).

    The normal at each point is perpendicular to the line joining its two neighbours, so the
    shape (including dots the user placed) is kept; only its size changes.
    """
    from ..contours import polygon_area

    pts = control.astype(float)
    if len(pts) < 3 or d == 0:
        return pts.copy()
    tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    norm = np.column_stack([tang[:, 1], -tang[:, 0]])
    norm /= np.maximum(np.linalg.norm(norm, axis=1, keepdims=True), 1e-9)
    out = pts + d * norm
    probe = pts + 0.1 * norm  # these normals point out for one winding direction and in for the other
    if polygon_area(probe) < polygon_area(pts):
        out = pts - d * norm
    return out


def _inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path

    return Path(spline_through(poly, 200)).contains_points(pts)


def insert_point(control: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Insert a new point into a closed polygon at the nearest edge."""
    a, b = control, np.roll(control, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
    d = np.linalg.norm(a + t[:, None] * ab - p, axis=1)
    i = int(np.argmin(d))
    return np.insert(control, i + 1, p, axis=0)


class CurveEditor:
    """Owns the 'control points' and 'curves' layers, the mouse editing and the undo stack."""

    HIT_SCREEN_PX = 12  # how close (on screen) a click must be to a dot or curve
    GRAB_SCREEN_PX = 30  # a drag starting this close to a dot grabs the nearest one (easy on a trackpad)

    def __init__(self, viewer, scale: tuple[float, float, float, float]):
        self.viewer = viewer
        self.scale = scale  # (time, slice, y, x) world units per index
        self.curves: dict[Key, np.ndarray] = {}
        self.edited: set[Key] = set()
        self._keys: list[Key] = []  # key of each row in the points layer
        self._updating = False
        self.points = None
        self.shapes = None
        self.add_to = "endo"  # contour that a new drawing makes
        self._undo: list[tuple[dict, set, set]] = []
        self._redo: list[tuple[dict, set, set]] = []
        self.hidden: set[Key] = set()  # kept for older saves; dots are no longer hidden per contour
        self.show_dots = True  # Points toggle: off = clean contours to look at, no dot editing
        self.focus: str | None = "endo"  # the contour being edited (Endo / Epi / RV); None = all
        self.pending: tuple[int, int, str, list] | None = None  # contour being drawn: (t, z, name, points)
        self.active = False
        self.on_change = None  # callback(message) for the panel's status line
        self.on_edit = None  # callback() after any edit (e.g. to refresh per-slice results)
        self._mouse_attached = False
        self.on_window = None  # Ctrl + drag on the image: callback(d_contrast, d_brightness)
        self.pan_mode = False  # Pan button: left-drag moves the image instead of editing
        self.click_hook = None  # one-shot: the next left click goes here as (frame, slice, point_px), not to editing
        self.suspended = False  # another tool (ROI) owns the mouse: no contour editing
        self._dims_attached = False
        self._shape_of: dict[Key, int] = {}  # index of each contour in the curves layer (this slice)
        self._last_draw = 0.0
        self.load_frame = None  # callback(frame) -> {key: points}: a frame's contours, built when first viewed
        self.loaded_frames: set[int] | None = None  # frames whose contours are in self.curves (None = all)

    # ---------- layers ----------

    def show(self, curves: dict[Key, np.ndarray], keep: set[tuple[int, int]] | None = None,
             frames: set[int] | None = None) -> None:
        """Show these contours for editing. Slices (t, z) in `keep` are unchanged since the last show: they
        keep their current dots, and stay smoothed (dots hidden) if they were. `frames`: the frames `curves`
        covers; any other frame is built with load_frame when it is first viewed (None: curves covers all)."""
        keep = keep or set()
        self.loaded_frames = None if frames is None else set(frames)
        new = {k: v.copy() for k, v in curves.items() if k[:2] not in keep}
        new.update({k: v for k, v in self.curves.items() if k[:2] in keep})
        hidden = {k for k in self.hidden if k[:2] in keep and k in new}
        self.curves = new
        self.edited.clear()
        self.hidden = hidden
        self.pending = None
        self._undo.clear()
        self._redo.clear()
        for name in ("curves", "control points"):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        self.shapes = self.viewer.add_shapes(name="curves", scale=self.scale, ndim=4)
        self.points = self.viewer.add_points(np.empty((0, 4)), name="control points", scale=self.scale,
                                             ndim=4, size=self.DOT_SIZE, border_width=0.25, face_color="white",
                                             out_of_slice_display=False)
        self.points.events.data.connect(self._on_points_changed)
        if not self._mouse_attached:
            self.viewer.mouse_drag_callbacks.append(self._on_mouse)
            self._mouse_attached = True
        if not self._dims_attached:
            self.viewer.dims.events.current_step.connect(self._on_step)
            self._dims_attached = True
        self._view = self._here()
        self._load_frame(self._view[0])
        self._refresh_layers()
        self.set_active(True)

    def detach(self) -> None:
        """Stop listening to the viewer (a new mouse was opened and gets a new editor)."""
        self.active = False
        if self._mouse_attached:
            try:
                self.viewer.mouse_drag_callbacks.remove(self._on_mouse)
            except ValueError:
                pass
            self._mouse_attached = False
        if self._dims_attached:
            try:
                self.viewer.dims.events.current_step.disconnect(self._on_step)
            except (TypeError, ValueError, RuntimeError):
                pass
            self._dims_attached = False
        self.points = self.shapes = None

    def set_active(self, on: bool) -> None:
        """Curve editing on (mouse edits contours, right-drag pans) or off (napari's normal mouse)."""
        self.active = on
        for layer in (self.points, self.shapes):
            if layer is not None:
                layer.visible = on
                layer.mode = "pan_zoom"
        for layer in self.viewer.layers:  # napari follows the active layer's setting; left-drag edits instead
            if hasattr(layer, "mouse_pan"):
                layer.mouse_pan = not on
        if on and "image" in self.viewer.layers:  # keep napari's own point/shape shortcuts out of the way
            self.viewer.layers.selection.active = self.viewer.layers["image"]

    @property
    def camera(self):
        scene = getattr(self.viewer, "scene", None)
        return scene.camera if scene is not None and hasattr(scene, "camera") else self.viewer.camera

    def _focused(self, key: Key) -> bool:
        return self.focus is None or key[2] == self.focus

    def _visible(self, key: Key) -> bool:
        """Its dots show and can be grabbed: Points is on, and it is the contour being edited."""
        return self.show_dots and key not in self.hidden and self._focused(key)

    def set_show_dots(self, on: bool) -> None:
        """Points toggle: show the selected contour's dots (and edit with them), or a clean view without dots."""
        self.show_dots = bool(on)
        self._refresh_layers()

    def set_focus(self, name: str | None) -> None:
        """Edit only this contour ('endo', 'epi', 'rv'): only its dots are shown and can be grabbed."""
        self.focus = name
        if name is not None:
            self.add_to = name
        self._refresh_layers()

    def _here(self) -> tuple[int, int]:
        step = self.viewer.dims.current_step
        return int(step[0]), int(step[1])

    def _on_step(self, event=None):
        if self._updating:  # our own redraw moved the position: don't redraw again (that can loop)
            return
        if self.points is not None and self._here() != getattr(self, "_view", None):
            self._view = self._here()
            self._load_frame(self._view[0])
            self._refresh_layers()

    def _load_frame(self, t: int) -> None:
        """Bring in the contours of frame t the first time it is viewed. They count as having been there all
        along, so they go into the undo history too (undo must never drop or clear them)."""
        if self.loaded_frames is None or t in self.loaded_frames or self.load_frame is None:
            return
        self.loaded_frames.add(t)
        try:
            extra = {k: np.array(v, dtype=float) for k, v in (self.load_frame(t) or {}).items() if k[0] == t}
        except Exception:  # a contour that can't be turned into dots stays as it is in the labels
            extra = {}
        extra = {k: v for k, v in extra.items() if k not in self.curves}
        self.curves.update(extra)
        for snap in self._undo + self._redo + list(getattr(self, "_redo_saved", [])):
            for k, v in extra.items():
                snap[0].setdefault(k, v.copy())

    def _rows(self) -> tuple[np.ndarray, list[Key]]:
        t0, z0 = self._here()
        coords, keys = [], []
        for key, ctrl in sorted(self.curves.items()):
            t, z, _ = key
            if (t, z) != (t0, z0) or not self._visible(key):
                continue
            for x, y in ctrl:
                coords.append((t, z, y, x))
                keys.append(key)
        return (np.array(coords) if coords else np.empty((0, 4))), keys

    def _refresh_layers(self) -> None:
        if self.points is None:
            return
        self._updating = True
        try:
            coords, self._keys = self._rows()
            colours = [COLOURS[k[2]] for k in self._keys]
            if self.pending is not None and self.pending[:2] == self._here():
                t, z, name, pts = self.pending
                extra = np.array([(t, z, y, x) for x, y in pts]).reshape(-1, 4)
                coords = np.vstack([coords, extra])
                colours += [COLOURS[name]] * len(pts)
            self.points.data = coords
            if colours:  # dots filled with their contour's colour, thin white rim to stand out on the image
                self.points.face_color = colours
                self.points.border_color = "white"
            self._redraw_shapes()
        finally:
            self._updating = False

    def _redraw_shapes(self, only: set[Key] | None = None) -> None:
        t0, z0 = self._here()
        polys, colours, types = [], [], []
        self._shape_of = {}
        for key, ctrl in sorted(self.curves.items()):
            t, z, name = key
            if (t, z) != (t0, z0):
                continue
            self._shape_of[key] = len(polys)
            polys.append(self._outline(key))
            colours.append(COLOURS[name])
            types.append("path")  # a closed path, not a polygon: no fill to triangulate on every redraw
        if self.pending is not None and self.pending[:2] == (t0, z0) and len(self.pending[3]) >= 2:
            t, z, name, pts = self.pending
            a = np.array(pts)
            polys.append(np.column_stack([np.full(len(a), t), np.full(len(a), z), a[:, 1], a[:, 0]]))
            colours.append(COLOURS[name])
            types.append("path")
        self.shapes.data = []
        if polys:
            self.shapes.add(polys, shape_type=types, edge_color=colours, face_color=(0, 0, 0, 0), edge_width=LINE_WIDTH)

    def _outline(self, key: Key) -> np.ndarray:
        """The contour as a closed line in layer coordinates (t, z, y, x)."""
        t, z, _ = key
        d = spline_through(self.curves[key], 200)
        d = np.vstack([d, d[:1]])
        return np.column_stack([np.full(len(d), t), np.full(len(d), z), d[:, 1], d[:, 0]])

    def _drag_update(self, key: Key, i: int) -> None:
        """Redraw while a dot is dragged: move just that dot and redraw just that contour (a full redraw
        rebuilds every dot and line on the slice, too slow to keep up with the mouse). At most ~80 times a
        second; the release does a full redraw."""
        now = time.perf_counter()
        if now - self._last_draw < 0.012:
            return
        self._last_draw = now
        rows = [r for r, k in enumerate(self._keys) if k == key]
        idx = self._shape_of.get(key)
        if i >= len(rows) or idx is None or len(rows) != len(self.curves[key]):
            self._refresh_layers()
            return
        self._updating = True
        try:
            x, y = self.curves[key][i]
            self.points.data[rows[i], 2:] = (y, x)
            light_refresh(self.points)
            self.shapes._data_view.edit(idx, self._outline(key))
            light_refresh(self.shapes)
        except Exception:  # napari internals changed: fall back to the full redraw
            self._refresh_layers()
        finally:
            self._updating = False

    # ---------- undo ----------

    def _snapshot(self):
        return ({k: v.copy() for k, v in self.curves.items()}, set(self.edited), set(self.hidden),
                None if self.pending is None else (*self.pending[:3], list(self.pending[3])))

    def push_undo(self) -> None:
        """Remember the state before an edit. A new edit ends the redo history (as in every editor)."""
        self._undo.append(self._snapshot())
        del self._undo[:-50]
        self._redo_saved, self._redo = self._redo, []

    def _drop_undo(self) -> None:
        """The edit didn't change anything after all: forget its undo step and keep the redo history."""
        if self._undo:
            self._undo.pop()
        self._redo = getattr(self, "_redo_saved", self._redo)

    def _restore(self, snap) -> None:
        # Contours that differ between now and the restored state must be written back to the labels: the
        # panel may already have applied the edit being undone (live results) and cleared `edited`.
        cur, old = self.curves, snap[0]
        changed = {k for k in set(cur) | set(old)
                   if k not in cur or k not in old or not np.array_equal(cur[k], old[k])}
        self.curves, self.edited = snap[0], set(snap[1]) | changed
        if len(snap) > 2:
            self.hidden = snap[2]
        self.pending = snap[3] if len(snap) > 3 else None
        self._refresh_layers()
        self._notify()

    def undo(self) -> bool:
        """Revert the last edit of any kind (Ctrl+Z)."""
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        """Put back the last undone edit (Ctrl+Y or Ctrl+Shift+Z)."""
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        return True

    # ---------- geometry helpers ----------

    DOT_SIZE = 1.1  # the dots' diameter in image pixels (the points layer's `size`)

    def _tol_px(self) -> float:
        """How far (image pixels) a click may be from a dot's centre and still take it: 12 screen pixels, or
        the dot itself plus a little when zoomed in so far that the drawn dot is bigger than that. Without the
        second part a click visibly on a big dot could miss it."""
        zoom = float(self.camera.zoom) or 1.0  # screen pixels per world unit
        screen = self.HIT_SCREEN_PX / zoom / float(min(self.scale[2], self.scale[3]))
        return max(screen, self.DOT_SIZE / 2 + 0.25)

    def _to_px(self, world_pos) -> np.ndarray:
        """Screen (world) position -> image pixels (x, y), through the layer's scale and any turn of the
        picture (orient.py), so clicks land on the right pixel however the image is rotated or mirrored."""
        lay = self.points if self.points is not None else self.shapes
        if lay is not None:
            try:
                d = np.asarray(lay.world_to_data(tuple(world_pos)), dtype=float)
                return np.array([d[-1], d[-2]])
            except Exception:
                pass
        return np.array([world_pos[-1] / self.scale[3], world_pos[-2] / self.scale[2]])

    def hit(self, t: int, z: int, p: np.ndarray, tol: float):
        """What is under point p (pixels): ('point', key, i), ('curve', key, i) or None.

        For 'curve', i is the index the new point would get (insert after control point i-1).
        """
        best = None
        for key, ctrl in self.curves.items():
            if key[:2] != (t, z) or not self._visible(key):
                continue
            d = np.hypot(*(ctrl - p).T)
            i = int(np.argmin(d))
            if d[i] <= tol and (best is None or best[0] != "point" or d[i] < best[3]):
                best = ("point", key, i, float(d[i]))
        if best:
            return best[:3]
        for key, ctrl in self.curves.items():
            if key[:2] != (t, z) or not self._visible(key) or len(ctrl) < 3:
                continue
            dense = spline_through(ctrl, 400)
            dd = np.hypot(*(dense - p).T)
            j = int(np.argmin(dd))
            if dd[j] <= tol and (best is None or dd[j] < best[3]):
                per = max(2, 400 // len(ctrl))  # dense points per control segment, as in spline_through
                best = ("curve", key, int(j // per) + 1, float(dd[j]))
        return best[:3] if best else None

    # ---------- edits (each one undoable) ----------

    def _changed(self, key: Key) -> None:
        self.edited.add(key)

    def move_point(self, key: Key, i: int, p: np.ndarray) -> None:
        self.curves[key][i] = p
        self._changed(key)

    def insert_point_at(self, key: Key, i: int, p: np.ndarray) -> int:
        self.curves[key] = np.insert(self.curves[key], i, p, axis=0)
        self._changed(key)
        return i

    def delete_point(self, key: Key, i: int) -> bool:
        if len(self.curves[key]) <= 4:
            self._say("A contour needs at least 4 points. Use Clear to remove it.")
            return False
        self.curves[key] = np.delete(self.curves[key], i, axis=0)
        self._changed(key)
        return True

    def clear_slice(self, t: int, z: int, which: tuple[str, ...] | None = None) -> int:
        """Delete contours on one slice and frame (the trash can): the contour being edited, or `which`.
        Undo brings them back."""
        if which is None:
            which = (self.focus,) if self.focus else ("endo", "epi", "rv")
        keys = [k for k in self.curves if k[:2] == (t, z) and k[2] in which]
        if not keys:
            return 0
        self.push_undo()
        for k in keys:
            del self.curves[k]
            self.edited.add(k)
        self.hidden -= set(keys)
        self._refresh_layers()
        self._notify()
        return len(keys)

    def add_pending(self, t: int, z: int, p: np.ndarray) -> None:
        if self.pending is None or self.pending[:2] != (t, z):
            self.pending = (t, z, self.add_to, [])
        self.pending[3].append((float(p[0]), float(p[1])))

    def close_pending(self) -> bool:
        if self.pending is None or len(self.pending[3]) < 3:
            return False
        t, z, name, pts = self.pending
        self.curves[(t, z, name)] = np.array(pts, dtype=float)
        self.edited.add((t, z, name))
        self.pending = None
        return True

    def _notify(self) -> None:
        if self.on_edit:
            self.on_edit()

    def _say(self, msg: str) -> None:
        if self.on_change:
            self.on_change(msg)

    # ---------- mouse ----------

    def _on_mouse(self, viewer, event):
        """Segment-style mouse editing; see the module docstring."""
        if not self.active or self.suspended or self.points is None or len(event.position) < 4:
            return
        t, z = self._here()
        p = self._to_px(event.position)
        tol = self._tol_px()
        if event.button == 1 and "Control" in {str(getattr(m, "name", m)) for m in (getattr(event, "modifiers", None) or ())} \
                and self.on_window is not None:
            # Ctrl + drag: window/level like Segment (left/right = contrast, up/down = brightness)
            last = np.array(event.pos, dtype=float)
            yield
            while event.type == "mouse_move":
                now = np.array(event.pos, dtype=float)
                d = now - last
                if np.abs(d).max() >= 2:
                    self.on_window(int(round(d[0] / 2)), int(round(-d[1] / 2)))
                    last = now
                yield
            return
        if event.button == 1 and self.click_hook is not None:  # e.g. "click the septum"
            hook, self.click_hook = self.click_hook, None
            hook(t, z, p)
            return
        if event.button == 1 and (self.pan_mode or not self.show_dots):
            # Pan on, or Points off (a clean view): left-drag moves the image, nothing is edited
            yield from self._pan_drag(event)
            return
        if event.button == 1:
            grab = tol * self.GRAB_SCREEN_PX / self.HIT_SCREEN_PX
            # drawing a new contour: drag any of its points to fix a misplaced click
            if self.pending is not None and self.pending[:2] == (t, z) and self.pending[3]:
                pts = np.array(self.pending[3], dtype=float)
                d = np.hypot(*(pts - p).T)
                j = int(np.argmin(d))
                if d[j] <= tol:  # the small click radius: clicks a little further away still add points
                    self.push_undo()
                    yield
                    moved = False
                    while event.type == "mouse_move":
                        q = self._to_px(event.position)
                        self.pending[3][j] = (float(q[0]), float(q[1]))
                        moved = True
                        self._refresh_layers()
                        yield
                    if not moved:  # a plain click on a point you placed changes nothing
                        self._drop_undo()
                    return
            h = self.hit(t, z, p, tol)
            if h is None:
                near = self._nearest_dot(t, z, p, grab)
                if near is not None and self.pending is None:
                    h = ("point", *near)
            if h is None:
                # empty image: a drag pans (only when it starts well away from every dot and line, so a
                # near-miss on a dot never slides the image); a click draws a new contour where one is missing
                moved = yield from self._follow(event)  # editing never pans (Pan button or right-drag does)
                can_draw = self.pending is not None or (t, z, self.add_to) not in self.curves
                if not moved and can_draw:
                    self.push_undo()
                    self.add_pending(t, z, p)
                    self._refresh_layers()
                    self._say("Drawing: click to add points, drag one to adjust, right-click to close.")
                return
            kind, key, i = h
            self.push_undo()
            if kind == "curve":  # a new dot on the line: it has to appear before the drag
                i = self.insert_point_at(key, i, p)
                self._refresh_layers()
            # grabbing an existing dot changes nothing yet: no redraw before the drag starts (a full redraw
            # here was the pause between pressing and the dot following the mouse)
            yield
            moved = False
            while event.type == "mouse_move":
                self.move_point(key, i, self._to_px(event.position))
                moved = True
                self._drag_update(key, i)
                yield
            if moved:
                self._refresh_layers()  # the exact final position, everything in sync
            if kind == "point" and not moved:  # a plain click on a dot changes nothing
                self._drop_undo()
            else:
                self._notify()
        elif event.button in (2, 3):  # right (or middle) button
            moved = yield from self._pan_drag(event)
            if moved or event.button == 3:
                return
            if self.pending is not None and self.pending[:2] == (t, z):
                self.push_undo()
                if self.close_pending():
                    self._say("Contour closed.")
                    self._notify()
                else:
                    self._drop_undo()
                    self._say("Place at least 3 points first.")
                self._refresh_layers()
                return
            h = self.hit(t, z, p, tol)
            if h is not None and h[0] == "point":
                self.push_undo()
                if self.delete_point(h[1], h[2]):
                    self._refresh_layers()
                    self._notify()
                else:
                    self._drop_undo()

    def _nearest_dot(self, t: int, z: int, p: np.ndarray, radius: float):
        """(key, index) of the closest editable dot within `radius` pixels, or None."""
        best = None
        for key, ctrl in self.curves.items():
            if key[:2] != (t, z) or not self._visible(key):
                continue
            d = np.hypot(*(ctrl - p).T)
            i = int(np.argmin(d))
            if d[i] <= radius and (best is None or d[i] < best[2]):
                best = (key, i, float(d[i]))
        return None if best is None else best[:2]

    def _follow(self, event):
        """Follow a drag without doing anything; returns whether it moved more than a few pixels."""
        start = np.array(event.pos, dtype=float)
        moved = False
        yield
        while event.type == "mouse_move":
            if np.abs(np.array(event.pos, dtype=float) - start).max() > 3:
                moved = True
            yield
        return moved

    def _pan_drag(self, event):
        """Follow a drag, panning the view once it moves more than a few pixels; returns whether it moved."""
        start = np.array(event.pos, dtype=float)
        last = start.copy()
        moved = False
        yield
        while event.type == "mouse_move":
            now = np.array(event.pos, dtype=float)
            if np.abs(now - start).max() > 3:
                moved = True
            if moved:
                cam = self.camera
                c = list(cam.center)
                d = (now - last) / (float(cam.zoom) or 1.0)
                c[-1] -= d[0]
                c[-2] -= d[1]
                cam.center = tuple(c)
            last = now
            yield
        return moved

    # ---------- napari-driven edits (kept for scripts/tests that set the layer data) ----------

    def _on_points_changed(self, event=None) -> None:
        if self._updating or self.points is None:
            return
        data = np.asarray(self.points.data)
        n_old = len(self._keys)
        if len(data) < n_old:
            built = self._match_after_delete(data)
        else:
            new_curves: dict[Key, list] = {}
            for row, key in zip(data[:n_old], self._keys):
                new_curves.setdefault(key, []).append((row[3], row[2]))
            built = {k: np.array(v) for k, v in new_curves.items()}
            for row in data[n_old:]:
                key = (int(round(row[0])), int(round(row[1])), self.add_to)
                p = np.array([row[3], row[2]])
                built[key] = insert_point(built[key], p) if key in built else np.array([p])
        shown = set(self._keys)
        for k, v in self.curves.items():  # contours not in the points layer stay as they are
            if k not in shown and k not in built:
                built[k] = v
        changed = {k for k in set(built) | set(self.curves)
                   if k not in self.curves or k not in built or not np.array_equal(built[k], self.curves[k])}
        if changed:
            self.push_undo()
        self.curves = {k: v for k, v in built.items() if len(v) >= 4}
        self.edited |= changed
        self._refresh_layers()
        if changed:
            self._notify()

    def _match_after_delete(self, data: np.ndarray) -> dict[Key, np.ndarray]:
        """Assign each remaining point to the contour of the nearest old point."""
        old, keys = self._rows()
        built: dict[Key, list] = {}
        for row in data:
            same = np.flatnonzero((old[:, 0] == row[0]) & (old[:, 1] == row[1]))
            j = same[np.argmin(np.linalg.norm(old[same, 2:] - row[2:], axis=1))]
            built.setdefault(keys[j], []).append((row[3], row[2]))
        return {k: np.array(v) for k, v in built.items()}

    # ---------- smoothing ----------

    def smooth(self, frame: int, slice_: int, selected_rows: set[int] | None = None) -> int:
        """Smooth the selected points, or both contours on this slice and frame if none are selected."""
        changed = 0
        self.push_undo()
        rows_by_key: dict[Key, list[int]] = {}
        for row, key in enumerate(self._keys):
            rows_by_key.setdefault(key, []).append(row)
        targets = ([k for k in self.curves if k[:2] == (frame, slice_) and self._focused(k)]
                   if not selected_rows else list(rows_by_key))
        for key in targets:
            if selected_rows:
                which = np.array([r in selected_rows for r in rows_by_key[key]])
                if not which.any():
                    continue
                self.curves[key] = smooth_points(self.curves[key], which)
            else:
                self.curves[key] = smooth_points(self.curves[key])
            self.edited.add(key)
            changed += 1
        if changed:
            self._refresh_layers()
        else:
            self._drop_undo()
        return changed

    def smooth(self, frame: int, slice_: int, passes: int = 1, strength: float = 0.3) -> int:
        """Smooth the contour being edited on one slice (all contours if none is chosen). Its dots stay, so it
        can be adjusted straight away; Smooth again for smoother."""
        keys = [k for k in self.curves if (k[0], k[1]) == (frame, slice_) and self._focused(k)]
        if not keys:
            return 0
        self.push_undo()
        for key in keys:
            self.curves[key] = smooth_points(self.curves[key], strength=strength, passes=passes)
            self.edited.add(key)
        self.hidden -= set(keys)
        self._refresh_layers()
        self._notify()
        return len(keys)

    MIN_GAP_PX = 0.5  # grow/shrink keeps the endocardium at least this far inside the epicardium

    def offset(self, frame: int, slice_: int, d: float) -> tuple[str | None, float]:
        """Grow (d > 0) or shrink (d < 0) the selected contour on one slice by d pixels.

        The endocardium never crosses the epicardium (and vice versa): points that would cross stay
        where they are. Returns (contour name, area change in
        pixels), or (None, 0) if there was nothing to change.
        """
        from ..contours import polygon_area

        name = self.focus or "endo"
        key = (frame, slice_, name)
        if key not in self.curves:
            return None, 0.0
        old = self.curves[key]
        new = offset_points(old, d)
        if d < 0 and (polygon_area(new) < 4.0 or polygon_area(new) >= polygon_area(old)):
            return name, 0.0  # too small to shrink further (it would turn inside out)
        inner, outer = (frame, slice_, "endo"), (frame, slice_, "epi")
        if name == "endo" and outer in self.curves:
            gap = offset_points(self.curves[outer], -self.MIN_GAP_PX)
            ok = _inside(gap, new)
            new[~ok] = old[~ok]
        elif name == "epi" and inner in self.curves:
            gap = offset_points(self.curves[inner], self.MIN_GAP_PX)
            ok = ~_inside(gap, new)
            new[~ok] = old[~ok]
        change = polygon_area(new) - polygon_area(old)
        if abs(change) < 1e-6:
            return name, 0.0
        self.push_undo()
        self.curves[key] = new
        self.edited.add(key)
        self._refresh_layers()
        self._notify()
        return name, change

    smooth_and_hide = smooth  # older name

    def show_slice_points(self, frame: int, slice_: int) -> None:
        """Bring back the points on one slice for editing."""
        keys = {k for k in self.hidden if k[:2] == (frame, slice_)}
        if keys:
            self.hidden -= keys
            self._refresh_layers()

    def show_points(self, on: bool) -> None:
        self.set_active(on)

    def apply_to(self, label4d: np.ndarray) -> np.ndarray:
        """Write edited slices back into a (x, y, slice, time) label array; returns a new array (with the same
        memory layout as label4d: often a transposed view of the napari layer, which order="C" would reshuffle)."""
        out = label4d.copy(order="K")
        slices = {(t, z) for t, z, _ in self.edited}
        for t, z in slices:
            out[:, :, z, t] = labels_from_curves(self.curves.get((t, z, "endo")), self.curves.get((t, z, "epi")),
                                                 out.shape[:2], rv=self.curves.get((t, z, "rv")))
        return out
