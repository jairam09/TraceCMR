"""Segment-style contour editing in napari: drag control points, the curve follows.

Each contour (endocardium, epicardium) on each slice of the ED and ES frames is a
smooth closed curve through about 16-40 control points, placed
adaptively so tips and notches are kept. napari shows the control points
as a Points layer (drag to move, add mode to insert, Delete to remove) and the
curves as a Shapes layer. When the user computes, edited contours are turned
back into labels; untouched slices keep their original masks exactly.
"""

from __future__ import annotations

import numpy as np

from .. import labels as L
from ..contours import Curve, curve_to_mask, mask_to_curve, spline_through

START_POINTS = 18  # evenly spaced, plus a few at sharp tips, never bunched (docs/methods.md)
COLOURS = {"endo": (0.90, 0.29, 0.29, 1.0), "epi": (0.24, 0.78, 0.43, 1.0)}
Key = tuple[int, int, str]  # (frame, slice, "endo"/"epi")


def curves_from_labels(label4d: np.ndarray, frames: list[int], n_points: int = START_POINTS) -> dict[Key, np.ndarray]:
    """Control points for every contoured slice of the given frames. label4d is (x, y, slice, time)."""
    out: dict[Key, np.ndarray] = {}
    for t in frames:
        for z in range(label4d.shape[2]):
            lab = label4d[:, :, z, t]
            blood = lab == L.LV_BLOODPOOL
            lv = blood | (lab == L.LV_MYOCARDIUM) | (lab == L.SCAR)
            for name, mask in (("endo", blood), ("epi", lv)):
                # epicardium: smoother (anatomically smooth surface) and evenly spaced, no tip points
                c = mask_to_curve(mask, n_points) if name == "endo" else mask_to_curve(mask, 16, None, smooth_px=3.0)
                if c is not None:
                    out[(t, z, name)] = c.control
    return out


def labels_from_curves(endo: np.ndarray | None, epi: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """One slice of standard labels from endo/epi control points."""
    lab = np.zeros(shape, dtype=np.uint8)
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


def insert_point(control: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Insert a new point into a closed polygon at the nearest edge."""
    a, b = control, np.roll(control, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
    d = np.linalg.norm(a + t[:, None] * ab - p, axis=1)
    i = int(np.argmin(d))
    return np.insert(control, i + 1, p, axis=0)


class CurveEditor:
    """Owns the 'control points' and 'curves' layers and keeps them in sync."""

    def __init__(self, viewer, scale: tuple[float, float, float, float]):
        self.viewer = viewer
        self.scale = scale
        self.curves: dict[Key, np.ndarray] = {}
        self.edited: set[Key] = set()
        self._keys: list[Key] = []  # key of each row in the points layer
        self._updating = False
        self.points = None
        self.shapes = None
        self.add_to = "endo"  # contour that newly added points join
        self._undo: list[tuple[dict, set, set]] = []
        self.hidden: set[tuple[int, int]] = set()  # (frame, slice) whose points are hidden after smoothing

    # ---------- layers ----------

    def show(self, curves: dict[Key, np.ndarray]) -> None:
        self.curves = {k: v.copy() for k, v in curves.items()}
        self.edited.clear()
        self.hidden.clear()
        for name in ("curves", "control points"):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        self.shapes = self.viewer.add_shapes(name="curves", scale=self.scale, ndim=4)
        self.points = self.viewer.add_points(np.empty((0, 4)), name="control points", scale=self.scale,
                                             ndim=4, size=1.5, border_width=0.2, face_color="white",
                                             out_of_slice_display=False)
        self.points.events.data.connect(self._on_points_changed)
        self._refresh_layers()
        self.points.mode = "select"
        self.viewer.layers.selection.active = self.points

    def _rows(self) -> tuple[np.ndarray, list[Key]]:
        coords, keys = [], []
        for key, ctrl in sorted(self.curves.items()):
            t, z, _ = key
            if (t, z) in self.hidden:
                continue
            for x, y in ctrl:
                coords.append((t, z, y, x))
                keys.append(key)
        return (np.array(coords) if coords else np.empty((0, 4))), keys

    def _refresh_layers(self) -> None:
        self._updating = True
        try:
            coords, self._keys = self._rows()
            self.points.data = coords
            self.points.border_color = [COLOURS[k[2]] for k in self._keys] if self._keys else "white"
            self._redraw_shapes()
        finally:
            self._updating = False

    def _redraw_shapes(self, only: set[Key] | None = None) -> None:
        polys, colours = [], []
        for key, ctrl in sorted(self.curves.items()):
            t, z, name = key
            dense = spline_through(ctrl, 200)
            polys.append(np.column_stack([np.full(len(dense), t), np.full(len(dense), z), dense[:, 1], dense[:, 0]]))
            colours.append(COLOURS[name])
        self.shapes.data = []
        if polys:
            self.shapes.add(polys, shape_type="polygon", edge_color=colours, face_color=(0, 0, 0, 0),
                            edge_width=0.25)

    # ---------- edits ----------

    def _on_points_changed(self, event=None) -> None:
        if self._updating:
            return
        data = np.asarray(self.points.data)
        n_old = len(self._keys)
        new_curves: dict[Key, list] = {}
        if len(data) >= n_old:  # moved and/or added points
            for row, key in zip(data[:n_old], self._keys):
                new_curves.setdefault(key, []).append((row[3], row[2]))
            built = {k: np.array(v) for k, v in new_curves.items()}
            for row in data[n_old:]:  # added points join the chosen contour on that slice
                key = (int(round(row[0])), int(round(row[1])), self.add_to)
                p = np.array([row[3], row[2]])
                built[key] = insert_point(built[key], p) if key in built else np.array([p])
        else:  # deleted points: napari keeps the remaining rows in order but we lost the mapping
            built = self._match_after_delete(data)
        for k, v in self.curves.items():  # contours on hidden slices are not in the points layer
            if (k[0], k[1]) in self.hidden:
                built[k] = v
        changed = {k for k in set(built) | set(self.curves)
                   if k not in self.curves or k not in built or not np.array_equal(built[k], self.curves[k])}
        self.curves = {k: v for k, v in built.items() if len(v) >= 4}
        self.edited |= changed
        self._refresh_layers()

    def _match_after_delete(self, data: np.ndarray) -> dict[Key, np.ndarray]:
        """Assign each remaining point to the contour of the nearest old point."""
        old, keys = self._rows()
        built: dict[Key, list] = {}
        for row in data:
            same = np.flatnonzero((old[:, 0] == row[0]) & (old[:, 1] == row[1]))
            j = same[np.argmin(np.linalg.norm(old[same, 2:] - row[2:], axis=1))]
            built.setdefault(keys[j], []).append((row[3], row[2]))
        return {k: np.array(v) for k, v in built.items()}

    def smooth(self, frame: int, slice_: int, selected_rows: set[int] | None = None) -> int:
        """Smooth the selected points, or both contours on this slice and frame if none are selected.

        Returns the number of contours changed.
        """
        changed = 0
        snapshot = ({k: v.copy() for k, v in self.curves.items()}, set(self.edited), set(self.hidden))
        rows_by_key: dict[Key, list[int]] = {}
        for row, key in enumerate(self._keys):
            rows_by_key.setdefault(key, []).append(row)
        for key, rows in rows_by_key.items():
            t, z, _ = key
            if selected_rows:
                which = np.array([r in selected_rows for r in rows])
                if not which.any():
                    continue
                self.curves[key] = smooth_points(self.curves[key], which)
            elif (t, z) == (frame, slice_):
                self.curves[key] = smooth_points(self.curves[key])
            else:
                continue
            self.edited.add(key)
            changed += 1
        if changed:
            self._undo.append(snapshot)
            del self._undo[:-20]
            self._refresh_layers()
        return changed

    def smooth_and_hide(self, frame: int, slice_: int, passes: int = 1, strength: float = 0.3) -> int:
        """Segment-style: smooth both contours on one slice and hide that slice's points."""
        keys = [k for k in self.curves if (k[0], k[1]) == (frame, slice_)]
        if not keys:
            return 0
        self._undo.append(({k: v.copy() for k, v in self.curves.items()}, set(self.edited), set(self.hidden)))
        del self._undo[:-20]
        for key in keys:
            self.curves[key] = smooth_points(self.curves[key], strength=strength, passes=passes)
            self.edited.add(key)
        self.hidden.add((frame, slice_))
        self._refresh_layers()
        return len(keys)

    def show_slice_points(self, frame: int, slice_: int) -> None:
        """Bring back the points on one slice for editing."""
        if (frame, slice_) in self.hidden:
            self.hidden.discard((frame, slice_))
            self._refresh_layers()

    def show_points(self, on: bool) -> None:
        if self.points is not None:
            self.points.visible = on
            if on:
                self.viewer.layers.selection.active = self.points

    def undo(self) -> bool:
        """Revert the last smoothing step."""
        if not self._undo:
            return False
        snap = self._undo.pop()
        self.curves, self.edited = snap[0], snap[1]
        if len(snap) > 2:
            self.hidden = snap[2]
        self._refresh_layers()
        return True

    def apply_to(self, label4d: np.ndarray) -> np.ndarray:
        """Write edited slices back into a (x, y, slice, time) label array; returns a new array."""
        out = label4d.copy()
        slices = {(t, z) for t, z, _ in self.edited}
        for t, z in slices:
            out[:, :, z, t] = labels_from_curves(self.curves.get((t, z, "endo")), self.curves.get((t, z, "epi")),
                                                 out.shape[:2])
        return out
