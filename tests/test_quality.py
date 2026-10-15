"""Double-check flags: checking only some slices, and not re-checking unchanged ones."""

import numpy as np

from tracecmr import quality as Q
from tracecmr.synthetic import tapered_phantom


def _scan():
    ed = tapered_phantom(n_slices=6, base_blood_mm=2.4, apex_blood_mm=1.0, wall_mm=1.0)
    es = tapered_phantom(n_slices=6, base_blood_mm=1.6, apex_blood_mm=0.6, wall_mm=1.3)
    lab = np.stack([ed.label, es.label], -1)
    img = np.stack([ed.image, es.image], -1)
    lab[:, :, 3, 0] = np.roll(lab[:, :, 3, 0], 12, axis=0)  # one outline well off the heart in the image
    pairs = {"ED": [(0, z) for z in range(6)], "ES": [(1, z) for z in range(6)]}
    return lab, img, ed.spacing, pairs


def test_only_and_cache_give_the_same_flags(monkeypatch):
    lab, img, sp, pairs = _scan()
    order = list(range(6))
    full = Q.check_slices(lab, img, sp, pairs, order)
    assert (0, 3) in full
    some = {(0, 2), (0, 3), (1, 4)}
    part = Q.check_slices(lab, img, sp, pairs, order, only=some)
    for k in some:  # the per-slice checks of these slices are unchanged
        assert part.get(k, []) == full.get(k, [])

    calls = []
    real = Q._slice_checks
    monkeypatch.setattr(Q, "_slice_checks", lambda *a: calls.append(1) or real(*a))
    cache: dict = {}
    first = Q.check_slices(lab, img, sp, pairs, order, cache=cache)
    n_first = len(calls)
    assert first == full and n_first > 0
    lab2 = lab.copy()
    lab2[:, :, 1, 0][lab2[:, :, 1, 0] == 2] = 0  # edit one slice
    Q.check_slices(lab2, img, sp, pairs, order, cache=cache)
    assert len(calls) == n_first + 1  # only the edited slice was checked again
    assert Q.check_slices(lab, img, sp, pairs, order, cache=cache) == full
