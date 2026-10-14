"""Turning the picture: one matrix for the viewer, the ED | ES view, figures and exported images."""

import numpy as np

from tracecmr.app import orient


def test_clockwise_on_screen():
    # a point to the right of the centre moves down (y grows downward on screen) for +90
    p = orient.transform_points_xy([[1, 0]], 90, False, (0, 0))[0]
    assert np.allclose(p, [0, 1])
    # mirror then rotate: right goes left
    assert np.allclose(orient.transform_points_xy([[1, 0]], 0, True, (0, 0))[0], [-1, 0])


def test_image_and_points_agree():
    h, w = 41, 41
    img = np.zeros((h, w))
    x, y = 30, 12
    img[y, x] = 1.0
    for angle, flip in ((90, False), (-90, False), (180, False), (0, True), (90, True), (30, False)):
        out = orient.orient_image(img, angle, flip, order=1)
        yy, xx = np.unravel_index(np.argmax(out), out.shape)
        px, py = orient.transform_points_xy([[x, y]], angle, flip, ((w - 1) / 2, (h - 1) / 2))[0]
        assert abs(xx - px) <= 1 and abs(yy - py) <= 1, (angle, flip, (xx, yy), (px, py))


def test_napari_affine_matches_points():
    cy, cx = 12.8, 12.8  # world centre (mm)
    a = orient.napari_affine(35, True, (cy, cx))
    world = a @ np.array([3, 4, 5.0, 9.0, 1])  # (t, z, y, x, 1)
    px = orient.transform_points_xy([[9.0, 5.0]], 35, True, (cx, cy))[0]
    assert np.allclose(world[:2], [3, 4])
    assert np.allclose(world[3], px[0]) and np.allclose(world[2], px[1])


def test_identity_and_grow_box():
    img = np.arange(12.0).reshape(3, 4)
    assert orient.orient_image(img, 360, False) is img
    assert orient.grow_box((10, 20, 10, 20), (256, 256), 0, False) == (10, 20, 10, 20)
    g = orient.grow_box((10, 20, 10, 20), (256, 256), 45, False)
    assert g[0] < 10 and g[1] > 20 and g[2] < 10 and g[3] > 20
