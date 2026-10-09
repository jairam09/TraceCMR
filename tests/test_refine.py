import numpy as np



def test_fit_slice_shrinks_into_dark_and_grows_into_bright():
    """End-systole case: the outline covers dark muscle at the top and misses a bright horn on the side."""
    from tracecmr.refine import fit_slice

    n = 60
    yy, xx = np.mgrid[:n, :n]
    img = np.full((n, n), 100.0)
    epi = np.hypot(yy - 30, xx - 30) < 20
    blood_true = (np.abs(xx - 30) < 7) & (yy > 28) & (yy < 38)  # bright pool
    horn = (xx >= 18) & (xx < 24) & (yy > 22) & (yy < 36)  # bright strip on the left
    img[blood_true | horn] = 250.0
    lab = np.zeros((n, n), np.uint8)
    lab[epi] = 2
    model = (np.abs(xx - 30) < 7) & (yy > 20) & (yy < 38)  # reaches up into dark muscle, misses the horn
    lab[model] = 1
    out = fit_slice(lab, img, (0.1, 0.1))
    got = out == 1
    assert not got[(yy > 20) & (yy < 27) & (np.abs(xx - 30) < 5)].any()  # dark cap given back
    assert got[horn & (yy > 29)].mean() > 0.8  # horn added (it touches the pool)
    assert (out[epi] > 0).all() and ((out == 2) | (out == 1))[epi].all()  # epi unchanged


def test_fit_slice_leaves_low_contrast_alone():
    from tracecmr.refine import fit_slice

    lab = np.zeros((40, 40), np.uint8)
    yy, xx = np.mgrid[:40, :40]
    lab[np.hypot(yy - 20, xx - 20) < 15] = 2
    lab[np.hypot(yy - 20, xx - 20) < 6] = 1
    img = np.full((40, 40), 100.0) + np.random.default_rng(0).normal(0, 2, (40, 40))
    assert np.array_equal(fit_slice(lab, img, (0.1, 0.1)), lab)
