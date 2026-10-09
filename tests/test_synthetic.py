import pytest

from tracecmr import labels as L
from tracecmr.synthetic import ring_phantom, ring_phantom_truth


@pytest.mark.parametrize("spacing", [(0.1, 0.1, 1.0), (0.05, 0.05, 0.5), (0.2, 0.2, 1.0)])
def test_phantom_voxel_volumes_match_exact_geometry(spacing):
    """Counting voxels should approach the true volume; finer pixels -> closer."""
    shape = (int(12 / spacing[0]), int(12 / spacing[1]), 8)
    ci = ring_phantom(shape=shape, spacing=spacing)
    truth = ring_phantom_truth(shape=shape, spacing=spacing)
    counts = ci.labels_present()
    blood = counts[L.LV_BLOODPOOL] * ci.voxel_volume_mm3
    myo = counts[L.LV_MYOCARDIUM] * ci.voxel_volume_mm3
    # Pixelation error is a few percent at mouse resolution; tighter when pixels are smaller.
    tol = 0.03 if spacing[0] <= 0.1 else 0.06
    assert blood == pytest.approx(truth["blood_ul"], rel=tol)
    assert myo == pytest.approx(truth["myocardium_ul"], rel=tol)


def test_phantom_is_centered_and_on_every_slice():
    ci = ring_phantom(shape=(128, 128, 10))
    assert ci.slices_with(L.LV_BLOODPOOL) == list(range(10))
    assert ci.label[64, 64, 0] == L.LV_BLOODPOOL
    assert ci.label[0, 0, 0] == L.BACKGROUND
