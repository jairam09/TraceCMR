import numpy as np
import nibabel as nib
import pytest

from mcmr import labels as L
from mcmr.io import CardiacImage, load_nifti, remap_labels, sanity_check, save_nifti
from mcmr.synthetic import ring_phantom


def test_roundtrip_preserves_image_label_and_spacing(tmp_path):
    ci = ring_phantom(spacing=(0.12, 0.12, 0.8), noise_sd=0.05)
    save_nifti(ci, tmp_path / "img.nii.gz", tmp_path / "lab.nii.gz")
    back = load_nifti(tmp_path / "img.nii.gz", tmp_path / "lab.nii.gz")
    assert back.image.shape == ci.image.shape
    assert np.allclose(back.spacing, (0.12, 0.12, 0.8))
    assert np.allclose(back.image, ci.image, atol=1e-6)
    assert np.array_equal(back.label, ci.label)


def test_acdc_labels_are_remapped():
    acdc = np.array([0, 1, 2, 3])  # bg, RV, myo, LV in ACDC
    out = remap_labels(acdc, L.ACDC)
    assert out.tolist() == [L.BACKGROUND, L.RV_BLOODPOOL, L.LV_MYOCARDIUM, L.LV_BLOODPOOL]


def test_unknown_label_value_raises():
    with pytest.raises(ValueError, match="not in mapping"):
        remap_labels(np.array([0, 1, 7]), L.ACDC)


def test_4d_requires_frame(tmp_path):
    data = np.zeros((8, 8, 3, 5), dtype=np.float32)
    data[..., 2] = 1.0
    nib.save(nib.Nifti1Image(data, np.eye(4)), tmp_path / "cine.nii.gz")
    with pytest.raises(ValueError, match="frame"):
        load_nifti(tmp_path / "cine.nii.gz")
    ci = load_nifti(tmp_path / "cine.nii.gz", frame=2)
    assert ci.image.shape == (8, 8, 3)
    assert ci.image.max() == 1.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        CardiacImage(image=np.zeros((4, 4, 2)), spacing=(1, 1, 1), label=np.zeros((4, 4, 3)))


def test_voxel_volume_units():
    ci = CardiacImage(image=np.zeros((2, 2, 2)), spacing=(0.1, 0.1, 1.0))
    assert ci.voxel_volume_mm3 == pytest.approx(0.01)  # mm^3 == uL


def test_sanity_check_flags_human_spacing_for_mouse():
    ci = ring_phantom(spacing=(1.5, 1.5, 8.0), shape=(64, 64, 4), blood_radius_mm=20, wall_mm=8)
    assert any("unusual for mouse" in w for w in sanity_check(ci, species="mouse"))
    assert not any("unusual" in w for w in sanity_check(ci, species="human"))


def test_sanity_check_flags_gap_in_lv_slices():
    ci = ring_phantom()
    ci.label[:, :, 4][ci.label[:, :, 4] == L.LV_BLOODPOOL] = L.BACKGROUND
    assert any("not contiguous" in w for w in sanity_check(ci))


def test_clean_phantom_has_no_warnings():
    assert sanity_check(ring_phantom(noise_sd=0.05)) == []
