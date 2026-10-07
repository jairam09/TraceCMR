from ventrimap.synthetic import ring_phantom
from ventrimap.viz import save_overlay


def test_overlay_writes_png(tmp_path):
    out = save_overlay(ring_phantom(noise_sd=0.05), tmp_path / "overlay.png")
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_bullseye_writes_png(tmp_path):
    import numpy as np
    from ventrimap.viz import save_bullseye

    vals = {"basal": np.linspace(0.6, 1.2, 6), "mid": np.full(6, np.nan), "apical": np.array([1, -0.2, 0.9, 1.0])}
    out = save_bullseye(vals, tmp_path / "b.png", "t", "mm", 0.4, 1.6)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
