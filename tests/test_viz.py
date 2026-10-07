from mcmr.synthetic import ring_phantom
from mcmr.viz import save_overlay


def test_overlay_writes_png(tmp_path):
    out = save_overlay(ring_phantom(noise_sd=0.05), tmp_path / "overlay.png")
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
