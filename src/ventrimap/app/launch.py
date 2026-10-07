"""Start napari with the VentriMap panel docked: `ventrimap` or `ventrimap Mouse40.h5`."""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    import napari

    from ._widget import VentriMapWidget
    from .session import Session

    argv = sys.argv[1:] if argv is None else argv
    viewer = napari.Viewer(title="VentriMap")
    panel = VentriMapWidget(viewer)
    viewer.window.add_dock_widget(panel, name="VentriMap", area="right")
    if argv:
        p = Path(argv[0])
        panel.load_session(Session.from_dicom(p) if p.is_dir() else Session.from_ucl(p))
    napari.run()


if __name__ == "__main__":
    main()
