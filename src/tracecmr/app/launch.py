"""Start napari with the TraceCMR panel docked: `tracecmr` or `tracecmr Mouse40.h5`."""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    import napari

    from ._widget import TraceCMRWidget
    from .session import Session

    argv = sys.argv[1:] if argv is None else argv
    viewer = napari.Viewer(title="TraceCMR")
    panel = TraceCMRWidget(viewer)
    viewer.window.add_dock_widget(panel, name="TraceCMR", area="right")
    if argv:
        p = Path(argv[0])
        panel.load_session(Session.from_dicom(p) if p.is_dir() else Session.from_ucl(p))
    napari.run()


if __name__ == "__main__":
    main()
