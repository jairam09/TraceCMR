"""TraceCMR: open-source analysis for mouse cardiac MRI.

Planned outputs: LV volumes and function, regional wall thickness and
thickening, and LGE scar quantification.
"""

import os as _os

# Renamed from VentriMap (Oct 2026): settings made with the old VENTRIMAP_* names keep working.
for _k, _v in list(_os.environ.items()):
    if _k.startswith("VENTRIMAP_") and "TRACECMR_" + _k[10:] not in _os.environ:
        _os.environ["TRACECMR_" + _k[10:]] = _v

from .io import CardiacImage, load_nifti, save_nifti, sanity_check

__version__ = "0.3.7"
__all__ = ["CardiacImage", "load_nifti", "save_nifti", "sanity_check"]
