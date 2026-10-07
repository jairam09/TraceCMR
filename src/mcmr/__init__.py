"""mcmr: open-source analysis pipeline for mouse cardiac MRI.

Planned outputs: LV volumes and function, regional wall thickness and
thickening, and LGE scar quantification.
"""

from .io import CardiacImage, load_nifti, save_nifti, sanity_check

__version__ = "0.1.1"
__all__ = ["CardiacImage", "load_nifti", "save_nifti", "sanity_check"]
