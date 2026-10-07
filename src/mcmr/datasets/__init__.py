"""Loaders for specific public datasets. Each returns standard CardiacImage objects."""

from .ucl import UCL_SPACING, find_ed_es, labeled_frames, load_ucl

__all__ = ["UCL_SPACING", "find_ed_es", "labeled_frames", "load_ucl"]
