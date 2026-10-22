"""What TraceCMR found, and skipped, when scan folders were opened: for Help → Report a problem.

"It didn't open our files" is the feedback outside labs are most likely to send, and the hardest to act on, since
the scans stay on their computer. So opening a folder notes what was there and what was left out and why: each
scan's or series' parameters (scanner, software version, image size, how slices and frames are laid out). Never
pixels, and never animal, study or patient names. The notes go into the problem report and, as they happen, into
the log.
"""

from __future__ import annotations

import re
import sys
import time
from collections import Counter, deque

_NOTES: deque[str] = deque(maxlen=400)
# file paths inside error messages (their folder names can name the animal): quoted ones, which may hold spaces,
# and bare ones starting at a drive letter, a share or the root
_QUOTED_PATH = re.compile(r"""(['"])[^'"]*[\\/][^'"]*\1""")
_BARE_PATH = re.compile(r"(?<![\w.])(?:[A-Za-z]:[\\/]|\\\\|/(?=[^\s/]))\S*")


def scrub(text: str) -> str:
    """An error message with any file paths in it replaced by <path>."""
    return _BARE_PATH.sub("<path>", _QUOTED_PATH.sub(r"\1<path>\1", text))


def error(e: BaseException) -> str:
    return f"{type(e).__name__}: {scrub(str(e))}"


def note(msg: str) -> None:
    _NOTES.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}")
    try:
        if sys.stderr is not None:
            print(f"[open] {msg}", file=sys.stderr, flush=True)
    except Exception:  # never let a note break opening a scan
        pass


def notes() -> list[str]:
    return list(_NOTES)


def clear() -> None:
    _NOTES.clear()


def _val(ds, name: str) -> str:
    v = getattr(ds, name, None)
    if v is None or v == "":
        return "?"
    if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
        return "/".join(str(x) for x in v)
    return str(v)


def dicom_header(ds) -> str:
    """The parts of a DICOM header that decide whether and how TraceCMR reads it (no names, no IDs, no dates)."""
    sop = getattr(ds, "SOPClassUID", None)
    sop = getattr(sop, "name", None) or str(sop or "?")
    try:
        ts = ds.file_meta.TransferSyntaxUID
        ts = getattr(ts, "name", None) or str(ts)
    except Exception:
        ts = "?"
    multi = "PerFrameFunctionalGroupsSequence" in ds
    return (f"{_val(ds, 'Manufacturer')} {_val(ds, 'ManufacturerModelName')}, software {_val(ds, 'SoftwareVersions')}; "
            f"{_val(ds, 'Modality')}, {sop}, {ts}; {_val(ds, 'Rows')}x{_val(ds, 'Columns')}, "
            f"frames per file {_val(ds, 'NumberOfFrames') if multi or hasattr(ds, 'NumberOfFrames') else 1}, "
            f"{'enhanced (per-frame groups)' if multi else 'classic'}; "
            f"position {'yes' if hasattr(ds, 'ImagePositionPatient') or multi else 'no'}, "
            f"trigger time {'yes' if hasattr(ds, 'TriggerTime') else 'no'}")


def kinds_of_files(names: list[str]) -> str:
    """'12 .txt, 3 with no extension': what the unreadable files were, without their names."""
    c = Counter((n.rsplit(".", 1)[-1].lower() if "." in n else "") for n in names)
    return ", ".join(f"{k} .{ext}" if ext else f"{k} with no extension" for ext, k in c.most_common(6))


def bruker_scan(sc) -> str:
    """One ParaVision scan's layout: its number, acquisition protocol and data layout (the scan's free-text name,
    which can name the animal, is left out; no pixels)."""
    p = getattr(sc, "params", {}) or {}
    version = p.get("VisuCreatorVersion") or p.get("VisuCreator") or "?"
    groups = ", ".join(f"{n} {g}" for n, g in sc.groups) or "none"
    return (f"E{sc.number} [{sc.protocol or '?'}]: {sc.dim}D {sc.size[0]}x{sc.size[1]}, "
            f"{sc.n_slices} slice(s) x {sc.n_frames} frame(s); frame groups: {groups}; ParaVision {version}"
            + (f"; {sc.note}" if sc.note else ""))
