"""A long-lived process that runs the contouring models for the app.

The app never loads TensorFlow or PyTorch itself: a model problem (a crash in a native library, a
TensorFlow/PyTorch clash, running out of memory) then ends this worker, not the app, and the app reports
it and starts a fresh worker next time. Each model stays loaded between clicks.

Protocol (one line per request on stdin, one reply line on stdout):
    load <model>                       -> @@ok
    run <model> <in.npz> <out.npz>     -> @@ok   (in.npz: vol, spacing; out.npz: lab, and unc for UCL)
    quit
Errors reply "@@err <message>".
"""

import os
import sys
import traceback

import numpy as np


def main() -> None:
    os.environ["TRACECMR_INPROCESS"] = "1"
    from . import autocontour as ac

    out = sys.stdout
    sys.stdout = sys.stderr  # anything the libraries print goes to the log, not into the reply channel
    for line in sys.stdin:
        parts = line.strip().split(" ", 3)
        if not parts or not parts[0]:
            continue
        cmd = parts[0]
        try:
            if cmd == "quit":
                break
            if cmd == "load":
                ac._load_ucl() if parts[1] == "ucl" else ac._load_calico()
            elif cmd == "run":
                model, inp, dst = parts[1], parts[2], parts[3]
                with np.load(inp) as d:  # closed straight away: Windows cannot delete a file still open
                    vol, spacing = d["vol"].copy(), tuple(float(v) for v in d["spacing"])
                with open(dst, "wb") as f:
                    if model == "ucl":  # labels plus each slice's "model unsure" score (quality.py)
                        lab, unc = ac._ucl_volume_unsure(vol, spacing)
                        np.savez(f, lab=lab, unc=unc)
                    else:
                        np.savez(f, lab=ac._calico_volume(vol, spacing))
            else:
                raise ValueError(f"unknown command {cmd!r}")
            out.write("@@ok\n")
        except Exception as e:  # report and keep serving
            traceback.print_exc()
            out.write(f"@@err {type(e).__name__}: {e}".replace("\n", " ") + "\n")
        out.flush()


if __name__ == "__main__":
    main()
