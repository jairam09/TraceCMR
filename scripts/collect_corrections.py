"""Gather training pairs from TraceCMR results folders: for every image the model contoured, the model's
contour and the user's corrected contour (what fine-tuning learns from).

Only images the user actually reviewed count. By default an image is used when the user's contour differs
from the model's (a correction) or the mouse was saved after the user marked it reviewed; with --all,
every auto-contoured image with a contour is used (unchanged ones count as "the model was right").

Example:
    python scripts/collect_corrections.py "C:/Users/me/Documents/TraceCMR results" corrections.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--all", action="store_true", help="also keep images the user left unchanged")
    args = ap.parse_args()
    imgs, model, user, ids, changed = [], [], [], [], []
    for f in sorted(list(args.results.rglob("*.tracecmr.npz")) + list(args.results.rglob("*.ventrimap.npz"))):
        with np.load(f, allow_pickle=False) as d:
            if "model_images" not in d.files:
                continue
            meta = json.loads(str(d["meta"]))
            lab, ml, zt, im = d["label"], d["model_label"], d["model_zt"], d["model_images"]
        n_used = 0
        for (z, t), img in zip(zt, im):
            u, m = lab[:, :, z, t], ml[:, :, z, t]
            diff = int((u != m).sum())
            if not u.any() or (diff == 0 and not args.all):
                continue
            imgs.append(img)
            model.append(m)
            user.append(u)
            ids.append(f"{meta.get('subject_id', f.stem)}|z{z}|t{t}")
            changed.append(diff)
            n_used += 1
        print(f"{f.parent.name}: {n_used} image(s)")
    if not imgs:
        print("No corrected images found yet.")
        return
    arrays = {}  # one entry per image: scans differ in size
    for i, (a, m, u) in enumerate(zip(imgs, model, user)):
        arrays[f"image_{i}"], arrays[f"model_{i}"], arrays[f"user_{i}"] = a, m, u
    np.savez_compressed(args.out, ids=np.array(ids), pixels_changed=np.array(changed), **arrays)
    print(f"{len(imgs)} images -> {args.out} ({sum(c > 0 for c in changed)} corrected)")


if __name__ == "__main__":
    main()
