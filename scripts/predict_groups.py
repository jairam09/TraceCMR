"""Blind group prediction for the UCL cohort from TraceCMR measurements alone.

The rules below were written and committed BEFORE the 130-mouse results were
inspected (2026-10-07) and before the true group labels were available. Do not
change them after seeing the labels; report how they score as they are. Any
improved rules must be reported separately as post-hoc.

Rules (applied in this order; first match wins):
  MI      regional dysfunction: at least one mid/apical segment thickens < 5%
          while the mid/apical median is >= 35 percentage points higher
          (thick_min_seg_pct < 5 and thick_spread_pct >= 35)
  DCM     global dysfunction with dilation: EF < 45% and EDV >= 80 uL
  TAC     hypertrophy with preserved EF: EF >= 45% and
          (LV mass at ED >= 130 mg or mean ED wall thickness >= 1.15 mm)
  Control everything else

Physiological basis: an infarct impairs one territory, not the whole wall; the
transgenic DCM model dilates the LV and lowers EF globally; pressure overload
(TAC) thickens the wall and raises mass, with EF preserved until late failure.
Expected group sizes (paper): control 39, MI 11, DCM 51, TAC 29.

Example:
    python scripts/predict_groups.py results/cohort.csv results/predictions.csv
"""

from __future__ import annotations

import csv
import math
import sys


def f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return math.nan


def predict(row: dict) -> str:
    if f(row, "thick_min_seg_pct") < 5 and f(row, "thick_spread_pct") >= 35:
        return "MI"
    if f(row, "ef_pct") < 45 and f(row, "edv_ul") >= 80:
        return "DCM"
    if f(row, "ef_pct") >= 45 and (f(row, "mass_ed_mg") >= 130 or f(row, "edwt_mean_mm") >= 1.15):
        return "TAC"
    return "Control"


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    rows = list(csv.DictReader(open(src)))
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mouse", "predicted_group"])
        for r in rows:
            w.writerow([r["mouse"], predict(r)])
    counts: dict[str, int] = {}
    for r in rows:
        counts[predict(r)] = counts.get(predict(r), 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
