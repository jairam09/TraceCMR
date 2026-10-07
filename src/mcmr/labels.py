"""Standard label numbers used everywhere inside this package.

Every dataset uses its own label convention. Loaders translate each dataset's
numbers into these, so the rest of the code only ever sees one scheme.
"""

BACKGROUND = 0
LV_BLOODPOOL = 1
LV_MYOCARDIUM = 2
RV_BLOODPOOL = 3
SCAR = 4
MVO = 5  # microvascular obstruction (no-reflow); counts as infarct

NAMES = {
    BACKGROUND: "background",
    LV_BLOODPOOL: "LV blood pool",
    LV_MYOCARDIUM: "LV myocardium",
    RV_BLOODPOOL: "RV blood pool",
    SCAR: "scar",
    MVO: "MVO",
}

# Dataset conventions -> standard labels.
# ACDC: 0 background, 1 RV cavity, 2 myocardium, 3 LV cavity.
ACDC = {0: BACKGROUND, 1: RV_BLOODPOOL, 2: LV_MYOCARDIUM, 3: LV_BLOODPOOL}

# Already in the standard scheme (e.g. synthetic phantoms, our own exports).
IDENTITY = {k: k for k in NAMES}

LABEL_MAPS = {"acdc": ACDC, "standard": IDENTITY}
