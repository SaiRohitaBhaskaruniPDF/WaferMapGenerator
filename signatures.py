"""
Needs geometry's dies. Core "brain" of failure patterns. --- "Which dies fail, and why?'
Spatial signature generators — 25 real-world wafer map patterns.

Input die:  (dieX, dieY, cx_mm, cy_mm)
Output die: (dieX, dieY, cx_mm, cy_mm, bin_num)
"""
import numpy as np
import random
from typing import List, Tuple

from geometry import Die, WaferConfig

DieResult = Tuple[int, int, float, float, int]

# ---------------------------------------------------------------------------
# Bin definitions — bin_num → {name, state, color, description}
# ---------------------------------------------------------------------------

""" This maps each bin number --> its meaning(name, pass/fail, color for drawing, description for CSV). 
There are 25 bins defined. 
This is my single source of truth for what each bin means. """
BIN_DEFINITIONS = {
    1:  {"name": "PASS",            "state": "P", "color": "#4CAF50",
         "description": "Die passed all tests"},
    2:  {"name": "EDGE_RING",       "state": "F", "color": "#E74C3C",
         "description": "Edge ring failure — stress near wafer periphery"},
    3:  {"name": "CENTER_FAIL",     "state": "F", "color": "#E67E22",
         "description": "Center cluster failure — chuck or deposition non-uniformity"},
    4:  {"name": "SCRATCH",         "state": "F", "color": "#9B59B6",
         "description": "Linear scratch across the wafer surface"},
    5:  {"name": "RANDOM_FAIL",     "state": "F", "color": "#3498DB",
         "description": "Random scatter failure — particulate contamination or ESD"},
    6:  {"name": "QUADRANT_FAIL",   "state": "F", "color": "#F1C40F",
         "description": "Quadrant failure — process asymmetry or tool issue"},
    7:  {"name": "BULLS_EYE",       "state": "F", "color": "#1ABC9C",
         "description": "Bull's-eye pattern — alternating radial bands"},
    8:  {"name": "DONUT",           "state": "F", "color": "#FF6B6B",
         "description": "Donut / mid-ring failure — intermediate radius ring defect"},
    9:  {"name": "HALF_FAIL",       "state": "F", "color": "#C0392B",
         "description": "Half-wafer failure — lithography or CMP non-uniformity"},
    10: {"name": "CROSS_FAIL",      "state": "F", "color": "#8E44AD",
         "description": "Cross pattern — both horizontal and vertical bands fail"},
    11: {"name": "HOT_SPOT",        "state": "F", "color": "#E91E63",
         "description": "Hot spot / local cluster — isolated defect site"},
    12: {"name": "RETICLE",         "state": "F", "color": "#00BCD4",
         "description": "Reticle-systematic failure — repeating per reticle step"},
    13: {"name": "LOW_YIELD",       "state": "F", "color": "#FF5722",
         "description": "Low overall yield — high random fail rate across wafer"},
    14: {"name": "CORNER_FAIL",     "state": "F", "color": "#607D8B",
         "description": "Corner cluster failure — stress at reticle field corners"},
    15: {"name": "RING_CRACK",      "state": "F", "color": "#795548",
         "description": "Ring crack — mechanical stress ring pattern"},
    16: {"name": "WEDGE_FAIL",      "state": "F", "color": "#FF9800",
         "description": "Wedge / sector failure — pie-slice shaped region"},
    17: {"name": "SYSTEMATIC_GRID", "state": "F", "color": "#4DB6AC",
         "description": "Systematic grid — every Nth row or column fails"},
    18: {"name": "MULTI_CLUSTER",   "state": "F", "color": "#F06292",
         "description": "Multiple hot spots scattered across the wafer"},
    19: {"name": "TOP_EDGE",        "state": "F", "color": "#AED581",
         "description": "Top-edge arc failure — asymmetric edge ring (top)"},
    20: {"name": "BOT_EDGE",        "state": "F", "color": "#FFD54F",
         "description": "Bottom-edge arc failure — asymmetric edge ring (bottom)"},
    21: {"name": "DIAGONAL",        "state": "F", "color": "#BA68C8",
         "description": "Diagonal streak — 45° or 135° scratch or slippage line"},
    22: {"name": "CONCENTRIC",      "state": "F", "color": "#4FC3F7",
         "description": "Concentric ring pair — two concentric bands of failures"},
    23: {"name": "PERIPHERAL_SPOT", "state": "F", "color": "#A5D6A7",
         "description": "Peripheral spot — localized cluster near wafer edge"},
    24: {"name": "RADIAL_SPOKE",    "state": "F", "color": "#FFAB40",
         "description": "Radial spoke — fan-shaped radial defect streaks"},
    25: {"name": "MIXED_MODE",      "state": "F", "color": "#EF9A9A",
         "description": "Mixed-mode — combination of edge ring + center cluster"},
    # ----------------------------------------------------------------------
    # Bins 26-29: "scratch family" bins.
    # A plain scratch (bin 4) just says "there is a linear scratch."
    # These four bins go further and encode the *root cause* (which tool /
    # process made the scratch), because in a real fab the geometry of a
    # scratch is a fingerprint that points back to the equipment that caused it.
    # Each still has state "F" (fail) — the extra meaning lives in the name +
    # description, and in the SCRATCH_FAMILIES table just below.
    # ----------------------------------------------------------------------
    26: {"name": "HANDLER_SCRATCH", "state": "F", "color": "#7E57C2",
         "description": "Robotic-handler scratch — straight line, same angle on every "
                        "wafer in the lot (misaligned end-effector / aligner pin)"},
    27: {"name": "SLOT_SCRATCH",    "state": "F", "color": "#5C6BC0",
         "description": "Cassette / FOUP slot scratch — short mark near the wafer edge, "
                        "repeats on every wafer (worn or damaged carrier slot)"},
    28: {"name": "WAND_SCRATCH",    "state": "F", "color": "#AB47BC",
         "description": "Manual wafer-wand scratch — irregular / squiggly, random per "
                        "wafer, may appear on only some wafers (hand transfer by operator)"},
    29: {"name": "CMP_ARC_SCRATCH", "state": "F", "color": "#26A69A",
         "description": "CMP arc scratch — curved arc following the polish-pad sweep "
                        "radius (slurry agglomerate / pad debris dragged across surface)"},
    # ----------------------------------------------------------------------
    # Bins 30-33: added for the 2026.07.10 spec.
    #   30 = striping (lens-tilt yield loss along one stepping-field edge)
    #   31/32 = dies that passed the previous insertion but failed CP2/CP3
    #   33 = site-correlated loss in a multi-site setup (S2S)
    # ----------------------------------------------------------------------
    30: {"name": "STRIPE_FAIL",     "state": "F", "color": "#FF7043",
         "description": "Striping — yield loss along one edge of every stepping "
                        "field (caused by lens tilt in lithography)"},
    31: {"name": "CP2_FAIL",        "state": "F", "color": "#90A4AE",
         "description": "Failed at CP2 retest (passed CP1) — e.g. temperature-"
                        "sensitive marginal die"},
    32: {"name": "CP3_FAIL",        "state": "F", "color": "#78909C",
         "description": "Failed at CP3 retest (passed CP1 and CP2)"},
    33: {"name": "S2S_FAIL",        "state": "F", "color": "#B0BEC5",
         "description": "Site-to-site yield loss — die killed by a weak site in "
                        "the multi-site probe fixture"},
}

# Bin numbers the pipeline stages use when THEY (not a spatial signature)
# decide a die fails. Kept as named constants so the yield model, CP cascade
# and multi-site stages never hard-code magic numbers.
PASS_BIN = 1
RANDOM_FAIL_BIN = 5    # yield-model random kills reuse the Random Scatter bin
STRIPE_BIN = 30
CP2_FAIL_BIN = 31
CP3_FAIL_BIN = 32
S2S_FAIL_BIN = 33


# ---------------------------------------------------------------------------
# Scratch family metadata table
# ---------------------------------------------------------------------------
# This is a small "knowledge base" that maps each scratch family to:
#   - bin        : which bin number it paints failed dies with (ties back to
#                  BIN_DEFINITIONS above)
#   - tool       : the equipment / process a process engineer would suspect
#   - root_cause : the physical mechanism that creates this scratch shape
#   - lot_repeatable : True  -> the *same* scratch appears on every wafer in the
#                              lot (a fixed equipment fault touches each wafer
#                              identically), so its geometry is derived from a
#                              LOT-level seed.
#                      False -> it varies wafer-to-wafer (or may be absent on
#                              some wafers), so its geometry is derived from a
#                              per-WAFER seed.
# The UI reads this to show tooltips, and llm_agent.py reads it to teach the
# language model which family to pick from a described root cause.
SCRATCH_FAMILIES = {
    "Robotic Handler Scratch": {
        "bin": 26,
        "tool": "Wafer-handling robot / aligner",
        "root_cause": "Misaligned robot end-effector or aligner pin dragging across the wafer",
        "lot_repeatable": True,
    },
    "Cassette Slot Scratch": {
        "bin": 27,
        "tool": "Cassette / FOUP carrier slot",
        "root_cause": "Worn or damaged carrier slot contacting the wafer edge on load/unload",
        "lot_repeatable": True,
    },
    "Wafer-Wand Scratch": {
        "bin": 28,
        "tool": "Manual wafer wand (operator)",
        "root_cause": "Technician hand-transferring a wafer with a wand — irregular, inconsistent",
        "lot_repeatable": False,
    },
    "CMP Arc Scratch": {
        "bin": 29,
        "tool": "CMP (Chemical Mechanical Planarization) polisher",
        "root_cause": "Large particle / slurry agglomerate / pad debris dragged along the pad sweep",
        "lot_repeatable": True,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pass(die: Die) -> DieResult:
    return (*die, 1)

def _fail(die: Die, bin_num: int) -> DieResult:
    return (*die, bin_num)


# ---------------------------------------------------------------------------
# Individual signature functions  
# Each function takes the blank dies and returns them with a bin number attached. They all follow the same recipe.
# ---------------------------------------------------------------------------
""""
The pattern for every signature function:

Create a seeded random generator (seed = reproducibility — same seed always gives same wafer)
Loop over every die
Check a geometric condition (here: "is this die in the outer ring?")
If yes AND a random roll beats fail_rate → mark it failed with a bin number
Otherwise → mark it passed
The only thing that changes between signatures is the geometric condition:

Edge Ring → dist >= outer_threshold
Center Cluster → dist <= small_radius
Scratch → distance_from_a_line <= width
Quadrant → cx >= 0 and cy >= 0
The fail_rate (e.g. 0.88) means "88% of dies in this zone fail, 12% survive" — that randomness is what makes it look real instead of a perfect solid shape.


"""
def assign_edge_ring(dies, radius, edge_exclusion, ring_width=None,
                     fail_rate=0.88, seed=None):
    rng = random.Random(seed)
    active_radius = radius - edge_exclusion
    if ring_width is None:
        ring_width = radius * 0.18
    results = []
    for die in dies:
        _, _, cx, cy = die
        dist = np.sqrt(cx**2 + cy**2)
        if dist >= (active_radius - ring_width) and rng.random() < fail_rate:
            results.append(_fail(die, 2))
        else:
            results.append(_pass(die))
    return results


def assign_center_cluster(dies, cluster_radius=None, fail_rate=0.85, seed=None):
    rng = random.Random(seed)
    results = []
    for die in dies:
        _, _, cx, cy = die
        dist = np.sqrt(cx**2 + cy**2)
        if dist <= (cluster_radius or 20.0) and rng.random() < fail_rate:
            results.append(_fail(die, 3))
        else:
            results.append(_pass(die))
    return results


def assign_scratch(dies, angle_deg=45.0, width_mm=None, fail_rate=0.92, seed=None):
    rng = random.Random(seed)
    if width_mm is None:
        width_mm = 8.0
    angle_rad = np.radians(angle_deg)
    nx, ny = -np.sin(angle_rad), np.cos(angle_rad)
    results = []
    for die in dies:
        _, _, cx, cy = die
        dist_from_line = abs(cx * nx + cy * ny)
        if dist_from_line <= width_mm / 2 and rng.random() < fail_rate:
            results.append(_fail(die, 4))
        else:
            results.append(_pass(die))
    return results


def assign_random_scatter(dies, fail_fraction=0.12, seed=None):
    rng = random.Random(seed)
    results = []
    for die in dies:
        if rng.random() < fail_fraction:
            results.append(_fail(die, 5))
        else:
            results.append(_pass(die))
    return results


def assign_quadrant(dies, quadrant=1, fail_rate=0.85, seed=None):
    rng = random.Random(seed)
    results = []
    for die in dies:
        _, _, cx, cy = die
        in_q = (
            (quadrant == 1 and cx >= 0 and cy >= 0) or
            (quadrant == 2 and cx < 0  and cy >= 0) or
            (quadrant == 3 and cx < 0  and cy < 0)  or
            (quadrant == 4 and cx >= 0 and cy < 0)
        )
        if in_q and rng.random() < fail_rate:
            results.append(_fail(die, 6))
        else:
            results.append(_pass(die))
    return results


def assign_bulls_eye(dies, ring_width=None, fail_rate=0.85, seed=None):
    rng = random.Random(seed)
    results = []
    for die in dies:
        _, _, cx, cy = die
        dist = np.sqrt(cx**2 + cy**2)
        ring_idx = int(dist / (ring_width or 15.0))
        if ring_idx % 2 == 1 and rng.random() < fail_rate:
            results.append(_fail(die, 7))
        else:
            results.append(_pass(die))
    return results


def assign_full_pass(dies):
    return [_pass(die) for die in dies]


def assign_donut(dies, inner_r=None, outer_r=None, fail_rate=0.87, seed=None):
    """Mid-wafer ring — inner and outer radius bounds."""
    rng = random.Random(seed)
    results = []
    for die in dies:
        _, _, cx, cy = die
        dist = np.sqrt(cx**2 + cy**2)
        in_ring = (inner_r or 30.0) <= dist <= (outer_r or 70.0)
        if in_ring and rng.random() < fail_rate:
            results.append(_fail(die, 8))
        else:
            results.append(_pass(die))
    return results


def assign_half_wafer(dies, direction="top", fail_rate=0.87, seed=None):
    """Half-wafer fail: top, bottom, left, or right."""
    rng = random.Random(seed)
    results = []
    for die in dies:
        _, _, cx, cy = die
        in_half = (
            (direction == "top"    and cy >= 0) or
            (direction == "bottom" and cy < 0)  or
            (direction == "left"   and cx < 0)  or
            (direction == "right"  and cx >= 0)
        )
        if in_half and rng.random() < fail_rate:
            results.append(_fail(die, 9))
        else:
            results.append(_pass(die))
    return results


def assign_cross(dies, band_width=None, fail_rate=0.90, seed=None):
    """Horizontal + vertical band cross pattern."""
    rng = random.Random(seed)
    bw = band_width or 15.0
    results = []
    for die in dies:
        _, _, cx, cy = die
        in_cross = abs(cx) <= bw / 2 or abs(cy) <= bw / 2
        if in_cross and rng.random() < fail_rate:
            results.append(_fail(die, 10))
        else:
            results.append(_pass(die))
    return results


def assign_hot_spot(dies, spot_x=None, spot_y=None, spot_r=None,
                    fail_rate=0.92, seed=None):
    """Off-center localized cluster."""
    rng = random.Random(seed)
    results = []
    for die in dies:
        _, _, cx, cy = die
        dist = np.sqrt((cx - (spot_x or 40.0))**2 + (cy - (spot_y or 40.0))**2)
        if dist <= (spot_r or 20.0) and rng.random() < fail_rate:
            results.append(_fail(die, 11))
        else:
            results.append(_pass(die))
    return results


def assign_reticle(dies, dies_per_reticle_x=2, dies_per_reticle_y=2,
                   fail_die_x=0, fail_die_y=0, fail_rate=0.85, seed=None):
    """Repeating reticle-field pattern — same die position fails in every reticle shot."""
    rng = random.Random(seed)
    dpr_x = max(1, int(dies_per_reticle_x))
    dpr_y = max(1, int(dies_per_reticle_y))
    fail_x = int(fail_die_x) % dpr_x
    fail_y = int(fail_die_y) % dpr_y
    results = []
    for die in dies:
        dieX, dieY, cx, cy = die
        local_x = dieX % dpr_x
        local_y = dieY % dpr_y
        if local_x < 0:
            local_x += dpr_x
        if local_y < 0:
            local_y += dpr_y
        if local_x == fail_x and local_y == fail_y and rng.random() < fail_rate:
            results.append(_fail(die, 12))
        else:
            results.append(_pass(die))
    return results


def assign_stripe(dies, dies_per_reticle_x=2, dies_per_reticle_y=2,
                  edge="top", fail_rate=1.0, seed=None):
    """Striping (bin 30): yield loss along ONE edge of every stepping field.

    Real-world cause: "lens tilt" in lithography — one edge of the exposure
    field is slightly out of focus, so the same row/column of dies inside
    EVERY reticle shot underperforms.

    Geometry: compute each die's local position inside its stepping field
    (die index modulo dies-per-reticle). A die is in the stripe when that
    local position sits on the requested field edge:
        top    -> last local row      bottom -> first local row
        left   -> first local column  right  -> last local column

    fail_rate = 1.0 gives a hard stripe (every stripe die fails); below 1.0
    gives a soft-repeater-style stripe (spec: 10%..100% selectable).
    """
    rng = random.Random(seed)
    dpr_x = max(1, int(dies_per_reticle_x))
    dpr_y = max(1, int(dies_per_reticle_y))
    results = []
    for die in dies:
        dieX, dieY, _, _ = die
        # Local (column, row) inside the stepping field, wrapped to 0..dpr-1.
        local_x = dieX % dpr_x
        local_y = dieY % dpr_y
        in_stripe = (
            (edge == "top"    and local_y == dpr_y - 1) or
            (edge == "bottom" and local_y == 0) or
            (edge == "left"   and local_x == 0) or
            (edge == "right"  and local_x == dpr_x - 1)
        )
        if in_stripe and rng.random() < fail_rate:
            results.append(_fail(die, 30))
        else:
            results.append(_pass(die))
    return results


def assign_low_yield(dies, fail_fraction=0.60, seed=None):
    rng = random.Random(seed)
    results = []
    for die in dies:
        if rng.random() < fail_fraction:
            results.append(_fail(die, 13))
        else:
            results.append(_pass(die))
    return results


def assign_corner_cluster(dies, radius, corner_r=None, fail_rate=0.88, seed=None):
    """Fail dies near the four diagonal corners of the reticle grid."""
    rng = random.Random(seed)
    cr = corner_r or radius * 0.15
    corners = [
        (radius * 0.6,  radius * 0.6),
        (-radius * 0.6, radius * 0.6),
        (-radius * 0.6, -radius * 0.6),
        (radius * 0.6,  -radius * 0.6),
    ]
    results = []
    for die in dies:
        _, _, cx, cy = die
        near = any(np.sqrt((cx - px)**2 + (cy - py)**2) <= cr for px, py in corners)
        if near and rng.random() < fail_rate:
            results.append(_fail(die, 14))
        else:
            results.append(_pass(die))
    return results


def assign_ring_crack(dies, radius, crack_radius=None, crack_width=None,
                      fail_rate=0.90, seed=None):
    """Thin ring at a specific radius — simulates ring crack from dicing stress."""
    rng = random.Random(seed)
    cr = crack_radius or radius * 0.55
    cw = crack_width or radius * 0.06
    results = []
    for die in dies:
        _, _, cx, cy = die
        dist = np.sqrt(cx**2 + cy**2)
        if abs(dist - cr) <= cw / 2 and rng.random() < fail_rate:
            results.append(_fail(die, 15))
        else:
            results.append(_pass(die))
    return results


def assign_wedge(dies, angle_start=0.0, angle_span=90.0,
                 fail_rate=0.87, seed=None):
    """Pie-slice wedge sector failure."""
    rng = random.Random(seed)
    a_start = np.radians(angle_start)
    a_end   = np.radians(angle_start + angle_span)
    results = []
    for die in dies:
        _, _, cx, cy = die
        angle = np.arctan2(cy, cx)
        # Normalise to [0, 2π]
        if angle < 0:
            angle += 2 * np.pi
        a_s = a_start % (2 * np.pi)
        a_e = a_end   % (2 * np.pi)
        in_wedge = (a_s <= angle <= a_e) if a_e >= a_s else (angle >= a_s or angle <= a_e)
        if in_wedge and rng.random() < fail_rate:
            results.append(_fail(die, 16))
        else:
            results.append(_pass(die))
    return results


def assign_systematic_grid(dies, stride=3, axis="row", fail_rate=0.85, seed=None):
    """Every Nth row or column fails."""
    rng = random.Random(seed)
    results = []
    for die in dies:
        dieX, dieY, cx, cy = die
        idx = dieY if axis == "row" else dieX
        if (idx % stride == 0) and rng.random() < fail_rate:
            results.append(_fail(die, 17))
        else:
            results.append(_pass(die))
    return results


def assign_multi_cluster(dies, radius, n_clusters=3, cluster_r=None,
                         fail_rate=0.88, seed=None):
    """Several randomly-placed hot spots across the wafer."""
    rng = random.Random(seed)
    cr = cluster_r or radius * 0.12
    # Place cluster centres at reproducible positions inside the active area
    centres = []
    local_rng = random.Random((seed or 0) + 999)
    while len(centres) < n_clusters:
        angle = local_rng.uniform(0, 2 * np.pi)
        r_val = local_rng.uniform(radius * 0.15, radius * 0.75)
        centres.append((r_val * np.cos(angle), r_val * np.sin(angle)))

    results = []
    for die in dies:
        _, _, cx, cy = die
        near = any(np.sqrt((cx - px)**2 + (cy - py)**2) <= cr for px, py in centres)
        if near and rng.random() < fail_rate:
            results.append(_fail(die, 18))
        else:
            results.append(_pass(die))
    return results


def assign_top_edge_arc(dies, radius, edge_exclusion, arc_span=120.0,
                        fail_rate=0.88, seed=None):
    """Arc of failures along the top edge only."""
    rng = random.Random(seed)
    active_r = radius - edge_exclusion
    ring_w   = radius * 0.18
    half_arc = np.radians(arc_span / 2)
    results = []
    for die in dies:
        _, _, cx, cy = die
        dist  = np.sqrt(cx**2 + cy**2)
        angle = np.arctan2(cy, cx)
        in_edge = dist >= (active_r - ring_w)
        in_arc  = abs(angle - np.pi / 2) <= half_arc
        if in_edge and in_arc and rng.random() < fail_rate:
            results.append(_fail(die, 19))
        else:
            results.append(_pass(die))
    return results


def assign_bottom_edge_arc(dies, radius, edge_exclusion, arc_span=120.0,
                           fail_rate=0.88, seed=None):
    """Arc of failures along the bottom edge only."""
    rng = random.Random(seed)
    active_r = radius - edge_exclusion
    ring_w   = radius * 0.18
    half_arc = np.radians(arc_span / 2)
    results = []
    for die in dies:
        _, _, cx, cy = die
        dist  = np.sqrt(cx**2 + cy**2)
        angle = np.arctan2(cy, cx)
        in_edge = dist >= (active_r - ring_w)
        in_arc  = abs(angle + np.pi / 2) <= half_arc
        if in_edge and in_arc and rng.random() < fail_rate:
            results.append(_fail(die, 20))
        else:
            results.append(_pass(die))
    return results


def assign_diagonal(dies, angle_deg=45.0, width_mm=None, fail_rate=0.92, seed=None):
    """45° or 135° diagonal scratch / slip-line."""
    rng = random.Random(seed)
    width_mm = width_mm or 6.0
    angle_rad = np.radians(angle_deg)
    nx, ny = -np.sin(angle_rad), np.cos(angle_rad)
    results = []
    for die in dies:
        _, _, cx, cy = die
        if abs(cx * nx + cy * ny) <= width_mm / 2 and rng.random() < fail_rate:
            results.append(_fail(die, 21))
        else:
            results.append(_pass(die))
    return results


def assign_concentric_rings(dies, radius, r1=None, r2=None, width=None,
                            fail_rate=0.87, seed=None):
    """Two concentric ring bands of failures."""
    rng = random.Random(seed)
    r1 = r1 or radius * 0.25
    r2 = r2 or radius * 0.65
    w  = width or radius * 0.07
    results = []
    for die in dies:
        _, _, cx, cy = die
        dist = np.sqrt(cx**2 + cy**2)
        in_ring = abs(dist - r1) <= w / 2 or abs(dist - r2) <= w / 2
        if in_ring and rng.random() < fail_rate:
            results.append(_fail(die, 22))
        else:
            results.append(_pass(die))
    return results


def assign_peripheral_spot(dies, radius, edge_exclusion, spot_angle=0.0,
                           spot_r=None, fail_rate=0.90, seed=None):
    """Localised cluster near a specific point on the wafer edge."""
    rng = random.Random(seed)
    active_r  = radius - edge_exclusion
    sr = spot_r or radius * 0.14
    angle_rad = np.radians(spot_angle)
    spot_cx   = (active_r - sr) * np.cos(angle_rad)
    spot_cy   = (active_r - sr) * np.sin(angle_rad)
    results = []
    for die in dies:
        _, _, cx, cy = die
        dist = np.sqrt((cx - spot_cx)**2 + (cy - spot_cy)**2)
        if dist <= sr and rng.random() < fail_rate:
            results.append(_fail(die, 23))
        else:
            results.append(_pass(die))
    return results


def assign_radial_spokes(dies, n_spokes=4, spoke_width=None, fail_rate=0.88, seed=None):
    """Fan of radial spokes radiating from the center."""
    rng = random.Random(seed)
    spoke_angle = 2 * np.pi / n_spokes
    half_w = np.radians((spoke_width or 15.0) / 2)
    results = []
    for die in dies:
        _, _, cx, cy = die
        angle = np.arctan2(cy, cx) % (2 * np.pi)
        # Check proximity to any spoke
        in_spoke = any(
            abs((angle - i * spoke_angle) % (2 * np.pi)) <= half_w or
            abs((angle - i * spoke_angle) % (2 * np.pi) - 2 * np.pi) <= half_w
            for i in range(n_spokes)
        )
        if in_spoke and rng.random() < fail_rate:
            results.append(_fail(die, 24))
        else:
            results.append(_pass(die))
    return results


def assign_mixed_mode(dies, radius, edge_exclusion, fail_rate=0.85, seed=None):
    """Edge ring + center cluster combo."""
    rng = random.Random(seed)
    active_r   = radius - edge_exclusion
    ring_w     = radius * 0.15
    center_r   = radius * 0.20
    results = []
    for die in dies:
        _, _, cx, cy = die
        dist = np.sqrt(cx**2 + cy**2)
        in_edge   = dist >= (active_r - ring_w)
        in_center = dist <= center_r
        if (in_edge or in_center) and rng.random() < fail_rate:
            results.append(_fail(die, 25))
        else:
            results.append(_pass(die))
    return results


# ---------------------------------------------------------------------------
# Scratch family generators (bins 26-29)
# ---------------------------------------------------------------------------
# All four share the same core idea as assign_scratch: for every die, measure a
# geometric distance to some shape (a line, or an arc), and if the die is close
# enough to that shape AND a random roll beats fail_rate, mark it failed.
# What differs between them is (a) the *shape* and (b) *where the randomness
# comes from* (lot-level vs per-wafer), which is what makes each look like a
# different real-world tool fault.


def assign_handler_scratch(dies, angle_deg=45.0, offset_mm=0.0, width_mm=None,
                           fail_rate=0.90, seed=None):
    """Robotic-handler scratch (bin 26): one straight line across the wafer.

    Real-world cause: a misaligned wafer-handling robot / aligner pin. Because
    the SAME mechanical fault touches every wafer the same way, the caller is
    expected to pass a FIXED angle_deg + offset_mm for the whole lot (derived
    from a lot-level seed), so the scratch looks identical on every wafer.

    Geometry: a straight line is defined by its normal direction (nx, ny).
    The signed distance from a point (cx, cy) to a line through the origin is
    (cx*nx + cy*ny). We shift the line sideways by `offset_mm` so it need not
    pass exactly through the wafer center. A die fails if it lies within
    width_mm/2 of that line.

    Parameters
    ----------
    angle_deg : orientation of the scratch line (degrees). LOT-level (fixed).
    offset_mm : how far the line is shifted from the wafer center, along the
                line's normal. LOT-level (fixed). 0 = passes through center.
    width_mm  : thickness of the scratch band (dies within +/- width/2 fail).
    fail_rate : fraction of in-band dies that actually fail (adds realism —
                a scratch rarely kills 100% of the dies it crosses).
    seed      : per-WAFER seed, used ONLY for the fail_rate coin-flips so the
                speckle differs slightly wafer-to-wafer while the line stays put.
    """
    rng = random.Random(seed)
    if width_mm is None:
        width_mm = 6.0
    angle_rad = np.radians(angle_deg)
    # Normal vector of the line (perpendicular to the scratch direction).
    nx, ny = -np.sin(angle_rad), np.cos(angle_rad)
    results = []
    for die in dies:
        _, _, cx, cy = die
        # Signed distance to the line, then remove the lot-level offset.
        dist_from_line = abs(cx * nx + cy * ny - offset_mm)
        if dist_from_line <= width_mm / 2 and rng.random() < fail_rate:
            results.append(_fail(die, 26))
        else:
            results.append(_pass(die))
    return results


def assign_slot_scratch(dies, radius, edge_exclusion, angle_deg=0.0,
                        depth_mm=None, arc_span_deg=45.0, width_mm=None,
                        fail_rate=0.90, seed=None):
    """Cassette / FOUP slot scratch (bin 27): a SHORT mark near the wafer edge.

    Real-world cause: a worn/damaged carrier slot rubbing the wafer edge as it
    slides in or out. It is localized to the edge and to one side of the wafer,
    and (like the handler scratch) repeats on every wafer, so angle_deg is a
    LOT-level parameter.

    Geometry: instead of a full-diameter line, we fail dies that are BOTH
      (a) near the edge  -> distance from center between (active_r - depth) and active_r
      (b) near one side  -> polar angle within +/- arc_span/2 of angle_deg
    That combination carves out a short scratch hugging the rim.

    Parameters
    ----------
    radius, edge_exclusion : wafer geometry (to locate the usable edge).
    angle_deg    : which side of the wafer the slot mark sits on. LOT-level.
    depth_mm     : how far inward from the edge the mark reaches.
    arc_span_deg : angular length of the mark (bigger = longer scratch).
    width_mm     : unused placeholder kept for signature symmetry / future use.
    fail_rate    : fraction of in-region dies that fail.
    seed         : per-WAFER seed for the fail_rate coin-flips only.
    """
    rng = random.Random(seed)
    active_r = radius - edge_exclusion
    if depth_mm is None:
        depth_mm = radius * 0.18
    center_angle = np.radians(angle_deg)
    half_arc = np.radians(arc_span_deg / 2)
    results = []
    for die in dies:
        _, _, cx, cy = die
        dist = np.sqrt(cx**2 + cy**2)
        # Polar angle of this die, wrapped to [0, 2*pi).
        ang = np.arctan2(cy, cx) % (2 * np.pi)
        # Smallest angular gap between the die and the mark's center angle.
        d_ang = abs((ang - (center_angle % (2 * np.pi)) + np.pi) % (2 * np.pi) - np.pi)
        near_edge = dist >= (active_r - depth_mm)
        near_side = d_ang <= half_arc
        if near_edge and near_side and rng.random() < fail_rate:
            results.append(_fail(die, 27))
        else:
            results.append(_pass(die))
    return results


def assign_wand_scratch(dies, radius, width_mm=None, presence_prob=0.6,
                        fail_rate=0.88, seed=None):
    """Manual wafer-wand scratch (bin 28): irregular, squiggly, per-wafer.

    Real-world cause: an operator hand-transferring a wafer with a wand. Because
    it is done by hand, it is inconsistent: the angle, position, curviness and
    even whether it appears at all change from wafer to wafer. So EVERYTHING
    here is driven by the per-WAFER seed.

    Geometry: we start from a straight line (like a scratch) but add a sine-wave
    "wiggle" to it so it looks hand-drawn rather than machine-straight. We work
    in a rotated coordinate frame:
        along = distance measured ALONG the scratch direction
        perp  = distance measured PERPENDICULAR to it
    A die fails if its perpendicular distance to the wiggling center line is
    within width/2, i.e. abs(perp - wiggle(along)) <= width/2.

    Parameters
    ----------
    radius        : wafer radius (used to scale wiggle amplitude / frequency).
    width_mm      : thickness of the scratch band.
    presence_prob : probability THIS wafer has a wand scratch at all. With the
                    remaining probability the whole wafer comes back all-pass,
                    which is why wand scratches show up on only some wafers.
    fail_rate     : fraction of in-band dies that fail.
    seed          : per-WAFER seed controlling angle, position, wiggle AND
                    presence — all of it varies wafer-to-wafer.
    """
    rng = random.Random(seed)

    # Roll for presence first: some wafers simply have no wand scratch.
    if rng.random() > presence_prob:
        return [_pass(die) for die in dies]

    if width_mm is None:
        width_mm = 6.0

    # Random orientation and sideways offset for this wafer's scratch.
    angle_rad = np.radians(rng.uniform(0.0, 180.0))
    offset_mm = rng.uniform(-radius * 0.4, radius * 0.4)

    # Direction unit vector (along the scratch) and its normal (perpendicular).
    tx, ty = np.cos(angle_rad), np.sin(angle_rad)   # along
    nx, ny = -np.sin(angle_rad), np.cos(angle_rad)  # perpendicular

    # Random "hand-drawn" wiggle: amplitude in mm, spatial frequency in 1/mm.
    amp = rng.uniform(radius * 0.05, radius * 0.18)
    freq = rng.uniform(1.5, 4.0) / radius
    phase = rng.uniform(0.0, 2 * np.pi)

    results = []
    for die in dies:
        _, _, cx, cy = die
        along = cx * tx + cy * ty            # position along the scratch
        perp = cx * nx + cy * ny - offset_mm  # distance from the base line
        wiggle = amp * np.sin(freq * along + phase)  # the squiggle
        if abs(perp - wiggle) <= width_mm / 2 and rng.random() < fail_rate:
            results.append(_fail(die, 28))
        else:
            results.append(_pass(die))
    return results


def assign_cmp_arc_scratch(dies, radius, arc_radius=None, arc_width=None,
                           center_offset=None, angle_start=None,
                           arc_span_deg=140.0, fail_rate=0.90, seed=None):
    """CMP arc scratch (bin 29): a CURVED arc, not a straight line.

    Real-world cause: during Chemical Mechanical Planarization a large particle
    (slurry agglomerate / pad debris) gets dragged across the wafer. Because the
    pad and conditioner sweep in circles, the resulting scratch is a CURVE that
    follows the pad-sweep radius rather than a straight diameter line. It tends
    to recur at a similar radius across the lot (tool kinematics), so arc_radius
    and the arc center are LOT-level parameters.

    Geometry: define a circle of radius `arc_radius` centered at a point that is
    OFFSET from the wafer center (the pad's rotation center is not the wafer
    center). A die fails if its distance to that circle is within arc_width/2
    (so it hugs the circular band) AND it falls within the arc's angular span
    (an arc is only PART of a full circle).

    Parameters
    ----------
    radius        : wafer radius (used for sensible defaults).
    arc_radius    : radius of the pad-sweep circle the scratch follows. LOT-level.
    arc_width     : thickness of the arc band.
    center_offset : (dx, dy) offset of the arc's center from the wafer center.
                    LOT-level. Defaults to a point off to one side.
    angle_start   : starting polar angle of the visible arc (radians), measured
                    around the ARC's center. LOT-level.
    arc_span_deg  : angular length of the arc in degrees (how much of the circle
                    is actually scratched).
    fail_rate     : fraction of in-band dies that fail.
    seed          : per-WAFER seed for the fail_rate coin-flips only.
    """
    rng = random.Random(seed)
    if arc_radius is None:
        arc_radius = radius * 0.9
    if arc_width is None:
        arc_width = radius * 0.10
    if center_offset is None:
        # Push the arc center off to the side so the visible arc curves through
        # the wafer instead of being a concentric ring around the middle.
        center_offset = (radius * 0.7, 0.0)
    if angle_start is None:
        angle_start = np.pi * 0.6

    ox, oy = center_offset
    a_start = angle_start % (2 * np.pi)
    a_end = (angle_start + np.radians(arc_span_deg)) % (2 * np.pi)

    results = []
    for die in dies:
        _, _, cx, cy = die
        # Position of the die relative to the ARC's center (not wafer center).
        dx, dy = cx - ox, cy - oy
        r = np.sqrt(dx**2 + dy**2)
        ang = np.arctan2(dy, dx) % (2 * np.pi)
        # Close to the circular band of radius arc_radius?
        on_band = abs(r - arc_radius) <= arc_width / 2
        # Within the visible angular span? (handle the wrap-around case)
        if a_end >= a_start:
            in_span = a_start <= ang <= a_end
        else:
            in_span = ang >= a_start or ang <= a_end
        if on_band and in_span and rng.random() < fail_rate:
            results.append(_fail(die, 29))
        else:
            results.append(_pass(die))
    return results


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# "No pattern" — no spatial signature at all. The generator applies zero
# signature fails; yield comes purely from the yield model (requested yield /
# defect density, or the 93-97% per-wafer baseline when neither was given).
# Distinct from "Full Pass", which is the deliberate 100%-yield wafer.
NO_PATTERN_SIGNATURE = "None (Yield Model Only)"

SIGNATURE_NAMES = [
    "Edge Ring",
    "Center Cluster",
    "Scratch / Streak",
    "Random Scatter",
    "Quadrant Failure",
    "Bull's-Eye",
    "Full Pass",
    NO_PATTERN_SIGNATURE,
    "Donut (Mid-Ring)",
    "Half Wafer — Top",
    "Half Wafer — Bottom",
    "Half Wafer — Left",
    "Half Wafer — Right",
    "Cross Pattern",
    "Hot Spot",
    "Reticle Pattern",
    "Low Yield",
    "Corner Clusters",
    "Ring Crack",
    "Wedge / Sector",
    "Systematic Grid — Row",
    "Systematic Grid — Column",
    "Multi-Cluster",
    "Top Edge Arc",
    "Bottom Edge Arc",
    "Diagonal Scratch",
    "Concentric Rings",
    "Peripheral Spot",
    "Radial Spokes",
    "Mixed Mode (Edge + Center)",
    # Striping (bin 30) — lens-tilt yield loss along one edge of every
    # stepping field. Four variants, one per field edge (per spec).
    "Striping — Top",
    "Striping — Bottom",
    "Striping — Left",
    "Striping — Right",
    # Scratch families (bins 26-29) — each is a scratch whose SHAPE points to
    # the specific tool/process that caused it. Names here must exactly match
    # the keys of SCRATCH_FAMILIES above so the two tables stay in sync.
    "Robotic Handler Scratch",
    "Cassette Slot Scratch",
    "Wafer-Wand Scratch",
    "CMP Arc Scratch",
]

"""
apply_signature() the dispatcher 
This is a big if/elif that takes a signature name (string) and calls the right function. 
It's the "switchboard" — the app just says apply_signature(dies, "Edge Ring", config) and doesn't need to know which function handles it.
There's also SIGNATURE_NAMES — a list of all 29 pattern names, used to populate the dropdown in the UI.

"""
def apply_signature(dies: List[Die], signature_type: str,
                    config: WaferConfig, seed: int = None,
                    lot_seed: int = None) -> List[DieResult]:
    """Apply a named spatial signature and return die results.

    Parameters
    ----------
    seed : the PER-WAFER seed. Controls wafer-to-wafer variation (the random
           fail_rate coin-flips, and per-wafer patterns like the wand scratch).
    lot_seed : the LOT-level seed, shared by every wafer in the lot. Used to
           derive the geometry of "lot-repeatable" defects (handler scratch
           angle, slot-scratch side, CMP arc radius) so those look IDENTICAL on
           every wafer. If not given, we fall back to `seed` so old callers that
           only pass a per-wafer seed keep working exactly as before.
    """
    radius = config.diameter / 2.0
    ee     = config.edge_exclusion

    # Backward compatibility: if the caller didn't split lot vs wafer seeds,
    # treat the wafer seed as the lot seed too.
    if lot_seed is None:
        lot_seed = seed

    # A dedicated RNG seeded from the LOT seed. Every wafer in the lot builds
    # this the same way, so any geometry drawn from `lot_rng` is stable across
    # the lot (that is what makes an equipment fault "repeat" wafer-to-wafer).
    lot_rng = random.Random(lot_seed)

    s = signature_type
    if s == "Edge Ring":
        return assign_edge_ring(dies, radius, ee, ring_width=radius * 0.18, seed=seed)
    elif s == "Center Cluster":
        return assign_center_cluster(dies, cluster_radius=radius * 0.22, seed=seed)
    elif s == "Scratch / Streak":
        angle = 30.0 + ((seed or 0) % 5) * 18.0
        return assign_scratch(dies, angle_deg=angle,
                              width_mm=max(config.die_width, config.die_height) * 1.2, seed=seed)
    elif s == "Random Scatter":
        return assign_random_scatter(dies, fail_fraction=0.12, seed=seed)
    elif s == "Quadrant Failure":
        quadrant = ((seed or 0) % 4) + 1
        return assign_quadrant(dies, quadrant=quadrant, seed=seed)
    elif s == "Bull's-Eye":
        return assign_bulls_eye(dies, ring_width=radius * 0.15, seed=seed)
    elif s == "Full Pass":
        return assign_full_pass(dies)
    elif s == NO_PATTERN_SIGNATURE:
        # No spatial fails here — the yield model decides which dies die
        # (see generator.py: baseline 93-97% per wafer when no yield given).
        return assign_full_pass(dies)
    elif s == "Donut (Mid-Ring)":
        return assign_donut(dies, inner_r=radius * 0.25, outer_r=radius * 0.60, seed=seed)
    elif s == "Half Wafer — Top":
        return assign_half_wafer(dies, direction="top", seed=seed)
    elif s == "Half Wafer — Bottom":
        return assign_half_wafer(dies, direction="bottom", seed=seed)
    elif s == "Half Wafer — Left":
        return assign_half_wafer(dies, direction="left", seed=seed)
    elif s == "Half Wafer — Right":
        return assign_half_wafer(dies, direction="right", seed=seed)
    elif s == "Cross Pattern":
        return assign_cross(dies, band_width=max(config.die_width, config.die_height) * 1.5, seed=seed)
    elif s == "Hot Spot":
        angle = np.radians(((seed or 0) * 73) % 360)
        r_val = radius * 0.45
        return assign_hot_spot(dies,
                               spot_x=r_val * np.cos(angle),
                               spot_y=r_val * np.sin(angle),
                               spot_r=radius * 0.14, seed=seed)
    elif s == "Reticle Pattern":
        # Repeaters: fail rate 1.0 = classic hard repeater (the same reticle
        # position fails on every shot). Below 1.0 = "soft repeater" — the
        # position fails only part of the time (spec: 10%..100% selectable).
        return assign_reticle(
            dies,
            dies_per_reticle_x=config.dies_per_reticle_x,
            dies_per_reticle_y=config.dies_per_reticle_y,
            fail_die_x=config.reticle_fail_die_x,
            fail_die_y=config.reticle_fail_die_y,
            fail_rate=getattr(config, "repeater_fail_rate", 1.0),
            seed=seed,
        )
    elif s in ("Striping — Top", "Striping — Bottom",
               "Striping — Left", "Striping — Right"):
        edge = s.split("—")[1].strip().lower()
        return assign_stripe(
            dies,
            dies_per_reticle_x=config.dies_per_reticle_x,
            dies_per_reticle_y=config.dies_per_reticle_y,
            edge=edge,
            fail_rate=getattr(config, "stripe_fail_rate", 1.0),
            seed=seed,
        )
    elif s == "Low Yield":
        return assign_low_yield(dies, fail_fraction=0.55, seed=seed)
    elif s == "Corner Clusters":
        return assign_corner_cluster(dies, radius, corner_r=radius * 0.15, seed=seed)
    elif s == "Ring Crack":
        return assign_ring_crack(dies, radius, crack_radius=radius * 0.55,
                                 crack_width=radius * 0.08, seed=seed)
    elif s == "Wedge / Sector":
        start = ((seed or 0) * 37) % 360
        return assign_wedge(dies, angle_start=start, angle_span=90.0, seed=seed)
    elif s == "Systematic Grid — Row":
        return assign_systematic_grid(dies, stride=3, axis="row", seed=seed)
    elif s == "Systematic Grid — Column":
        return assign_systematic_grid(dies, stride=3, axis="col", seed=seed)
    elif s == "Multi-Cluster":
        n = 3 + ((seed or 0) % 3)
        return assign_multi_cluster(dies, radius, n_clusters=n,
                                    cluster_r=radius * 0.10, seed=seed)
    elif s == "Top Edge Arc":
        return assign_top_edge_arc(dies, radius, ee, arc_span=130.0, seed=seed)
    elif s == "Bottom Edge Arc":
        return assign_bottom_edge_arc(dies, radius, ee, arc_span=130.0, seed=seed)
    elif s == "Diagonal Scratch":
        angle = 135.0 + ((seed or 0) % 3) * 15.0
        return assign_diagonal(dies, angle_deg=angle,
                               width_mm=max(config.die_width, config.die_height), seed=seed)
    elif s == "Concentric Rings":
        return assign_concentric_rings(dies, radius, r1=radius * 0.25,
                                       r2=radius * 0.65, width=radius * 0.08, seed=seed)
    elif s == "Peripheral Spot":
        spot_angle = ((seed or 0) * 47) % 360
        return assign_peripheral_spot(dies, radius, ee,
                                      spot_angle=spot_angle, spot_r=radius * 0.14, seed=seed)
    elif s == "Radial Spokes":
        return assign_radial_spokes(dies, n_spokes=4, spoke_width=20.0, seed=seed)
    elif s == "Mixed Mode (Edge + Center)":
        return assign_mixed_mode(dies, radius, ee, seed=seed)

    # ----- Scratch families (bins 26-29) --------------------------------
    # For the three "lot_repeatable" families we draw the geometry from
    # `lot_rng` (fixed across the lot) but pass the per-wafer `seed` for the
    # fail_rate speckle. For the wand scratch, everything comes from `seed`.
    elif s == "Robotic Handler Scratch":
        # One fixed angle + sideways offset for the whole lot.
        angle = lot_rng.uniform(0.0, 180.0)
        offset = lot_rng.uniform(-radius * 0.3, radius * 0.3)
        return assign_handler_scratch(
            dies, angle_deg=angle, offset_mm=offset,
            width_mm=max(config.die_width, config.die_height) * 1.2, seed=seed)
    elif s == "Cassette Slot Scratch":
        # One fixed side (angle) of the wafer for the whole lot.
        angle = lot_rng.uniform(0.0, 360.0)
        return assign_slot_scratch(
            dies, radius, ee, angle_deg=angle,
            depth_mm=radius * 0.18, arc_span_deg=45.0, seed=seed)
    elif s == "Wafer-Wand Scratch":
        # Fully per-wafer: angle, position, wiggle and presence all use `seed`.
        return assign_wand_scratch(
            dies, radius,
            width_mm=max(config.die_width, config.die_height) * 1.1, seed=seed)
    elif s == "CMP Arc Scratch":
        # Fixed pad-sweep radius + arc center + start angle for the whole lot.
        arc_radius = radius * lot_rng.uniform(0.7, 1.0)
        center_off = radius * lot_rng.uniform(0.5, 0.9)
        start_ang = lot_rng.uniform(0.0, 2 * np.pi)
        return assign_cmp_arc_scratch(
            dies, radius, arc_radius=arc_radius, arc_width=radius * 0.10,
            center_offset=(center_off, 0.0), angle_start=start_ang,
            arc_span_deg=140.0, seed=seed)

    else:
        return assign_random_scatter(dies, seed=seed)


# ---------------------------------------------------------------------------
# Multi-signature compositor
# ---------------------------------------------------------------------------

def compose_signatures(dies: List[Die], signature_types: List[str],
                       config: WaferConfig, seed: int = None,
                       lot_seed: int = None) -> List[DieResult]:
    """Overlay several signatures on ONE wafer and merge them per die.

    Why this exists
    ---------------
    Real wafers often show more than one failure signature at once (e.g. an
    edge ring from a process problem AND a scratch from a handling problem),
    because the root causes are unrelated. This function lets us stack any
    number of the existing signatures on a single wafer.

    How it works
    ------------
    1. Run each requested signature independently via apply_signature(). Each
       returns a full list of per-die results in the SAME die order.
    2. Walk the dies position-by-position and decide a single final bin:
         - If NO signature failed a die, it stays PASS.
         - If one or more signatures failed it, the EARLIER signature in the
           list wins (it has higher priority). So order matters: put the most
           important / most visually dominant signature first.

    This needs zero changes to the individual assign_* functions — it purely
    combines their outputs, so every one of the existing signatures (including
    the new scratch families) can be layered for free.

    Parameters
    ----------
    signature_types : ordered list of signature names. Index 0 = highest
                      priority (wins ties on a shared die).
    seed, lot_seed  : forwarded to each apply_signature call (see that function).
    """
    # Degenerate cases: nothing selected -> all pass; one selected -> just it.
    if not signature_types:
        return [_pass(die) for die in dies]
    if len(signature_types) == 1:
        return apply_signature(dies, signature_types[0], config,
                               seed=seed, lot_seed=lot_seed)

    # Step 1: generate every layer. We nudge each layer's per-wafer seed by its
    # index so two layers don't produce identical random speckle, while staying
    # fully reproducible for a given (seed, lot_seed).
    layers = []
    for i, name in enumerate(signature_types):
        layer_seed = None if seed is None else seed + i * 101
        layers.append(apply_signature(dies, name, config,
                                      seed=layer_seed, lot_seed=lot_seed))

    # Step 2: merge per die. Because every layer preserves die order, we can
    # simply zip them together index-by-index.
    merged: List[DieResult] = []
    for die_idx, die in enumerate(dies):
        final_bin = 1  # assume PASS until some layer says otherwise
        for layer in layers:
            bin_num = layer[die_idx][4]  # the 5th field is the bin number
            if bin_num != 1:  # this layer failed the die
                final_bin = bin_num
                break  # earliest (highest-priority) failing layer wins
        merged.append((*die, final_bin))
    return merged
