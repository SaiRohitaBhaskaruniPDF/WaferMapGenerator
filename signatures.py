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
# Dispatcher
# ---------------------------------------------------------------------------

SIGNATURE_NAMES = [
    "Edge Ring",
    "Center Cluster",
    "Scratch / Streak",
    "Random Scatter",
    "Quadrant Failure",
    "Bull's-Eye",
    "Full Pass",
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
]

"""
apply_signature() the dispatcher 
This is a big if/elif that takes a signature name (string) and calls the right function. 
It's the "switchboard" — the app just says apply_signature(dies, "Edge Ring", config) and doesn't need to know which function handles it.
There's also SIGNATURE_NAMES — a list of all 29 pattern names, used to populate the dropdown in the UI.

"""
def apply_signature(dies: List[Die], signature_type: str,
                    config: WaferConfig, seed: int = None) -> List[DieResult]:
    """Apply a named spatial signature and return die results."""
    radius = config.diameter / 2.0
    ee     = config.edge_exclusion

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
        return assign_reticle(
            dies,
            dies_per_reticle_x=config.dies_per_reticle_x,
            dies_per_reticle_y=config.dies_per_reticle_y,
            fail_die_x=config.reticle_fail_die_x,
            fail_die_y=config.reticle_fail_die_y,
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
    else:
        return assign_random_scatter(dies, seed=seed)
