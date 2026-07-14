"""
Wafer geometry — "where does each die go?"

Given a round wafer and rectangular dies, this module answers:
  * which integer grid positions (dieX, dieY) fit inside the usable circle,
  * what the physical center of each die is in millimeters,
  * which edge marker the wafer carries (notch vs flat) based on its diameter,
  * how big the lithography stepping field (reticle shot) is for a die size.

All measurements are in millimeters. The die grid is an integer coordinate
system centered at (0, 0): negative dieX = left of wafer center, positive =
right; same convention for dieY.

Spec constraints implemented here (2026.07.10 Synthetic Wafer Map Generator):
  * Wafer diameter: 150 / 200 / 300 mm only.
  * 150 mm wafers carry a FLAT; 200/300 mm wafers carry a NOTCH (auto-select).
  * Edge exclusion: 1 to 10 mm.
  * Die size: 1x1 mm up to 25x35 mm with aspect ratio between 1:2 and 2:1.
  * Scribe street: 0.05 to 0.2 mm, default 0.1 mm.
  * Stepping field is auto-derived from the die size (users know their die
    size, not their reticle layout).
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Spec constants
# ---------------------------------------------------------------------------

# The only wafer sizes the spec allows. 150 mm (6"), 200 mm (8"), 300 mm (12").
STANDARD_DIAMETERS = (150.0, 200.0, 300.0)

# Edge exclusion band allowed by the spec (mm).
EDGE_EXCLUSION_MIN = 1.0
EDGE_EXCLUSION_MAX = 10.0

# Scribe street (the saw lane between dies) allowed by the spec (mm).
STREET_MIN = 0.05
STREET_MAX = 0.20
STREET_DEFAULT = 0.10

# Die size box: the smaller side may not exceed 25 mm, the larger side 35 mm,
# and the aspect ratio must stay between 1:2 and 2:1 (6x3 OK, 12x3 not).
DIE_MIN_MM = 1.0
DIE_MAX_SHORT_MM = 25.0
DIE_MAX_LONG_MM = 35.0
ASPECT_MIN = 0.5   # 1:2
ASPECT_MAX = 2.0   # 2:1

# A full-field lithography scanner exposes at most ~26 x 33 mm per shot.
# The auto stepping field packs as many dies as possible into this box.
RETICLE_FIELD_W_MM = 26.0
RETICLE_FIELD_H_MM = 33.0


def auto_edge_type(diameter: float) -> str:
    """Spec rule: 150 mm wafers have a flat, 200/300 mm wafers have a notch."""
    return "flat" if diameter <= 150.0 else "notch"


def snap_diameter(diameter: float) -> float:
    """Snap an arbitrary requested diameter to the nearest standard size."""
    return min(STANDARD_DIAMETERS, key=lambda d: abs(d - float(diameter)))


def validate_die_size(die_width: float, die_height: float) -> Tuple[bool, str]:
    """Check a die size against the spec box and aspect-ratio rule.

    Returns (ok, message). `message` explains the violation when ok is False.
    """
    w, h = float(die_width), float(die_height)
    if w < DIE_MIN_MM or h < DIE_MIN_MM:
        return False, f"Die must be at least {DIE_MIN_MM}x{DIE_MIN_MM} mm."
    short, long_ = min(w, h), max(w, h)
    if short > DIE_MAX_SHORT_MM or long_ > DIE_MAX_LONG_MM:
        return False, (
            f"Die may not exceed {DIE_MAX_SHORT_MM:.0f}x{DIE_MAX_LONG_MM:.0f} mm "
            f"(got {w:g}x{h:g})."
        )
    ratio = w / h
    if ratio < ASPECT_MIN or ratio > ASPECT_MAX:
        return False, (
            f"Aspect ratio must stay between 1:2 and 2:1 "
            f"(got {w:g}x{h:g} = {ratio:.2f})."
        )
    return True, ""


def clamp_die_size(die_width: float, die_height: float) -> Tuple[float, float]:
    """Force a die size into the legal box, fixing aspect ratio if needed.

    Used for LLM/keyword-parsed requests where we would rather correct the
    value than raise an error at the user.
    """
    w = max(DIE_MIN_MM, float(die_width))
    h = max(DIE_MIN_MM, float(die_height))
    # Cap each side against the box (short side 25 mm, long side 35 mm).
    if w <= h:
        w = min(w, DIE_MAX_SHORT_MM)
        h = min(h, DIE_MAX_LONG_MM)
    else:
        h = min(h, DIE_MAX_SHORT_MM)
        w = min(w, DIE_MAX_LONG_MM)
    # Pull the aspect ratio back inside 1:2 .. 2:1 by shrinking the long side.
    if w / h > ASPECT_MAX:
        w = h * ASPECT_MAX
    elif w / h < ASPECT_MIN:
        h = w * ASPECT_MAX
    return round(w, 3), round(h, 3)


def clamp_street(street_width: float) -> float:
    """Force the scribe street into the spec range 0.05..0.2 mm."""
    return max(STREET_MIN, min(STREET_MAX, float(street_width)))


def auto_stepping_field(die_width: float, die_height: float,
                        street_width: float = STREET_DEFAULT) -> Tuple[int, int]:
    """Derive dies-per-reticle from the die size (spec: auto-generate).

    Packs as many full dies (incl. street) as fit inside a standard 26x33 mm
    scanner field. Always at least 1x1 (dies bigger than the field are exposed
    one per shot).
    """
    pitch_x = die_width + street_width
    pitch_y = die_height + street_width
    dpr_x = max(1, int(RETICLE_FIELD_W_MM // pitch_x))
    dpr_y = max(1, int(RETICLE_FIELD_H_MM // pitch_y))
    return dpr_x, dpr_y


# ---------------------------------------------------------------------------
# Wafer configuration
# ---------------------------------------------------------------------------

@dataclass
class WaferConfig:
    """Everything geometric about one wafer design.

    `edge_orientation` applies to BOTH edge marker types: it says where the
    notch or the flat sits on the rendered map (down / up / left / right, i.e.
    6 / 12 / 9 / 3 o'clock — 90 degree increments only, per spec).
    """
    diameter: float          # mm: 150, 200 or 300
    edge_type: str           # 'notch' or 'flat' (see auto_edge_type)
    edge_exclusion: float    # mm, 1..10 — dies this close to the edge are excluded
    die_width: float         # mm
    die_height: float        # mm
    x_offset: float = 0.0    # mm, shifts the die grid horizontally from center
    y_offset: float = 0.0    # mm, shifts the die grid vertically from center
    street_width: float = STREET_DEFAULT  # mm scribe/street gap between dies
    dies_per_reticle_x: int = 2   # dies across one stepping field (X)
    dies_per_reticle_y: int = 2   # dies across one stepping field (Y)
    reticle_fail_die_x: int = 0   # which die column inside the field repeats-fails
    reticle_fail_die_y: int = 0   # which die row inside the field repeats-fails
    edge_orientation: str = "down"   # 'down' | 'up' | 'left' | 'right'
    # Repeaters: 1.0 = classic "hard" repeater (always fails). Below 1.0 it is
    # a "soft repeater" — the same reticle position fails only part of the time
    # (spec allows 10%..100% in 10% steps).
    repeater_fail_rate: float = 1.0
    # Striping: yield loss along ONE edge of every stepping field (lens tilt).
    # 1.0 = hard fail along the stripe, <1.0 = soft-repeater style.
    stripe_fail_rate: float = 1.0

    @property
    def pitch_x(self) -> float:
        """Center-to-center spacing between dies in X (die + street)."""
        return self.die_width + self.street_width

    @property
    def pitch_y(self) -> float:
        """Center-to-center spacing between dies in Y (die + street)."""
        return self.die_height + self.street_width

    @property
    def reticle_width_mm(self) -> float:
        """Stepping-field width derived from dies-per-reticle and pitch."""
        return self.dies_per_reticle_x * self.pitch_x

    @property
    def reticle_height_mm(self) -> float:
        """Stepping-field height derived from dies-per-reticle and pitch."""
        return self.dies_per_reticle_y * self.pitch_y

    @property
    def die_area_cm2(self) -> float:
        """Die area in cm² — the A in the Poisson yield model Y = e^(-A·D)."""
        return (self.die_width / 10.0) * (self.die_height / 10.0)


# Type alias: each die is (dieX, dieY, center_x_mm, center_y_mm)
Die = Tuple[int, int, float, float]


def compute_die_grid(config: WaferConfig) -> List[Die]:
    """
    Compute all valid die positions that fit within the wafer.

    A die is included if its center falls within (radius - edge_exclusion) of
    the wafer center. Die centers are spaced by pitch = die size + street.
    """
    radius = config.diameter / 2.0
    active_radius = radius - config.edge_exclusion
    pitch_x = config.pitch_x
    pitch_y = config.pitch_y

    max_ix = int(np.ceil(radius / pitch_x)) + 1
    max_iy = int(np.ceil(radius / pitch_y)) + 1

    dies: List[Die] = []
    for ix in range(-max_ix, max_ix + 1):
        for iy in range(-max_iy, max_iy + 1):
            cx = ix * pitch_x + config.x_offset
            cy = iy * pitch_y + config.y_offset

            dist = np.sqrt(cx ** 2 + cy ** 2)
            if dist <= active_radius:
                dies.append((ix, iy, cx, cy))

    return dies


def gross_die_per_wafer(config: WaferConfig) -> int:
    """Gross Die Per Wafer (GDPW) — total die positions on one wafer."""
    return len(compute_die_grid(config))
