"""
where does each die go?
Answers -- given a round and rectangular dies, which grid positions fit inside the circle?
Wafer geometry: computes which die positions fit within a wafer circle.
All measurements in millimeters.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class WaferConfig: #used as a container that holds all the wafer settings in one object. Instead of passing 7 seperate arguments everywhere, you pass one config.
    diameter: float        # mm, e.g. 300
    edge_type: str         # 'notch' or 'flat'
    edge_exclusion: float  # mm, e.g. 3.0 — dies this close to the edge are excluded
    die_width: float       # mm, e.g. 10.0
    die_height: float      # mm, e.g. 10.0
    x_offset: float = 0.0  # mm, shifts the die grid horizontally from center
    y_offset: float = 0.0  # mm, shifts the die grid vertically from center


# Type alias: each die is (dieX, dieY, center_x_mm, center_y_mm)
Die = Tuple[int, int, float, float]


def compute_die_grid(config: WaferConfig) -> List[Die]:
    """
    Compute all valid die positions that fit within the wafer.

    The die grid is an integer coordinate system (dieX, dieY) centered at (0,0).
    Negative dieX = left of center, positive = right. Same convention for dieY.
    A die is included if its center falls within (radius - edge_exclusion) of the wafer center.

    Returns a list of (dieX, dieY, cx_mm, cy_mm) tuples.

    * calculate active_radius = radius minus edge exclusion(the usable zone)
    * Loop over a grid of integer positions(ix,iy) covering the whole wafer
    * For each position, compute the die's physical center in mm
    * measure distance from wafer center
    * keep the die only if it fits inside active_radius
    * Output: a list of (dieX, dieY, cx_mm, cy_mm) tuples. This is the "blank" wafer — no pass/fail yet.
    """
    radius = config.diameter / 2.0
    active_radius = radius - config.edge_exclusion

    # Compute the max integer index range to check in each direction
    max_ix = int(np.ceil(radius / config.die_width)) + 1
    max_iy = int(np.ceil(radius / config.die_height)) + 1

    dies: List[Die] = []
    for ix in range(-max_ix, max_ix + 1):
        for iy in range(-max_iy, max_iy + 1):
            # Physical center of this die in mm coordinates
            cx = ix * config.die_width + config.x_offset
            cy = iy * config.die_height + config.y_offset

            # Include die if its center is within the active (non-excluded) radius
            dist = np.sqrt(cx ** 2 + cy ** 2)
            if dist <= active_radius:
                dies.append((ix, iy, cx, cy))

    return dies
