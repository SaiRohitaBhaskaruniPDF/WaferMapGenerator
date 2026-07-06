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
class WaferConfig:
    diameter: float        # mm, e.g. 300
    edge_type: str         # 'notch' or 'flat'
    edge_exclusion: float  # mm, e.g. 3.0 — dies this close to the edge are excluded
    die_width: float       # mm, e.g. 10.0
    die_height: float      # mm, e.g. 10.0
    x_offset: float = 0.0  # mm, shifts the die grid horizontally from center
    y_offset: float = 0.0  # mm, shifts the die grid vertically from center
    street_width: float = 0.0  # mm scribe/street gap between adjacent dies
    dies_per_reticle_x: int = 2  # dies across one reticle field (X)
    dies_per_reticle_y: int = 2  # dies across one reticle field (Y)
    reticle_fail_die_x: int = 0  # which die column inside the reticle fails (0-based)
    reticle_fail_die_y: int = 0  # which die row inside the reticle fails (0-based)

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
        """Reticle field width derived from dies-per-reticle and pitch."""
        return self.dies_per_reticle_x * self.pitch_x

    @property
    def reticle_height_mm(self) -> float:
        """Reticle field height derived from dies-per-reticle and pitch."""
        return self.dies_per_reticle_y * self.pitch_y


# Type alias: each die is (dieX, dieY, center_x_mm, center_y_mm)
Die = Tuple[int, int, float, float]


def compute_die_grid(config: WaferConfig) -> List[Die]:
    """
    Compute all valid die positions that fit within the wafer.

    The die grid is an integer coordinate system (dieX, dieY) centered at (0,0).
    Negative dieX = left of center, positive = right. Same convention for dieY.
    A die is included if its center falls within (radius - edge_exclusion) of the wafer center.

    Die centers are spaced by pitch = die size + street width.
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
