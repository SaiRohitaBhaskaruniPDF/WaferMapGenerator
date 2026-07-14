"""
Fab realism helpers (2026.07.10 spec, "Nice To Have" section):

  * Lot numbers   — FYYWWSSSS: Fab letter, 2-digit Year, Work Week (1..53),
                    4-digit Sequential number. Splits/child lots get .01, .02
                    suffixes.
  * Lot sequences — multiple lots need DIFFERENT sort-start timestamps
                    (1/month, 1/week, 1/day, several/day) so downstream trend
                    charts have something to plot.
  * Test time     — 1..600 seconds per touchdown; drives per-die elapsed
                    time and wafer start/finish timestamps.
  * Multi-site    — 1..16 sites probed simultaneously. Site count is derived
                    from Gross Die Per Wafer, and dies are assigned to sites
                    by a layout pattern (side-by-side, stacked, block,
                    checkerboard, diagonal).
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import List, Sequence, Tuple

from geometry import Die

# ---------------------------------------------------------------------------
# Lot numbers — FYYWWSSSS
# ---------------------------------------------------------------------------

def make_lot_id(fab_letter: str = "A", when: datetime | None = None,
                sequence: int = 0, split: int | None = None) -> str:
    """Build a fab-style lot number: FYYWWSSSS (+ .NN for split lots).

    F    = fab designator (one alphabetic character)
    YY   = last two digits of the year
    WW   = ISO work week, 1..53 (some years genuinely have 53)
    SSSS = sequential lot number 0000..9999
    .NN  = optional split / child-lot suffix (e.g. FYYWWSSSS.01)
    """
    when = when or datetime.now()
    week = when.isocalendar()[1]  # ISO week number handles 53-week years
    lot = f"{fab_letter[:1].upper()}{when.year % 100:02d}{week:02d}{sequence % 10000:04d}"
    if split is not None:
        lot += f".{split:02d}"
    return lot


# Cadence choices for multi-lot sequences: label -> gap between lot starts.
LOT_CADENCES = {
    "1 lot per month":      timedelta(days=30),
    "1 lot per week":       timedelta(weeks=1),
    "1 lot per day":        timedelta(days=1),
    "multiple lots per day": timedelta(hours=6),
}


def lot_schedule(num_lots: int, cadence: str,
                 fab_letter: str = "A",
                 end: datetime | None = None) -> List[Tuple[str, datetime]]:
    """Plan a sequence of lots ENDING near `end` (default: now).

    Returns [(lot_id, sort_start_datetime), ...] oldest first. Working
    backwards from "now" makes the data look like recent history, which is
    what a trend-chart demo wants. Each lot gets a fresh FYYWWSSSS number
    whose YY/WW fields match its own start date.
    """
    end = end or datetime.now()
    gap = LOT_CADENCES.get(cadence, timedelta(weeks=1))
    schedule: List[Tuple[str, datetime]] = []
    rng = random.Random(num_lots * 7919)  # stable sequence numbers per run size
    seq = rng.randint(0, 9999 - num_lots)
    for i in range(num_lots):
        start = end - gap * (num_lots - 1 - i)
        schedule.append((make_lot_id(fab_letter, start, seq + i), start))
    return schedule


# ---------------------------------------------------------------------------
# Multi-site — site count from GDPW, plus site-assignment layouts
# ---------------------------------------------------------------------------

# Spec's GDPW -> parallelism table ("imperfect but roughly correct").
_GDPW_SITE_TABLE = (
    (200, 1),    # < 200 gross die per wafer -> single site
    (400, 2),    # 200..399 -> 2 sites
    (800, 4),    # 400..799 -> 4 sites
    (1600, 8),   # 800..1599 -> 8 sites
)

SITE_PATTERNS = ("side by side", "top & bottom", "block", "checkerboard", "diagonal")


def auto_site_count(gross_die_per_wafer: int) -> int:
    """Pick the parallelism (1..16 sites) from Gross Die Per Wafer."""
    for limit, sites in _GDPW_SITE_TABLE:
        if gross_die_per_wafer < limit:
            return sites
    return 16  # 1600+ GDPW


def _block_dims(site_count: int, pattern: str) -> Tuple[int, int]:
    """Choose the (cols, rows) footprint of the probe-card site array.

    side by side  -> all sites in one horizontal row       (N x 1)
    top & bottom  -> all sites in one vertical column      (1 x N)
    block         -> squarish rectangle, e.g. 2x2, 2x4     (probe-card PCB
                     routing likes compact rectangles)
    checkerboard / diagonal reuse the block footprint but scatter the site
    numbers inside it (also a wire-routing trick on real probe cards).
    """
    if pattern == "side by side":
        return site_count, 1
    if pattern == "top & bottom":
        return 1, site_count
    # Squarish block: split the count into the most compact rectangle.
    cols = 1
    for c in range(1, site_count + 1):
        if site_count % c == 0 and c <= site_count // c:
            cols = c
    rows = cols
    cols = site_count // rows
    return cols, rows


def assign_sites(dies: Sequence[Die], site_count: int,
                 pattern: str = "block") -> List[int]:
    """Assign every die a probe site number (1..site_count).

    The tester steps the whole site array across the wafer, so a die's site
    is fixed by its position inside the repeating array footprint:

      * side by side / top & bottom / block — the site index simply follows
        the die's local (x, y) inside the footprint.
      * checkerboard — neighbouring positions alternate between the low and
        high halves of the site numbers, so adjacent dies are wired to far
        apart probe channels (avoids wire congestion on the probe card PCB).
      * diagonal — the site index shifts by one every row, producing the
        diagonal striping seen on real diagonal-layout probe cards.

    Returns a list of 1-based site numbers in the same order as `dies`.
    """
    if site_count <= 1:
        return [1] * len(dies)

    cols, rows = _block_dims(site_count, pattern)
    sites: List[int] = []
    for dieX, dieY, _, _ in dies:
        lx = dieX % cols
        ly = dieY % rows
        if pattern == "checkerboard":
            base = (ly * cols + lx)
            # Even positions take low site numbers, odd take high ones.
            half = (site_count + 1) // 2
            idx = (base // 2) % half if base % 2 == 0 else half + (base // 2) % (site_count - half)
        elif pattern == "diagonal":
            idx = (lx + ly) % site_count
        else:
            idx = ly * cols + lx
        sites.append(int(idx) % site_count + 1)
    return sites


# ---------------------------------------------------------------------------
# Test time — touchdown math
# ---------------------------------------------------------------------------

def wafer_test_seconds(n_dies: int, site_count: int,
                       seconds_per_touchdown: float) -> float:
    """How long one wafer takes to sort.

    A "touchdown" tests `site_count` dies at once, so the number of
    touchdowns is ceil(dies / sites) and total time = touchdowns x s/td.
    """
    touchdowns = -(-n_dies // max(1, site_count))  # ceiling division
    return touchdowns * seconds_per_touchdown
