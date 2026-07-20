"""
Story 1g: Low yield at FT caused by CP clusters ("missed" CP test coverage).

Background (spec 1.g): CP test programs sometimes miss failures that show up
at FT -- either the CP limits were set too loose, or an item could not be
[easily/cheaply] tested at CP. Two demo cases:

  Dramatic case (`gdbn_dramatic`):
    - CP shows NO pattern (clean, baseline yield only -- "no pattern needed").
    - FT has a yield problem that, once the FT units are mapped back onto
      their CP (X, Y) via ECID, reveals a spatial pattern (default: a donut /
      mid-ring band) that was completely invisible at CP. Every FT unit
      (pass or fail) was a CP pass.

  Less-dramatic / "Good Die Bad Neighborhood" case (`gdbn_neighbor`):
    - CP DOES show a pattern that causes real CP fails (default: a scratch).
    - Die that PASSED CP but sit within `growth` dies (default 1, i.e. the
      failing region grown/dilated by one die in every dimension) of a
      CP-failing die have a default 50% chance of failing FT anyway. This is
      the classic data set that justifies Good-Die-Bad-Neighborhood (GDBN)
      binning in Exensio Test Ops.

Both cases assemble units CORRECTLY (no mis-pick) -- the extra hazard is
layered on as `AssembledUnit.extra_fail_pct`, which
`final_test.run_final_test()` rolls before the ordinary baseline FT fallout.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from assembly import AssembledUnit, AssemblyManifest, make_ft_lot_id
from ecid import assign_ecids_for_wafer, ECID_MODE_PLAIN
from signatures import PASS_BIN, BIN_DEFINITIONS, DieResult, NO_PATTERN_SIGNATURE

GDBN_MODE_DRAMATIC = "dramatic"
GDBN_MODE_NEIGHBOR = "neighbor"
GDBN_MODES = (GDBN_MODE_DRAMATIC, GDBN_MODE_NEIGHBOR)

# Default knobs, per spec 1.g.
DEFAULT_NEIGHBOR_FAIL_PCT = 0.5
DEFAULT_NEIGHBOR_GROWTH = 1
DEFAULT_DRAMATIC_FAIL_PCT = 0.87   # matches signatures.assign_donut default

GDBN_SCENARIOS = {
    "gdbn_dramatic": {
        "gdbn_mode": GDBN_MODE_DRAMATIC,
        "cp_signature": NO_PATTERN_SIGNATURE,
        "fail_pct": DEFAULT_DRAMATIC_FAIL_PCT,
        "growth": 0,
    },
    "gdbn_neighbor": {
        "gdbn_mode": GDBN_MODE_NEIGHBOR,
        "cp_signature": "Scratch / Streak",
        "fail_pct": DEFAULT_NEIGHBOR_FAIL_PCT,
        "growth": DEFAULT_NEIGHBOR_GROWTH,
    },
}


def _is_pass_bin(bin_num: int) -> bool:
    return BIN_DEFINITIONS.get(bin_num, {}).get("state") == "P" or bin_num == PASS_BIN


def _ship_insertion(insertion_names: Sequence[str], ship_insertion: str = "") -> str:
    if ship_insertion and ship_insertion in insertion_names:
        return ship_insertion
    return insertion_names[-1]


def donut_hazard_mask(
    results: Sequence[DieResult],
    radius_mm: float,
    inner_frac: float = 0.25,
    outer_frac: float = 0.60,
) -> Dict[Tuple[int, int], bool]:
    """(x,y) -> True if this die sits in the 'invisible-at-CP' ring band.

    Mirrors `signatures.assign_donut`'s default geometry, but is evaluated
    independently of the CP bin -- it decides an FT-only outcome for die
    that CP already marked PASS.
    """
    inner_r = radius_mm * inner_frac
    outer_r = radius_mm * outer_frac
    mask: Dict[Tuple[int, int], bool] = {}
    for die_x, die_y, cx, cy, _bin in results:
        dist = math.hypot(cx, cy)
        mask[(die_x, die_y)] = inner_r <= dist <= outer_r
    return mask


def neighbor_hazard_mask(
    results: Sequence[DieResult],
    growth: int = DEFAULT_NEIGHBOR_GROWTH,
) -> Dict[Tuple[int, int], bool]:
    """(x,y) -> True if a CP-PASSING die is within `growth` dies (Chebyshev
    distance, i.e. the fail region dilated by `growth` in every dimension)
    of any CP-failing die."""
    fail_xy = {(x, y) for x, y, _cx, _cy, b in results if not _is_pass_bin(b)}
    mask: Dict[Tuple[int, int], bool] = {}
    for die_x, die_y, _cx, _cy, bin_num in results:
        if not _is_pass_bin(bin_num):
            continue
        hazard = False
        for dx in range(-growth, growth + 1):
            for dy in range(-growth, growth + 1):
                if dx == 0 and dy == 0:
                    continue
                if (die_x + dx, die_y + dy) in fail_xy:
                    hazard = True
                    break
            if hazard:
                break
        mask[(die_x, die_y)] = hazard
    return mask


def pick_gdbn_units(
    lots,
    insertion_names: Sequence[str],
    mode: str = GDBN_MODE_NEIGHBOR,
    radius_mm: float = 150.0,
    growth: int = DEFAULT_NEIGHBOR_GROWTH,
    fail_pct: float = DEFAULT_NEIGHBOR_FAIL_PCT,
    ft_lot_id: str = "",
    ship_insertion: str = "",
    seed: int = 0,
    ecid_mode: str = ECID_MODE_PLAIN,
) -> Tuple[List[AssembledUnit], AssemblyManifest]:
    """Correct 1:1 assembly (every CP passer -> one FT unit), with a spatial
    FT-only hazard flag applied per story 1g."""
    manifest = AssemblyManifest()
    ins = _ship_insertion(insertion_names, ship_insertion)
    if not lots:
        manifest.warnings.append("No CP lots to assemble")
        return [], manifest

    cp_lot = lots[0]
    ft_lot_id = ft_lot_id or make_ft_lot_id([cp_lot.lot_id], "one_to_one", seed)
    units: List[AssembledUnit] = []
    seq = 0
    n_hazard = 0
    reason = "gdbn_dramatic" if mode == GDBN_MODE_DRAMATIC else "gdbn_neighbor"

    for wafer in cp_lot.wafers:
        results = wafer.insertions[ins]
        ecids = assign_ecids_for_wafer(results, cp_lot.lot_id, wafer.number, mode=ecid_mode)
        if mode == GDBN_MODE_DRAMATIC:
            hazard_mask = donut_hazard_mask(results, radius_mm)
        else:
            hazard_mask = neighbor_hazard_mask(results, growth)

        for (die_x, die_y, _cx, _cy, bin_num), ecid in zip(results, ecids):
            if not _is_pass_bin(bin_num):
                continue
            seq += 1
            hazard = hazard_mask.get((die_x, die_y), False)
            if hazard:
                n_hazard += 1
            units.append(AssembledUnit(
                unit_id=f"{ft_lot_id}_U{seq:05d}",
                ft_lot_id=ft_lot_id,
                cp_lot_id=cp_lot.lot_id,
                cp_wafer_id=wafer.wafer_id,
                cp_wafer_number=wafer.number,
                intended_die_x=die_x,
                intended_die_y=die_y,
                actual_die_x=die_x,
                actual_die_y=die_y,
                cp_bin=bin_num,
                ecid=ecid,
                pick_error="none",
                extra_fail_pct=(fail_pct if hazard else 0.0),
                extra_fail_reason=(reason if hazard else ""),
                notes=("Spatial hazard (GDBN) — CP pass, FT-only risk" if hazard else ""),
            ))
    manifest.stats["units"] = float(len(units))
    manifest.stats["hazard_units"] = float(n_hazard)
    manifest.stats["hazard_pct_actual"] = (n_hazard / len(units)) if units else 0.0
    return units, manifest


def apply_gdbn_ft_loss(
    lots,
    insertion_names: Sequence[str],
    scenario_id: str,
    radius_mm: float = 150.0,
    growth: Optional[int] = None,
    fail_pct: Optional[float] = None,
    ship_insertion: str = "",
    seed: int = 0,
    ecid_mode: str = ECID_MODE_PLAIN,
) -> Tuple[List[AssembledUnit], AssemblyManifest, dict]:
    """Build assembly picks for a named Story 1g (GDBN) scenario."""
    if scenario_id not in GDBN_SCENARIOS:
        raise ValueError(f"Unknown GDBN scenario: {scenario_id}")
    spec = dict(GDBN_SCENARIOS[scenario_id])
    if growth is not None:
        spec["growth"] = growth
    if fail_pct is not None:
        spec["fail_pct"] = fail_pct

    units, manifest = pick_gdbn_units(
        lots, insertion_names,
        mode=spec["gdbn_mode"],
        radius_mm=radius_mm,
        growth=int(spec.get("growth", DEFAULT_NEIGHBOR_GROWTH)),
        fail_pct=float(spec.get("fail_pct", DEFAULT_NEIGHBOR_FAIL_PCT)),
        ship_insertion=ship_insertion,
        seed=seed,
        ecid_mode=ecid_mode,
    )
    return units, manifest, spec
