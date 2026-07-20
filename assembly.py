"""
Assembly pick model for Story 1 (CP → FT traceability).

Builds the list of units that leave the wafer(s) for Final Test:
  - correct 1:1 (passers only)
  - sweeper (passers from multiple CP lots → one FT lot)
  - wrong bin (fail dies with/without ECID)
  - wrong XY (origin shift ±1)
  - assembly wrecks (strip ECID on a small % of units)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from signatures import BIN_DEFINITIONS, PASS_BIN, DieResult
from ecid import (
    assign_ecids_for_wafer,
    should_burn_ecid,
    ECID_MODE_PLAIN,
)


PickError = str  # "none" | "wrong_bin" | "wrong_xy" | "assembly_wreck"


@dataclass
class AssembledUnit:
    """One packaged unit headed to (or at) Final Test, with provenance."""
    unit_id: str
    ft_lot_id: str
    cp_lot_id: str
    cp_wafer_id: str
    cp_wafer_number: int
    intended_die_x: int
    intended_die_y: int
    actual_die_x: int
    actual_die_y: int
    cp_bin: int
    ecid: str
    pick_error: PickError = "none"
    notes: str = ""
    # Story 1g (GDBN / missed-CP-cluster): extra FT-only fail probability for
    # units that were correctly picked (no mispick) but sit in a spatial
    # hazard zone invisible to the CP test program. 0.0 = no extra hazard.
    extra_fail_pct: float = 0.0
    extra_fail_reason: str = ""


@dataclass
class AssemblyManifest:
    """Diagnostics for demos / tests (shortfalls, fallbacks, GDPW, etc.)."""
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)


def _is_pass(bin_num: int) -> bool:
    return BIN_DEFINITIONS.get(bin_num, {}).get("state") == "P" or bin_num == PASS_BIN


def _ship_insertion(insertion_names: Sequence[str], ship_insertion: str = "") -> str:
    if ship_insertion and ship_insertion in insertion_names:
        return ship_insertion
    return insertion_names[-1]


def _die_index(results: Sequence[DieResult]) -> Dict[Tuple[int, int], DieResult]:
    return {(d[0], d[1]): d for d in results}


def _xy_shift_candidates(rng: random.Random) -> List[Tuple[int, int]]:
    """±1 in X or Y (not both), shuffled."""
    opts = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rng.shuffle(opts)
    return opts


def make_ft_lot_id(cp_lot_ids: Sequence[str], mode: str = "one_to_one",
                   seed: int = 0) -> str:
    """FT lot number that never equals any source CP lot."""
    rng = random.Random(seed + 4242)
    if mode == "sweeper":
        base = f"FTSWP{rng.randint(1000, 9999)}"
    else:
        src = cp_lot_ids[0] if cp_lot_ids else "CPLOT"
        base = f"FT_{src}"
    # Guarantee uniqueness vs CP lots
    if base not in cp_lot_ids:
        return base
    n = 1
    while f"{base}_{n}" in cp_lot_ids:
        n += 1
    return f"{base}_{n}"


def pick_passers_one_to_one(
    lots,  # List[LotResult] — duck typed
    insertion_names: Sequence[str],
    ft_lot_id: str = "",
    ship_insertion: str = "",
    seed: int = 0,
    ecid_mode: str = ECID_MODE_PLAIN,
) -> Tuple[List[AssembledUnit], AssemblyManifest]:
    """Correct assembly: every CP passer becomes one FT unit (1 CP lot → 1 FT lot)."""
    manifest = AssemblyManifest()
    ins = _ship_insertion(insertion_names, ship_insertion)
    if not lots:
        manifest.warnings.append("No CP lots to assemble")
        return [], manifest

    cp_lot = lots[0]
    ft_lot_id = ft_lot_id or make_ft_lot_id([cp_lot.lot_id], "one_to_one", seed)
    units: List[AssembledUnit] = []
    seq = 0
    for wafer in cp_lot.wafers:
        results = wafer.insertions[ins]
        ecids = assign_ecids_for_wafer(results, cp_lot.lot_id, wafer.number, mode=ecid_mode)
        for (die_x, die_y, _cx, _cy, bin_num), ecid in zip(results, ecids):
            if not _is_pass(bin_num):
                continue
            seq += 1
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
            ))
    manifest.stats["units"] = float(len(units))
    manifest.stats["cp_lots"] = 1.0
    return units, manifest


def pick_sweeper(
    lots,
    insertion_names: Sequence[str],
    ft_lot_id: str = "",
    ship_insertion: str = "",
    seed: int = 0,
    ecid_mode: str = ECID_MODE_PLAIN,
) -> Tuple[List[AssembledUnit], AssemblyManifest]:
    """Sweeper: pass dies from multiple CP lots into one FT lot."""
    manifest = AssemblyManifest()
    ins = _ship_insertion(insertion_names, ship_insertion)
    if len(lots) < 2:
        manifest.warnings.append(
            "Sweeper expected multiple CP lots; falling back to all available lots"
        )
    cp_ids = [l.lot_id for l in lots]
    ft_lot_id = ft_lot_id or make_ft_lot_id(cp_ids, "sweeper", seed)
    if ft_lot_id in cp_ids:
        # Should not happen, but enforce the story rule hard.
        ft_lot_id = make_ft_lot_id(cp_ids + [ft_lot_id], "sweeper", seed + 1)
        manifest.warnings.append(f"Adjusted FT lot id to {ft_lot_id} to avoid CP collision")

    units: List[AssembledUnit] = []
    seq = 0
    for lot in lots:
        for wafer in lot.wafers:
            results = wafer.insertions[ins]
            ecids = assign_ecids_for_wafer(results, lot.lot_id, wafer.number, mode=ecid_mode)
            for (die_x, die_y, _cx, _cy, bin_num), ecid in zip(results, ecids):
                if not _is_pass(bin_num):
                    continue
                seq += 1
                units.append(AssembledUnit(
                    unit_id=f"{ft_lot_id}_U{seq:05d}",
                    ft_lot_id=ft_lot_id,
                    cp_lot_id=lot.lot_id,
                    cp_wafer_id=wafer.wafer_id,
                    cp_wafer_number=wafer.number,
                    intended_die_x=die_x,
                    intended_die_y=die_y,
                    actual_die_x=die_x,
                    actual_die_y=die_y,
                    cp_bin=bin_num,
                    ecid=ecid,
                    pick_error="none",
                ))
    manifest.stats["units"] = float(len(units))
    manifest.stats["cp_lots"] = float(len(lots))
    manifest.stats["distinct_cp_lots_in_ft"] = float(len({u.cp_lot_id for u in units}))
    return units, manifest


def apply_assembly_wrecks(
    units: List[AssembledUnit],
    blank_ecid_pct: float,
    seed: int = 0,
) -> Tuple[List[AssembledUnit], AssemblyManifest]:
    """Strip ECID on ~blank_ecid_pct of units (cracked die / power short in Assy)."""
    manifest = AssemblyManifest()
    pct = max(0.0, min(1.0, float(blank_ecid_pct)))
    if pct <= 0 or not units:
        manifest.stats["wrecked"] = 0.0
        return units, manifest

    rng = random.Random(seed + 911)
    n_wreck = max(1, int(round(len(units) * pct))) if pct > 0 else 0
    # Cap so we stay under 2% when caller asks for detail default, but allow
    # exact requested fraction for tests.
    idxs = list(range(len(units)))
    rng.shuffle(idxs)
    wrecked = set(idxs[:n_wreck])
    out: List[AssembledUnit] = []
    for i, u in enumerate(units):
        if i in wrecked:
            out.append(AssembledUnit(
                unit_id=u.unit_id,
                ft_lot_id=u.ft_lot_id,
                cp_lot_id=u.cp_lot_id,
                cp_wafer_id=u.cp_wafer_id,
                cp_wafer_number=u.cp_wafer_number,
                intended_die_x=u.intended_die_x,
                intended_die_y=u.intended_die_y,
                actual_die_x=u.actual_die_x,
                actual_die_y=u.actual_die_y,
                cp_bin=u.cp_bin,
                ecid="",
                pick_error="assembly_wreck",
                notes="ECID stripped — assembly wreck (short/crack)",
            ))
        else:
            out.append(u)
    manifest.stats["wrecked"] = float(len(wrecked))
    manifest.stats["blank_ecid_pct_requested"] = pct
    manifest.stats["blank_ecid_pct_actual"] = (
        len(wrecked) / len(units) if units else 0.0
    )
    return out, manifest


def pick_wrong_bin(
    lots,
    insertion_names: Sequence[str],
    valid_ecid_mix: float = 0.5,
    ft_lot_id: str = "",
    ship_insertion: str = "",
    seed: int = 0,
    max_units: Optional[int] = None,
    ecid_mode: str = ECID_MODE_PLAIN,
) -> Tuple[List[AssembledUnit], AssemblyManifest]:
    """Assembly error: ship fail-bin dies instead of passers.

    Mix of fails with valid ECID (post-write) and blank ECID (pre-write).
    """
    manifest = AssemblyManifest()
    ins = _ship_insertion(insertion_names, ship_insertion)
    if not lots:
        return [], manifest
    cp_lot = lots[0]
    ft_lot_id = ft_lot_id or make_ft_lot_id([cp_lot.lot_id], "one_to_one", seed)
    mix = max(0.0, min(1.0, float(valid_ecid_mix)))

    with_ecid: List[Tuple] = []
    blank_ecid: List[Tuple] = []
    for wafer in cp_lot.wafers:
        results = wafer.insertions[ins]
        ecids = assign_ecids_for_wafer(results, cp_lot.lot_id, wafer.number, mode=ecid_mode)
        for die, ecid in zip(results, ecids):
            die_x, die_y, _cx, _cy, bin_num = die
            if _is_pass(bin_num):
                continue
            entry = (wafer, die_x, die_y, bin_num, ecid)
            if ecid:
                with_ecid.append(entry)
            else:
                blank_ecid.append(entry)

    rng = random.Random(seed + 5150)
    rng.shuffle(with_ecid)
    rng.shuffle(blank_ecid)

    # How many to pick: default = all passers count (replace good build volume)
    n_passers = sum(
        1 for w in cp_lot.wafers for d in w.insertions[ins] if _is_pass(d[4])
    )
    target = max_units if max_units is not None else n_passers
    n_valid = int(round(target * mix))
    n_blank = target - n_valid

    chosen: List[Tuple] = []
    chosen.extend(with_ecid[:n_valid])
    chosen.extend(blank_ecid[:n_blank])
    # Top up from whichever pool has remainder
    shortfall = target - len(chosen)
    if shortfall > 0:
        rest = with_ecid[n_valid:] + blank_ecid[n_blank:]
        chosen.extend(rest[:shortfall])
        if len(chosen) < target:
            manifest.warnings.append(
                f"Wrong-bin pick shortfall: wanted {target}, got {len(chosen)}"
            )
    if not with_ecid:
        manifest.warnings.append("No post-write fail dies with valid ECID on CP lot")
    if not blank_ecid:
        manifest.warnings.append("No pre-write fail dies with blank ECID on CP lot")

    units: List[AssembledUnit] = []
    for seq, (wafer, die_x, die_y, bin_num, ecid) in enumerate(chosen, start=1):
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
            pick_error="wrong_bin",
            notes="Fail bin picked at assembly",
        ))
    manifest.stats["units"] = float(len(units))
    manifest.stats["with_ecid"] = float(sum(1 for u in units if u.ecid))
    manifest.stats["blank_ecid"] = float(sum(1 for u in units if not u.ecid))
    return units, manifest


def pick_wrong_xy(
    lots,
    insertion_names: Sequence[str],
    xy_shift: Optional[Tuple[int, int]] = None,
    valid_ecid_mix: float = 0.5,
    ft_lot_id: str = "",
    ship_insertion: str = "",
    seed: int = 0,
    ecid_mode: str = ECID_MODE_PLAIN,
) -> Tuple[List[AssembledUnit], AssemblyManifest]:
    """Assembly error: intend passer (x,y) but ship neighbour (x±1,y) or (x,y±1).

    Intended coordinates are the good die; actual is the shifted fail/neighbour.
    If the shifted die is off-wafer, try other ±1 shifts; if none work, skip
    that unit and record a warning.
    """
    manifest = AssemblyManifest()
    ins = _ship_insertion(insertion_names, ship_insertion)
    if not lots:
        return [], manifest
    cp_lot = lots[0]
    ft_lot_id = ft_lot_id or make_ft_lot_id([cp_lot.lot_id], "one_to_one", seed)
    rng = random.Random(seed + 777)
    fixed_shift = xy_shift
    mix = max(0.0, min(1.0, float(valid_ecid_mix)))

    # First pass: collect candidate (intended_pass, actual_die) pairs
    candidates: List[Tuple] = []  # (wafer, ix, iy, ax, ay, abin, aecid)
    skipped = 0
    for wafer in cp_lot.wafers:
        results = wafer.insertions[ins]
        index = _die_index(results)
        ecid_map = {
            (d[0], d[1]): (ecid if should_burn_ecid(d[4]) else "")
            for d, ecid in zip(
                results,
                assign_ecids_for_wafer(
                    results, cp_lot.lot_id, wafer.number, mode=ecid_mode),
            )
        }
        for die_x, die_y, _cx, _cy, bin_num in results:
            if not _is_pass(bin_num):
                continue
            shifts = [fixed_shift] if fixed_shift else _xy_shift_candidates(rng)
            placed = False
            for dx, dy in shifts:
                if dx == 0 and dy == 0:
                    continue
                ax, ay = die_x + dx, die_y + dy
                actual = index.get((ax, ay))
                if actual is None:
                    continue
                a_bin = actual[4]
                a_ecid = ecid_map.get((ax, ay), "")
                # Prefer neighbours that are fails for horror demos, but any
                # on-wafer neighbour is a mis-pick.
                candidates.append(
                    (wafer, die_x, die_y, ax, ay, a_bin, a_ecid, dx, dy)
                )
                placed = True
                break
            if not placed:
                skipped += 1

    if skipped:
        manifest.warnings.append(
            f"Wrong-XY: {skipped} passers had no valid ±1 neighbour on wafer"
        )

    # Apply ECID mix: for units whose actual die has ECID, optionally blank
    # a portion so the fail population is 50/50 valid vs blank (story default).
    rng2 = random.Random(seed + 888)
    rng2.shuffle(candidates)
    n = len(candidates)
    n_keep_valid = int(round(n * mix))
    units: List[AssembledUnit] = []
    for seq, (wafer, ix, iy, ax, ay, a_bin, a_ecid, dx, dy) in enumerate(
            candidates, start=1):
        # Force mix: first n_keep_valid keep their natural ECID; rest blanked
        # if they had one, or left blank if already blank.
        if seq <= n_keep_valid:
            ecid = a_ecid
        else:
            ecid = ""
        units.append(AssembledUnit(
            unit_id=f"{ft_lot_id}_U{seq:05d}",
            ft_lot_id=ft_lot_id,
            cp_lot_id=cp_lot.lot_id,
            cp_wafer_id=wafer.wafer_id,
            cp_wafer_number=wafer.number,
            intended_die_x=ix,
            intended_die_y=iy,
            actual_die_x=ax,
            actual_die_y=ay,
            cp_bin=a_bin,
            ecid=ecid,
            pick_error="wrong_xy",
            notes=f"XY shift ({dx:+d},{dy:+d}) from intended ({ix},{iy})",
        ))
    manifest.stats["units"] = float(len(units))
    manifest.stats["with_ecid"] = float(sum(1 for u in units if u.ecid))
    manifest.stats["blank_ecid"] = float(sum(1 for u in units if not u.ecid))
    manifest.stats["skipped_no_neighbour"] = float(skipped)
    return units, manifest


# ---------------------------------------------------------------------------
# Scenario orchestration
# ---------------------------------------------------------------------------

STORY1_SCENARIOS = {
    "one_to_one_simple": {
        "mode": "one_to_one",
        "blank_ecid_pct": 0.0,
        "pick": "passers",
    },
    "one_to_one_detail": {
        "mode": "one_to_one",
        "blank_ecid_pct": 0.015,
        "pick": "passers",
    },
    "sweeper_simple": {
        "mode": "sweeper",
        "blank_ecid_pct": 0.0,
        "pick": "passers",
        "min_cp_lots": 2,
    },
    "sweeper_detail": {
        "mode": "sweeper",
        "blank_ecid_pct": 0.015,
        "pick": "passers",
        "min_cp_lots": 2,
    },
    "wrong_bin": {
        "mode": "one_to_one",
        "blank_ecid_pct": 0.0,
        "pick": "wrong_bin",
        "valid_ecid_mix": 0.5,
    },
    "wrong_xy_horror_simple": {
        "mode": "one_to_one",
        "blank_ecid_pct": 0.0,
        "pick": "wrong_xy",
        "valid_ecid_mix": 0.5,
        "gdpw_target": 100,
        "mispick_ft_fail_pct": 1.0,
    },
    "wrong_xy_horror_subtle_ft": {
        "mode": "one_to_one",
        "blank_ecid_pct": 0.0,
        "pick": "wrong_xy",
        "valid_ecid_mix": 0.5,
        "gdpw_target": 100,
        "mispick_ft_fail_pct": 0.8,
    },
    "wrong_xy_1000_simple": {
        "mode": "one_to_one",
        "blank_ecid_pct": 0.0,
        "pick": "wrong_xy",
        "valid_ecid_mix": 0.5,
        "gdpw_target": 1000,
        "mispick_ft_fail_pct": 1.0,
    },
    "wrong_xy_1000_subtle_ft": {
        "mode": "one_to_one",
        "blank_ecid_pct": 0.0,
        "pick": "wrong_xy",
        "valid_ecid_mix": 0.5,
        "gdpw_target": 1000,
        "mispick_ft_fail_pct": 0.8,
    },
}


def assemble_for_scenario(
    lots,
    insertion_names: Sequence[str],
    scenario_id: str,
    blank_ecid_pct: Optional[float] = None,
    valid_ecid_mix: Optional[float] = None,
    xy_shift: Optional[Tuple[int, int]] = None,
    ship_insertion: str = "",
    seed: int = 0,
    ecid_mode: str = ECID_MODE_PLAIN,
) -> Tuple[List[AssembledUnit], AssemblyManifest, dict]:
    """Build assembly picks for a named Story 1 scenario."""
    if scenario_id not in STORY1_SCENARIOS:
        raise ValueError(f"Unknown Story 1 scenario: {scenario_id}")
    spec = dict(STORY1_SCENARIOS[scenario_id])
    if blank_ecid_pct is not None:
        spec["blank_ecid_pct"] = blank_ecid_pct
    if valid_ecid_mix is not None:
        spec["valid_ecid_mix"] = valid_ecid_mix

    pick = spec.get("pick", "passers")
    mode = spec.get("mode", "one_to_one")
    manifest = AssemblyManifest()

    if pick == "wrong_bin":
        units, m = pick_wrong_bin(
            lots, insertion_names,
            valid_ecid_mix=spec.get("valid_ecid_mix", 0.5),
            ship_insertion=ship_insertion,
            seed=seed,
            ecid_mode=ecid_mode,
        )
    elif pick == "wrong_xy":
        units, m = pick_wrong_xy(
            lots, insertion_names,
            xy_shift=xy_shift,
            valid_ecid_mix=spec.get("valid_ecid_mix", 0.5),
            ship_insertion=ship_insertion,
            seed=seed,
            ecid_mode=ecid_mode,
        )
    elif mode == "sweeper":
        units, m = pick_sweeper(
            lots, insertion_names,
            ship_insertion=ship_insertion,
            seed=seed,
            ecid_mode=ecid_mode,
        )
    else:
        units, m = pick_passers_one_to_one(
            lots, insertion_names,
            ship_insertion=ship_insertion,
            seed=seed,
            ecid_mode=ecid_mode,
        )

    manifest.warnings.extend(m.warnings)
    manifest.stats.update(m.stats)

    wreck_pct = float(spec.get("blank_ecid_pct", 0.0))
    if wreck_pct > 0 and pick == "passers":
        units, wm = apply_assembly_wrecks(units, wreck_pct, seed=seed)
        manifest.warnings.extend(wm.warnings)
        manifest.stats.update(wm.stats)

    return units, manifest, spec
