"""
Final Test outcomes + ground-truth match tables for Story 1.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from assembly import AssembledUnit, AssemblyManifest
from ecid import ecid_components, blank_ecid_components


# Internal FT bins (kept separate from CP signature bins).
FT_PASS_BIN = 1
FT_FAIL_BASELINE = 101      # natural FT fallout on good units
FT_FAIL_NO_ECID = 102       # blank ECID → fail
FT_FAIL_MISPICK = 103       # wrong bin / wrong XY
FT_FAIL_GDBN = 104          # spatial "good die bad neighborhood" fallout (1.g)

FT_BIN_INFO = {
    FT_PASS_BIN: {"name": "FT_PASS", "state": "P", "description": "Passed Final Test"},
    FT_FAIL_BASELINE: {
        "name": "FT_BASELINE_FAIL", "state": "F",
        "description": "Final Test functional / parametric fail",
    },
    FT_FAIL_NO_ECID: {
        "name": "FT_NO_ECID", "state": "F",
        "description": "Failed FT — no readable ECID (assembly wreck / short)",
    },
    FT_FAIL_MISPICK: {
        "name": "FT_MISPICK", "state": "F",
        "description": "Failed FT — mis-picked die (wrong bin or wrong XY)",
    },
    FT_FAIL_GDBN: {
        "name": "FT_GDBN_FAIL", "state": "F",
        "description": "Failed FT — spatial fallout invisible at CP (GDBN / missed CP cluster)",
    },
}


@dataclass
class FtUnitResult:
    unit: AssembledUnit
    ft_bin: int
    ft_hardbin: int
    ft_softbin: int
    ft_pass: bool
    fail_reason: str  # "none" | "baseline" | "no_ecid" | "mispick"


def run_final_test(
    units: Sequence[AssembledUnit],
    baseline_ft_fallout: float = 0.03,
    mispick_ft_fail_pct: float = 1.0,
    seed: int = 0,
) -> List[FtUnitResult]:
    """Assign FT pass/fail from assembly provenance + knobs.

    Rules (Story 1):
      - Any unit with no ECID → fail
      - Any mis-picked unit (wrong_bin / wrong_xy) → fail with mispick_ft_fail_pct
        (1.0 = simple, 0.8 = subtle default)
      - Correctly-picked units with a spatial hazard flag (GDBN, story 1g) →
        fail with their own extra_fail_pct, checked before the baseline roll
      - Correct picks with valid ECID → fail only via baseline_ft_fallout
      - Assembly wrecks already have blank ECID → fail as no_ecid
    """
    rng = random.Random(seed + 3331)
    baseline = max(0.0, min(1.0, float(baseline_ft_fallout)))
    mispick_p = max(0.0, min(1.0, float(mispick_ft_fail_pct)))
    out: List[FtUnitResult] = []

    for u in units:
        if not u.ecid:
            ft_bin = FT_FAIL_NO_ECID
            reason = "no_ecid"
            passed = False
        elif u.pick_error in ("wrong_bin", "wrong_xy"):
            if rng.random() < mispick_p:
                ft_bin = FT_FAIL_MISPICK
                reason = "mispick"
                passed = False
            else:
                # Subtle case: some mis-picks still pass FT (bad for quality!)
                ft_bin = FT_PASS_BIN
                reason = "none"
                passed = True
        elif u.extra_fail_pct > 0 and rng.random() < u.extra_fail_pct:
            # GDBN / missed-CP-cluster: correctly-picked die, valid ECID, but
            # fails FT because of a spatial hazard the CP program never saw.
            ft_bin = FT_FAIL_GDBN
            reason = u.extra_fail_reason or "gdbn"
            passed = False
        else:
            if rng.random() < baseline:
                ft_bin = FT_FAIL_BASELINE
                reason = "baseline"
                passed = False
            else:
                ft_bin = FT_PASS_BIN
                reason = "none"
                passed = True

        # Hardbin: 1 = pass, 2+ = fail causes
        if passed:
            hard, soft = 1, 1
        else:
            hard = {
                FT_FAIL_BASELINE: 2, FT_FAIL_NO_ECID: 3, FT_FAIL_MISPICK: 4,
                FT_FAIL_GDBN: 5,
            }.get(ft_bin, 2)
            soft = hard * 10 + 1

        out.append(FtUnitResult(
            unit=u,
            ft_bin=ft_bin,
            ft_hardbin=hard,
            ft_softbin=soft,
            ft_pass=passed,
            fail_reason=reason,
        ))
    return out


def build_ft_df(
    ft_results: Sequence[FtUnitResult],
    program: str = "DEMO",
    ecid_representation: str = "single",
) -> pd.DataFrame:
    """Unit-level Final Test CSV.

    `ecid_representation="split_items"` adds the 4 decomposed ECID test
    items (spec 1.b.iii) alongside the convenience ECID column.
    """
    split_items = ecid_representation == "split_items"
    rows = []
    for r in ft_results:
        u = r.unit
        info = FT_BIN_INFO.get(r.ft_bin, {})
        row = {
            "Program": program,
            "FtLot": u.ft_lot_id,
            "UnitId": u.unit_id,
            "ECID": u.ecid,
            "FtBin": r.ft_bin,
            "FtHardBin": r.ft_hardbin,
            "FtSoftBin": r.ft_softbin,
            "FtBinName": info.get("name", f"BIN{r.ft_bin}"),
            "FtBinState": info.get("state", "F"),
            "FtPass": int(r.ft_pass),
            "FailReason": r.fail_reason,
            "PickError": u.pick_error,
            "SpatialHazard": u.extra_fail_reason or "",
            "CpLot": u.cp_lot_id,
            "CpWafer": u.cp_wafer_id,
            "CpWaferNumber": u.cp_wafer_number,
            "IntendedDieX": u.intended_die_x,
            "IntendedDieY": u.intended_die_y,
            "ActualDieX": u.actual_die_x,
            "ActualDieY": u.actual_die_y,
            "CpBin": u.cp_bin,
            "Notes": u.notes,
        }
        if split_items:
            row.update(
                ecid_components(u.cp_lot_id, u.cp_wafer_number,
                                 u.actual_die_x, u.actual_die_y)
                if u.ecid else blank_ecid_components()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_match_df(ft_results: Sequence[FtUnitResult]) -> pd.DataFrame:
    """Ground-truth match table — never join on ECID alone.

    expected_match:
      true         — correct pick, valid ECID, should match exactly one CP die
      false        — mis-pick (wrong bin / wrong XY); ECID may match a *fail* die
      unmatchable  — blank ECID; must NOT match via ECID join
    """
    rows = []
    for r in ft_results:
        u = r.unit
        if not u.ecid:
            expected = "unmatchable"
        elif u.pick_error in ("wrong_bin", "wrong_xy"):
            expected = "false"
        else:
            expected = "true"
        rows.append({
            "UnitId": u.unit_id,
            "FtLot": u.ft_lot_id,
            "ECID": u.ecid,
            "ExpectedMatch": expected,
            "PickError": u.pick_error,
            "CpLot": u.cp_lot_id,
            "CpWafer": u.cp_wafer_id,
            "MatchDieX": u.actual_die_x,
            "MatchDieY": u.actual_die_y,
            "IntendedDieX": u.intended_die_x,
            "IntendedDieY": u.intended_die_y,
            "FtPass": int(r.ft_pass),
            "FailReason": r.fail_reason,
        })
    return pd.DataFrame(rows)


def naive_ecid_join_explosion(
    cp_df: pd.DataFrame,
    ft_df: pd.DataFrame,
) -> Tuple[int, int]:
    """Demonstrate the blank-ECID join hazard.

    Returns (blank_ft_count, naive_join_rows) where naive_join_rows is how
    many rows a SQL-style join on ECID produces when matching blank FT units
    to blank CP dies (cartesian product over blanks).
    """
    cp_blank = cp_df[cp_df["ECID"].fillna("") == ""]
    ft_blank = ft_df[ft_df["ECID"].fillna("") == ""]
    n_ft = len(ft_blank)
    n_cp = len(cp_blank)
    return n_ft, n_ft * n_cp


def story1_summary(
    ft_results: Sequence[FtUnitResult],
    manifest: Optional[AssemblyManifest] = None,
) -> Dict[str, float]:
    """Quick metrics for UI / tests."""
    n = len(ft_results) or 1
    passed = sum(1 for r in ft_results if r.ft_pass)
    blank = sum(1 for r in ft_results if not r.unit.ecid)
    mispick = sum(1 for r in ft_results if r.unit.pick_error in ("wrong_bin", "wrong_xy"))
    wreck = sum(1 for r in ft_results if r.unit.pick_error == "assembly_wreck")
    stats = {
        "ft_units": float(len(ft_results)),
        "ft_yield_pct": 100.0 * passed / n,
        "blank_ecid_pct": 100.0 * blank / n,
        "mispick_pct": 100.0 * mispick / n,
        "wreck_pct": 100.0 * wreck / n,
    }
    if manifest:
        stats.update({f"asm_{k}": v for k, v in manifest.stats.items()})
    return stats
