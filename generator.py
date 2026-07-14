"""
Generation pipeline — turns one parsed request into finished lots.

This module is the single place where all the spec stages are chained, so
the Streamlit UI (manual form) and the chat agent share EXACTLY the same
behavior. The stages, in order:

    die grid            (geometry.py — diameter, die size, street, exclusion)
      -> spatial signature   (signatures.py — which dies fail, and why)
      -> yield target        (yield_model.py — direct % or Y = e^(-A*D))
      -> S2S loss            (yield_model.py — weak probe site, optional)
      -> CP cascade          (yield_model.py — CP2/CP3 retest fallout)
      -> bin mapping         (binning.py — internal bin -> hardbin/softbin)
      -> exports             (CSV rows here, STDF via stdf_writer.py)

Everything is seeded from (lot index, wafer index, signature set) so a
given request always regenerates identical data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from geometry import WaferConfig, compute_die_grid, auto_stepping_field
from signatures import BIN_DEFINITIONS, DieResult, compose_signatures
from yield_model import (
    resolve_target_yield, apply_yield_target, cascade_insertions,
    s2s_factors, apply_s2s, INSERTION_TEMPS,
)
from binning import build_bin_map, map_wafer_bins
from test_items import TestPlan, generate_die_results
from fab import lot_schedule, auto_site_count, assign_sites, make_lot_id
from stdf_writer import write_stdf


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class WaferResult:
    """One wafer, fully processed through every insertion."""
    wafer_id: str
    number: int                                   # 1..25 sequential (spec)
    insertions: Dict[str, List[DieResult]]        # "CP1" -> die results
    hard_bins: Dict[str, List[int]]               # per insertion, per die
    soft_bins: Dict[str, List[int]]
    sites: List[int]                              # per die probe site (1-based)


@dataclass
class LotResult:
    """One lot: identity, sort-start time, and its wafers."""
    lot_id: str
    start_time: datetime
    wafers: List[WaferResult]

    def insertion_wafers(self, insertion: str) -> List[List[DieResult]]:
        """Die results for every wafer at one insertion (for rendering)."""
        return [w.insertions[insertion] for w in self.wafers]

    @property
    def wafer_ids(self) -> List[str]:
        return [w.wafer_id for w in self.wafers]


@dataclass
class GenerationResult:
    """Everything the UI needs after one generation run."""
    config: WaferConfig
    lots: List[LotResult]
    insertion_names: List[str]                    # e.g. ["CP1", "CP2"]
    site_count: int
    site_pattern: str
    s2s: Optional[List[float]]                    # per-site factors or None
    test_plan: Optional[TestPlan]
    include_test_data: bool
    seconds_per_touchdown: float
    program: str
    df: pd.DataFrame = field(default=None)        # master die-level CSV
    param_df: Optional[pd.DataFrame] = None       # long-format test results

    @property
    def primary_lot(self) -> LotResult:
        """First lot — what the UI previews."""
        return self.lots[0]

    def stdf_files(self) -> Dict[str, bytes]:
        """Build one STDF per lot per insertion: {filename: bytes}."""
        files: Dict[str, bytes] = {}
        for lot in self.lots:
            for ins in self.insertion_names:
                files[f"{lot.lot_id}_{ins}.stdf"] = write_stdf(
                    lot.lot_id,
                    self.program,
                    lot.wafer_ids,
                    lot.insertion_wafers(ins),
                    insertion=ins,
                    temperature=INSERTION_TEMPS.get(ins, "25C"),
                    hard_bins=[w.hard_bins[ins] for w in lot.wafers],
                    soft_bins=[w.soft_bins[ins] for w in lot.wafers],
                    sites=[w.sites for w in lot.wafers],
                    site_count=self.site_count,
                    start_time=lot.start_time,
                    seconds_per_touchdown=self.seconds_per_touchdown,
                    test_plan=self.test_plan,
                    include_test_data=self.include_test_data,
                )
        return files


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def generate(req, config: WaferConfig) -> GenerationResult:
    """Run the full pipeline for a parsed request (WaferGenRequest).

    `req` is duck-typed: any object with the WaferGenRequest fields works.
    `config` is the finished WaferConfig (already validated / clamped).
    """
    # ---- Stepping field: auto-derive from die size unless overridden ------
    # Spec: "more likely that die size is entered than stepping field config,
    # so tool must autogenerate a stepping field".
    if getattr(req, "auto_reticle", True):
        dpr_x, dpr_y = auto_stepping_field(
            config.die_width, config.die_height, config.street_width)
        config.dies_per_reticle_x = dpr_x
        config.dies_per_reticle_y = dpr_y
        config.reticle_fail_die_x %= dpr_x
        config.reticle_fail_die_y %= dpr_y

    dies = compute_die_grid(config)
    gdpw = len(dies)

    # ---- Multi-site setup --------------------------------------------------
    # Site count comes from the spec's GDPW table unless multi-site is off.
    if getattr(req, "multi_site", True):
        site_count = auto_site_count(gdpw)
    else:
        site_count = 1
    site_pattern = getattr(req, "site_pattern", "block")
    sites = assign_sites(dies, site_count, site_pattern)

    # S2S: per-site survival factors. "healthy" keeps every site above 95%;
    # unhealthy drags one site down so the loss shows up in per-site yields.
    s2s = None
    if getattr(req, "s2s_enabled", False) and site_count > 1:
        s2s = s2s_factors(site_count,
                          healthy=getattr(req, "s2s_healthy", True),
                          seed=hash(req.lot_id) % 100000)

    # ---- Yield target ------------------------------------------------------
    target_yield = resolve_target_yield(
        config,
        getattr(req, "yield_mode", "signature"),
        getattr(req, "target_yield_pct", None),
        getattr(req, "defect_density", None),
    )

    # ---- Insertions and bins ----------------------------------------------
    num_insertions = max(1, min(3, int(getattr(req, "num_insertions", 1))))
    insertion_names = [f"CP{i}" for i in range(1, num_insertions + 1)]
    bin_map = build_bin_map(getattr(req, "hardbin_count", 16),
                            getattr(req, "softbin_multiplier", 4))

    # ---- Test plan ----------------------------------------------------------
    test_plan = None
    include_test_data = bool(getattr(req, "include_test_data", False))
    if getattr(req, "test_count", 0):
        test_plan = TestPlan(
            count=int(req.test_count),
            parametric_pct=int(getattr(req, "parametric_pct", 50)),
            value_shape=getattr(req, "value_shape", "uniform"),
            naming_style=getattr(req, "naming_style", "simple"),
            name_length=int(getattr(req, "name_length", 31)),
        )

    # ---- Lot schedule --------------------------------------------------------
    # One lot by default; multiple lots get fab-style IDs and spaced start
    # times (the trend-chart nice-to-have). A single lot keeps the user's ID
    # unless they asked for an auto fab-style number.
    num_lots = max(1, int(getattr(req, "num_lots", 1)))
    cadence = getattr(req, "lot_cadence", "1 lot per week")
    fab_letter = getattr(req, "fab_letter", "A")
    if num_lots > 1 or getattr(req, "auto_lot_id", False):
        schedule = lot_schedule(num_lots, cadence, fab_letter)
    else:
        schedule = [(req.lot_id, datetime.now())]

    signatures = req.signatures if isinstance(req.signatures, list) else [req.signatures]
    # Lot-repeatable defects (handler scratch etc.) derive their geometry
    # from this seed, so they look identical on every wafer of a lot.
    base_lot_seed = hash(tuple(signatures)) % 100000

    lots: List[LotResult] = []
    for lot_idx, (lot_id, lot_start) in enumerate(schedule):
        lot_seed = base_lot_seed + lot_idx * 7717
        wafers: List[WaferResult] = []
        for w in range(req.num_wafers):
            # Per-wafer seed drives wafer-to-wafer variation (random speckle,
            # per-wafer patterns like the wand scratch, retest survival).
            wafer_seed = lot_seed + w * 137

            # 1. Spatial signature(s) decide which dies fail and why.
            cp1 = compose_signatures(dies, signatures, config,
                                     seed=wafer_seed, lot_seed=lot_seed)
            # 2. Converge on the requested yield (adds/removes random fails).
            cp1 = apply_yield_target(cp1, target_yield, seed=wafer_seed + 11)
            # 3. Optional site-correlated loss on top of the yield model.
            if s2s is not None:
                cp1 = apply_s2s(cp1, sites, s2s, seed=wafer_seed + 23)
            # 4. Retest cascade: CP2/CP3 keep 90..99.9% of prior passers.
            insertions = cascade_insertions(cp1, num_insertions,
                                            seed=wafer_seed + 37)
            # 5. Map internal bins to the configured hardbin/softbin spaces.
            hard_bins: Dict[str, List[int]] = {}
            soft_bins: Dict[str, List[int]] = {}
            for ins, results in insertions.items():
                hb, sb = map_wafer_bins(results, bin_map)
                hard_bins[ins] = hb
                soft_bins[ins] = sb

            wafers.append(WaferResult(
                wafer_id=f"{lot_id}_{w + 1:02d}",
                number=w + 1,
                insertions=insertions,
                hard_bins=hard_bins,
                soft_bins=soft_bins,
                sites=sites,
            ))
        lots.append(LotResult(lot_id=lot_id, start_time=lot_start, wafers=wafers))

    result = GenerationResult(
        config=config,
        lots=lots,
        insertion_names=insertion_names,
        site_count=site_count,
        site_pattern=site_pattern,
        s2s=s2s,
        test_plan=test_plan,
        include_test_data=include_test_data,
        seconds_per_touchdown=float(getattr(req, "seconds_per_touchdown", 1.0)),
        program=req.program,
    )
    result.df = _build_master_df(result)
    if include_test_data and test_plan is not None:
        result.param_df = _build_param_df(result)
    return result


# ---------------------------------------------------------------------------
# DataFrames for CSV export
# ---------------------------------------------------------------------------

def _build_master_df(result: GenerationResult) -> pd.DataFrame:
    """Master die-level CSV: one row per die per wafer per insertion.

    Keeps every column the old exporter had (Exensio loaders depend on them)
    and adds the spec columns: Insertion, HardBin, SoftBin, Site.
    """
    config = result.config
    rows = []
    for lot in result.lots:
        start_str = lot.start_time.strftime("%m/%d/%Y %H:%M")
        for wafer in lot.wafers:
            for ins in result.insertion_names:
                results = wafer.insertions[ins]
                hard = wafer.hard_bins[ins]
                soft = wafer.soft_bins[ins]
                for i, (dieX, dieY, cx, cy, bin_num) in enumerate(results):
                    info = BIN_DEFINITIONS.get(bin_num, {})
                    rows.append({
                        "Program":        result.program,
                        "Lot":            lot.lot_id,
                        "Wafer":          wafer.wafer_id,
                        "WaferNumber":    wafer.number,
                        "Insertion":      ins,
                        "start_time":     start_str,
                        "rework_flag":    0,
                        "Bin":            bin_num,
                        "HardBin":        hard[i],
                        "SoftBin":        soft[i],
                        "Site":           wafer.sites[i],
                        "dieX":           dieX,
                        "dieY":           dieY,
                        "X_mm":           round(cx, 4),
                        "Y_mm":           round(cy, 4),
                        "BinName":        info.get("name", f"BIN{bin_num}"),
                        "BinState":       info.get("state", "F"),
                        "BinDesc":        info.get("description", ""),
                        "StreetWidth_mm": config.street_width,
                        "DieWidth_mm":    config.die_width,
                        "DieHeight_mm":   config.die_height,
                    })
    return pd.DataFrame(rows)


def _build_param_df(result: GenerationResult) -> pd.DataFrame:
    """Long-format per-test CSV (CP1 only, to keep the file size sane).

    One row per die per test item: which test, its name, the value and the
    pass flag. Uses the same per-die seeds as the STDF writer so both
    exports contain identical numbers.
    """
    plan = result.test_plan
    rows = []
    for lot in result.lots:
        for w_idx, wafer in enumerate(lot.wafers):
            cp1 = wafer.insertions["CP1"]
            for die_idx, (dieX, dieY, _cx, _cy, bin_num) in enumerate(cp1):
                die_passed = BIN_DEFINITIONS.get(bin_num, {}).get("state") == "P"
                die_seed = (w_idx * 100003) + die_idx  # matches stdf_writer
                for t_num, is_param, value, t_pass in generate_die_results(
                        plan, die_passed, die_seed):
                    rows.append({
                        "Lot":      lot.lot_id,
                        "Wafer":    wafer.wafer_id,
                        "dieX":     dieX,
                        "dieY":     dieY,
                        "TestNum":  t_num,
                        "TestName": plan.names[t_num - 1],
                        "Type":     "PARAMETRIC" if is_param else "PASS_FAIL",
                        "Value":    value,
                        "Pass":     int(t_pass),
                    })
    return pd.DataFrame(rows)
