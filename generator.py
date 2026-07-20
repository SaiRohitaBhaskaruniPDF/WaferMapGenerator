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

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from geometry import WaferConfig, compute_die_grid, auto_stepping_field
from signatures import (
    BIN_DEFINITIONS, DieResult, compose_signatures, NO_PATTERN_SIGNATURE,
)
from yield_model import (
    resolve_target_yield, apply_yield_target, cascade_insertions,
    s2s_factors, apply_s2s, INSERTION_TEMPS, poisson_yield,
    NO_PATTERN_YIELD_MIN, NO_PATTERN_YIELD_MAX,
)
from binning import build_bin_map, map_wafer_bins
from test_items import TestPlan, generate_die_results
from fab import lot_schedule, auto_site_count, assign_sites, make_lot_id
from stdf_writer import write_stdf
from ecid import (
    assign_ecids_for_wafer, assert_nonblank_unique, ecid_components,
    blank_ecid_components, ECID_MODE_PLAIN,
)
from assembly import assemble_for_scenario
from final_test import (
    run_final_test, build_ft_df, build_match_df, story1_summary,
)
from gdbn import apply_gdbn_ft_loss
from multidie import assemble_multidie_products, build_multidie_df


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
    ecids: Dict[str, List[str]] = field(default_factory=dict)  # per insertion


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
    ecid_representation: str = "single"           # "single" | "split_items"
    # Story 1 (ECID / FT) — None when story not enabled
    ft_df: Optional[pd.DataFrame] = None
    match_df: Optional[pd.DataFrame] = None
    story_manifest: Optional[dict] = None
    story_summary: Optional[dict] = None
    # Story 1c (multi-die products) / Story 1g (GDBN) — None unless enabled
    multidie_df: Optional[pd.DataFrame] = None

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
    ecid_mode = getattr(req, "ecid_mode", ECID_MODE_PLAIN) or ECID_MODE_PLAIN

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
            # "No pattern" with no yield given: apply a per-wafer baseline
            # (93-97%) so the wafers look like ordinary production material
            # instead of all-pass.
            wafer_target = target_yield
            variation = float(getattr(req, "yield_variation_pct", 0.0) or 0.0)
            if wafer_target is not None and variation > 0.0:
                # Per-wafer jitter: vary the defect density (or the direct
                # yield target) by a random factor inside the ±% band.
                factor = 1.0 + random.Random(wafer_seed + 3).uniform(
                    -variation, variation) / 100.0
                if (getattr(req, "yield_mode", "signature") == "defect_density"
                        and getattr(req, "defect_density", None) is not None):
                    wafer_target = poisson_yield(
                        config.die_area_cm2, req.defect_density * factor)
                else:
                    wafer_target = max(0.0, min(1.0, wafer_target * factor))
            if wafer_target is None and NO_PATTERN_SIGNATURE in signatures:
                wafer_target = random.Random(wafer_seed + 5).uniform(
                    NO_PATTERN_YIELD_MIN, NO_PATTERN_YIELD_MAX)
            cp1 = apply_yield_target(cp1, wafer_target, seed=wafer_seed + 11)
            # 3. Optional site-correlated loss on top of the yield model.
            if s2s is not None:
                cp1 = apply_s2s(cp1, sites, s2s, seed=wafer_seed + 23)
            # 4. Retest cascade: CP2/CP3 keep 90..99.9% of prior passers.
            insertions = cascade_insertions(cp1, num_insertions,
                                            seed=wafer_seed + 37)
            # 5. Map internal bins to the configured hardbin/softbin spaces.
            hard_bins: Dict[str, List[int]] = {}
            soft_bins: Dict[str, List[int]] = {}
            ecids: Dict[str, List[str]] = {}
            for ins, results in insertions.items():
                hb, sb = map_wafer_bins(results, bin_map)
                hard_bins[ins] = hb
                soft_bins[ins] = sb
                ecids[ins] = assign_ecids_for_wafer(
                    results, lot_id, w + 1, mode=ecid_mode)
                assert_nonblank_unique(ecids[ins])

            wafers.append(WaferResult(
                wafer_id=f"{lot_id}_{w + 1:02d}",
                number=w + 1,
                insertions=insertions,
                hard_bins=hard_bins,
                soft_bins=soft_bins,
                sites=sites,
                ecids=ecids,
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
        ecid_representation=getattr(req, "ecid_representation", "single") or "single",
    )
    result.df = _build_master_df(result)
    if include_test_data and test_plan is not None:
        result.param_df = _build_param_df(result)

    # ---- Story 1: ECID matching / FT traceability (optional) ---------------
    story_id = getattr(req, "story_id", "none")
    if story_id == "story1":
        _apply_story1(result, req)
    elif story_id == "story1_gdbn":
        _apply_gdbn(result, req)
    elif story_id == "story1_multidie":
        _apply_multidie(result, req)

    return result


def _apply_story1(result: GenerationResult, req) -> None:
    """Assemble FT units + match ground truth for a Story 1 scenario."""
    scenario = getattr(req, "story1_scenario", "one_to_one_simple") or "one_to_one_simple"
    seed = hash((scenario, req.lot_id, tuple(req.signatures))) % 100000
    blank = getattr(req, "blank_ecid_pct", None)
    mix = getattr(req, "valid_ecid_mix", None)
    xy = getattr(req, "xy_shift", None)
    if isinstance(xy, (list, tuple)) and len(xy) == 2:
        xy_shift = (int(xy[0]), int(xy[1]))
    else:
        xy_shift = None

    ecid_mode = getattr(req, "ecid_mode", ECID_MODE_PLAIN) or ECID_MODE_PLAIN
    units, manifest, spec = assemble_for_scenario(
        result.lots,
        result.insertion_names,
        scenario_id=scenario,
        blank_ecid_pct=blank,
        valid_ecid_mix=mix,
        xy_shift=xy_shift,
        ship_insertion=getattr(req, "ship_insertion", "") or "",
        seed=seed,
        ecid_mode=ecid_mode,
    )

    mispick_fail = getattr(req, "mispick_ft_fail_pct", None)
    if mispick_fail is None:
        mispick_fail = float(spec.get("mispick_ft_fail_pct", 1.0))
    baseline = float(getattr(req, "baseline_ft_fallout", 0.03))

    ft_results = run_final_test(
        units,
        baseline_ft_fallout=baseline,
        mispick_ft_fail_pct=float(mispick_fail),
        seed=seed,
    )
    result.ft_df = build_ft_df(
        ft_results, program=result.program,
        ecid_representation=result.ecid_representation,
    )
    result.match_df = build_match_df(ft_results)
    result.story_summary = story1_summary(ft_results, manifest)
    result.story_manifest = {
        "story_id": "story1",
        "scenario_id": scenario,
        "spec": spec,
        "warnings": list(manifest.warnings),
        "stats": dict(manifest.stats),
        "summary": dict(result.story_summary),
        "seed": seed,
        "ft_lot_ids": sorted({u.ft_lot_id for u in units}),
        "cp_lot_ids": [l.lot_id for l in result.lots],
        "gdpw": len(result.lots[0].wafers[0].insertions[result.insertion_names[0]])
                if result.lots and result.lots[0].wafers else 0,
    }


def _apply_gdbn(result: GenerationResult, req) -> None:
    """Story 1g: low yield at FT caused by CP clusters (GDBN / missed coverage)."""
    scenario = getattr(req, "gdbn_scenario", "gdbn_neighbor") or "gdbn_neighbor"
    seed = hash((scenario, req.lot_id, tuple(req.signatures))) % 100000
    ecid_mode = getattr(req, "ecid_mode", ECID_MODE_PLAIN) or ECID_MODE_PLAIN
    radius_mm = result.config.diameter / 2.0
    growth = getattr(req, "gdbn_growth", None)
    fail_pct = getattr(req, "gdbn_fail_pct", None)

    units, manifest, spec = apply_gdbn_ft_loss(
        result.lots,
        result.insertion_names,
        scenario_id=scenario,
        radius_mm=radius_mm,
        growth=growth,
        fail_pct=fail_pct,
        ship_insertion=getattr(req, "ship_insertion", "") or "",
        seed=seed,
        ecid_mode=ecid_mode,
    )

    baseline = float(getattr(req, "baseline_ft_fallout", 0.03))
    ft_results = run_final_test(
        units,
        baseline_ft_fallout=baseline,
        mispick_ft_fail_pct=1.0,  # unused: GDBN units are never mis-picked
        seed=seed,
    )
    result.ft_df = build_ft_df(
        ft_results, program=result.program,
        ecid_representation=result.ecid_representation,
    )
    result.match_df = build_match_df(ft_results)
    result.story_summary = story1_summary(ft_results, manifest)
    result.story_manifest = {
        "story_id": "story1_gdbn",
        "scenario_id": scenario,
        "spec": spec,
        "warnings": list(manifest.warnings),
        "stats": dict(manifest.stats),
        "summary": dict(result.story_summary),
        "seed": seed,
        "ft_lot_ids": sorted({u.ft_lot_id for u in units}),
        "cp_lot_ids": [l.lot_id for l in result.lots],
        "gdpw": len(result.lots[0].wafers[0].insertions[result.insertion_names[0]])
                if result.lots and result.lots[0].wafers else 0,
    }


def _apply_multidie(result: GenerationResult, req) -> None:
    """Story 1c (Case B): package multi-die products from 3 component roles."""
    mode = getattr(req, "multidie_mode", "full_trace") or "full_trace"
    n_products = int(getattr(req, "num_multidie_products", 0) or 0)
    if n_products <= 0:
        n_products = max(10, min(2000, int(getattr(req, "num_wafers", 5)) * 40))
    seed = hash((mode, req.lot_id)) % 100000
    ecid_mode = getattr(req, "ecid_mode", ECID_MODE_PLAIN) or ECID_MODE_PLAIN

    products, manifest = assemble_multidie_products(
        base_lot_id=req.lot_id,
        n_products=n_products,
        mode=mode,
        baseline_ft_fallout=float(getattr(req, "baseline_ft_fallout", 0.05)),
        seed=seed,
        ecid_mode=ecid_mode,
    )
    result.multidie_df = build_multidie_df(products)
    result.story_summary = {
        "products": float(len(products)),
        "ft_yield_pct": 100.0 * manifest["stats"].get("ft_pass", 0.0) / max(1, len(products)),
        "fully_traceable_pct": 100.0 * manifest["stats"].get("fully_traceable", 0.0)
                                / max(1, len(products)),
    }
    result.story_manifest = {
        "story_id": "story1_multidie",
        "scenario_id": mode,
        "warnings": list(manifest["warnings"]),
        "stats": dict(manifest["stats"]),
        "summary": dict(result.story_summary),
        "seed": seed,
        "cp_lot_ids": [l.lot_id for l in result.lots],
        "gdpw": len(result.lots[0].wafers[0].insertions[result.insertion_names[0]])
                if result.lots and result.lots[0].wafers else 0,
    }


# ---------------------------------------------------------------------------
# DataFrames for CSV export
# ---------------------------------------------------------------------------

def _build_master_df(result: GenerationResult) -> pd.DataFrame:
    """Master die-level CSV: one row per die per wafer per insertion.

    Keeps every column the old exporter had (Exensio loaders depend on them)
    and adds the spec columns: Insertion, HardBin, SoftBin, Site.
    """
    config = result.config
    split_items = result.ecid_representation == "split_items"
    rows = []
    for lot in result.lots:
        start_str = lot.start_time.strftime("%m/%d/%Y %H:%M")
        for wafer in lot.wafers:
            for ins in result.insertion_names:
                results = wafer.insertions[ins]
                hard = wafer.hard_bins[ins]
                soft = wafer.soft_bins[ins]
                ecid_list = wafer.ecids.get(ins) if wafer.ecids else None
                for i, (dieX, dieY, cx, cy, bin_num) in enumerate(results):
                    info = BIN_DEFINITIONS.get(bin_num, {})
                    ecid = ecid_list[i] if ecid_list else ""
                    row = {
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
                        "ECID":           ecid,
                        "BinName":        info.get("name", f"BIN{bin_num}"),
                        "BinState":       info.get("state", "F"),
                        "BinDesc":        info.get("description", ""),
                        "StreetWidth_mm": config.street_width,
                        "DieWidth_mm":    config.die_width,
                        "DieHeight_mm":   config.die_height,
                    }
                    if split_items:
                        # Spec 1.b.iii: some programs report ECID as 4
                        # separate test items instead of one burned value.
                        row.update(
                            ecid_components(lot.lot_id, wafer.number, dieX, dieY)
                            if ecid else blank_ecid_components()
                        )
                    rows.append(row)
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
