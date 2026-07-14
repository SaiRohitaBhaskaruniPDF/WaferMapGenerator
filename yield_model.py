"""
Yield model and CP insertion cascade (2026.07.10 spec, "Yield" and
"Test insertions" sections).

This module sits BETWEEN the spatial signatures and the exporters:

    spatial signature  ->  yield target (CP1)  ->  CP2 / CP3 cascade

1. Yield target (CP1)
   The spec allows two ways to set wafer yield:
     * directly ("give me 92% yield"), or
     * via defect density with the simple Poisson model  Y = e^(-A*D)
       where A = die area in cm² and D = defects/cm².
   The spatial signature decides WHICH dies fail first; this module then
   nudges the wafer toward the requested yield:
     * yield too high -> kill additional random dies (bin 5, RANDOM_FAIL),
     * yield too low  -> revive a random subset of failed dies (this thins
       the signature pattern instead of erasing it).

2. CP insertion cascade (CP1 / CP2 / CP3)
   CP = Circuit Probe = wafer sort. Each insertion is typically run at a
   different temperature (CP1 room, CP2 cold, CP3 hot). Spec rules:
     * CP1 pass/fail comes from the yield model + signature.
     * CP2 yield = 90%..99.9% OF CP1, and only CP1 passers can pass CP2.
     * CP3 same rule applied to CP2.
   Dies that fall out at CP2/CP3 get dedicated bins (31/32) so the retest
   loss is visible on the wafer maps and in the bin summaries.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence

from geometry import Die, WaferConfig
from signatures import (
    DieResult, BIN_DEFINITIONS,
    PASS_BIN, RANDOM_FAIL_BIN, CP2_FAIL_BIN, CP3_FAIL_BIN, S2S_FAIL_BIN,
)

# Names of the wafer-sort insertions, in test order.
INSERTION_NAMES = ("CP1", "CP2", "CP3")

# Nominal test temperature per insertion — written into the STDF MIR so a
# downstream tool can tell the insertions apart (CP1 room, CP2 cold, CP3 hot).
INSERTION_TEMPS = {"CP1": "25C", "CP2": "-40C", "CP3": "125C"}

# Retest survival band from the spec: CP2 yield is 90%..99.9% of CP1 yield
# (and CP3 of CP2). The spec explicitly says to ignore the rare "CP2 worse
# than CP1" pathologies.
RETEST_SURVIVAL_MIN = 0.90
RETEST_SURVIVAL_MAX = 0.999


# ---------------------------------------------------------------------------
# Yield math
# ---------------------------------------------------------------------------

def poisson_yield(die_area_cm2: float, defect_density: float) -> float:
    """Poisson yield model from the spec:  Y = e^(-A*D).

    A = die area in cm², D = defect density in defects/cm² (the spec says to
    ignore the defects/in² convention). Returns yield as a 0..1 fraction.
    """
    return math.exp(-die_area_cm2 * max(0.0, defect_density))


def resolve_target_yield(config: WaferConfig, yield_mode: str,
                         target_yield_pct: Optional[float],
                         defect_density: Optional[float]) -> Optional[float]:
    """Turn the user's yield input into a single 0..1 target (or None).

    yield_mode:
      'signature'      -> no target; whatever the spatial pattern produces
      'direct'         -> use target_yield_pct as-is
      'defect_density' -> compute from Y = e^(-A*D)
    """
    if yield_mode == "direct" and target_yield_pct is not None:
        return max(0.0, min(1.0, target_yield_pct / 100.0))
    if yield_mode == "defect_density" and defect_density is not None:
        return poisson_yield(config.die_area_cm2, defect_density)
    return None


# ---------------------------------------------------------------------------
# CP1: converge the signature wafer on the target yield
# ---------------------------------------------------------------------------

def apply_yield_target(die_results: List[DieResult],
                       target_yield: Optional[float],
                       seed: Optional[int] = None) -> List[DieResult]:
    """Adjust a signature-binned wafer so its yield converges on the target.

    The spatial signature stays visually intact:
      * If the wafer yields MORE than the target, extra random dies are killed
        with RANDOM_FAIL (bin 5) — like background defectivity on a real wafer.
      * If the wafer yields LESS than the target, a random subset of failed
        dies is revived (pass). This thins the signature instead of erasing it.

    target_yield is a 0..1 fraction; None means "leave the wafer alone".
    """
    if target_yield is None or not die_results:
        return die_results

    rng = random.Random(seed)
    total = len(die_results)
    target_pass = round(total * target_yield)

    pass_idx = [i for i, d in enumerate(die_results) if d[4] == PASS_BIN]
    fail_idx = [i for i, d in enumerate(die_results) if d[4] != PASS_BIN]

    results = list(die_results)
    if len(pass_idx) > target_pass:
        # Too healthy: kill random passing dies until we hit the target.
        n_extra_kills = len(pass_idx) - target_pass
        for i in rng.sample(pass_idx, n_extra_kills):
            d = results[i]
            results[i] = (d[0], d[1], d[2], d[3], RANDOM_FAIL_BIN)
    elif len(pass_idx) < target_pass:
        # Too sick: revive random failed dies until we hit the target.
        n_revives = min(target_pass - len(pass_idx), len(fail_idx))
        for i in rng.sample(fail_idx, n_revives):
            d = results[i]
            results[i] = (d[0], d[1], d[2], d[3], PASS_BIN)
    return results


# ---------------------------------------------------------------------------
# CP2 / CP3 cascade
# ---------------------------------------------------------------------------

def cascade_insertions(cp1_results: List[DieResult],
                       num_insertions: int = 1,
                       seed: Optional[int] = None) -> Dict[str, List[DieResult]]:
    """Run the CP retest cascade and return per-insertion die results.

    Spec rules implemented:
      * Only dies that passed the PREVIOUS insertion can pass the next one.
      * Each retest keeps 90%..99.9% of the previous insertion's passers
        (the survival rate is drawn once per wafer per insertion, so every
        wafer loses a slightly different fraction).
      * Dies already failed carry their original bin forward unchanged —
        in a real flow they simply are not retested.

    Returns {"CP1": [...], "CP2": [...], ...} with num_insertions entries.
    """
    rng = random.Random(seed)
    num_insertions = max(1, min(3, int(num_insertions)))

    out: Dict[str, List[DieResult]] = {"CP1": cp1_results}
    prev = cp1_results
    for level in range(2, num_insertions + 1):
        name = f"CP{level}"
        retest_bin = CP2_FAIL_BIN if level == 2 else CP3_FAIL_BIN
        # Fraction of the previous insertion's PASSERS that survive this one.
        survival = rng.uniform(RETEST_SURVIVAL_MIN, RETEST_SURVIVAL_MAX)
        current: List[DieResult] = []
        for d in prev:
            if d[4] == PASS_BIN and rng.random() > survival:
                current.append((d[0], d[1], d[2], d[3], retest_bin))
            else:
                current.append(d)  # still passing, or already failed earlier
        out[name] = current
        prev = current
    return out


# ---------------------------------------------------------------------------
# Site-to-site (S2S) yield loss — multi-site nice-to-have
# ---------------------------------------------------------------------------

def s2s_factors(site_count: int, healthy: bool = True,
                seed: Optional[int] = None) -> List[float]:
    """Per-site survival factors for the S2S model (spec "S2S" section).

    Actual per-site yield = yield model * S2S factor. A HEALTHY setup has all
    factors above 0.95 (results look random across the fixture). A PROBLEM
    setup drags one random site down hard so the loss is clearly visible.
    """
    rng = random.Random(seed)
    if healthy:
        return [rng.uniform(0.95, 1.0) for _ in range(site_count)]
    factors = [rng.uniform(0.95, 1.0) for _ in range(site_count)]
    bad_site = rng.randrange(site_count)
    factors[bad_site] = rng.uniform(0.40, 0.80)  # the problem site
    return factors


def apply_s2s(die_results: List[DieResult], sites: Sequence[int],
              factors: Sequence[float],
              seed: Optional[int] = None) -> List[DieResult]:
    """Kill passing dies according to their probe site's S2S factor.

    `sites` holds the 1-based site number per die (same order as die_results);
    `factors` holds one 0..1 survival factor per site. A passing die on site s
    survives with probability factors[s-1]; casualties get bin 33 (S2S_FAIL).
    """
    rng = random.Random(seed)
    results: List[DieResult] = []
    for d, site in zip(die_results, sites):
        if d[4] == PASS_BIN and rng.random() > factors[site - 1]:
            results.append((d[0], d[1], d[2], d[3], S2S_FAIL_BIN))
        else:
            results.append(d)
    return results
