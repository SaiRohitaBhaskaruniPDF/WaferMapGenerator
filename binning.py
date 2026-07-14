"""
Hardbin / softbin mapping (2026.07.10 spec, "Number of bins" section).

The generator keeps its own INTERNAL bin numbers (1..33 in BIN_DEFINITIONS)
because those carry the signature meaning and the map colors. Real testers,
however, report two separate bin spaces:

  * HARDBIN — the physical handler bin. Spec sizes: 16, 64 or 256.
  * SOFTBIN — the finer software bin. Spec sizes: hardbins x4, x16 or x64.
    (The spec says to ignore any strict softbin->hardbin mapping for now.)

This module translates each internal bin into a (hardbin, softbin) pair:

  * PASS (internal 1)   -> hardbin 1, softbin 1 (the universal convention).
  * Each internal FAIL bin gets a STABLE hardbin in 2..hardbin_count, so the
    same failure cause always lands in the same hardbin across wafers/lots.
  * The softbin is derived from the hardbin deterministically: each hardbin
    owns a block of `multiplier` softbins and the internal bin picks one
    inside the block. Deterministic = reproducible files.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from signatures import BIN_DEFINITIONS, DieResult, PASS_BIN

# Allowed sizes from the spec.
"""Your generator uses ~33 internal bins (in BIN_DEFINITIONS) that carry the signature meaning and map colors.
 build_bin_map translates those internal bins into the user's chosen hardbin/softbin space. Pass is pinned to (1, 1); 
fail bins cycle through the available fail hardbins:"""
HARDBIN_CHOICES = (16, 64, 256)
SOFTBIN_MULTIPLIERS = (4, 16, 64)


def build_bin_map(hardbin_count: int = 16,
                  softbin_multiplier: int = 4) -> Dict[int, Tuple[int, int]]:
    """Build {internal_bin: (hardbin, softbin)} for every defined bin.

    Fail hardbins are assigned by cycling the internal fail bins through
    2..hardbin_count, so with 16 hardbins the 32 internal fail causes share
    the 15 fail hardbins, while with 256 hardbins each cause gets its own.
    """
    hardbin_count = int(hardbin_count)
    softbin_multiplier = int(softbin_multiplier)
    n_fail_bins = hardbin_count - 1  # hardbin 1 is reserved for PASS
    """One nuance: with only 16 hardbins, your ~32 internal fail causes share the 15 fail hardbins (they wrap via % n_fail_bins); with 256 hardbins each cause gets its own.
      The mapping is deterministic, so the same failure cause always lands in the same bin across wafers — that's what makes the exports reproducible. map_wafer_bins (lines 59–72) then applies it per wafer, called from generator.py lines 222–225.
    """
    mapping: Dict[int, Tuple[int, int]] = {}
    for internal in sorted(BIN_DEFINITIONS):
        if internal == PASS_BIN:
            mapping[internal] = (1, 1)
            continue
        # Stable fail hardbin: cycle internal fail bins across 2..hardbin_count.
        hard = 2 + (internal - 2) % n_fail_bins
        # Softbin: each hardbin owns a block of `multiplier` consecutive
        # softbins; the internal bin selects a slot inside that block.
        slot = (internal - 2) % softbin_multiplier
        soft = (hard - 1) * softbin_multiplier + 1 + slot
        mapping[internal] = (hard, soft)
    return mapping


def map_wafer_bins(die_results: List[DieResult],
                   bin_map: Dict[int, Tuple[int, int]]) -> Tuple[List[int], List[int]]:
    """Translate one wafer's internal bins into parallel hard/soft bin lists.

    Returns (hard_bins, soft_bins) in the same die order as die_results, so
    callers can zip them with the original tuples without reshaping anything.
    """
    hard_bins: List[int] = []
    soft_bins: List[int] = []
    for d in die_results:
        hard, soft = bin_map.get(d[4], bin_map[PASS_BIN])
        hard_bins.append(hard)
        soft_bins.append(soft)
    return hard_bins, soft_bins


def bin_name(internal_bin: int) -> str:
    """Human-readable name for an internal bin (used in HBR/SBR records)."""
    return BIN_DEFINITIONS.get(internal_bin, {}).get("name", f"BIN{internal_bin}")
