"""
Story 1 scenario presets — geometry knobs and request overlays.

Die sizes follow the Yield Stories doc examples:
  ~100 GDPW  → 23×23 mm on 300 mm
  ~1000 GDPW → 7.9×8 mm on 300 mm
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from assembly import STORY1_SCENARIOS
from gdbn import GDBN_SCENARIOS
from multidie import MULTIDIE_MODES


# (die_width, die_height) presets keyed by nominal GDPW target
GDPW_DIE_PRESETS: Dict[int, Tuple[float, float]] = {
    100: (23.0, 23.0),
    1000: (7.9, 8.0),
}

SCENARIO_LABELS = {
    "one_to_one_simple": "1:1 lot — simple (all valid ECID)",
    "one_to_one_detail": "1:1 lot — detail (<2% blank ECID)",
    "sweeper_simple": "Sweeper — simple (all valid ECID)",
    "sweeper_detail": "Sweeper — detail (<2% blank ECID)",
    "wrong_bin": "Assembly error — wrong bin picked",
    "wrong_xy_horror_simple": "Wrong XY — ~100 GDPW, simple FT (100% mispick fail)",
    "wrong_xy_horror_subtle_ft": "Wrong XY — ~100 GDPW, subtle FT (80% mispick fail)",
    "wrong_xy_1000_simple": "Wrong XY — ~1000 GDPW, simple FT",
    "wrong_xy_1000_subtle_ft": "Wrong XY — ~1000 GDPW, subtle FT (80% mispick fail)",
}

GDBN_SCENARIO_LABELS = {
    "gdbn_dramatic": "GDBN — dramatic (clean CP, donut only visible at FT)",
    "gdbn_neighbor": "GDBN — good-die-bad-neighborhood (scratch @ CP, neighbours risk FT fail)",
}

MULTIDIE_SCENARIO_LABELS = {
    "full_trace": "Multi-die — B.1 full traceability (all 3 components)",
    "partial_trace": "Multi-die — B.2 partial traceability (1 of 3 untraceable)",
}

ENCODING_MODE_LABELS = {
    "plain": "Plain concatenated (lot+wafer+X+Y)",
    "rot13": "ROT13 'encrypted' (still unique, not map-readable)",
}

REPRESENTATION_LABELS = {
    "single": "Single ECID value",
    "split_items": "Split into 4 test items (lot/wafer/X/Y)",
}


def scenario_defaults(scenario_id: str) -> dict:
    if scenario_id not in STORY1_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_id}")
    return dict(STORY1_SCENARIOS[scenario_id])


def apply_scenario_to_request(req: Any, scenario_id: str) -> Any:
    """Mutate / overlay a WaferGenRequest for the chosen Story 1 scenario."""
    spec = scenario_defaults(scenario_id)
    req.story_id = "story1"
    req.story1_scenario = scenario_id

    # Knobs from scenario when not already overridden
    if getattr(req, "blank_ecid_pct", None) is None:
        req.blank_ecid_pct = float(spec.get("blank_ecid_pct", 0.0))
    if getattr(req, "valid_ecid_mix", None) is None:
        req.valid_ecid_mix = float(spec.get("valid_ecid_mix", 0.5))
    if getattr(req, "mispick_ft_fail_pct", None) is None:
        req.mispick_ft_fail_pct = float(spec.get("mispick_ft_fail_pct", 1.0))

    gdpw = spec.get("gdpw_target")
    if gdpw and gdpw in GDPW_DIE_PRESETS:
        w, h = GDPW_DIE_PRESETS[gdpw]
        req.die_width = w
        req.die_height = h
        req.diameter = 300.0

    if spec.get("mode") == "sweeper":
        min_lots = int(spec.get("min_cp_lots", 2))
        if int(getattr(req, "num_lots", 1)) < min_lots:
            req.num_lots = min_lots
            req.auto_lot_id = True
    elif spec.get("mode") == "one_to_one":
        # Azure often misreads "CP lot + FT lot" as num_lots=2. 1:1 assembly
        # only consumes the first CP lot, so force a single CP lot.
        req.num_lots = 1

    # Wrong-bin / wrong-XY demos need enough fails and ~50% yield horror case
    if scenario_id.startswith("wrong_xy_horror") or scenario_id == "wrong_bin":
        if getattr(req, "yield_mode", "signature") == "signature":
            req.yield_mode = "direct"
            req.target_yield_pct = 50.0

    return req


def apply_gdbn_scenario_to_request(req: Any, scenario_id: str) -> Any:
    """Mutate / overlay a WaferGenRequest for a Story 1g (GDBN) scenario."""
    if scenario_id not in GDBN_SCENARIOS:
        raise ValueError(f"Unknown GDBN scenario: {scenario_id}")
    spec = GDBN_SCENARIOS[scenario_id]
    req.story_id = "story1_gdbn"
    req.gdbn_scenario = scenario_id
    req.num_lots = 1
    # CP-side signature per spec 1.g: dramatic = clean CP, neighbor = scratch.
    req.signatures = [spec["cp_signature"]]
    req.signature = spec["cp_signature"]
    if scenario_id == "gdbn_dramatic" and getattr(req, "yield_mode", "signature") == "signature":
        # "No pattern" already gets a 93-97% baseline; nothing else needed --
        # the donut-shaped fallout is layered on at FT, not at CP.
        pass
    return req


def apply_multidie_scenario_to_request(req: Any, scenario_id: str) -> Any:
    """Mutate / overlay a WaferGenRequest for a Story 1c (multi-die) scenario."""
    if scenario_id not in MULTIDIE_MODES:
        raise ValueError(f"Unknown multi-die scenario: {scenario_id}")
    req.story_id = "story1_multidie"
    req.multidie_mode = scenario_id
    return req
