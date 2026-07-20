"""
Tests for Story 1g: low yield at FT caused by CP clusters (GDBN /
missed-CP-cluster coverage).

Run:  py -m pytest tests/test_story1_gdbn.py -v
"""
from __future__ import annotations

from llm_agent import WaferGenRequest, request_to_config
from generator import generate
from story1_presets import apply_gdbn_scenario_to_request
from signatures import BIN_DEFINITIONS, PASS_BIN
from gdbn import donut_hazard_mask, neighbor_hazard_mask


def _is_cp_pass(bin_num: int) -> bool:
    return BIN_DEFINITIONS.get(bin_num, {}).get("state") == "P" or bin_num == PASS_BIN


def _gen_gdbn(scenario: str, **overrides) -> object:
    req = WaferGenRequest(
        num_wafers=overrides.pop("num_wafers", 3),
        test_count=0,
        multi_site=False,
        lot_id=overrides.pop("lot_id", "GDBNLOT1"),
        program="GDBN_TEST",
    )
    apply_gdbn_scenario_to_request(req, scenario)
    for k, v in overrides.items():
        setattr(req, k, v)
    config = request_to_config(req)
    return generate(req, config)


# ---------------------------------------------------------------------------
# Hazard mask geometry (pure functions, no generation needed)
# ---------------------------------------------------------------------------

def test_neighbor_hazard_mask_flags_only_passers_next_to_a_fail():
    # A 3x3 grid, centre die fails, all others pass.
    results = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            bin_num = 4 if (x, y) == (0, 0) else 1  # SCRATCH vs PASS
            results.append((x, y, float(x), float(y), bin_num))
    mask = neighbor_hazard_mask(results, growth=1)
    # The failing die itself is never entered in the mask (only passers are).
    assert mask.get((0, 0), False) is False
    for x, y in ((-1, -1), (1, 1), (0, 1), (1, 0)):
        assert mask[(x, y)] is True
    # Corner far away in a bigger grid would be False, but our grid only has
    # immediate neighbours of the centre, so nothing else to check here.


def test_neighbor_hazard_mask_growth_zero_flags_nothing():
    results = [(-1, 0, -1.0, 0.0, 4), (0, 0, 0.0, 0.0, 1), (1, 0, 1.0, 0.0, 1)]
    mask = neighbor_hazard_mask(results, growth=0)
    assert not any(mask.values())


def test_donut_hazard_mask_flags_ring_band():
    radius = 100.0
    results = [
        (0, 0, 0.0, 0.0, 1),      # centre -- inside inner radius, not in ring
        (1, 0, 45.0, 0.0, 1),     # inside the 25-60% ring band
        (2, 0, 95.0, 0.0, 1),     # outside the ring band (near edge)
    ]
    mask = donut_hazard_mask(results, radius_mm=radius)
    assert mask[(0, 0)] is False
    assert mask[(1, 0)] is True
    assert mask[(2, 0)] is False


# ---------------------------------------------------------------------------
# End-to-end: gdbn_neighbor (good-die-bad-neighborhood)
# ---------------------------------------------------------------------------

def test_gdbn_neighbor_units_are_all_cp_passers():
    result = _gen_gdbn("gdbn_neighbor", num_wafers=2)
    assert result.ft_df is not None and len(result.ft_df) > 0
    for b in result.ft_df["CpBin"]:
        assert _is_cp_pass(b)
    # No mis-picks in this story -- units are always correctly assembled.
    assert (result.ft_df["PickError"] == "none").all()


def test_gdbn_neighbor_has_spatial_hazard_fails():
    result = _gen_gdbn("gdbn_neighbor", num_wafers=3)
    hazard_all = result.ft_df[result.ft_df["SpatialHazard"] == "gdbn_neighbor"]
    assert len(hazard_all) > 0, "Scratch CP pattern should create at least one neighbour"
    hazard_fails = result.ft_df[result.ft_df["FailReason"] == "gdbn_neighbor"]
    assert len(hazard_fails) > 0
    assert (hazard_fails["FtPass"] == 0).all()
    # Every hazard fail must be a subset of the flagged hazard population.
    assert set(hazard_fails["UnitId"]).issubset(set(hazard_all["UnitId"]))


def test_gdbn_neighbor_zero_growth_means_no_hazard():
    result = _gen_gdbn("gdbn_neighbor", num_wafers=2, gdbn_growth=0)
    hazard_all = result.ft_df[result.ft_df["SpatialHazard"] == "gdbn_neighbor"]
    assert len(hazard_all) == 0


def test_gdbn_fail_pct_knob_changes_hazard_fail_rate():
    low = _gen_gdbn("gdbn_neighbor", num_wafers=4, gdbn_fail_pct=0.05, gdbn_growth=1)
    high = _gen_gdbn("gdbn_neighbor", num_wafers=4, gdbn_fail_pct=0.95, gdbn_growth=1)

    def hazard_fail_rate(result):
        hazard_all = result.ft_df[result.ft_df["SpatialHazard"] == "gdbn_neighbor"]
        if len(hazard_all) == 0:
            return None
        return 1.0 - hazard_all["FtPass"].mean()

    low_rate = hazard_fail_rate(low)
    high_rate = hazard_fail_rate(high)
    assert low_rate is not None and high_rate is not None
    assert high_rate > low_rate


# ---------------------------------------------------------------------------
# End-to-end: gdbn_dramatic
# ---------------------------------------------------------------------------

def test_gdbn_dramatic_cp_yield_is_high_and_clean():
    result = _gen_gdbn("gdbn_dramatic", num_wafers=2)
    cp1 = result.lots[0].wafers[0].insertions["CP1"]
    passed = sum(1 for d in cp1 if _is_cp_pass(d[4]))
    cp_yield = passed / len(cp1)
    # "No pattern" baseline is 93-97%; allow some slack for small samples.
    assert cp_yield > 0.85


def test_gdbn_dramatic_has_ring_shaped_ft_only_fails_invisible_at_cp():
    result = _gen_gdbn("gdbn_dramatic", num_wafers=3)
    hazard_fails = result.ft_df[result.ft_df["FailReason"] == "gdbn_dramatic"]
    assert len(hazard_fails) > 0
    assert (hazard_fails["FtPass"] == 0).all()
    # Every one of these was a CP PASS -- the whole point of the dramatic case.
    for b in hazard_fails["CpBin"]:
        assert _is_cp_pass(b)


def test_gdbn_scenarios_reproducible():
    a = _gen_gdbn("gdbn_neighbor", lot_id="REPG1", num_wafers=2)
    b = _gen_gdbn("gdbn_neighbor", lot_id="REPG1", num_wafers=2)
    assert list(a.ft_df["ECID"]) == list(b.ft_df["ECID"])
    assert list(a.ft_df["FtPass"]) == list(b.ft_df["FtPass"])
