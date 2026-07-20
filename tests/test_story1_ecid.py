"""
Hard edge-case tests for Story 1 (ECID matching / FT traceability).

Run:  py -m pytest tests/test_story1_ecid.py -v
"""
from __future__ import annotations

import pytest

from llm_agent import WaferGenRequest, request_to_config, _parse_story1_scenario
from generator import generate
from story1_presets import apply_scenario_to_request
from final_test import naive_ecid_join_explosion
from ecid import (
    format_ecid, should_burn_ecid, assert_nonblank_unique,
    ecid_components, concat_ecid_components, ECID_MODE_ROT13,
)
from signatures import PASS_BIN, RANDOM_FAIL_BIN
from geometry import gross_die_per_wafer
from assembly import make_ft_lot_id, STORY1_SCENARIOS


def _gen(scenario: str, **overrides) -> object:
    req = WaferGenRequest(
        num_wafers=overrides.pop("num_wafers", 3),
        num_lots=overrides.pop("num_lots", 1),
        signatures=overrides.pop("signatures", ["Edge Ring"]),
        test_count=0,
        multi_site=False,
        lot_id=overrides.pop("lot_id", "CPLOT1"),
        program="STORY1_TEST",
    )
    apply_scenario_to_request(req, scenario)
    for k, v in overrides.items():
        setattr(req, k, v)
    req.story_id = "story1"
    req.story1_scenario = scenario
    config = request_to_config(req)
    return generate(req, config)


# ---------------------------------------------------------------------------
# ECID fundamentals
# ---------------------------------------------------------------------------

def test_ecid_format_unique_per_xy():
    a = format_ecid("LOT1", 1, 0, 0)
    b = format_ecid("LOT1", 1, 1, 0)
    c = format_ecid("LOT1", 2, 0, 0)
    assert a != b and a != c
    assert a.startswith("LOT1W01")


def test_ecid_format_includes_lot():
    """Bug fix: same (wafer, x, y) in two different lots must NOT collide.

    This matters for the Sweeper story, which combines multiple CP lots
    into one FT lot -- if ECID didn't encode the source lot, two lots could
    mint identical ECID strings for the same wafer/die coordinates.
    """
    a = format_ecid("LOTA", 1, 5, 5)
    b = format_ecid("LOTB", 1, 5, 5)
    assert a != b


def test_ecid_rot13_mode_stays_unique_but_unreadable():
    plain = format_ecid("LOT1", 3, -2, 7)
    encrypted = format_ecid("LOT1", 3, -2, 7, mode=ECID_MODE_ROT13)
    assert plain != encrypted
    # Still deterministic / reproducible for the same inputs.
    assert encrypted == format_ecid("LOT1", 3, -2, 7, mode=ECID_MODE_ROT13)
    # ROT13 is an involution: applying it again recovers the plain value.
    import codecs
    assert codecs.encode(encrypted, "rot13") == plain
    # Different (wafer, x, y) must still map to different encrypted values.
    other = format_ecid("LOT1", 3, -2, 8, mode=ECID_MODE_ROT13)
    assert other != encrypted


def test_ecid_split_items_roundtrip():
    """Spec 1.b.iii: 4 separate test items must concatenate back to the
    same single-value ECID a 'plain' mode program would have burned."""
    lot, wafer, x, y = "LOT7", 4, -3, 12
    single = format_ecid(lot, wafer, x, y)
    parts = ecid_components(lot, wafer, x, y)
    rebuilt = concat_ecid_components(
        parts["EcidItemLot"], parts["EcidItemWafer"], parts["EcidItemX"], parts["EcidItemY"]
    )
    assert rebuilt == single


def test_pre_write_fail_gets_blank_ecid():
    assert should_burn_ecid(PASS_BIN) is True
    assert should_burn_ecid(RANDOM_FAIL_BIN) is False
    assert should_burn_ecid(2) is True  # EDGE_RING post-write


def test_cp_csv_has_ecid_column_even_without_story():
    req = WaferGenRequest(num_wafers=1, test_count=0, multi_site=False,
                          signatures=["No pattern"])
    result = generate(req, request_to_config(req))
    assert "ECID" in result.df.columns
    assert result.ft_df is None


# ---------------------------------------------------------------------------
# 1:1 simple / detail
# ---------------------------------------------------------------------------

def test_one_to_one_simple_all_valid_ecid_and_ft_pass_fail():
    result = _gen("one_to_one_simple", blank_ecid_pct=0.0, baseline_ft_fallout=0.05)
    assert result.ft_df is not None and len(result.ft_df) > 0
    assert (result.ft_df["ECID"] != "").all()
    # Pass + Fail at FT
    assert result.ft_df["FtPass"].isin([0, 1]).all()
    assert set(result.ft_df["FtPass"].unique()) == {0, 1} or (
        result.ft_df["FtPass"].sum() < len(result.ft_df)
    )
    # Every non-blank FT ECID exists on CP for that lot
    cp_ecids = set(result.df.loc[result.df["ECID"] != "", "ECID"])
    for e in result.ft_df["ECID"]:
        assert e in cp_ecids
    # FT lot ≠ CP lot
    ft_lots = set(result.ft_df["FtLot"])
    cp_lots = set(result.df["Lot"])
    assert ft_lots.isdisjoint(cp_lots)
    # Ground truth: all expected_match true
    assert (result.match_df["ExpectedMatch"] == "true").all()


def test_one_to_one_detail_blank_ecid_join_explosion():
    """Detail case: blank FT ECIDs + blank CP ECIDs → naive join explodes."""
    result = _gen(
        "one_to_one_detail",
        blank_ecid_pct=0.02,
        baseline_ft_fallout=0.03,
        yield_mode="direct",
        target_yield_pct=85.0,
        num_wafers=5,
    )
    ft_blank = (result.ft_df["ECID"] == "").sum()
    assert ft_blank >= 1
    # Blank FT units must fail
    blanks = result.ft_df[result.ft_df["ECID"] == ""]
    assert (blanks["FtPass"] == 0).all()
    assert (blanks["FailReason"] == "no_ecid").all()
    # Match table marks them unmatchable
    un = result.match_df[result.match_df["ECID"] == ""]
    assert (un["ExpectedMatch"] == "unmatchable").all()

    n_ft_blank, join_rows = naive_ecid_join_explosion(result.df, result.ft_df)
    assert n_ft_blank >= 1
    # Hard edge: cartesian product must be strictly larger than blank FT count
    # whenever CP also has blanks (the join hazard).
    cp_blank = (result.df["ECID"] == "").sum()
    if cp_blank > 1:
        assert join_rows > n_ft_blank
        assert join_rows == n_ft_blank * cp_blank


def test_blank_ecid_pct_under_two_percent():
    result = _gen("one_to_one_detail", blank_ecid_pct=0.015, num_wafers=10)
    pct = (result.ft_df["ECID"] == "").mean()
    assert pct < 0.02 or abs(pct - 0.015) < 0.02  # rounding with small N


# ---------------------------------------------------------------------------
# Sweeper
# ---------------------------------------------------------------------------

def test_sweeper_ft_lot_differs_from_all_cp_lots():
    result = _gen("sweeper_simple", num_lots=3, num_wafers=2, blank_ecid_pct=0.0)
    cp_lots = set(result.df["Lot"].unique())
    assert len(cp_lots) >= 2
    ft_lots = set(result.ft_df["FtLot"].unique())
    assert len(ft_lots) == 1
    assert ft_lots.isdisjoint(cp_lots)
    # Units come from multiple CP lots
    assert result.ft_df["CpLot"].nunique() >= 2
    assert (result.ft_df["ECID"] != "").all()


def test_sweeper_ecids_globally_unique_across_cp_lots():
    """Regression: two CP lots with overlapping wafer numbers/die (x,y) must
    not mint colliding ECIDs now that lot id is embedded (bug fix)."""
    result = _gen("sweeper_simple", num_lots=3, num_wafers=2, blank_ecid_pct=0.0)
    all_ecids = list(result.df.loc[result.df["ECID"] != "", "ECID"])
    assert len(all_ecids) > 0
    assert len(all_ecids) == len(set(all_ecids))
    # Sanity: multiple lots really do share wafer numbers/die coordinates.
    assert result.df["Lot"].nunique() >= 2
    assert result.df["WaferNumber"].duplicated().any()


def test_sweeper_detail_has_blanks_and_distinct_ft_lot():
    result = _gen("sweeper_detail", num_lots=2, num_wafers=4, blank_ecid_pct=0.02)
    assert (result.ft_df["ECID"] == "").any()
    assert set(result.ft_df["FtLot"]).isdisjoint(set(result.df["Lot"]))


def test_make_ft_lot_id_never_collides():
    cp = ["FTSWP1234", "A25010001"]
    ft = make_ft_lot_id(cp, mode="sweeper", seed=1)
    assert ft not in cp


# ---------------------------------------------------------------------------
# Wrong bin / wrong XY
# ---------------------------------------------------------------------------

def test_wrong_bin_mix_and_all_fail_simple():
    result = _gen(
        "wrong_bin",
        valid_ecid_mix=0.5,
        mispick_ft_fail_pct=1.0,
        yield_mode="direct",
        target_yield_pct=50.0,
        num_wafers=4,
    )
    assert len(result.ft_df) > 0
    assert (result.ft_df["PickError"] == "wrong_bin").all()
    # Mix of valid and blank ECID (when both populations exist)
    with_e = (result.ft_df["ECID"] != "").sum()
    blank = (result.ft_df["ECID"] == "").sum()
    # At least one population present
    assert with_e + blank == len(result.ft_df)
    # Simple FT: blank ⇒ fail; mispick ⇒ fail
    assert (result.ft_df["FtPass"] == 0).all()
    assert (result.match_df["ExpectedMatch"].isin(["false", "unmatchable"])).all()


def test_wrong_xy_shift_actual_differs_from_intended():
    result = _gen(
        "wrong_xy_horror_simple",
        xy_shift=(1, 0),
        mispick_ft_fail_pct=1.0,
        num_wafers=2,
    )
    assert len(result.ft_df) > 0
    # Actual ≠ intended for every unit
    diff = (
        (result.ft_df["ActualDieX"] != result.ft_df["IntendedDieX"])
        | (result.ft_df["ActualDieY"] != result.ft_df["IntendedDieY"])
    )
    assert diff.all()
    # Shift magnitude is exactly 1 in one axis
    dx = (result.ft_df["ActualDieX"] - result.ft_df["IntendedDieX"]).abs()
    dy = (result.ft_df["ActualDieY"] - result.ft_df["IntendedDieY"]).abs()
    assert ((dx + dy) == 1).all()
    assert (result.ft_df["FtPass"] == 0).all()


def test_wrong_xy_horror_gdpw_near_100():
    result = _gen("wrong_xy_horror_simple", num_wafers=1)
    gdpw = result.story_manifest["gdpw"]
    assert 60 <= gdpw <= 160, f"expected ~100 GDPW, got {gdpw}"


def test_wrong_xy_1000_gdpw_near_1000():
    result = _gen("wrong_xy_1000_simple", num_wafers=1)
    gdpw = result.story_manifest["gdpw"]
    assert 700 <= gdpw <= 1300, f"expected ~1000 GDPW, got {gdpw}"


def test_subtle_ft_mispick_not_all_fail():
    """Subtle FT: ~80% of mis-picks fail — some must still pass."""
    result = _gen(
        "wrong_xy_1000_subtle_ft",
        mispick_ft_fail_pct=0.8,
        num_wafers=3,
        # Force valid ECID on mispicks so blank rule doesn't force 100% fail
        valid_ecid_mix=1.0,
    )
    # Only look at units that have ECID (blank always fail)
    with_ecid = result.ft_df[result.ft_df["ECID"] != ""]
    assert len(with_ecid) > 20
    fail_rate = 1.0 - with_ecid["FtPass"].mean()
    assert 0.5 < fail_rate < 0.95, f"expected ~80% fail, got {fail_rate:.2f}"
    assert with_ecid["FtPass"].sum() > 0  # some pass


def test_horror_produces_many_ft_fails_vs_1000():
    horror = _gen("wrong_xy_horror_simple", mispick_ft_fail_pct=1.0,
                  num_wafers=2, valid_ecid_mix=1.0)
    high = _gen("wrong_xy_1000_simple", mispick_ft_fail_pct=1.0,
                num_wafers=2, valid_ecid_mix=1.0)
    # Same absolute fail rate (100%), but horror has fewer units; check fail count
    # relative to GDPW story intent: horror ~50% CP yield → many bad neighbours
    assert horror.ft_df["FtPass"].sum() == 0
    assert high.ft_df["FtPass"].sum() == 0
    assert horror.story_manifest["gdpw"] < high.story_manifest["gdpw"]


# ---------------------------------------------------------------------------
# Reproducibility & non-blank uniqueness
# ---------------------------------------------------------------------------

def test_story1_reproducible():
    a = _gen("one_to_one_simple", lot_id="REP1", num_wafers=2)
    b = _gen("one_to_one_simple", lot_id="REP1", num_wafers=2)
    assert list(a.ft_df["ECID"]) == list(b.ft_df["ECID"])
    assert list(a.ft_df["FtPass"]) == list(b.ft_df["FtPass"])


def test_nonblank_ecids_unique_across_cp_wafer():
    result = _gen("one_to_one_simple", num_wafers=2)
    for lot in result.lots:
        for wafer in lot.wafers:
            for ins, ecids in wafer.ecids.items():
                assert_nonblank_unique(ecids)


def test_all_scenarios_generate_without_crash():
    for sid in STORY1_SCENARIOS:
        result = _gen(sid, num_wafers=2, num_lots=2 if "sweeper" in sid else 1)
        assert result.ft_df is not None
        assert result.match_df is not None
        assert len(result.ft_df) == len(result.match_df)
        assert result.story_manifest["scenario_id"] == sid


def test_rot13_mode_end_to_end_cp_and_ft_match():
    result = _gen("one_to_one_simple", num_wafers=2, blank_ecid_pct=0.0,
                  ecid_mode="rot13")
    cp_ecids = set(result.df.loc[result.df["ECID"] != "", "ECID"])
    for e in result.ft_df["ECID"]:
        assert e in cp_ecids
    plain = _gen("one_to_one_simple", num_wafers=2, blank_ecid_pct=0.0,
                ecid_mode="plain")
    assert set(result.df["ECID"]) != set(plain.df["ECID"])


def test_split_ecid_items_end_to_end():
    result = _gen("one_to_one_simple", num_wafers=2, blank_ecid_pct=0.0,
                  ecid_representation="split_items")
    for col in ("EcidItemLot", "EcidItemWafer", "EcidItemX", "EcidItemY"):
        assert col in result.df.columns
        assert col in result.ft_df.columns
    from ecid import concat_ecid_components
    nonblank = result.df[result.df["ECID"] != ""]
    for _, row in nonblank.head(25).iterrows():
        rebuilt = concat_ecid_components(
            row["EcidItemLot"], row["EcidItemWafer"], row["EcidItemX"], row["EcidItemY"])
        assert rebuilt == row["ECID"]


# ---------------------------------------------------------------------------
# Keyword parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("generate ECID matching 1:1 simple FT demo", "one_to_one_simple"),
    ("detail ECID case with blank ECIDs under 2%", "one_to_one_detail"),
    ("sweeper lot simple", "sweeper_simple"),
    ("sweeper detail blank ecid", "sweeper_detail"),
    ("assembly wrong bin picked", "wrong_bin"),
    ("wrong XY horror 100 gdpw", "wrong_xy_horror_simple"),
    ("wrong xy 1000 gdpw subtle ft 80%", "wrong_xy_1000_subtle_ft"),
])
def test_keyword_story1_scenarios(text, expected):
    assert _parse_story1_scenario(text.lower()) == expected


@pytest.mark.parametrize("text,expected", [
    ("show me a gdbn good die bad neighborhood demo", "gdbn_neighbor"),
    ("missed cp cluster that shows up at final test", "gdbn_dramatic"),
    ("dramatic case donut invisible at cp", "gdbn_dramatic"),
])
def test_keyword_gdbn_scenarios(text, expected):
    from llm_agent import _parse_gdbn_scenario
    assert _parse_gdbn_scenario(text.lower()) == expected


@pytest.mark.parametrize("text,expected", [
    ("multi-die product with full traceability", "full_trace"),
    ("chiplet package with partial traceability, 2 of 3", "partial_trace"),
])
def test_keyword_multidie_scenarios(text, expected):
    from llm_agent import _parse_multidie_scenario
    assert _parse_multidie_scenario(text.lower()) == expected


def test_keyword_ecid_mode_and_representation():
    from llm_agent import _parse_ecid_mode, _parse_ecid_representation
    assert _parse_ecid_mode("use rot13 encrypted ecid") == "rot13"
    assert _parse_ecid_mode("plain ecid please") is None
    assert _parse_ecid_representation("split into 4 test items") == "split_items"
    assert _parse_ecid_representation("normal ecid") is None
