"""
Tests for Story 1c: multi-die product traceability (Case B of the 2x2
matrix -- multiple die at CP, one packaged product at FT).

Run:  py -m pytest tests/test_story1_multidie.py -v
"""
from __future__ import annotations

from llm_agent import WaferGenRequest, request_to_config
from generator import generate
from story1_presets import apply_multidie_scenario_to_request
from multidie import COMPONENT_ROLE_NAMES


def _gen_multidie(scenario: str, **overrides) -> object:
    req = WaferGenRequest(
        num_wafers=overrides.pop("num_wafers", 3),
        test_count=0,
        multi_site=False,
        lot_id=overrides.pop("lot_id", "MDLOT1"),
        program="MULTIDIE_TEST",
    )
    apply_multidie_scenario_to_request(req, scenario)
    for k, v in overrides.items():
        setattr(req, k, v)
    config = request_to_config(req)
    return generate(req, config)


def test_multidie_df_has_one_row_per_product_with_3_components():
    result = _gen_multidie("full_trace", num_multidie_products=50)
    assert result.multidie_df is not None
    assert len(result.multidie_df) > 0
    for role in COMPONENT_ROLE_NAMES:
        prefix = role.capitalize()
        assert f"{prefix}CpLot" in result.multidie_df.columns
        assert f"{prefix}Ecid" in result.multidie_df.columns
        assert f"{prefix}FtPass" in result.multidie_df.columns


def test_full_trace_all_components_traceable():
    result = _gen_multidie("full_trace", num_multidie_products=50)
    df = result.multidie_df
    assert df["FullyTraceable"].all()
    for role in COMPONENT_ROLE_NAMES:
        assert (df[f"{role.capitalize()}Ecid"] != "").all()


def test_partial_trace_exactly_one_component_untraceable():
    result = _gen_multidie("partial_trace", num_multidie_products=50)
    df = result.multidie_df
    # Per spec 1.c.ii.3.b: 2 of 3 have traceability, one does not.
    assert not df["FullyTraceable"].any()
    traceable_roles = [
        role for role in COMPONENT_ROLE_NAMES
        if (df[f"{role.capitalize()}Ecid"] != "").all()
    ]
    untraceable_roles = [
        role for role in COMPONENT_ROLE_NAMES
        if (df[f"{role.capitalize()}Ecid"] == "").all()
    ]
    assert len(traceable_roles) == 2
    assert len(untraceable_roles) == 1


def test_product_ft_pass_is_and_of_all_components():
    result = _gen_multidie("full_trace", num_multidie_products=80,
                           baseline_ft_fallout=0.15)
    df = result.multidie_df
    expected = df["LogicFtPass"] & df["MemoryFtPass"] & df["RfFtPass"]
    assert (df["ProductFtPass"] == expected).all()
    # With a 15% per-component fallout across 3 components, at least one
    # product should fail overall (otherwise the AND logic isn't exercised).
    assert (df["ProductFtPass"] == 0).any()


def test_component_cp_lots_are_distinct_per_role():
    result = _gen_multidie("full_trace", num_multidie_products=20)
    df = result.multidie_df
    logic_lots = set(df["LogicCpLot"])
    memory_lots = set(df["MemoryCpLot"])
    rf_lots = set(df["RfCpLot"])
    assert logic_lots.isdisjoint(memory_lots)
    assert logic_lots.isdisjoint(rf_lots)
    assert memory_lots.isdisjoint(rf_lots)


def test_num_products_knob_is_respected():
    result = _gen_multidie("full_trace", num_multidie_products=37)
    # High per-component yields (85-92%) mean we should build close to the
    # requested count within the attempt cap.
    assert 30 <= len(result.multidie_df) <= 37


def test_ecid_mode_rot13_applies_to_components():
    plain = _gen_multidie("full_trace", num_multidie_products=10, ecid_mode="plain")
    rot13 = _gen_multidie("full_trace", num_multidie_products=10, ecid_mode="rot13")
    plain_ecids = set(plain.multidie_df["LogicEcid"])
    rot13_ecids = set(rot13.multidie_df["LogicEcid"])
    assert plain_ecids.isdisjoint(rot13_ecids)


def test_multidie_reproducible():
    a = _gen_multidie("partial_trace", lot_id="REPM1", num_multidie_products=15)
    b = _gen_multidie("partial_trace", lot_id="REPM1", num_multidie_products=15)
    assert list(a.multidie_df["ProductId"]) == list(b.multidie_df["ProductId"])
    assert list(a.multidie_df["ProductFtPass"]) == list(b.multidie_df["ProductFtPass"])
