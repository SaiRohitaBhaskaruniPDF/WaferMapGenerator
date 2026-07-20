"""
ECID burn / blank rules for Story 1 (CP -> FT traceability demos).

Valid ECIDs are unique among non-blank values, GLOBALLY (across every lot in
a generation run) -- not just within one wafer. Blank ECIDs are the empty
string intentionally -- detail scenarios use that to demonstrate the
cartesian-join hazard when tools join CP to FT on ECID alone.

Encoding variants (spec 1.b): a real ECID is a single burned value, but
customers implement it a few different ways:
  i.   Single concatenated value: (fab lot)(wafer #)(x)(y)          -- "plain"
  ii.  Same value, lightly "encrypted" so it still joins uniquely but no
       longer maps visually onto a CP map                          -- "rot13"
  iii. Same 4 components, but reported as 4 SEPARATE test items that must be
       concatenated back together before a join will work           -- see
       `ecid_components()` / `concat_ecid_components()` below.
"""
from __future__ import annotations

import codecs
from typing import Dict, Iterable, List, Sequence, Tuple

from signatures import BIN_DEFINITIONS, PASS_BIN, RANDOM_FAIL_BIN, DieResult

# Fail bins that typically happen BEFORE ECID is written (power short, etc.).
# These dies keep a blank ECID even though they appear on the CP wafer map.
PRE_ECID_WRITE_FAIL_BINS = frozenset({
    RANDOM_FAIL_BIN,  # 5 -- particulate / early random / short proxy
    13,               # LOW_YIELD -- early random kills from yield model
})

# Everything else that fails is treated as post-write (e.g. Idd) and keeps ECID.

# ---------------------------------------------------------------------------
# Encoding modes (spec 1.b.i / 1.b.ii)
# ---------------------------------------------------------------------------
ECID_MODE_PLAIN = "plain"
ECID_MODE_ROT13 = "rot13"
ECID_MODES = (ECID_MODE_PLAIN, ECID_MODE_ROT13)


def sanitize_lot_token(lot_id: str) -> str:
    """Alnum-only token for a lot id, so it is safe to embed in an ECID."""
    token = "".join(ch for ch in str(lot_id) if ch.isalnum())
    return token or "LOT"


def _encode(raw: str, mode: str) -> str:
    if mode == ECID_MODE_ROT13:
        # Real customer encryption is much stronger; ROT13 is intentionally
        # "good enough for demo purposes" per the spec -- it keeps the value
        # unique (so DB joins still work) while making it unreadable as a
        # CP (lot, wafer, x, y) map coordinate.
        return codecs.encode(raw, "rot13")
    return raw


def format_ecid(
    lot_id: str,
    wafer_number: int,
    die_x: int,
    die_y: int,
    mode: str = ECID_MODE_PLAIN,
) -> str:
    """Stable ECID unique per (lot, wafer, x, y) -- globally, not just per wafer.

    Concatenated form matches spec 1.b.i: (fab lot)(wafer #)(x)(y).
    """
    raw = (
        f"{sanitize_lot_token(lot_id)}"
        f"W{wafer_number:02d}"
        f"X{die_x:+04d}"
        f"Y{die_y:+04d}"
    )
    return _encode(raw, mode)


def should_burn_ecid(bin_num: int) -> bool:
    """True if this CP bin result should carry a valid burned ECID."""
    info = BIN_DEFINITIONS.get(bin_num, {})
    if info.get("state") == "P" or bin_num == PASS_BIN:
        return True
    if bin_num in PRE_ECID_WRITE_FAIL_BINS:
        return False
    # Post-write fail (signature / CP2 / CP3 / S2S / etc.)
    return info.get("state") == "F"


def assign_ecids_for_wafer(
    die_results: Sequence[DieResult],
    lot_id: str,
    wafer_number: int,
    mode: str = ECID_MODE_PLAIN,
) -> List[str]:
    """Return one ECID (or "") per die, same order as die_results."""
    out: List[str] = []
    for die_x, die_y, _cx, _cy, bin_num in die_results:
        if should_burn_ecid(bin_num):
            out.append(format_ecid(lot_id, wafer_number, die_x, die_y, mode=mode))
        else:
            out.append("")
    return out


def build_ecid_lookup(
    lot_id: str,
    wafer_number: int,
    die_results: Sequence[DieResult],
    mode: str = ECID_MODE_PLAIN,
) -> Dict[Tuple[int, int], str]:
    """Map (dieX, dieY) -> ECID for one wafer (blanks included as "")."""
    ecids = assign_ecids_for_wafer(die_results, lot_id, wafer_number, mode=mode)
    lookup: Dict[Tuple[int, int], str] = {}
    for (die_x, die_y, *_rest), ecid in zip(die_results, ecids):
        lookup[(die_x, die_y)] = ecid
    return lookup


def assert_nonblank_unique(ecids: Iterable[str]) -> None:
    """Raise if any non-blank ECID is duplicated."""
    seen = set()
    for e in ecids:
        if not e:
            continue
        if e in seen:
            raise ValueError(f"Duplicate non-blank ECID: {e}")
        seen.add(e)


# ---------------------------------------------------------------------------
# Split-into-test-items representation (spec 1.b.iii)
# ---------------------------------------------------------------------------
# Some customers (e.g. Analog Devices, per the spec) don't burn/read ECID as
# one string -- they read 4 separate test items (lot #, wafer #, X, Y) that
# have to be concatenated back into a single value before a CP<->FT join
# works. These helpers let the exporter add that representation alongside
# (or instead of) the convenience single-value ECID column.

ECID_ITEM_COLUMNS = ("EcidItemLot", "EcidItemWafer", "EcidItemX", "EcidItemY")


def ecid_components(lot_id: str, wafer_number: int, die_x: int, die_y: int) -> Dict[str, object]:
    """The 4 independent 'test items' a split-representation program reports."""
    return {
        "EcidItemLot": sanitize_lot_token(lot_id),
        "EcidItemWafer": int(wafer_number),
        "EcidItemX": int(die_x),
        "EcidItemY": int(die_y),
    }


def blank_ecid_components() -> Dict[str, object]:
    """The 4 test items when there is no ECID at all (nothing was written)."""
    return {col: "" for col in ECID_ITEM_COLUMNS}


def concat_ecid_components(
    lot_token: str, wafer_number: object, die_x: object, die_y: object,
) -> str:
    """Reassemble the single-value (plain-mode) ECID from its 4 test items.

    This is the join step a customer using the split-item method (1.b.iii)
    must perform themselves before CP<->FT ECID matching will work.
    """
    return (
        f"{sanitize_lot_token(lot_token)}"
        f"W{int(wafer_number):02d}"
        f"X{int(die_x):+04d}"
        f"Y{int(die_y):+04d}"
    )
