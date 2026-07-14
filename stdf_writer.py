"""
STDF v4 binary writer — the "Exensio-ready" export.

Produces one valid STDF file PER LOT PER INSERTION (a real sort run writes
one file per insertion), containing:

  FAR  – File Attributes Record
  MIR  – Master Information Record  (lot, program, insertion, temperature)
  SDR  – Site Description Record    (multi-site parallelism, when > 1 site)
  per wafer:
    WIR  – Wafer Information Record (start, real timestamp)
    per die:
      PIR  – Part Information Record  (head + probe site)
      PTR* – Parametric Test Records  (optional, one per test item)
      PRR  – Part Results Record      (hard/soft bin, coords, test time)
    WRR  – Wafer Results Record (finish timestamp, pass/fail counts)
  TSR* – Test Synopsis Records (per test item, when test data is included)
  HBR/SBR – hard/soft bin summaries using the MAPPED bin numbers
  MRR  – Master Results Record

Reference: STDF Specification Version 4 (Teradyne).
All multi-byte integers are little-endian (CPU type 2 = x86).
"""
from __future__ import annotations

import struct
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from signatures import BIN_DEFINITIONS, DieResult, PASS_BIN
from binning import build_bin_map, map_wafer_bins, bin_name
from test_items import TestPlan, generate_die_results
from fab import wafer_test_seconds


# ---------------------------------------------------------------------------
# Low-level field encoders
# ---------------------------------------------------------------------------
# STDF is a strict binary format: every field must be packed to an exact byte
# size. These helpers do the low-level encoding:
#   _u1/_u2/_u4 : unsigned ints of 1/2/4 bytes      _i2 : signed 2-byte int
#   _r4         : 4-byte IEEE float                 _cn : length-prefixed str
#   _record     : wrap a record body with its (length, type, subtype) header

def _u1(v: int) -> bytes:
    return struct.pack("<B", v & 0xFF)

def _u2(v: int) -> bytes:
    return struct.pack("<H", v & 0xFFFF)

def _u4(v: int) -> bytes:
    return struct.pack("<I", v & 0xFFFFFFFF)

def _i2(v: int) -> bytes:
    return struct.pack("<h", v)

def _i1(v: int) -> bytes:
    return struct.pack("<b", v)

def _r4(v: float) -> bytes:
    return struct.pack("<f", v)

def _cn(s: str, max_len: int = 255) -> bytes:
    """STDF Cn string: 1-byte length prefix + ASCII data."""
    encoded = s[:max_len].encode("ascii", errors="replace")
    return bytes([len(encoded)]) + encoded

def _ts(dt: Optional[datetime] = None) -> bytes:
    """4-byte Unix timestamp (U4) from a datetime (default: now)."""
    stamp = dt.timestamp() if dt else time.time()
    return _u4(int(stamp))

def _record(rec_typ: int, rec_sub: int, body: bytes) -> bytes:
    """Wrap body bytes in an STDF record header."""
    return _u2(len(body)) + _u1(rec_typ) + _u1(rec_sub) + body


# ---------------------------------------------------------------------------
# Individual record builders
# ---------------------------------------------------------------------------

def _far() -> bytes:
    """FAR — File Attributes Record (REC_TYP=0, REC_SUB=10)."""
    return _record(0, 10, _u1(2) + _u1(4))  # CPU_TYPE=2 (x86 LE), STDF_VER=4


def _mir(lot_id: str, program: str, insertion: str, temperature: str,
         start: Optional[datetime]) -> bytes:
    """MIR — Master Information Record (REC_TYP=1, REC_SUB=10).

    TEST_COD carries the insertion name (CP1/CP2/CP3) and TST_TEMP the
    nominal temperature, which is how downstream tools tell the files of a
    multi-insertion flow apart.
    """
    ts = _ts(start)
    body  = ts            # SETUP_T
    body += ts            # START_T
    body += _u1(1)        # STAT_NUM (station)
    body += _u1(ord(" ")) # MODE_COD
    body += _u1(ord(" ")) # RTST_COD
    body += _u1(ord(" ")) # PROT_COD
    body += _u2(0)        # BURN_TIM
    body += _u1(ord(" ")) # CMOD_COD
    body += _cn(lot_id)   # LOT_ID
    body += _cn(program)  # PART_TYP
    body += _cn("")        # NODE_NAM
    body += _cn("TP1")    # TSTR_TYP
    body += _cn("")        # JOB_NAM
    body += _cn("1.0")    # JOB_REV
    body += _cn("")        # SBLOT_ID
    body += _cn("SpatialSig-Bot") # OPER_NAM
    body += _cn("")        # EXEC_TYP
    body += _cn("")        # EXEC_VER
    body += _cn(insertion) # TEST_COD  (CP1 / CP2 / CP3)
    body += _cn(temperature) # TST_TEMP (e.g. 25C / -40C / 125C)
    body += _cn("")        # USER_TXT
    body += _cn("")        # AUX_FILE
    body += _cn("")        # PKG_TYP
    body += _cn("")        # FAMLY_ID
    body += _cn("")        # DATE_COD
    body += _cn("")        # FACIL_ID
    body += _cn("")        # FLOOR_ID
    body += _cn("SyntheticFab") # PROC_ID
    return _record(1, 10, body)


def _sdr(site_count: int) -> bytes:
    """SDR — Site Description Record (REC_TYP=1, REC_SUB=80).

    Declares the multi-site setup: how many sites are probed in parallel
    and their site numbers (1..N).
    """
    body  = _u1(1)              # HEAD_NUM
    body += _u1(1)              # SITE_GRP
    body += _u1(site_count)     # SITE_CNT
    for s in range(1, site_count + 1):
        body += _u1(s)          # SITE_NUM array
    return _record(1, 80, body)


def _wir(wafer_id: str, start: Optional[datetime], head_num: int = 1) -> bytes:
    """WIR — Wafer Information Record (REC_TYP=2, REC_SUB=10)."""
    body  = _u1(head_num)   # HEAD_NUM
    body += _u1(255)         # SITE_GRP
    body += _ts(start)       # START_T (real sort-start time for trend charts)
    body += _cn(wafer_id)    # WAFER_ID
    return _record(2, 10, body)


def _wrr(wafer_id: str, head_num: int, total: int, passed: int,
         finish: Optional[datetime]) -> bytes:
    """WRR — Wafer Results Record (REC_TYP=2, REC_SUB=20)."""
    body  = _u1(head_num)     # HEAD_NUM
    body += _u1(255)           # SITE_GRP
    body += _ts(finish)        # FINISH_T
    body += _u4(total)         # PART_CNT
    body += _u4(0xFFFFFFFF)    # RTST_CNT (missing)
    body += _u4(0xFFFFFFFF)    # ABRT_CNT (missing)
    body += _u4(passed)        # GOOD_CNT
    body += _u4(total - passed)  # FUNC_CNT
    body += _cn(wafer_id)      # WAFER_ID
    return _record(2, 20, body)


def _pir(head_num: int = 1, site_num: int = 1) -> bytes:
    """PIR — Part Information Record (REC_TYP=5, REC_SUB=10)."""
    return _record(5, 10, _u1(head_num) + _u1(site_num))


def _prr(head_num: int, site_num: int, pass_fail: bool,
         hard_bin: int, soft_bin: int,
         die_x: int, die_y: int, part_id: str,
         num_tests: int = 0, test_ms: int = 0) -> bytes:
    """PRR — Part Results Record (REC_TYP=5, REC_SUB=20).

    hard_bin / soft_bin are the MAPPED bins (binning.py), not the internal
    signature bins. test_ms carries the touchdown time in milliseconds.
    """
    part_flg = 0x00 if pass_fail else 0x08  # bit 3 set = part failed
    body  = _u1(head_num)
    body += _u1(site_num)
    body += _u1(part_flg)    # PART_FLG
    body += _u2(num_tests)   # NUM_TEST
    body += _u2(hard_bin)    # HARD_BIN
    body += _u2(soft_bin)    # SOFT_BIN
    body += _i2(die_x)       # X_COORD
    body += _i2(die_y)       # Y_COORD
    body += _u4(test_ms)     # TEST_T (elapsed test time, ms)
    body += _cn(part_id)     # PART_ID
    body += _cn("")           # PART_TXT
    body += bytes([0])        # PART_FIX (empty B*n)
    return _record(5, 20, body)


def _ptr(test_num: int, head: int, site: int, passed: bool, value: float,
         test_name: str = "", limits: Optional[Tuple[float, float]] = None) -> bytes:
    """PTR — Parametric Test Record (REC_TYP=15, REC_SUB=10).

    Used for BOTH kinds of spec test items:
      * parametric items report their shaped real value,
      * pass/fail items report literally 0.0 or 1.0.
    Limits (LO_LIMIT/HI_LIMIT) are emitted only on the FIRST occurrence of a
    test number; STDF readers cache them as defaults for later records,
    which keeps multi-million-record files small.
    """
    test_flg = 0x00 if passed else 0x80  # bit 7 set = test failed
    body  = _u4(test_num)   # TEST_NUM
    body += _u1(head)        # HEAD_NUM
    body += _u1(site)        # SITE_NUM
    body += _u1(test_flg)    # TEST_FLG
    body += _u1(0x00)        # PARM_FLG
    body += _r4(value)       # RESULT
    body += _cn(test_name)   # TEST_TXT (test item name)
    body += _cn("")           # ALARM_ID
    if limits is not None:
        lo, hi = limits
        body += _u1(0x00)    # OPT_FLAG (all limit fields valid)
        body += _i1(0)       # RES_SCAL
        body += _i1(0)       # LLM_SCAL
        body += _i1(0)       # HLM_SCAL
        body += _r4(lo)      # LO_LIMIT
        body += _r4(hi)      # HI_LIMIT
    return _record(15, 10, body)


def _tsr(test_num: int, exec_cnt: int, fail_cnt: int, test_name: str) -> bytes:
    """TSR — Test Synopsis Record (REC_TYP=10, REC_SUB=30)."""
    body  = _u1(255)          # HEAD_NUM (summary over all heads)
    body += _u1(255)          # SITE_NUM (summary over all sites)
    body += _u1(ord("P"))     # TEST_TYP ('P' = parametric)
    body += _u4(test_num)     # TEST_NUM
    body += _u4(exec_cnt)     # EXEC_CNT
    body += _u4(fail_cnt)     # FAIL_CNT
    body += _u4(0)            # ALRM_CNT
    body += _cn(test_name)    # TEST_NAM
    return _record(10, 30, body)


def _hbr(bin_num: int, pass_flag: bool, count: int, name: str) -> bytes:
    """HBR — Hard Bin Record (REC_TYP=1, REC_SUB=40)."""
    body  = _u1(255) + _u1(255)           # summary over heads/sites
    body += _u2(bin_num) + _u4(count)
    body += _u1(ord("P") if pass_flag else ord("F"))
    body += _cn(name)
    return _record(1, 40, body)


def _sbr(bin_num: int, pass_flag: bool, count: int, name: str) -> bytes:
    """SBR — Software Bin Record (REC_TYP=1, REC_SUB=50)."""
    body  = _u1(255) + _u1(255)
    body += _u2(bin_num) + _u4(count)
    body += _u1(ord("P") if pass_flag else ord("F"))
    body += _cn(name)
    return _record(1, 50, body)


def _mrr(finish: Optional[datetime]) -> bytes:
    """MRR — Master Results Record (REC_TYP=1, REC_SUB=20)."""
    body  = _ts(finish)      # FINISH_T
    body += _u1(ord(" "))    # DISP_COD
    body += _cn("") + _cn("")  # USR_DESC, EXC_DESC
    return _record(1, 20, body)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_stdf(
    lot_id: str,
    program: str,
    wafer_ids: List[str],
    all_wafers: List[List[DieResult]],
    *,
    insertion: str = "CP1",
    temperature: str = "25C",
    hard_bins: Optional[List[List[int]]] = None,
    soft_bins: Optional[List[List[int]]] = None,
    sites: Optional[List[List[int]]] = None,
    site_count: int = 1,
    start_time: Optional[datetime] = None,
    seconds_per_touchdown: float = 0.0,
    test_plan: Optional[TestPlan] = None,
    include_test_data: bool = False,
) -> bytes:
    """Build a complete STDF binary blob for ONE lot at ONE insertion.

    Parameters
    ----------
    lot_id / program / wafer_ids / all_wafers
        Same as before: identity strings plus per-wafer die-result lists
        (each die is (dieX, dieY, cx, cy, internal_bin)).
    insertion / temperature
        Which CP insertion this file represents; written into the MIR.
    hard_bins / soft_bins
        Mapped bin numbers, one list per wafer parallel to all_wafers.
        When omitted, a default 16/x4 mapping is built on the fly.
    sites / site_count
        Per-die probe-site numbers and total parallelism. site_count > 1
        also emits an SDR record.
    start_time / seconds_per_touchdown
        Lot sort-start and test time. Wafer i starts when wafer i-1 ends,
        so WIR/WRR carry a believable time series for trend charts.
    test_plan / include_test_data
        When include_test_data is True, one PTR is written per die per test
        item (can be huge — the app warns first), plus TSR summaries.

    Returns the STDF file content as bytes.
    """
    # Fall back to a default bin mapping when the caller didn't map bins.
    if hard_bins is None or soft_bins is None:
        default_map = build_bin_map(16, 4)
        hard_bins, soft_bins = [], []
        for wafer in all_wafers:
            hb, sb = map_wafer_bins(wafer, default_map)
            hard_bins.append(hb)
            soft_bins.append(sb)

    start_time = start_time or datetime.now()
    buf = bytearray()
    buf += _far()
    buf += _mir(lot_id, program, insertion, temperature, start_time)
    if site_count > 1:
        buf += _sdr(site_count)

    # Summary accumulators for the trailer records.
    hard_counts: Dict[int, Tuple[bool, int]] = {}   # hardbin -> (is_pass, count)
    soft_counts: Dict[int, Tuple[bool, int]] = {}
    hard_names: Dict[int, str] = {}
    soft_names: Dict[int, str] = {}
    test_exec: Dict[int, int] = {}                   # test_num -> executions
    test_fail: Dict[int, int] = {}
    limits_written: set = set()                       # test numbers with limits emitted

    head_num = 1
    touchdown_ms = int(seconds_per_touchdown * 1000)
    wafer_clock = start_time  # rolling wafer start time

    for w_idx, (wafer_id, wafer_dies) in enumerate(zip(wafer_ids, all_wafers)):
        buf += _wir(wafer_id, wafer_clock, head_num)

        passed_count = 0
        w_hard = hard_bins[w_idx]
        w_soft = soft_bins[w_idx]
        w_sites = sites[w_idx] if sites else [1] * len(wafer_dies)

        for die_idx, die in enumerate(wafer_dies):
            die_x, die_y, _cx, _cy, internal_bin = die
            info    = BIN_DEFINITIONS.get(internal_bin, BIN_DEFINITIONS[5])
            is_pass = info["state"] == "P"
            site    = w_sites[die_idx]
            part_id = f"{wafer_id}_X{die_x}_Y{die_y}"

            buf += _pir(head_num, site)

            # Optional per-test PTR records for this die.
            num_tests = 0
            if include_test_data and test_plan is not None:
                # Deterministic per-die seed -> identical data on re-export.
                die_seed = (w_idx * 100003) + die_idx
                for t_num, is_param, value, t_pass in generate_die_results(
                        test_plan, is_pass, die_seed):
                    name = test_plan.names[t_num - 1]
                    # Emit limits only the first time a test number appears.
                    limits = None
                    if t_num not in limits_written:
                        limits = test_plan.limits if is_param else (0.5, 1.5)
                        limits_written.add(t_num)
                    buf += _ptr(t_num, head_num, site, t_pass, value,
                                test_name=name, limits=limits)
                    test_exec[t_num] = test_exec.get(t_num, 0) + 1
                    if not t_pass:
                        test_fail[t_num] = test_fail.get(t_num, 0) + 1
                    num_tests += 1

            buf += _prr(
                head_num=head_num,
                site_num=site,
                pass_fail=is_pass,
                hard_bin=w_hard[die_idx],
                soft_bin=w_soft[die_idx],
                die_x=die_x,
                die_y=die_y,
                part_id=part_id,
                num_tests=num_tests,
                test_ms=touchdown_ms,
            )

            if is_pass:
                passed_count += 1
            # Accumulate bin summaries under the MAPPED numbers, remembering
            # a representative internal-bin name for each.
            hb, sb = w_hard[die_idx], w_soft[die_idx]
            hard_counts[hb] = (is_pass, hard_counts.get(hb, (is_pass, 0))[1] + 1)
            soft_counts[sb] = (is_pass, soft_counts.get(sb, (is_pass, 0))[1] + 1)
            hard_names.setdefault(hb, bin_name(internal_bin))
            soft_names.setdefault(sb, bin_name(internal_bin))

        # Wafer finish time: start + touchdowns x seconds/touchdown. The next
        # wafer starts where this one finished (a realistic serial sort flow).
        elapsed = wafer_test_seconds(len(wafer_dies), max(1, site_count),
                                     seconds_per_touchdown)
        wafer_finish = wafer_clock + timedelta(seconds=elapsed)
        buf += _wrr(wafer_id, head_num, len(wafer_dies), passed_count, wafer_finish)
        wafer_clock = wafer_finish

    # Trailer summaries: TSR per test item, then HBR/SBR per mapped bin.
    if include_test_data and test_plan is not None:
        for t_num in sorted(test_exec):
            buf += _tsr(t_num, test_exec[t_num], test_fail.get(t_num, 0),
                        test_plan.names[t_num - 1])

    for hb in sorted(hard_counts):
        is_p, count = hard_counts[hb]
        buf += _hbr(hb, is_p, count, hard_names[hb])
    for sb in sorted(soft_counts):
        is_p, count = soft_counts[sb]
        buf += _sbr(sb, is_p, count, soft_names[sb])

    buf += _mrr(wafer_clock)
    return bytes(buf)
