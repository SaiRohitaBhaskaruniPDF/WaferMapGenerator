"""
Minimal STDF (Standard Test Data Format) writer.

Produces a valid STDF v4 binary file containing:
  FAR  – File Attributes Record
  MIR  – Master Information Record
  per wafer:
    WIR  – Wafer Information Record (start)
    per die:
      PIR  – Part Information Record
      PRR  – Part Results Record
    WRR  – Wafer Results Record (end)
  MRR  – Master Results Record

Reference: STDF Specification Version 4 (Teradyne).

All multi-byte integers are little-endian (CPU type 2 = x86).
"""
from __future__ import annotations

import struct
import time
from datetime import datetime
from typing import List

from signatures import BIN_DEFINITIONS, DieResult


# ---------------------------------------------------------------------------
# Low-level record helpers
# ---------------------------------------------------------------------------

def _u1(v: int) -> bytes:
    return struct.pack("<B", v & 0xFF)

def _u2(v: int) -> bytes:
    return struct.pack("<H", v & 0xFFFF)

def _u4(v: int) -> bytes:
    return struct.pack("<I", v & 0xFFFFFFFF)

def _i2(v: int) -> bytes:
    return struct.pack("<h", v)

def _cn(s: str, max_len: int = 255) -> bytes:
    """STDF Cn string: 1-byte length prefix + ASCII data."""
    encoded = s[:max_len].encode("ascii", errors="replace")
    return bytes([len(encoded)]) + encoded

def _timestamp() -> bytes:
    """4-byte Unix timestamp (U4)."""
    return _u4(int(time.time()))

def _record(rec_typ: int, rec_sub: int, body: bytes) -> bytes:
    """Wrap body bytes in an STDF record header."""
    length = len(body)
    header = _u2(length) + _u1(rec_typ) + _u1(rec_sub)
    return header + body


# ---------------------------------------------------------------------------
# Individual record builders
# ---------------------------------------------------------------------------

def _far() -> bytes:
    """FAR — File Attributes Record (REC_TYP=0, REC_SUB=10)."""
    body = _u1(2)  # CPU_TYPE = 2 (little-endian x86)
    body += _u1(4) # STDF_VER = 4
    return _record(0, 10, body)


def _mir(lot_id: str, program: str, wafer_count: int) -> bytes:
    """MIR — Master Information Record (REC_TYP=1, REC_SUB=10)."""
    ts = _timestamp()
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
    body += _cn("WAFER")  # TEST_COD
    body += _cn("")        # TST_TEMP
    body += _cn("")        # USER_TXT
    body += _cn("")        # AUX_FILE
    body += _cn("")        # PKG_TYP
    body += _cn("")        # FAMLY_ID
    body += _cn("")        # DATE_COD
    body += _cn("")        # FACIL_ID
    body += _cn("")        # FLOOR_ID
    body += _cn("SyntheticFab") # PROC_ID
    return _record(1, 10, body)


def _wir(wafer_id: str, head_num: int = 1) -> bytes:
    """WIR — Wafer Information Record (REC_TYP=2, REC_SUB=10)."""
    body  = _u1(head_num)  # HEAD_NUM
    body += _u1(255)        # SITE_GRP
    body += _timestamp()    # START_T
    body += _cn(wafer_id)   # WAFER_ID
    return _record(2, 10, body)


def _wrr(wafer_id: str, head_num: int, total: int, passed: int) -> bytes:
    """WRR — Wafer Results Record (REC_TYP=2, REC_SUB=20)."""
    failed = total - passed
    body  = _u1(head_num)    # HEAD_NUM
    body += _u1(255)          # SITE_GRP
    body += _timestamp()      # FINISH_T
    body += _u4(total)        # PART_CNT
    body += _u4(0xFFFFFFFF)   # RTST_CNT (missing)
    body += _u4(0xFFFFFFFF)   # ABRT_CNT (missing)
    body += _u4(passed)       # GOOD_CNT
    body += _u4(failed)       # FUNC_CNT
    body += _cn(wafer_id)     # WAFER_ID
    return _record(2, 20, body)


def _pir(head_num: int = 1, site_num: int = 1) -> bytes:
    """PIR — Part Information Record (REC_TYP=5, REC_SUB=10)."""
    body  = _u1(head_num)
    body += _u1(site_num)
    return _record(5, 10, body)


def _prr(head_num: int, site_num: int, pass_fail: bool,
         hard_bin: int, soft_bin: int,
         die_x: int, die_y: int, part_id: str) -> bytes:
    """PRR — Part Results Record (REC_TYP=5, REC_SUB=20)."""
    # PART_FLG: bit 3 set = fail
    part_flg = 0x00 if pass_fail else 0x08
    body  = _u1(head_num)
    body += _u1(site_num)
    body += _u1(part_flg)    # PART_FLG
    body += _u2(0)            # NUM_TEST
    body += _u2(hard_bin)     # HARD_BIN
    body += _u2(soft_bin)     # SOFT_BIN
    body += _i2(die_x)        # X_COORD
    body += _i2(die_y)        # Y_COORD
    body += _u4(0)            # TEST_T  (elapsed ms)
    body += _cn(part_id)      # PART_ID
    body += _cn("")            # PART_TXT
    body += bytes([0])         # PART_FIX (empty B*n)
    return _record(5, 20, body)


def _mrr() -> bytes:
    """MRR — Master Results Record (REC_TYP=1, REC_SUB=20)."""
    body  = _timestamp()    # FINISH_T
    body += _u1(ord(" "))   # DISP_COD
    body += _cn("")          # USR_DESC
    body += _cn("")          # EXC_DESC
    return _record(1, 20, body)


def _sbr(hard_bin: int, pass_flag: bool, count: int, name: str) -> bytes:
    """SBR — Software Bin Record (REC_TYP=1, REC_SUB=50)."""
    body  = _u1(255)        # HEAD_NUM (summarised)
    body += _u1(255)        # SITE_NUM
    body += _u2(hard_bin)   # SBIN_NUM
    body += _u4(count)      # SBIN_CNT
    body += _u1(ord("P") if pass_flag else ord("F"))  # SBIN_PF
    body += _cn(name)       # SBIN_NAM
    return _record(1, 50, body)


def _hbr(hard_bin: int, pass_flag: bool, count: int, name: str) -> bytes:
    """HBR — Hard Bin Record (REC_TYP=1, REC_SUB=40)."""
    body  = _u1(255)
    body += _u1(255)
    body += _u2(hard_bin)
    body += _u4(count)
    body += _u1(ord("P") if pass_flag else ord("F"))
    body += _cn(name)
    return _record(1, 40, body)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_stdf(
    lot_id: str,
    program: str,
    wafer_ids: List[str],
    all_wafers: List[List[DieResult]],
) -> bytes:
    """
    Build a complete STDF binary blob for the given wafer lot.

    Parameters
    ----------
    lot_id      : e.g. 'LOT_001'
    program     : e.g. 'HBN_PRD020'
    wafer_ids   : list of wafer ID strings (one per wafer)
    all_wafers  : list of wafer die-result lists from apply_signature()

    Returns
    -------
    bytes  — complete STDF file content ready to write to disk.
    """
    buf = bytearray()

    buf += _far()
    buf += _mir(lot_id, program, len(all_wafers))

    # Accumulate bin counts for summary records
    global_bin_counts: dict[int, int] = {}

    for w_idx, (wafer_id, wafer_dies) in enumerate(zip(wafer_ids, all_wafers)):
        head_num = 1
        buf += _wir(wafer_id, head_num)

        passed_count = 0
        wafer_bin_counts: dict[int, int] = {}

        for part_idx, die in enumerate(wafer_dies):
            die_x, die_y, _cx, _cy, bin_num = die
            info      = BIN_DEFINITIONS.get(bin_num, BIN_DEFINITIONS[5])
            is_pass   = info["state"] == "P"
            part_id   = f"{wafer_id}_X{die_x}_Y{die_y}"

            buf += _pir(head_num, 1)
            buf += _prr(
                head_num=head_num,
                site_num=1,
                pass_fail=is_pass,
                hard_bin=bin_num,
                soft_bin=bin_num,
                die_x=die_x,
                die_y=die_y,
                part_id=part_id,
            )

            if is_pass:
                passed_count += 1
            wafer_bin_counts[bin_num] = wafer_bin_counts.get(bin_num, 0) + 1
            global_bin_counts[bin_num] = global_bin_counts.get(bin_num, 0) + 1

        buf += _wrr(wafer_id, head_num, len(wafer_dies), passed_count)

    # Summary bin records (once at end)
    for bin_num, count in sorted(global_bin_counts.items()):
        info = BIN_DEFINITIONS.get(bin_num, {})
        is_p = info.get("state", "F") == "P"
        name = info.get("name", f"BIN{bin_num}")
        buf += _hbr(bin_num, is_p, count, name)
        buf += _sbr(bin_num, is_p, count, name)

    buf += _mrr()
    return bytes(buf)
