"""
LLM Agent for Wafer Map Generation.

Converts natural-language user prompts into structured WaferGenRequest
objects using Azure OpenAI (preferred) or OpenAI chat completions with
function calling. Falls back to a simple keyword parser when no API
credentials are available.

The WaferGenRequest carries EVERY parameter from the 2026.07.10 spec:
geometry, yield model, CP insertions, bin counts, test items, multi-site
and lot sequencing. request_to_config() turns the geometric part into a
validated WaferConfig; generator.generate() consumes the rest.

Environment variables (Azure — recommended):
  AZURE_OPENAI_API_KEY
  AZURE_OPENAI_ENDPOINT      e.g. https://your-resource.openai.azure.com/
  AZURE_OPENAI_DEPLOYMENT    your GPT-4.1 deployment name in Azure
  AZURE_OPENAI_API_VERSION   optional, default 2024-12-01-preview

Environment variables (OpenAI — optional fallback):
  OPENAI_API_KEY
  OPENAI_MODEL               optional, default gpt-4o-mini

Credentials can also be loaded from a `.env` file in the project root
(copy `.env.example` to `.env`).
"""
from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from geometry import (
    WaferConfig, STANDARD_DIAMETERS, auto_edge_type, snap_diameter,
    clamp_die_size, clamp_street, EDGE_EXCLUSION_MIN, EDGE_EXCLUSION_MAX,
    STREET_DEFAULT,
)
from signatures import SIGNATURE_NAMES, SCRATCH_FAMILIES, NO_PATTERN_SIGNATURE
from test_items import TEST_COUNT_CHOICES
from binning import HARDBIN_CHOICES, SOFTBIN_MULTIPLIERS
from fab import LOT_CADENCES

# Spec: a fab lot is at most 25 wafers (one standard FOUP).
MAX_WAFERS = 25
"""_clamp_wafers (lines 69–70) forces any request into 1–25. 
The generation loop honors the count at generator.py line 203 (for w in range(req.num_wafers))."""
# ---------------------------------------------------------------------------
# Lot-size presets
# ---------------------------------------------------------------------------
# In a real fab, wafers travel in a carrier called a FOUP (Front-Opening
# Unified Pod). The carrier's capacity sets the lot size:
#   - 25 wafers is the industry-standard FOUP (300 mm).
#   - 13 wafers is a special variant for thin / bonded wafers.
# Fabs often run PARTIAL lots (fewer than capacity), but essentially never
# more. The value is the wafer count; None means "custom number".
LOT_SIZE_PRESETS = {
    "Standard FOUP (25 wafers)": 25,
    "Thin/Bonded FOUP (13 wafers)": 13,
    "Partial Lot (custom count)": None,
}

# The four scratch-family signature names, kept as a set so we can reason
# about them (e.g. "if a specific family was detected, drop the generic").
_SCRATCH_FAMILY_NAMES = set(SCRATCH_FAMILIES.keys())


def _clamp_wafers(n: int) -> int:
    return max(1, min(MAX_WAFERS, int(n)))


def _load_env_file() -> None:
    """Load variables from project-root `.env` if present."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


_load_env_file()

# ---------------------------------------------------------------------------
# Data model returned by the agent
# ---------------------------------------------------------------------------

@dataclass
class WaferGenRequest:
    """Structured generation request parsed from natural language.

    Grouped to mirror the spec document. Every field has a sensible default
    so a bare request ("give me edge ring wafers") still produces a full,
    valid dataset.
    """
    # ---- Wafer geometry (spec "Must Have" section) -------------------------
    diameter: float = 300.0          # mm: 150 / 200 / 300
    edge_type: str = ""              # 'notch' | 'flat' | '' = auto from diameter
    edge_exclusion: float = 3.0      # mm, 1..10
    die_width: float = 10.0          # mm (1..25/35, aspect 1:2..2:1)
    die_height: float = 10.0         # mm
    x_offset: float = 0.0
    y_offset: float = 0.0
    street_width: float = STREET_DEFAULT   # mm scribe street, 0.05..0.2
    edge_orientation: str = "down"   # notch/flat position: down/up/left/right

    # ---- Stepping field / repeaters ----------------------------------------
    auto_reticle: bool = True        # spec: auto-generate the stepping field
    dies_per_reticle_x: int = 2      # used only when auto_reticle is False
    dies_per_reticle_y: int = 2
    reticle_fail_die_x: int = 0      # which field position repeats-fails
    reticle_fail_die_y: int = 0
    repeater_fail_rate: float = 1.0  # 1.0 = hard repeater, <1.0 = soft
    stripe_fail_rate: float = 1.0    # striping hardness (1.0 = 100% fail)

    # ---- Lot / generation ----------------------------------------------------
    lot_id: str = "LOT_001"
    program: str = "DEMO"
    num_wafers: int = 25             # sequential 1..25, max one FOUP
    num_lots: int = 1                # >1 = trend-chart lot sequence
    lot_cadence: str = "1 lot per week"  # spacing between lot start times
    fab_letter: str = "A"            # the F in FYYWWSSSS
    auto_lot_id: bool = False        # True = generate FYYWWSSSS lot numbers

    # ---- Yield model (spec "Yield" section) ---------------------------------
    yield_mode: str = "signature"    # 'signature' | 'direct' | 'defect_density'
    target_yield_pct: Optional[float] = None   # for 'direct' (0..100)
    defect_density: Optional[float] = None     # defects/cm² for Y = e^(-A*D)
    # ±% per-wafer jitter: each wafer varies its defect density (or direct
    # yield target) by a random amount inside this band. 0 = no variation.
    yield_variation_pct: float = 0.0

    # ---- Test insertions (spec "Test insertions" section) -------------------
    num_insertions: int = 1          # 1 = CP1, 2 = +CP2, 3 = +CP3

    # ---- Bins (spec "Number of bins" section) --------------------------------
    hardbin_count: int = 16          # 16 / 64 / 256
    softbin_multiplier: int = 4      # softbins = hardbins x4 / x16 / x64

    # ---- Test items (spec "Test items" section) ------------------------------
    test_count: int = 100            # 100 / 1000 / ... orders of magnitude
    parametric_pct: int = 50         # % parametric vs pass/fail, 10% steps
    value_shape: str = "uniform"     # uniform/exponential/quantized/signed/...
    naming_style: str = "simple"     # simple / obnoxious / chunked
    name_length: int = 31            # 31 / 63 / 127 / 255 for verbose styles
    include_test_data: bool = False  # write per-test PTRs (can be huge)

    # ---- Test time & multi-site (nice-to-have) -------------------------------
    seconds_per_touchdown: float = 1.0   # 1..600 s
    multi_site: bool = True          # auto site count from GDPW when True
    site_pattern: str = "block"      # side by side / top & bottom / block / ...
    s2s_enabled: bool = False        # site-to-site yield loss on/off
    s2s_healthy: bool = True         # True = all sites > 95%

    # ---- Signature(s) ---------------------------------------------------------
    # `signatures` is the source of truth: ordered list, index 0 wins on any
    # shared die. `signature` (singular) mirrors signatures[0] for legacy code.
    signatures: List[str] = field(default_factory=lambda: ["Edge Ring"])
    signature: str = "Edge Ring"

    # ---- Story 1: ECID matching / FT traceability ------------------------------
    # story_id: "none" keeps legacy CP-only behaviour.
    #   "story1"          -> assembly + FT (1:1 / sweeper / wrong-bin / wrong-xy)
    #   "story1_gdbn"      -> low yield at FT caused by CP clusters (spec 1.g)
    #   "story1_multidie"  -> multi-die product traceability, Case B (spec 1.c)
    story_id: str = "none"
    story1_scenario: str = "one_to_one_simple"
    blank_ecid_pct: Optional[float] = None   # None = use scenario default
    valid_ecid_mix: Optional[float] = None   # fraction of mis-picks with valid ECID
    mispick_ft_fail_pct: Optional[float] = None  # 1.0 simple, 0.8 subtle
    baseline_ft_fallout: float = 0.03        # FT fail rate on correctly picked units
    xy_shift: Optional[tuple] = None         # e.g. (1, 0); None = random ±1
    ship_insertion: str = ""                 # "" = last CP insertion

    # ECID encoding variants (spec 1.b)
    ecid_mode: str = "plain"                 # "plain" | "rot13"
    ecid_representation: str = "single"      # "single" | "split_items" (1.b.iii)

    # Story 1g: GDBN / missed-CP-cluster knobs
    gdbn_scenario: str = "gdbn_neighbor"      # "gdbn_neighbor" | "gdbn_dramatic"
    gdbn_growth: Optional[int] = None         # None = scenario default (1)
    gdbn_fail_pct: Optional[float] = None     # None = scenario default (0.5)

    # Story 1c: multi-die product (Case B) knobs
    multidie_mode: str = "full_trace"         # "full_trace" (B.1) | "partial_trace" (B.2)
    num_multidie_products: int = 0            # 0 = derive from num_wafers

    # Explanation the LLM provides (shown to user)
    explanation: str = ""

    # Whether the LLM was actually used (False = keyword fallback)
    used_llm: bool = False

    def __post_init__(self):
        """Keep the singular `signature` and the `signatures` list in sync,
        and resolve the auto edge type from the diameter."""
        if self.signatures:
            self.signature = self.signatures[0]
        elif self.signature:
            self.signatures = [self.signature]
        if not self.edge_type:
            # Spec auto-rule: 150 mm = flat, 200/300 mm = notch.
            self.edge_type = auto_edge_type(self.diameter)


def request_to_config(req: WaferGenRequest) -> WaferConfig:
    """Build a validated WaferConfig from a parsed generation request.

    All the spec geometry rules are enforced here so BOTH input paths (LLM
    and keyword parser) produce legal wafers: diameter snapped to 150/200/
    300, edge type auto-selected, die size clamped into the aspect-ratio
    box, street and edge exclusion clamped to their ranges.
    """
    diameter = snap_diameter(req.diameter)
    edge_type = req.edge_type or auto_edge_type(diameter)
    die_w, die_h = clamp_die_size(req.die_width, req.die_height)
    return WaferConfig(
        diameter=diameter,
        edge_type=edge_type,
        edge_exclusion=max(EDGE_EXCLUSION_MIN,
                           min(EDGE_EXCLUSION_MAX, req.edge_exclusion)),
        die_width=die_w,
        die_height=die_h,
        x_offset=req.x_offset,
        y_offset=req.y_offset,
        street_width=clamp_street(req.street_width),
        dies_per_reticle_x=req.dies_per_reticle_x,
        dies_per_reticle_y=req.dies_per_reticle_y,
        reticle_fail_die_x=req.reticle_fail_die_x,
        reticle_fail_die_y=req.reticle_fail_die_y,
        edge_orientation=req.edge_orientation,
        repeater_fail_rate=req.repeater_fail_rate,
        stripe_fail_rate=req.stripe_fail_rate,
    )


# ---------------------------------------------------------------------------
# JSON schema passed to the LLM as a function / tool
# ---------------------------------------------------------------------------

_FUNCTION_SCHEMA = {
    "name": "generate_wafer_maps",
    "description": (
        "Return structured parameters to generate synthetic semiconductor wafer maps "
        "based on the user's natural language request."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "diameter": {
                "type": "number",
                "enum": list(STANDARD_DIAMETERS),
                "description": "Wafer diameter in mm: 150, 200 or 300 only.",
            },
            "edge_type": {
                "type": "string",
                "enum": ["notch", "flat"],
                "description": (
                    "Wafer edge marker. Leave unset for the standard rule: "
                    "150 mm wafers have a flat, 200/300 mm wafers a notch."
                ),
            },
            "edge_exclusion": {
                "type": "number",
                "description": "Edge exclusion zone in mm (1-10).",
            },
            "die_width": {
                "type": "number",
                "description": "Die width in mm (1-35; aspect ratio 1:2..2:1).",
            },
            "die_height": {
                "type": "number",
                "description": "Die height in mm (1-35; aspect ratio 1:2..2:1).",
            },
            "street_width": {
                "type": "number",
                "description": (
                    "Scribe/street width between dies in mm (0.05-0.2). "
                    "Default 0.1 mm."
                ),
            },
            "edge_orientation": {
                "type": "string",
                "enum": ["down", "up", "left", "right"],
                "description": (
                    "Which side the notch/flat sits on (90-degree steps). "
                    "'down' = 6 o'clock (default). Phrases like 'notch up' -> up."
                ),
            },
            "lot_id": {
                "type": "string",
                "description": "Lot identifier string, e.g. 'LOT_A42'.",
            },
            "program": {
                "type": "string",
                "description": "Program / product name.",
            },
            "num_wafers": {
                "type": "integer",
                "description": (
                    f"Wafers per lot (1-{MAX_WAFERS}). 25 = standard full FOUP "
                    "(default), 13 = thin/bonded lot. Never more than 25."
                ),
            },
            "num_lots": {
                "type": "integer",
                "description": (
                    "Number of fab lots to generate (default 1). More than 1 "
                    "creates a time sequence of lots for trend charts."
                ),
            },
            "lot_cadence": {
                "type": "string",
                "enum": list(LOT_CADENCES.keys()),
                "description": "Spacing between lot start times when num_lots > 1.",
            },
            "yield_mode": {
                "type": "string",
                "enum": ["signature", "direct", "defect_density"],
                "description": (
                    "'direct' when the user gives a yield percentage, "
                    "'defect_density' when they give defects/cm2 (Y = e^(-A*D)), "
                    "'signature' to let the spatial pattern set the yield."
                ),
            },
            "target_yield_pct": {
                "type": "number",
                "description": "Target wafer yield in percent (yield_mode='direct').",
            },
            "defect_density": {
                "type": "number",
                "description": "Defect density in defects/cm2 (yield_mode='defect_density').",
            },
            "yield_variation_pct": {
                "type": "number",
                "description": (
                    "Per-wafer yield variation in ±percent (0-50). E.g. "
                    "'yield variation +/- 20% defect density' -> 20. Each "
                    "wafer jitters its defect density (or direct yield "
                    "target) randomly inside this band."
                ),
            },
            "num_insertions": {
                "type": "integer",
                "description": (
                    "Wafer sort insertions: 1 = CP1 only, 2 = CP1+CP2, 3 = "
                    "CP1+CP2+CP3. CP2/CP3 keep 90-99.9% of the prior passers."
                ),
            },
            "hardbin_count": {
                "type": "integer",
                "enum": list(HARDBIN_CHOICES),
                "description": "Number of hardbins: 16, 64 or 256.",
            },
            "softbin_multiplier": {
                "type": "integer",
                "enum": list(SOFTBIN_MULTIPLIERS),
                "description": "Softbins = hardbins x this factor (4, 16 or 64).",
            },
            "test_count": {
                "type": "integer",
                "enum": list(TEST_COUNT_CHOICES),
                "description": "Number of test items (orders of magnitude only).",
            },
            "parametric_pct": {
                "type": "integer",
                "description": (
                    "Percent of test items that are parametric (vs pass/fail), "
                    "in 10% steps. Default 50."
                ),
            },
            "seconds_per_touchdown": {
                "type": "number",
                "description": "Sort test time in seconds per touchdown (1-600).",
            },
            "multi_site": {
                "type": "boolean",
                "description": (
                    "True (default) = derive parallelism from gross die per "
                    "wafer; False = force single-site."
                ),
            },
            "s2s_enabled": {
                "type": "boolean",
                "description": "Enable site-to-site yield loss (one weak probe site).",
            },
            "repeater_fail_rate": {
                "type": "number",
                "description": (
                    "For Reticle Pattern: 1.0 = hard repeater (always fails), "
                    "0.1-0.9 = soft repeater failing only part of the time."
                ),
            },
            "signatures": {
                "type": "array",
                "items": {"type": "string", "enum": SIGNATURE_NAMES},
                "description": (
                    "One or more spatial defect patterns to apply to each wafer, in "
                    "PRIORITY ORDER (first = most dominant; it wins on any die shared "
                    "with another). Use a single item for one defect, or list several "
                    "when the user describes co-occurring defects."
                ),
            },
            "explanation": {
                "type": "string",
                "description": (
                    "One-sentence friendly explanation of the parameter choices "
                    "shown back to the user."
                ),
            },
        },
        # Only the explanation is mandatory: on follow-up ("change X, keep the
        # rest") messages the model should return JUST the fields that change,
        # and the previous request supplies everything else.
        "required": ["explanation"],
    },
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# Build a human-readable "scratch family" cheat-sheet straight from the
# metadata table in signatures.py, so the prompt never drifts out of sync.
_SCRATCH_FAMILY_GUIDE = chr(10).join(
    f"  - {name}: caused by {info['tool']} ({info['root_cause']})."
    for name, info in SCRATCH_FAMILIES.items()
)

_SYSTEM_PROMPT = f"""You are an expert semiconductor process engineer assistant that
generates synthetic wafer map data for demonstration and pre-sales use.

When the user describes what they want, extract the appropriate parameters and
call the `generate_wafer_maps` function. Use these guidelines:

SIGNATURE OPTIONS (pick the best match(es)):
{chr(10).join(f"  - {s}" for s in SIGNATURE_NAMES)}

MULTIPLE SIGNATURES: `signatures` is a LIST. Return several, in priority order
(most dominant first), whenever the user describes more than one defect on the
same wafers — e.g. "edge ring with a scratch" -> ["Edge Ring", "Robotic Handler
Scratch"]. Return a single-item list for one defect.

SCRATCH FAMILIES: a scratch's shape reveals the tool that made it. Pick the
specific family when the user hints at a root cause or tool; otherwise fall back
to the generic "Scratch / Streak".
{_SCRATCH_FAMILY_GUIDE}
  Hints: "robot"/"handler"/"aligner" -> Robotic Handler Scratch;
  "cassette"/"FOUP"/"slot"/"carrier" -> Cassette Slot Scratch;
  "wand"/"manual"/"by hand"/"operator" -> Wafer-Wand Scratch;
  "CMP"/"polish"/"slurry"/"pad" -> CMP Arc Scratch.

NO PATTERN: if the user asks for "no pattern", "no signature", "no spatial
signature", or plain/ordinary wafers, use "{NO_PATTERN_SIGNATURE}" — do NOT
substitute Random Scatter or any other pattern. Yield then comes purely from
the yield model; if the user also gave no yield, the tool applies a 93-97%
per-wafer baseline (mention that in your explanation). "Full Pass" is only
for explicitly perfect / 100%-yield wafers.

REPEATERS & STRIPING: "repeater"/"repeating bad die" -> Reticle Pattern.
"soft repeater" -> Reticle Pattern with repeater_fail_rate 0.1-0.9.
"striping"/"stripe"/"lens tilt" -> one of the Striping signatures
(Striping — Top / Bottom / Left / Right).

WAFER DIAMETERS: only 150 mm (6"), 200 mm (8"), 300 mm (12"). Default 300 mm.
EDGE TYPE: auto rule — 150 mm wafers have a FLAT, 200/300 mm have a NOTCH.
Only set edge_type when the user explicitly overrides that.

DIE SIZES: 1x1 mm up to 25x35 mm, aspect ratio between 1:2 and 2:1
(6x3 OK, 12x3 NOT). Default 10x10 mm.

STREET WIDTH: scribe street 0.05-0.2 mm, default 0.1 mm. Users may say
"50 um" or "50 micron" — convert to mm (50 um = 0.05 mm).

EDGE ORIENTATION: where the notch/flat points: down (default), up, left, right.

YIELD: if the user gives a yield percentage ("92% yield") set
yield_mode='direct' and target_yield_pct. If they give a defect density
("0.5 defects per cm2") set yield_mode='defect_density' and defect_density
(the tool applies Y = e^(-A*D)). Otherwise leave yield_mode='signature'.
If they ask for per-wafer yield/defect-density variation ("+/- 20%",
"vary by 10%"), set yield_variation_pct. If they ask for anything the
schema cannot express, say so plainly in the explanation instead of
claiming it was applied.

INSERTIONS: "CP1 and CP2" or "two insertions" -> num_insertions=2;
"CP1, CP2, CP3" / "hot cold room" -> 3. Default 1.

BINS: hardbin_count 16 (default) / 64 / 256; softbin_multiplier 4/16/64.

TEST ITEMS: test_count in orders of magnitude (100 default, 1000, ...).
parametric_pct = percent of parametric vs pass/fail items (default 50).

TEST TIME: seconds_per_touchdown 1-600 if the user mentions test time.

MULTI-SITE: leave multi_site=true (parallelism is derived from gross die per
wafer). Set s2s_enabled=true if the user asks for site-to-site yield loss or
a "bad site".

NUM WAFERS: wafers ship in FOUP carriers. Default 25 (standard full lot).
13 for thin/bonded. Never more than 25 per lot. Map "full lot"/"standard
lot" -> 25, "thin lot"/"bonded lot" -> 13.

MULTIPLE LOTS: "10 lots", "a quarter of weekly lots" -> num_lots and
lot_cadence ("1 lot per month" / "1 lot per week" / "1 lot per day" /
"multiple lots per day"). Lot IDs become fab-style FYYWWSSSS automatically.

LOT / PROGRAM: invent plausible IDs if not specified (e.g. LOT_A01, PRD_HBN20).

FOLLOW-UP MESSAGES: when a "CURRENT PARAMETERS" block is provided, the user is
modifying an earlier request. Return ONLY the parameters that should change;
every omitted parameter keeps its current value. Do NOT re-guess signatures,
die size, wafer count etc. unless the user explicitly changes them. Note that
"edge exclusion" is the keep-out band in mm (edge_exclusion) — it is NOT the
Edge Ring signature.

Write a concise one-sentence explanation that tells the user what you chose and why.
"""


# ---------------------------------------------------------------------------
# LLM calls (Azure OpenAI preferred, then OpenAI)
# ---------------------------------------------------------------------------

_DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"
_DEFAULT_AZURE_DEPLOYMENT = "gpt-4.1"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def _parse_tool_args(response, has_previous: bool = False) -> dict:
    """Extract JSON arguments from a chat completion tool call."""
    message = response.choices[0].message
    tool_calls = message.tool_calls or []
    if not tool_calls:
        raise RuntimeError("LLM did not return a tool call")

    args = json.loads(tool_calls[0].function.arguments)

    # Normalize the signature(s). The schema asks for a `signatures` array,
    # but be defensive: accept an old-style singular `signature` string too,
    # and keep only names we actually recognize (drop anything hallucinated).
    sigs = args.get("signatures")
    if not sigs:
        single = args.get("signature")
        sigs = [single] if single else []
    if isinstance(sigs, str):
        sigs = [sigs]
    sigs = [s for s in sigs if s in SIGNATURE_NAMES]
    if sigs:
        args["signatures"] = sigs
        args["signature"] = sigs[0]
    elif has_previous:
        # Follow-up message with no (valid) signature mentioned: drop the key
        # entirely so the merge keeps the previous request's signatures.
        args.pop("signatures", None)
        args.pop("signature", None)
    else:
        args["signatures"] = ["Edge Ring"]
        args["signature"] = "Edge Ring"

    if "num_wafers" in args:
        args["num_wafers"] = _clamp_wafers(args["num_wafers"])
    return args


def _args_to_request(args: dict,
                     base: Optional[WaferGenRequest] = None) -> WaferGenRequest:
    """Map raw LLM tool arguments onto a WaferGenRequest, clamping every
    numeric field into its spec range so a hallucinated value can never
    produce an illegal wafer.

    When `base` is given (a follow-up message in the chat), every field the
    LLM did NOT return keeps its value from the previous request instead of
    resetting to the factory default — "change X, keep the rest" behavior.
    """
    d = base if base is not None else WaferGenRequest()

    def _num(key, default, lo=None, hi=None, cast=float):
        v = cast(args.get(key, default))
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        return v

    def _choice(key, default, choices):
        v = args.get(key, default)
        return v if v in choices else default

    num_lots = _num("num_lots", d.num_lots, 1, 60, cast=int)
    s2s_enabled = bool(args.get("s2s_enabled", d.s2s_enabled))
    return WaferGenRequest(
        diameter=snap_diameter(args.get("diameter", d.diameter)),
        edge_type=args.get("edge_type", d.edge_type if base is not None else ""),
        edge_exclusion=_num("edge_exclusion", d.edge_exclusion,
                            EDGE_EXCLUSION_MIN, EDGE_EXCLUSION_MAX),
        die_width=_num("die_width", d.die_width, 1.0, 35.0),
        die_height=_num("die_height", d.die_height, 1.0, 35.0),
        street_width=clamp_street(args.get("street_width", d.street_width)),
        edge_orientation=args.get("edge_orientation", d.edge_orientation),
        lot_id=args.get("lot_id", d.lot_id),
        program=args.get("program", d.program),
        num_wafers=_clamp_wafers(args.get("num_wafers", d.num_wafers)),
        num_lots=num_lots,
        lot_cadence=args.get("lot_cadence", d.lot_cadence),
        auto_lot_id=num_lots > 1,
        yield_mode=args.get("yield_mode", d.yield_mode),
        target_yield_pct=args.get("target_yield_pct", d.target_yield_pct),
        defect_density=args.get("defect_density", d.defect_density),
        yield_variation_pct=_num("yield_variation_pct", d.yield_variation_pct,
                                 0.0, 50.0),
        num_insertions=_num("num_insertions", d.num_insertions, 1, 3, cast=int),
        hardbin_count=_choice("hardbin_count", d.hardbin_count, HARDBIN_CHOICES),
        softbin_multiplier=_choice("softbin_multiplier", d.softbin_multiplier,
                                   SOFTBIN_MULTIPLIERS),
        test_count=_choice("test_count", d.test_count, TEST_COUNT_CHOICES),
        parametric_pct=_num("parametric_pct", d.parametric_pct, 0, 100,
                            cast=int) // 10 * 10,
        seconds_per_touchdown=_num("seconds_per_touchdown",
                                   d.seconds_per_touchdown, 1.0, 600.0),
        multi_site=bool(args.get("multi_site", d.multi_site)),
        s2s_enabled=s2s_enabled,
        s2s_healthy=not s2s_enabled,
        repeater_fail_rate=_num("repeater_fail_rate", d.repeater_fail_rate,
                                0.1, 1.0),
        signatures=list(args.get("signatures", d.signatures)),
        explanation=args.get("explanation", ""),
        used_llm=True,
    )


# Fields echoed back to the LLM as context for follow-up messages. Keep in
# sync with _FUNCTION_SCHEMA property names so the model can mirror them.
_CONTEXT_FIELDS = (
    "diameter", "edge_type", "edge_exclusion", "die_width", "die_height",
    "street_width", "edge_orientation", "lot_id", "program", "num_wafers",
    "num_lots", "lot_cadence", "yield_mode", "target_yield_pct",
    "defect_density", "yield_variation_pct",
    "num_insertions", "hardbin_count", "softbin_multiplier",
    "test_count", "parametric_pct", "seconds_per_touchdown", "multi_site",
    "s2s_enabled", "repeater_fail_rate", "signatures",
)


def _request_context_json(req: WaferGenRequest) -> str:
    """Compact JSON of the previous request, for the follow-up context block."""
    return json.dumps({f: getattr(req, f) for f in _CONTEXT_FIELDS})


def _chat_completion_kwargs(
    user_prompt: str,
    previous_request: Optional[WaferGenRequest] = None,
) -> dict:
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if previous_request is not None:
        messages.append({
            "role": "system",
            "content": (
                "CURRENT PARAMETERS (from the user's previous request):\n"
                f"{_request_context_json(previous_request)}\n"
                "The next user message may be a follow-up modification. "
                "Return ONLY the parameters that should change; omitted "
                "parameters keep the values above."
            ),
        })
    messages.append({"role": "user", "content": user_prompt})
    return {
        "messages": messages,
        "tools": [{"type": "function", "function": _FUNCTION_SCHEMA}],
        "tool_choice": {"type": "function", "function": {"name": "generate_wafer_maps"}},
        "temperature": 0.3,
    }


def _call_azure_openai(
    user_prompt: str,
    *,
    api_key: str,
    azure_endpoint: str,
    deployment: str,
    api_version: str,
    previous_request: Optional[WaferGenRequest] = None,
) -> WaferGenRequest:
    """Send the user message to Azure OpenAI and parse the function-call response."""
    try:
        from openai import AzureOpenAI  # type: ignore
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=azure_endpoint.rstrip("/"),
        api_version=api_version,
    )

    response = client.chat.completions.create(
        model=deployment,
        **_chat_completion_kwargs(user_prompt, previous_request),
    )
    args = _parse_tool_args(response, has_previous=previous_request is not None)
    return _args_to_request(args, base=previous_request)


def _call_openai(
    user_prompt: str,
    api_key: str,
    model: str,
    previous_request: Optional[WaferGenRequest] = None,
) -> WaferGenRequest:
    """Send the user message to OpenAI and parse the function-call response."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        **_chat_completion_kwargs(user_prompt, previous_request),
    )
    args = _parse_tool_args(response, has_previous=previous_request is not None)
    return _args_to_request(args, base=previous_request)


# ---------------------------------------------------------------------------
# Keyword-based fallback parser (no API key required)
# ---------------------------------------------------------------------------

_KEYWORD_MAP = {
    # Pattern keywords → signature name
    # "No pattern" FIRST: an explicit request for no spatial signature must
    # never fall through to a pattern (or to the Edge Ring default).
    r"no.?pattern|no.?signature|no.?spatial|patternless|plain.?wafer|clean.?wafer|blank.?wafer":
        NO_PATTERN_SIGNATURE,
    r"edge.?ring|peripheral.?ring|edge.?fail": "Edge Ring",
    r"center|centre|chuck|middle.?fail":        "Center Cluster",
    # --- Scratch families FIRST, so a specific tool word wins over the
    #     generic "scratch" pattern below.
    r"robot|handler|end.?effector|aligner":     "Robotic Handler Scratch",
    r"cassette|foup|carrier|slot":              "Cassette Slot Scratch",
    r"wand|manual|by.?hand|hand.?transfer|operator": "Wafer-Wand Scratch",
    r"cmp|polish|slurry|pad.?debris|planariz":  "CMP Arc Scratch",
    # --- Generic scratch (fallback when no specific tool is mentioned)
    r"scratch|streak|linear|slip":              "Scratch / Streak",
    r"random|scatter|particle|noise":           "Random Scatter",
    r"quadrant|quarter":                        "Quadrant Failure",
    r"bull.?s.?eye|bullseye|alternating.?ring": "Bull's-Eye",
    r"full.?pass|all.?pass|100.?%":             "Full Pass",
    r"donut|mid.?ring|annular":                 "Donut (Mid-Ring)",
    r"half.?top|top.?half":                     "Half Wafer — Top",
    r"half.?bot|bottom.?half":                  "Half Wafer — Bottom",
    r"half.?left|left.?half":                   "Half Wafer — Left",
    r"half.?right|right.?half":                 "Half Wafer — Right",
    r"cross|plus.?shape":                       "Cross Pattern",
    r"hot.?spot|local.?cluster|point.?defect":  "Hot Spot",
    # Repeaters are the reticle/stepping-field pattern (spec terminology).
    r"reticle|systematic.?repeat|shot|repeater|repeating.?bad": "Reticle Pattern",
    # Striping = lens-tilt yield loss along one stepping-field edge.
    r"strip(?:e|ing).?top|top.?strip":          "Striping — Top",
    r"strip(?:e|ing).?bottom|bottom.?strip":    "Striping — Bottom",
    r"strip(?:e|ing).?left|left.?strip":        "Striping — Left",
    r"strip(?:e|ing).?right|right.?strip":      "Striping — Right",
    r"strip(?:e|ing)|lens.?tilt":               "Striping — Top",
    r"low.?yield|bad.?wafer|high.?fail":        "Low Yield",
    r"corner|corner.?cluster":                  "Corner Clusters",
    r"ring.?crack|crack":                       "Ring Crack",
    r"wedge|sector|pie":                        "Wedge / Sector",
    r"grid.?row|row.?fail|every.?row":          "Systematic Grid — Row",
    r"grid.?col|column.?fail|every.?col":       "Systematic Grid — Column",
    r"multi.?cluster|multiple.?spot":           "Multi-Cluster",
    r"top.?edge|notch.?side":                   "Top Edge Arc",
    r"bot.?edge|bottom.?edge":                  "Bottom Edge Arc",
    r"diagonal|45.?deg|135.?deg":              "Diagonal Scratch",
    r"concentric|two.?ring|dual.?ring":         "Concentric Rings",
    r"peripheral.?spot|edge.?spot":             "Peripheral Spot",
    r"spoke|radial|fan":                        "Radial Spokes",
    r"mixed|combo|combined|both":               "Mixed Mode (Edge + Center)",
}

# Striping names, used to skip the generic fallback when a specific edge hit.
_STRIPE_NAMES = {"Striping — Top", "Striping — Bottom",
                 "Striping — Left", "Striping — Right"}


def _parse_die_size_mm(text: str) -> tuple[float, float] | None:
    """Parse die size from phrases like '9 mm x 5 mm', '9x5 mm', or 'die size 9 x 5'."""
    patterns = (
        r"(\d+(?:\.\d+)?)\s*mm\s*[x×]\s*(\d+(?:\.\d+)?)\s*mm",
        r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*mm",
        r"die\s+size\s+(\d+(?:\.\d+)?)\s*(?:mm\s*)?[x×]\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
    )
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


def _parse_street_width_mm(text: str) -> float | None:
    """Parse scribe/street width; accepts mm, um, µm, or micron units."""
    t = text.lower()
    m = re.search(
        r"(?:scribe|street)(?:\s*width)?\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:um|µm|microns?)"
        r"|(\d+(?:\.\d+)?)\s*(?:um|µm|microns?)\s*(?:scribe|street|width)",
        t,
    )
    if m:
        return float(m.group(1) or m.group(2)) / 1000.0

    m = re.search(
        r"(\d+(?:\.\d+)?)\s*mm\s*(?:scribe|street)"
        r"|(?:scribe|street)(?:\s*width)?\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*mm",
        t,
    )
    if m:
        return float(m.group(1) or m.group(2))
    return None


def _parse_edge_type(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\bflat\b", t):
        return "flat"
    if re.search(r"\bnotch\b", t):
        return "notch"
    return None


def _parse_edge_orientation(text: str) -> str | None:
    """Notch/flat position: up/left/right/down, or None if not mentioned."""
    t = text.lower()
    if re.search(r"(?:notch|flat)\s*up|(?:notch|flat)\s*top|notch\s*12", t):
        return "up"
    if re.search(r"(?:notch|flat)\s*left|notch\s*9", t):
        return "left"
    if re.search(r"(?:notch|flat)\s*right|notch\s*3", t):
        return "right"
    if re.search(r"(?:notch|flat)\s*(?:down|bottom)|notch\s*6", t):
        return "down"
    return None


def _parse_yield(text: str) -> tuple[str, float | None, float | None]:
    """Detect a yield spec: returns (mode, target_yield_pct, defect_density).

    "92% yield" / "yield of 92" -> direct; "0.5 defects/cm2" / "defect
    density 0.5" -> defect_density; neither -> signature mode.
    """
    t = text.lower()
    m = re.search(r"defect\s*density\s*(?:of\s*)?(\d+(?:\.\d+)?)"
                  r"|(\d+(?:\.\d+)?)\s*defects?\s*(?:/|per)\s*cm", t)
    if m:
        return "defect_density", None, float(m.group(1) or m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*%?\s*yield|yield\s*(?:of\s*)?(\d+(?:\.\d+)?)", t)
    if m:
        return "direct", float(m.group(1) or m.group(2)), None
    return "signature", None, None


def _parse_yield_variation(text: str) -> float | None:
    """Detect per-wafer yield variation: '+/- 20%', '±20%', 'vary by 10%',
    '20% variation'. Returns the ±percent, or None if not mentioned."""
    t = text.lower()
    m = re.search(r"(?:±|\+/-|\+-)\s*(\d+(?:\.\d+)?)\s*%", t)
    if m:
        return float(m.group(1))
    m = re.search(
        r"(?:variation|vary(?:ing)?)\s*(?:of|by)?\s*(\d+(?:\.\d+)?)\s*%"
        r"|(\d+(?:\.\d+)?)\s*%\s*(?:yield\s*|defect\s*density\s*)?variation",
        t,
    )
    if m:
        return float(m.group(1) or m.group(2))
    return None


def _parse_insertions(text: str) -> int | None:
    """How many CP insertions the user wants (1-3), or None if not mentioned."""
    t = text.lower()
    if re.search(r"cp3|three\s*insertions|3\s*insertions|hot.*cold|cold.*hot", t):
        return 3
    if re.search(r"cp2|two\s*insertions|2\s*insertions|retest", t):
        return 2
    if re.search(r"cp1|one\s*insertion|1\s*insertion|single\s*insertion", t):
        return 1
    return None


def _parse_lots(text: str) -> tuple[int | None, str | None]:
    """Detect multi-lot requests: (num_lots, cadence), None where unmentioned."""
    t = text.lower()
    m = re.search(r"(\d+)\s*lots", t)
    num_lots = max(1, min(60, int(m.group(1)))) if m else None
    cadence = None
    if re.search(r"per\s*month|monthly", t):
        cadence = "1 lot per month"
    elif re.search(r"lots\s*per\s*day|multiple\s*lots", t):
        # "3 lots per day" — check BEFORE the generic per-day pattern below.
        cadence = "multiple lots per day"
    elif re.search(r"per\s*day|daily", t):
        cadence = "1 lot per day"
    elif re.search(r"per\s*week|weekly", t):
        cadence = "1 lot per week"
    return num_lots, cadence


def _format_street_width(street_width: float) -> str:
    if street_width <= 0:
        return ""
    if street_width < 0.1:
        return f" Street width: {street_width * 1000:.0f} µm ({street_width} mm)."
    return f" Street width: {street_width} mm."


def _keyword_parse(text: str,
                   base: Optional[WaferGenRequest] = None) -> WaferGenRequest:
    """Best-effort keyword extraction without an LLM.

    When `base` is given (a follow-up chat message), start from a copy of the
    previous request and only overwrite fields this message actually mentions.
    """
    t = text.lower()
    is_followup = base is not None
    req = copy.deepcopy(base) if is_followup else WaferGenRequest()

    # Signature(s): collect EVERY pattern that matches (not just the first),
    # so "edge ring and a scratch" yields multiple signatures. Keep insertion
    # order and de-duplicate.
    matched: list[str] = []
    for pattern, sig in _KEYWORD_MAP.items():
        if re.search(pattern, t) and sig not in matched:
            matched.append(sig)

    # If any specific scratch FAMILY was detected, drop the generic scratch
    # so we don't produce both "Scratch / Streak" and, say, "CMP Arc Scratch".
    if _SCRATCH_FAMILY_NAMES & set(matched):
        matched = [m for m in matched if m != "Scratch / Streak"]
    # Same idea for striping: keep only the first (most specific) edge hit.
    stripe_hits = [m for m in matched if m in _STRIPE_NAMES]
    if len(stripe_hits) > 1:
        matched = [m for m in matched
                   if m not in _STRIPE_NAMES or m == stripe_hits[0]]

    if matched:
        req.signatures = matched
        req.signature = matched[0]

    # Soft repeater: a repeater with a fail rate below 100%.
    m = re.search(r"soft\s*repeater", t)
    if m:
        req.repeater_fail_rate = 0.5
        if "Reticle Pattern" not in req.signatures:
            req.signatures.insert(0, "Reticle Pattern")
            req.signature = req.signatures[0]

    # Lot size: recognize carrier-based phrases first, then a raw count.
    if re.search(r"thin.?lot|bonded|13.?wafer|small.?foup", t):
        req.num_wafers = 13
    elif re.search(r"full.?lot|standard.?lot|standard.?foup|full.?foup", t):
        req.num_wafers = 25
    m = re.search(r"(\d+)\s*(?:wafer|map)", t)
    if m:
        req.num_wafers = _clamp_wafers(int(m.group(1)))

    die_size = _parse_die_size_mm(text)
    if die_size:
        req.die_width, req.die_height = die_size

    # Diameter — e.g. "200 mm wafer"; snapped to 150/200/300 later anyway.
    diameter_parsed = False
    m = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:wafer|diameter)", t)
    if m:
        req.diameter = snap_diameter(float(m.group(1)))
        diameter_parsed = True
    else:
        for d in (300, 200, 150):
            if re.search(rf"\b{d}\s*mm\b", t):
                req.diameter = float(d)
                diameter_parsed = True
                break

    edge_type = _parse_edge_type(text)
    if edge_type:
        req.edge_type = edge_type
    elif not is_followup or diameter_parsed:
        req.edge_type = auto_edge_type(req.diameter)

    orientation = _parse_edge_orientation(text)
    if orientation is not None:
        req.edge_orientation = orientation

    street_width = _parse_street_width_mm(text)
    if street_width is not None:
        req.street_width = clamp_street(street_width)

    # Edge exclusion — "5 mm edge exclusion" / "edge exclusion of 5 mm".
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*mm\s*edge\s*exclusion"
        r"|edge\s*exclusion\s*(?:zone\s*)?(?:of\s*|to\s*|=\s*)?(\d+(?:\.\d+)?)\s*mm",
        t,
    )
    if m:
        req.edge_exclusion = max(EDGE_EXCLUSION_MIN,
                                 min(EDGE_EXCLUSION_MAX,
                                     float(m.group(1) or m.group(2))))

    # Yield / insertions / lots — the spec "must have" numeric knobs.
    # On follow-ups, only overwrite what this message actually mentions.
    yield_mode, target_yield, density = _parse_yield(text)
    if not is_followup or yield_mode != "signature":
        req.yield_mode = yield_mode
        req.target_yield_pct = target_yield
        req.defect_density = density
    variation = _parse_yield_variation(text)
    if variation is not None:
        req.yield_variation_pct = max(0.0, min(50.0, variation))
    num_insertions = _parse_insertions(text)
    if num_insertions is not None:
        req.num_insertions = num_insertions
    num_lots, lot_cadence = _parse_lots(text)
    if num_lots is not None:
        req.num_lots = num_lots
    if lot_cadence is not None:
        req.lot_cadence = lot_cadence
    req.auto_lot_id = req.num_lots > 1

    # Test time — "5 seconds per touchdown" / "test time 30 s".
    m = re.search(r"(\d+(?:\.\d+)?)\s*s(?:ec(?:onds)?)?\s*(?:per\s*)?(?:touchdown|test\s*time)"
                  r"|test\s*time\s*(?:of\s*)?(\d+(?:\.\d+)?)", t)
    if m:
        req.seconds_per_touchdown = max(1.0, min(600.0, float(m.group(1) or m.group(2))))

    # S2S — "bad site", "site to site loss".
    if re.search(r"s2s|site.?to.?site|bad\s*site|weak\s*site", t):
        req.s2s_enabled = True
        req.s2s_healthy = False

    # Story 1 — ECID / FT / sweeper / assembly errors
    gdbn_scenario = _parse_gdbn_scenario(t)
    multidie_scenario = _parse_multidie_scenario(t)
    story_scenario = _parse_story1_scenario(t)
    if gdbn_scenario:
        req.story_id = "story1_gdbn"
        req.gdbn_scenario = gdbn_scenario
    elif multidie_scenario:
        req.story_id = "story1_multidie"
        req.multidie_mode = multidie_scenario
    elif story_scenario:
        from story1_presets import apply_scenario_to_request
        req.story_id = "story1"
        req.story1_scenario = story_scenario
        apply_scenario_to_request(req, story_scenario)

    ecid_mode = _parse_ecid_mode(t)
    if ecid_mode:
        req.ecid_mode = ecid_mode
    ecid_repr = _parse_ecid_representation(t)
    if ecid_repr:
        req.ecid_representation = ecid_repr

    # Lot ID
    m = re.search(r"lot[_\s-]?(\w+)", t, re.IGNORECASE)
    if m and not req.auto_lot_id:
        req.lot_id = f"LOT_{m.group(1).upper()}"

    edge_str = (
        f"notch {req.edge_orientation}" if req.edge_type == "notch"
        else f"flat {req.edge_orientation}"
    )
    sig_str = " + ".join(req.signatures)
    req.explanation = (
        f"[Keyword parser] Detected signature(s): **{sig_str}**, "
        f"{req.num_wafers} wafer(s), {int(req.diameter)} mm diameter, "
        f"{req.die_width}×{req.die_height} mm dies, {edge_str} edge."
    )
    req.explanation += _format_street_width(req.street_width)
    if req.yield_mode == "direct":
        req.explanation += f" Target yield {req.target_yield_pct:g}%."
    elif req.yield_mode == "defect_density":
        req.explanation += f" Defect density {req.defect_density:g}/cm² (Y = e^(-A·D))."
    if req.yield_variation_pct > 0 and req.yield_mode != "signature":
        req.explanation += f" ±{req.yield_variation_pct:g}% per-wafer variation."
    if req.yield_mode == "signature" and NO_PATTERN_SIGNATURE in req.signatures:
        req.explanation += (
            " No yield specified — using a 93-97% per-wafer baseline "
            "(give a yield % or defect density to override)."
        )
    if req.num_insertions > 1:
        req.explanation += f" Insertions: CP1..CP{req.num_insertions}."
    if req.num_lots > 1:
        req.explanation += f" {req.num_lots} lots, {req.lot_cadence}."
    if req.story_id == "story1":
        req.explanation += f" Story 1 scenario: {req.story1_scenario}."
    elif req.story_id == "story1_gdbn":
        req.explanation += f" Story 1g (GDBN) scenario: {req.gdbn_scenario}."
    elif req.story_id == "story1_multidie":
        req.explanation += f" Story 1c (multi-die) scenario: {req.multidie_mode}."
    if req.ecid_mode != "plain":
        req.explanation += f" ECID mode: {req.ecid_mode}."
    if req.ecid_representation != "single":
        req.explanation += f" ECID representation: {req.ecid_representation}."
    req.used_llm = False
    return req


def _parse_story1_scenario(t: str) -> Optional[str]:
    """Map natural-language phrases onto a Story 1 scenario id."""
    if re.search(r"sweeper", t):
        if re.search(r"detail|blank\s*ecid|no\s*ecid|< ?2%|assembly\s*wreck", t):
            return "sweeper_detail"
        return "sweeper_simple"
    if re.search(r"wrong\s*bin|mis-?pick(?:ed)?\s*bin|fail\s*bin\s*picked", t):
        return "wrong_bin"
    if re.search(r"wrong\s*(?:x|y|xy)|origin\s*shift|shifted\s*(?:x|y|xy|origin)", t):
        horror = bool(re.search(r"horror|100\s*gdpw|low\s*gdpw|large\s*die", t))
        high = bool(re.search(r"1000\s*gdpw|high\s*gdpw|subtle\s*case|small\s*die", t))
        subtle_ft = bool(re.search(r"subtle\s*ft|80\s*%|adjustable\s*fail", t))
        if high or (not horror and re.search(r"subtle", t) and not subtle_ft):
            return ("wrong_xy_1000_subtle_ft" if subtle_ft
                    else "wrong_xy_1000_simple")
        if horror or re.search(r"100\s*gdpw", t):
            return ("wrong_xy_horror_subtle_ft" if subtle_ft
                    else "wrong_xy_horror_simple")
        return "wrong_xy_horror_simple"
    if re.search(r"ecid|traceability|final\s*test|\bft\b|1\s*:\s*1|one.to.one", t):
        if re.search(r"detail|blank\s*ecid|no\s*ecid|< ?2%|assembly\s*wreck", t):
            return "one_to_one_detail"
        if re.search(r"simple|match|trace", t) or re.search(r"ecid|final\s*test|\bft\b", t):
            return "one_to_one_simple"
    return None


def _parse_ecid_mode(t: str) -> Optional[str]:
    """Detect spec 1.b.ii/iii ECID encoding requests ('rot13', 'split test items')."""
    if re.search(r"rot ?13|encrypt", t):
        return "rot13"
    return None


def _parse_ecid_representation(t: str) -> Optional[str]:
    if re.search(r"split.*(?:test\s*item|ecid)|4\s*test\s*items|multiple\s*test\s*items", t):
        return "split_items"
    return None


def _parse_gdbn_scenario(t: str) -> Optional[str]:
    """Spec 1.g: low yield at FT caused by CP clusters (GDBN)."""
    if re.search(r"gdbn|bad\s*neighbo(?:u)?rhood|good\s*die\s*bad", t):
        return "gdbn_neighbor"
    if re.search(r"missed?\s*(?:cp\s*)?cluster|cp\s*cluster.*(?:ft|final\s*test)"
                 r"|dramatic\s*case|donut.*(?:invisible|missed)|invisible.*cp", t):
        return "gdbn_dramatic"
    return None


def _parse_multidie_scenario(t: str) -> Optional[str]:
    """Spec 1.c: multi-die product traceability (Case B of the 2x2 matrix)."""
    if not re.search(r"multi.?die|chiplet|multi.?chip|package[d]?\s*product|\bmcm\b|case\s*b", t):
        return None
    if re.search(r"partial|annoying|missing\s*trace|no\s*trace|2\s*of\s*3", t):
        return "partial_trace"
    return "full_trace"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _apply_story1_from_prompt(req: WaferGenRequest, user_prompt: str) -> WaferGenRequest:
    """Enable Story 1 when the user prompt asks for ECID / FT / sweeper / etc.

    Applied after BOTH LLM and keyword parsing so Azure GPT runs still get
    FT + match exports (the tool schema does not yet expose story fields).
    """
    t = user_prompt.lower()
    ecid_mode = _parse_ecid_mode(t)
    if ecid_mode:
        req.ecid_mode = ecid_mode
    ecid_repr = _parse_ecid_representation(t)
    if ecid_repr:
        req.ecid_representation = ecid_repr

    gdbn_scenario = _parse_gdbn_scenario(t)
    multidie_scenario = _parse_multidie_scenario(t)
    scenario = _parse_story1_scenario(t)

    if gdbn_scenario:
        req.story_id = "story1_gdbn"
        req.gdbn_scenario = gdbn_scenario
        req.explanation = (req.explanation or "").rstrip() + (
            f" Story 1g (GDBN) scenario: {gdbn_scenario}.")
    elif multidie_scenario:
        req.story_id = "story1_multidie"
        req.multidie_mode = multidie_scenario
        req.explanation = (req.explanation or "").rstrip() + (
            f" Story 1c (multi-die) scenario: {multidie_scenario}.")
    elif scenario:
        from story1_presets import apply_scenario_to_request
        # Preserve LLM-chosen lot/wafer counts when already set sensibly;
        # presets still bump sweeper to ≥2 lots if needed.
        apply_scenario_to_request(req, scenario)
        if "Story 1 scenario" not in (req.explanation or ""):
            req.explanation = (req.explanation or "").rstrip() + f" Story 1 scenario: {scenario}."
    return req


def parse_user_request(
    user_prompt: str,
    *,
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_deployment: Optional[str] = None,
    azure_api_version: Optional[str] = None,
    previous_request: Optional[WaferGenRequest] = None,
) -> WaferGenRequest:
    """
    Parse a natural-language wafer map request.

    `previous_request` is the last request generated in this chat session (or
    None for the first message). When set, the new message is treated as a
    modification: unmentioned parameters keep their previous values.

    Priority:
      1. Azure OpenAI if endpoint + key are available
      2. OpenAI if OPENAI_API_KEY / api_key is available
      3. Keyword parser fallback
    """
    azure_key = (api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")).strip()
    azure_endpoint = (
        azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    ).strip()
    azure_deployment = (
        azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", _DEFAULT_AZURE_DEPLOYMENT)
    ).strip()
    azure_api_version = (
        azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION", _DEFAULT_AZURE_API_VERSION)
    ).strip()

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

    try:
        if azure_key and azure_endpoint:
            req = _call_azure_openai(
                user_prompt,
                api_key=azure_key,
                azure_endpoint=azure_endpoint,
                deployment=azure_deployment,
                api_version=azure_api_version,
                previous_request=previous_request,
            )
            return _apply_story1_from_prompt(req, user_prompt)
        if openai_key:
            model = os.environ.get("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL).strip()
            req = _call_openai(user_prompt, openai_key, model,
                               previous_request=previous_request)
            return _apply_story1_from_prompt(req, user_prompt)
    except Exception as exc:
        req = _keyword_parse(user_prompt, base=previous_request)
        req.explanation = (
            f"⚠️ LLM call failed ({exc}). Using keyword parser instead. "
            + req.explanation
        )
        return req

    return _keyword_parse(user_prompt, base=previous_request)
