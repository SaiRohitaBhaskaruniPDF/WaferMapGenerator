"""
LLM Agent for Wafer Map Generation.

Converts natural-language user prompts into structured WaferGenRequest objects
using Azure OpenAI (preferred) or OpenAI chat completions with function calling.
Falls back to a simple keyword parser when no API credentials are available.

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

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from geometry import WaferConfig
from signatures import SIGNATURE_NAMES

MAX_WAFERS = 100


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
    """Structured generation request parsed from natural language."""
    # Wafer geometry
    diameter: float = 300.0          # mm
    edge_type: str = "notch"         # 'notch' | 'flat'
    edge_exclusion: float = 3.0      # mm
    die_width: float = 10.0          # mm
    die_height: float = 10.0         # mm
    x_offset: float = 0.0
    y_offset: float = 0.0
    street_width: float = 0.0
    dies_per_reticle_x: int = 2
    dies_per_reticle_y: int = 2
    reticle_fail_die_x: int = 0
    reticle_fail_die_y: int = 0

    # Wafer orientation
    notch_orientation: str = "down"  # 'down' | 'up' | 'left' | 'right'

    # Lot / generation
    lot_id: str = "LOT_001"
    program: str = "DEMO"
    num_wafers: int = 4

    # Signature
    signature: str = "Edge Ring"

    # Explanation the LLM provides (shown to user)
    explanation: str = ""

    # Whether the LLM was actually used (False = keyword fallback)
    used_llm: bool = False


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
                "description": "Wafer diameter in mm. Typical values: 150, 200, 300.",
            },
            "edge_type": {
                "type": "string",
                "enum": ["notch", "flat"],
                "description": "Wafer edge marker type.",
            },
            "edge_exclusion": {
                "type": "number",
                "description": "Edge exclusion zone in mm (1–5 typical).",
            },
            "die_width": {
                "type": "number",
                "description": "Die width in mm.",
            },
            "die_height": {
                "type": "number",
                "description": "Die height in mm.",
            },
            "x_offset": {
                "type": "number",
                "description": "Horizontal shift of the die grid from center (mm).",
            },
            "y_offset": {
                "type": "number",
                "description": "Vertical shift of the die grid from center (mm).",
            },
            "street_width": {
                "type": "number",
                "description": "Scribe/street width between dies in mm (0–3 typical). Default 0.",
            },
            "notch_orientation": {
                "type": "string",
                "enum": ["down", "up", "left", "right"],
                "description": (
                    "Which direction the notch points on the wafer map image. "
                    "'down' = 6 o'clock (default/standard). "
                    "'up' = 12 o'clock. 'left' = 9 o'clock. 'right' = 3 o'clock. "
                    "Phrases like 'notch down' → down, 'notch up' → up."
                ),
            },
            "dies_per_reticle_x": {
                "type": "integer",
                "description": "Number of dies across one reticle field in X (1–6). Default 2.",
            },
            "dies_per_reticle_y": {
                "type": "integer",
                "description": "Number of dies across one reticle field in Y (1–6). Default 2.",
            },
            "reticle_fail_die_x": {
                "type": "integer",
                "description": "0-based die column within reticle that fails (for Reticle Pattern).",
            },
            "reticle_fail_die_y": {
                "type": "integer",
                "description": "0-based die row within reticle that fails (for Reticle Pattern).",
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
                "description": f"Number of wafers to generate (1–{MAX_WAFERS}).",
            },
            "signature": {
                "type": "string",
                "enum": SIGNATURE_NAMES,
                "description": "Spatial defect pattern to apply.",
            },
            "explanation": {
                "type": "string",
                "description": (
                    "One-sentence friendly explanation of the parameter choices "
                    "shown back to the user."
                ),
            },
        },
        "required": ["signature", "num_wafers", "explanation"],
    },
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""You are an expert semiconductor process engineer assistant that
generates synthetic wafer map data for demonstration and pre-sales use.

When the user describes what they want, extract the appropriate parameters and
call the `generate_wafer_maps` function. Use these guidelines:

SIGNATURE OPTIONS (pick the best match):
{chr(10).join(f"  - {s}" for s in SIGNATURE_NAMES)}

WAFER DIAMETERS: 150 mm (6"), 200 mm (8"), 300 mm (12"). Default to 300 mm.

DIE SIZES: typical range 5–20 mm. Default 10×10 mm.

STREET WIDTH: scribe/street gap between dies. Always return street_width in mm.
  - Users may say "50 um", "50 µm", or "50 micron" scribe → convert to mm (50 µm = 0.05 mm).
  - Typical range 0–3 mm. Default 0.

EDGE TYPE: 'notch' (default) or 'flat'. Phrases like "notch down" mean edge_type = notch.

NOTCH ORIENTATION: where the notch points on the image.
  - 'notch down' or just 'notch' → notch_orientation = 'down' (default, 6 o'clock)
  - 'notch up'    → notch_orientation = 'up'
  - 'notch left'  → notch_orientation = 'left'
  - 'notch right' → notch_orientation = 'right'

RETICLE LAYOUT: dies_per_reticle_x/y (default 2×2). For Reticle Pattern signature,
set reticle_fail_die_x/y to pick which die position fails in every reticle shot.

NUM WAFERS: default 4, max {MAX_WAFERS}.

LOT / PROGRAM: invent plausible IDs if not specified (e.g. LOT_A01, PRD_HBN20).

Always pick the signature that best matches the described failure mode. If the
user describes something like "ring around the edge" → Edge Ring. "Bad dies in
the middle" → Center Cluster. "Scratches on the wafer" → Scratch / Streak, etc.

Write a concise one-sentence explanation that tells the user what you chose and why.
"""


# ---------------------------------------------------------------------------
# LLM calls (Azure OpenAI preferred, then OpenAI)
# ---------------------------------------------------------------------------

_DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"
_DEFAULT_AZURE_DEPLOYMENT = "gpt-4.1"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def _parse_tool_args(response) -> dict:
    """Extract JSON arguments from a chat completion tool call."""
    message = response.choices[0].message
    tool_calls = message.tool_calls or []
    if not tool_calls:
        raise RuntimeError("LLM did not return a tool call")

    args = json.loads(tool_calls[0].function.arguments)
    args["num_wafers"] = _clamp_wafers(args.get("num_wafers", 4))
    args["diameter"] = float(args.get("diameter", 300))
    args["die_width"] = float(args.get("die_width", 10))
    args["die_height"] = float(args.get("die_height", 10))
    args["street_width"] = max(0.0, min(5.0, float(args.get("street_width", 0))))
    args["dies_per_reticle_x"] = max(1, min(6, int(args.get("dies_per_reticle_x", 2))))
    args["dies_per_reticle_y"] = max(1, min(6, int(args.get("dies_per_reticle_y", 2))))
    args["reticle_fail_die_x"] = max(0, int(args.get("reticle_fail_die_x", 0)))
    args["reticle_fail_die_y"] = max(0, int(args.get("reticle_fail_die_y", 0)))
    return args


def _args_to_request(args: dict) -> WaferGenRequest:
    dpr_x = max(1, min(6, int(args.get("dies_per_reticle_x", 2))))
    dpr_y = max(1, min(6, int(args.get("dies_per_reticle_y", 2))))
    return WaferGenRequest(
        diameter=args.get("diameter", 300.0),
        edge_type=args.get("edge_type", "notch"),
        edge_exclusion=args.get("edge_exclusion", 3.0),
        die_width=args.get("die_width", 10.0),
        die_height=args.get("die_height", 10.0),
        x_offset=args.get("x_offset", 0.0),
        y_offset=args.get("y_offset", 0.0),
        street_width=max(0.0, min(5.0, float(args.get("street_width", 0.0)))),
        dies_per_reticle_x=dpr_x,
        dies_per_reticle_y=dpr_y,
        reticle_fail_die_x=int(args.get("reticle_fail_die_x", 0)) % dpr_x,
        reticle_fail_die_y=int(args.get("reticle_fail_die_y", 0)) % dpr_y,
        lot_id=args.get("lot_id", "LOT_001"),
        program=args.get("program", "DEMO"),
        num_wafers=args.get("num_wafers", 4),
        signature=args.get("signature", "Edge Ring"),
        notch_orientation=args.get("notch_orientation", "down"),
        explanation=args.get("explanation", ""),
        used_llm=True,
    )


def request_to_config(req: WaferGenRequest) -> WaferConfig:
    """Build WaferConfig from a parsed generation request."""
    return WaferConfig(
        diameter=req.diameter,
        edge_type=req.edge_type,
        edge_exclusion=req.edge_exclusion,
        die_width=req.die_width,
        die_height=req.die_height,
        x_offset=req.x_offset,
        y_offset=req.y_offset,
        street_width=req.street_width,
        dies_per_reticle_x=req.dies_per_reticle_x,
        dies_per_reticle_y=req.dies_per_reticle_y,
        reticle_fail_die_x=req.reticle_fail_die_x,
        reticle_fail_die_y=req.reticle_fail_die_y,
        notch_orientation=req.notch_orientation,
    )


def _chat_completion_kwargs(user_prompt: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
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
        **_chat_completion_kwargs(user_prompt),
    )
    return _args_to_request(_parse_tool_args(response))


def _call_openai(user_prompt: str, api_key: str, model: str) -> WaferGenRequest:
    """Send the user message to OpenAI and parse the function-call response."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        **_chat_completion_kwargs(user_prompt),
    )
    return _args_to_request(_parse_tool_args(response))


# ---------------------------------------------------------------------------
# Keyword-based fallback parser (no API key required)
# ---------------------------------------------------------------------------

_KEYWORD_MAP = {
    # Pattern keywords → signature name
    r"edge.?ring|peripheral.?ring|edge.?fail": "Edge Ring",
    r"center|centre|chuck|middle.?fail":        "Center Cluster",
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
    r"reticle|systematic.?repeat|shot":         "Reticle Pattern",
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


def _parse_notch_orientation(text: str) -> str:
    t = text.lower()
    if re.search(r"notch\s*up|notch\s*top|notch\s*12", t):
        return "up"
    if re.search(r"notch\s*left|notch\s*9", t):
        return "left"
    if re.search(r"notch\s*right|notch\s*3", t):
        return "right"
    return "down"


def _format_street_width(street_width: float) -> str:
    if street_width <= 0:
        return ""
    if street_width < 0.1:
        return f" Street width: {street_width * 1000:.0f} µm ({street_width} mm)."
    return f" Street width: {street_width} mm."


def _keyword_parse(text: str) -> WaferGenRequest:
    """Best-effort keyword extraction without an LLM."""
    t = text.lower()
    req = WaferGenRequest()

    # Signature
    for pattern, sig in _KEYWORD_MAP.items():
        if re.search(pattern, t):
            req.signature = sig
            break

    # Number of wafers — "100 wafers", "generate 100 wafer maps"
    m = re.search(r"(\d+)\s*(?:wafer|map)", t)
    if m:
        req.num_wafers = _clamp_wafers(int(m.group(1)))

    die_size = _parse_die_size_mm(text)
    if die_size:
        req.die_width, req.die_height = die_size

    # Diameter — e.g. "200 mm wafer" or "100mm"
    m = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:wafer|diameter)", t)
    if m:
        req.diameter = float(m.group(1))
    else:
        for d in [300, 200, 150, 100]:
            if re.search(rf"\b{d}\s*mm\b", t):
                req.diameter = float(d)
                break

    edge_type = _parse_edge_type(text)
    if edge_type:
        req.edge_type = edge_type

    req.notch_orientation = _parse_notch_orientation(text)

    street_width = _parse_street_width_mm(text)
    if street_width is not None:
        req.street_width = max(0.0, min(5.0, street_width))
    elif re.search(r"\bstreet\b|\bscribe\b", t):
        req.street_width = 0.1

    # Reticle layout — e.g. "3x3 reticle" or "2x2 dies per reticle"
    m = re.search(
        r"(\d+)\s*[x×]\s*(\d+)\s*(?:dies?\s*)?(?:per\s*)?reticle|reticle\s*(\d+)\s*[x×]\s*(\d+)",
        t,
    )
    if m:
        req.dies_per_reticle_x = max(1, min(6, int(m.group(1) or m.group(3))))
        req.dies_per_reticle_y = max(1, min(6, int(m.group(2) or m.group(4))))

    # Lot ID
    m = re.search(r"lot[_\s-]?(\w+)", t, re.IGNORECASE)
    if m:
        req.lot_id = f"LOT_{m.group(1).upper()}"

    edge_str = (
        f"notch {req.notch_orientation}" if req.edge_type == "notch" else "flat"
    )
    req.explanation = (
        f"[Keyword parser] Detected signature: **{req.signature}**, "
        f"{req.num_wafers} wafer(s), {int(req.diameter)} mm diameter, "
        f"{req.die_width}×{req.die_height} mm dies, {edge_str} edge."
    )
    if req.street_width > 0:
        req.explanation += _format_street_width(req.street_width)
    if req.signature == "Reticle Pattern":
        req.explanation += (
            f" Reticle layout: {req.dies_per_reticle_x}×{req.dies_per_reticle_y} dies/shot."
        )
    req.used_llm = False
    return req


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_user_request(
    user_prompt: str,
    *,
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_deployment: Optional[str] = None,
    azure_api_version: Optional[str] = None,
) -> WaferGenRequest:
    """
    Parse a natural-language wafer map request.

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
            return _call_azure_openai(
                user_prompt,
                api_key=azure_key,
                azure_endpoint=azure_endpoint,
                deployment=azure_deployment,
                api_version=azure_api_version,
            )
        if openai_key:
            model = os.environ.get("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL).strip()
            return _call_openai(user_prompt, openai_key, model)
    except Exception as exc:
        req = _keyword_parse(user_prompt)
        req.explanation = (
            f"⚠️ LLM call failed ({exc}). Using keyword parser instead. "
            + req.explanation
        )
        return req

    return _keyword_parse(user_prompt)
