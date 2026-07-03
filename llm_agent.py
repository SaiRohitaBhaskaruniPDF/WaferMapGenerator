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

from signatures import SIGNATURE_NAMES


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
                "description": "Number of wafers to generate (1–25).",
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

NUM WAFERS: default 4, max 25.

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
    args["num_wafers"] = max(1, min(25, int(args.get("num_wafers", 4))))
    args["diameter"] = float(args.get("diameter", 300))
    args["die_width"] = float(args.get("die_width", 10))
    args["die_height"] = float(args.get("die_height", 10))
    return args


def _args_to_request(args: dict) -> WaferGenRequest:
    return WaferGenRequest(
        diameter=args.get("diameter", 300.0),
        edge_type=args.get("edge_type", "notch"),
        edge_exclusion=args.get("edge_exclusion", 3.0),
        die_width=args.get("die_width", 10.0),
        die_height=args.get("die_height", 10.0),
        x_offset=args.get("x_offset", 0.0),
        y_offset=args.get("y_offset", 0.0),
        lot_id=args.get("lot_id", "LOT_001"),
        program=args.get("program", "DEMO"),
        num_wafers=args.get("num_wafers", 4),
        signature=args.get("signature", "Edge Ring"),
        explanation=args.get("explanation", ""),
        used_llm=True,
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


def _keyword_parse(text: str) -> WaferGenRequest:
    """Best-effort keyword extraction without an LLM."""
    t = text.lower()
    req = WaferGenRequest()

    # Signature
    for pattern, sig in _KEYWORD_MAP.items():
        if re.search(pattern, t):
            req.signature = sig
            break

    # Number of wafers
    m = re.search(r"(\d+)\s*(wafer|map)", t)
    if m:
        req.num_wafers = max(1, min(25, int(m.group(1))))

    # Diameter
    for d in [300, 200, 150]:
        if str(d) in t:
            req.diameter = float(d)
            break

    # Die size
    m = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*mm", t)
    if m:
        req.die_width  = float(m.group(1))
        req.die_height = float(m.group(2))

    # Lot ID
    m = re.search(r"lot[_\s-]?(\w+)", t, re.IGNORECASE)
    if m:
        req.lot_id = f"LOT_{m.group(1).upper()}"

    req.explanation = (
        f"[Keyword parser] Detected signature: **{req.signature}**, "
        f"{req.num_wafers} wafer(s), {int(req.diameter)} mm diameter."
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
