"""
Wafer Map Bot — Streamlit Web App
LLM-driven chatbot that generates synthetic wafer maps for pre-sales demos.

Two ways in, one pipeline out:
  * Chat page   — natural language -> llm_agent.parse_user_request()
  * Manual page — a form that fills the same WaferGenRequest directly

Both paths call generator.generate(), which runs the full spec pipeline
(signature -> yield model -> CP cascade -> bin mapping -> multi-site) and
returns a GenerationResult that this file renders and offers for download
(CSV, per-test CSV, PNG/SVG/JPEG/TIFF maps, STDF per lot per insertion).
"""
from __future__ import annotations

import io
import os
import zipfile
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def _inject_streamlit_secrets() -> None:
    """Load Streamlit Cloud secrets into os.environ for llm_agent."""
    try:
        for key in (
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_API_VERSION",
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
        ):
            value = st.secrets.get(key)
            if value and not os.environ.get(key):
                os.environ[key] = str(value)
    except Exception:
        pass


_inject_streamlit_secrets()

from geometry import (
    STANDARD_DIAMETERS, auto_edge_type, validate_die_size,
    EDGE_EXCLUSION_MIN, EDGE_EXCLUSION_MAX,
    STREET_MIN, STREET_MAX, STREET_DEFAULT,
)
from renderer import render_wafermaps, figure_to_bytes, IMAGE_FORMATS
from signatures import SIGNATURE_NAMES, BIN_DEFINITIONS, SCRATCH_FAMILIES
from llm_agent import (
    parse_user_request, WaferGenRequest, request_to_config,
    MAX_WAFERS, LOT_SIZE_PRESETS,
)
from binning import HARDBIN_CHOICES, SOFTBIN_MULTIPLIERS
from test_items import (
    TEST_COUNT_CHOICES, NAMING_STYLES, VERBOSE_LENGTHS, VALUE_SHAPES,
    estimate_result_count,
)
from fab import LOT_CADENCES, SITE_PATTERNS
from generator import generate, GenerationResult

EXAMPLE_PROMPTS = [
    "Give me 6 wafers on a 300 mm wafer with edge ring failures — lot ID DEMO_A01",
    "5 wafers with center cluster defects at 92% yield, 8×8 mm dies, 200 mm wafer",
    "Full lot with a soft repeater pattern, CP1 and CP2 insertions",
    "8 wafers with striping on the top of each reticle field — lens tilt demo",
    "4 lots per week of edge ring wafers with defect density 0.4 per cm2",
    "Show me a bull's-eye pattern on 10 wafers with 5 mm dies and a bad probe site",
]

_SIG_BIN = {
    "Edge Ring": 2, "Center Cluster": 3, "Scratch / Streak": 4,
    "Random Scatter": 5, "Quadrant Failure": 6, "Bull's-Eye": 7,
    "Full Pass": 1, "Donut (Mid-Ring)": 8,
    "Half Wafer — Top": 9, "Half Wafer — Bottom": 9,
    "Half Wafer — Left": 9, "Half Wafer — Right": 9,
    "Cross Pattern": 10, "Hot Spot": 11, "Reticle Pattern": 12,
    "Low Yield": 13, "Corner Clusters": 14, "Ring Crack": 15,
    "Wedge / Sector": 16, "Systematic Grid — Row": 17,
    "Systematic Grid — Column": 17, "Multi-Cluster": 18,
    "Top Edge Arc": 19, "Bottom Edge Arc": 20,
    "Diagonal Scratch": 21, "Concentric Rings": 22,
    "Peripheral Spot": 23, "Radial Spokes": 24,
    "Mixed Mode (Edge + Center)": 25,
    "Striping — Top": 30, "Striping — Bottom": 30,
    "Striping — Left": 30, "Striping — Right": 30,
    # Scratch families (bins 26-29), pulled straight from SCRATCH_FAMILIES so
    # the bin numbers can never drift out of sync with signatures.py.
    **{name: info["bin"] for name, info in SCRATCH_FAMILIES.items()},
}


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wafer Map Bot",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="auto",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _init_session_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "page" not in st.session_state:
        st.session_state["page"] = "chat"


def _compute_yields(all_wafers):
    """Per-wafer yield % from internal-bin die results."""
    yields = []
    for wafer in all_wafers:
        total = len(wafer)
        passed = sum(1 for d in wafer if BIN_DEFINITIONS.get(d[4], {}).get("state") == "P")
        yields.append(passed / total * 100 if total else 0.0)
    return yields


def _config_summary(result: GenerationResult, sig_name: str = "") -> str:
    """Short human-readable config line for captions and chat replies."""
    config = result.config
    parts = [
        f"{int(config.diameter)} mm",
        f"{config.die_width}×{config.die_height} mm dies",
        (f"notch {config.edge_orientation}" if config.edge_type == "notch"
         else f"flat {config.edge_orientation}"),
        f"{config.edge_exclusion:g} mm edge excl",
        f"{config.street_width * 1000:.0f} µm street",
        f"reticle {config.dies_per_reticle_x}×{config.dies_per_reticle_y}",
    ]
    if len(result.insertion_names) > 1:
        parts.append(" + ".join(result.insertion_names))
    if result.site_count > 1:
        parts.append(f"{result.site_count} sites")
    if len(result.lots) > 1:
        parts.append(f"{len(result.lots)} lots")
    return " · ".join(parts)


def _build_zip_bytes(all_wafers, titles, config, fmt: str) -> bytes:
    """Render each wafer individually and pack into a ZIP archive."""
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for wafer, title in zip(all_wafers, titles):
            fig_w = render_wafermaps([wafer], config, titles=[title])
            zf.writestr(f"{title}.{fmt}", figure_to_bytes(fig_w, fmt=fmt))
            plt.close(fig_w)
    zip_buf.seek(0)
    return zip_buf.getvalue()


def _stdf_download_bytes(result: GenerationResult) -> tuple[bytes, str]:
    """STDF export: a single .stdf when there is exactly one file, otherwise
    a ZIP holding one STDF per lot per insertion (a real sort run produces
    one file per insertion)."""
    files = result.stdf_files()
    if len(files) == 1:
        name, data = next(iter(files.items()))
        return data, name
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    zip_buf.seek(0)
    return zip_buf.getvalue(), f"{result.primary_lot.lot_id}_stdf.zip"


def _render_downloads(result: GenerationResult, key_prefix: str, precomputed=None):
    """Compact download rows for chat or manual results.

    Uses a per-entry, per-format session-state cache so images and ZIPs are
    never recomputed on Streamlit re-renders — only when a new image format
    is selected for the first time.
    """
    lot = result.primary_lot
    cp1_wafers = lot.insertion_wafers("CP1")
    titles = lot.wafer_ids
    config = result.config

    img_fmt = st.selectbox(
        "Image format",
        IMAGE_FORMATS,
        format_func=str.upper,
        key=f"{key_prefix}_img_fmt",
    )

    cache_key = f"_dl_cache_{key_prefix}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = dict(precomputed) if precomputed else {}
    cache = st.session_state[cache_key]

    # Format-agnostic exports (CSV / per-test CSV / STDF), cached once.
    if "csv" not in cache:
        cache["csv"] = result.df.to_csv(index=False).encode("utf-8")
    if "stdf" not in cache:
        with st.spinner("Building STDF…"):
            cache["stdf"], cache["stdf_name"] = _stdf_download_bytes(result)
    if "param_csv" not in cache and result.param_df is not None:
        cache["param_csv"] = result.param_df.to_csv(index=False).encode("utf-8")
    if "ft_csv" not in cache and result.ft_df is not None:
        cache["ft_csv"] = result.ft_df.to_csv(index=False).encode("utf-8")
    if "match_csv" not in cache and result.match_df is not None:
        cache["match_csv"] = result.match_df.to_csv(index=False).encode("utf-8")
    if "manifest_json" not in cache and result.story_manifest is not None:
        import json
        cache["manifest_json"] = json.dumps(
            result.story_manifest, indent=2).encode("utf-8")
    if "multidie_csv" not in cache and result.multidie_df is not None:
        cache["multidie_csv"] = result.multidie_df.to_csv(index=False).encode("utf-8")

    # Image grid + per-wafer ZIP for the selected format (first lot, CP1).
    fmt_cache = cache.get(img_fmt, {})
    if "grid" not in fmt_cache:
        with st.spinner(f"Rendering {img_fmt.upper()} grid…"):
            fig_grid = render_wafermaps(cp1_wafers, config, titles=titles)
            fmt_cache["grid"] = figure_to_bytes(fig_grid, fmt=img_fmt)
            plt.close(fig_grid)
        cache[img_fmt] = fmt_cache
    if "zip" not in fmt_cache:
        with st.spinner(f"Building {img_fmt.upper()} ZIP ({len(cp1_wafers)} wafers)…"):
            fmt_cache["zip"] = _build_zip_bytes(cp1_wafers, titles, config, fmt=img_fmt)
        cache[img_fmt] = fmt_cache

    lot_id = lot.lot_id
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.download_button(
            "CSV (die level)",
            data=cache["csv"],
            file_name=f"{lot_id}_wafer_data.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_csv",
        )
    with r1c2:
        st.download_button(
            "STDF" if cache["stdf_name"].endswith(".stdf") else "STDF (ZIP per insertion)",
            data=cache["stdf"],
            file_name=cache["stdf_name"],
            mime="application/octet-stream",
            use_container_width=True,
            key=f"{key_prefix}_stdf",
        )

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.download_button(
            f"Grid ({img_fmt.upper()})",
            data=cache[img_fmt]["grid"],
            file_name=f"{lot_id}_wafermap.{img_fmt}",
            mime=f"image/{img_fmt}" if img_fmt != "svg" else "image/svg+xml",
            use_container_width=True,
            key=f"{key_prefix}_grid",
        )
    with r2c2:
        st.download_button(
            f"ZIP per wafer ({img_fmt.upper()})",
            data=cache[img_fmt]["zip"],
            file_name=f"{lot_id}_individual_maps_{img_fmt}.zip",
            mime="application/zip",
            use_container_width=True,
            key=f"{key_prefix}_zip",
        )

    if "param_csv" in cache:
        st.download_button(
            "CSV (per-test results, CP1)",
            data=cache["param_csv"],
            file_name=f"{lot_id}_test_results.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_param_csv",
        )

    if "ft_csv" in cache or "match_csv" in cache:
        st.markdown("**Story 1 — FT / traceability**")
        c1, c2, c3 = st.columns(3)
        with c1:
            if "ft_csv" in cache:
                st.download_button(
                    "CSV (Final Test units)",
                    data=cache["ft_csv"],
                    file_name=f"{lot_id}_ft_units.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"{key_prefix}_ft_csv",
                )
        with c2:
            if "match_csv" in cache:
                st.download_button(
                    "CSV (match ground truth)",
                    data=cache["match_csv"],
                    file_name=f"{lot_id}_ecid_match.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"{key_prefix}_match_csv",
                )
        with c3:
            if "manifest_json" in cache:
                st.download_button(
                    "Manifest (JSON)",
                    data=cache["manifest_json"],
                    file_name=f"{lot_id}_story1_manifest.json",
                    mime="application/json",
                    use_container_width=True,
                    key=f"{key_prefix}_manifest",
                )

    if "multidie_csv" in cache:
        st.markdown("**Story 1c — multi-die product traceability**")
        st.download_button(
            "CSV (multi-die products)",
            data=cache["multidie_csv"],
            file_name=f"{lot_id}_multidie_products.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_multidie_csv",
        )
        if "manifest_json" in cache:
            st.download_button(
                "Manifest (JSON)",
                data=cache["manifest_json"],
                file_name=f"{lot_id}_multidie_manifest.json",
                mime="application/json",
                use_container_width=True,
                key=f"{key_prefix}_multidie_manifest",
            )


def _render_result(result: GenerationResult, sig_name: str, key_prefix: str,
                   preview_bytes=None, precomputed=None):
    """Inline wafer maps, summaries and downloads for one generation run."""
    lot = result.primary_lot
    cp1_wafers = lot.insertion_wafers("CP1")
    titles = lot.wafer_ids
    yields = _compute_yields(cp1_wafers)
    total_dies = len(cp1_wafers[0]) if cp1_wafers else 0
    avg_yield = sum(yields) / len(yields) if yields else 0

    st.caption(
        f"{lot.lot_id} · {sig_name} · {len(lot.wafers)} wafers · "
        f"{total_dies} dies/wafer · {avg_yield:.1f}% CP1 avg yield · "
        f"{_config_summary(result, sig_name)}"
    )

    # CP1 maps of the first lot (the primary preview).
    if preview_bytes:
        st.image(preview_bytes, use_container_width=True)
    else:
        fig = render_wafermaps(cp1_wafers, result.config, titles=titles)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Extra insertions: render on demand inside an expander (cheap for CP2/3
    # because they reuse the same geometry).
    if len(result.insertion_names) > 1:
        with st.expander("CP2 / CP3 maps (retest fallout)"):
            for ins in result.insertion_names[1:]:
                ins_wafers = lot.insertion_wafers(ins)
                ins_yields = _compute_yields(ins_wafers)
                avg = sum(ins_yields) / len(ins_yields) if ins_yields else 0
                st.markdown(f"**{ins}** — {avg:.1f}% avg yield")
                fig = render_wafermaps(ins_wafers, result.config,
                                       titles=[f"{t} {ins}" for t in titles])
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    _render_downloads(result, key_prefix, precomputed=precomputed)

    with st.expander("Yield summary & CSV preview"):
        # Per-wafer, per-insertion yield table.
        summary_rows = []
        for wafer in lot.wafers:
            row = {"Wafer ID": wafer.wafer_id, "Total Dies": total_dies}
            for ins in result.insertion_names:
                results = wafer.insertions[ins]
                passed = sum(1 for d in results
                             if BIN_DEFINITIONS.get(d[4], {}).get("state") == "P")
                row[f"{ins} Yield (%)"] = round(passed / len(results) * 100, 2)
            summary_rows.append(row)
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        if result.s2s is not None:
            st.caption(
                "S2S factors per site: "
                + ", ".join(f"site {i + 1}: {f:.2f}" for i, f in enumerate(result.s2s))
            )
        if len(result.lots) > 1:
            lots_df = pd.DataFrame([
                {"Lot": l.lot_id,
                 "Sort start": l.start_time.strftime("%Y-%m-%d %H:%M"),
                 "Wafers": len(l.wafers)}
                for l in result.lots
            ])
            st.dataframe(lots_df, use_container_width=True, hide_index=True)

        st.dataframe(result.df.head(100), use_container_width=True, hide_index=True)

    if result.story_summary:
        story_id = (result.story_manifest or {}).get("story_id", "story1")
        with st.expander("Story 1 — FT / ECID summary", expanded=True):
            s = result.story_summary
            if story_id == "story1_multidie":
                st.write(
                    f"Products: {int(s.get('products', 0))} · "
                    f"FT yield: {s.get('ft_yield_pct', 0):.1f}% · "
                    f"Fully traceable: {s.get('fully_traceable_pct', 0):.1f}%"
                )
            else:
                st.write(
                    f"FT units: {int(s.get('ft_units', 0))} · "
                    f"FT yield: {s.get('ft_yield_pct', 0):.1f}% · "
                    f"Blank ECID: {s.get('blank_ecid_pct', 0):.2f}% · "
                    f"Mispick: {s.get('mispick_pct', 0):.1f}%"
                    + (f" · Spatial hazard (GDBN) units: "
                       f"{100.0 * s.get('asm_hazard_pct_actual', 0):.1f}% "
                       f"({int(s.get('asm_hazard_units', 0))} die)"
                       if story_id == "story1_gdbn" else "")
                )
            if result.story_manifest:
                st.caption(
                    f"Scenario: {result.story_manifest.get('scenario_id')} · "
                    f"FT lots: {result.story_manifest.get('ft_lot_ids')} · "
                    f"GDPW: {result.story_manifest.get('gdpw')}"
                )
                if result.story_manifest.get("warnings"):
                    st.warning(" · ".join(result.story_manifest["warnings"]))
            if result.ft_df is not None:
                st.dataframe(result.ft_df.head(50), use_container_width=True,
                             hide_index=True)
            if result.multidie_df is not None:
                st.dataframe(result.multidie_df.head(50), use_container_width=True,
                             hide_index=True)


def _run_generation(req: WaferGenRequest) -> dict:
    """Shared chat/manual path: request -> config -> pipeline -> result dict."""
    config = request_to_config(req)
    result = generate(req, config)

    lot = result.primary_lot
    preview_fig = render_wafermaps(lot.insertion_wafers("CP1"), result.config,
                                   titles=lot.wafer_ids)
    preview_bytes = figure_to_bytes(preview_fig, fmt="png")
    plt.close(preview_fig)

    # Seed the download cache with what we already rendered (zero extra cost).
    precomputed = {"png": {"grid": preview_bytes}}

    return {
        "result": result,
        "preview_bytes": preview_bytes,
        "precomputed": precomputed,
        "sig": " + ".join(req.signatures),
    }


def _process_chat_request(user_input: str) -> dict:
    """Parse prompt, generate wafers, return assistant message dict."""
    api_key = st.session_state.get("ai_api_key") or os.environ.get("AZURE_OPENAI_API_KEY") or None
    azure_endpoint = (
        st.session_state.get("ai_azure_endpoint")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
        or None
    )
    azure_deployment = (
        st.session_state.get("ai_azure_deployment")
        or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        or None
    )

    req: WaferGenRequest = parse_user_request(
        user_input,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        previous_request=st.session_state.get("last_request"),
    )

    payload = _run_generation(req)
    # Remember the request so follow-up messages ("change X, keep the rest")
    # modify it instead of starting from defaults.
    st.session_state["last_request"] = req

    method = "Azure GPT" if req.used_llm else "keyword parser"
    content = (
        f"{req.explanation}\n\n"
        f"*{method}* · **{payload['sig']}** · {req.num_wafers} wafers · "
        f"Lot {payload['result'].primary_lot.lot_id} · "
        f"{_config_summary(payload['result'], payload['sig'])}"
    )
    result = payload["result"]
    if result.story_summary:
        s = result.story_summary
        manifest = result.story_manifest or {}
        story_id = manifest.get("story_id", "story1")
        scenario_id = manifest.get("scenario_id")
        if story_id == "story1_multidie":
            content += (
                f"\n\n**Story 1c multi-die ({scenario_id})** — "
                f"{int(s.get('products', 0))} products · "
                f"FT yield {s.get('ft_yield_pct', 0):.1f}% · "
                f"fully traceable {s.get('fully_traceable_pct', 0):.1f}%. "
                "Download **CSV (multi-die products)** below."
            )
        elif story_id == "story1_gdbn":
            hazard_pct = 100.0 * s.get("asm_hazard_pct_actual", 0)
            content += (
                f"\n\n**Story 1g GDBN ({scenario_id})** — "
                f"{int(s.get('ft_units', 0))} FT units · "
                f"FT yield {s.get('ft_yield_pct', 0):.1f}% · "
                f"spatial hazard die {hazard_pct:.1f}% · "
                f"FT lots {manifest.get('ft_lot_ids')}. "
                "Download **CSV (Final Test units)** and **CSV (match ground truth)** below."
            )
        else:
            content += (
                f"\n\n**Story 1 ({scenario_id})** — "
                f"{int(s.get('ft_units', 0))} FT units · "
                f"FT yield {s.get('ft_yield_pct', 0):.1f}% · "
                f"blank ECID {s.get('blank_ecid_pct', 0):.2f}% · "
                f"FT lots {manifest.get('ft_lot_ids')}. "
                "Download **CSV (Final Test units)** and **CSV (match ground truth)** below."
            )

    return {"role": "assistant", "content": content, "payload": payload}


def _handle_user_message(user_input: str) -> None:
    """Append user message, generate response, update chat history."""
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.spinner("Generating wafer maps…"):
        assistant_msg = _process_chat_request(user_input)
    st.session_state["chat_history"].append(assistant_msg)


def _render_signature_help() -> None:
    with st.expander(f"What can I ask for? ({len(SIGNATURE_NAMES)} spatial signatures)"):
        st.caption(
            "Describe any pattern in plain English — e.g. \"edge ring failures\", "
            "\"soft repeaters\", \"striping\". Combine several (\"edge ring and a "
            "scratch\") to layer them. You can also specify yield (\"92% yield\" or "
            "\"0.5 defects/cm2\"), insertions (\"CP1 and CP2\"), lot sequences "
            "(\"4 lots per week\"), street width, test time, and more."
        )
        for name in SIGNATURE_NAMES:
            bn = _SIG_BIN.get(name, 5)
            info = BIN_DEFINITIONS.get(bn, {})
            line = f"**{name}** — {info.get('description', '')}"
            fam = SCRATCH_FAMILIES.get(name)
            if fam:
                line += f"  \n_Tool:_ {fam['tool']}. _Root cause:_ {fam['root_cause']}."
            st.markdown(line)


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Settings")

        if st.button("Chat", use_container_width=True,
                       type="primary" if st.session_state["page"] == "chat" else "secondary"):
            st.session_state["page"] = "chat"
            st.rerun()

        if st.button("Manual configuration", use_container_width=True,
                       type="primary" if st.session_state["page"] == "manual" else "secondary"):
            st.session_state["page"] = "manual"
            st.rerun()

        if st.button("Stories (ECID / FT)", use_container_width=True,
                       type="primary" if st.session_state["page"] == "stories" else "secondary"):
            st.session_state["page"] = "stories"
            st.rerun()

        st.divider()

        with st.expander("Azure OpenAI (optional)"):
            st.caption(
                "Set credentials here, or via environment variables / Streamlit secrets. "
                "Without credentials the keyword parser is used."
            )
            azure_endpoint_input = st.text_input(
                "Endpoint",
                value=st.session_state.get(
                    "ai_azure_endpoint",
                    os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
                ),
                placeholder="https://your-resource.openai.azure.com/",
                key="sidebar_azure_endpoint",
            )
            azure_key_input = st.text_input(
                "API key",
                type="password",
                value=st.session_state.get("ai_api_key", ""),
                placeholder="Leave blank to use env var",
                key="sidebar_azure_key",
            )
            azure_deployment_input = st.text_input(
                "Deployment",
                value=st.session_state.get(
                    "ai_azure_deployment",
                    os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"),
                ),
                key="sidebar_azure_deployment",
            )
            if azure_endpoint_input:
                st.session_state["ai_azure_endpoint"] = azure_endpoint_input.strip()
            if azure_key_input:
                st.session_state["ai_api_key"] = azure_key_input
            if azure_deployment_input:
                st.session_state["ai_azure_deployment"] = azure_deployment_input.strip()

            has_azure = st.session_state.get("ai_azure_endpoint") and (
                st.session_state.get("ai_api_key") or os.environ.get("AZURE_OPENAI_API_KEY")
            )
            if has_azure:
                st.success("Azure OpenAI configured.")
            else:
                st.info("Using keyword parser fallback.")

        st.divider()

        if st.button("Clear conversation", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state.pop("last_request", None)
            st.rerun()


def _render_chat_page() -> None:
    st.markdown("## Wafer Map Bot")
    st.caption(
        "Describe the wafer maps you need in plain English. "
        "I'll generate CSV, images (PNG/SVG/JPEG/TIFF), and STDF outputs."
    )

    _render_signature_help()

    st.markdown("**Try an example:**")
    ex_cols = st.columns(3)
    for i, ex in enumerate(EXAMPLE_PROMPTS):
        if ex_cols[i % 3].button(ex[:55] + "…", key=f"ex_{i}", use_container_width=True):
            _handle_user_message(ex)
            st.rerun()

    pending = st.session_state.pop("pending_prompt", None)
    if pending:
        _handle_user_message(pending)
        st.rerun()

    for idx, msg in enumerate(st.session_state["chat_history"]):
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])
            if msg.get("payload"):
                p = msg["payload"]
                _render_result(
                    p["result"], p["sig"], key_prefix=f"chat_{idx}",
                    preview_bytes=p.get("preview_bytes"),
                    precomputed=p.get("precomputed"),
                )

    if not st.session_state["chat_history"]:
        st.info("Send a message below or click an example to get started.")

    user_input = st.chat_input("Describe the wafer maps you need…")
    if user_input:
        _handle_user_message(user_input.strip())
        st.rerun()


def _render_manual_page() -> None:
    st.markdown("## Manual Configuration")
    st.caption("Set parameters directly without the chatbot. Results appear below after you generate.")

    with st.form("manual_form"):
        # ---- Geometry (spec must-haves) --------------------------------------
        c1, c2 = st.columns(2)
        with c1:
            diameter = st.selectbox(
                "Wafer diameter (mm)", [int(d) for d in STANDARD_DIAMETERS],
                index=2,
                help="Spec sizes only. 150 mm wafers get a FLAT, 200/300 mm a NOTCH.",
            )
            edge_orientation = st.radio(
                "Notch/flat orientation",
                ["down", "up", "left", "right"],
                horizontal=True,
                help="Which side of the map the edge marker sits on (90° steps).",
            )
            edge_exclusion = st.slider(
                "Edge exclusion (mm)",
                EDGE_EXCLUSION_MIN, EDGE_EXCLUSION_MAX, 3.0, 0.5,
            )
            signature_types = st.multiselect(
                "Signature(s)", SIGNATURE_NAMES, default=["Edge Ring"],
                help="Pick one, or several to layer them (e.g. Edge Ring + a scratch).",
            )
        with c2:
            die_width = st.number_input(
                "Die width (mm)", 1.0, 35.0, 10.0, 0.5,
                help="1–25/35 mm; aspect ratio must stay between 1:2 and 2:1.",
            )
            die_height = st.number_input("Die height (mm)", 1.0, 35.0, 10.0, 0.5)
            street_width = st.number_input(
                "Street width (mm)", STREET_MIN, STREET_MAX, STREET_DEFAULT, 0.05,
                help="Scribe street between dies: 0.05–0.2 mm (spec).",
            )
            lot_id = st.text_input("Lot ID", "LOT_001")
            program = st.text_input("Program", "HBN_PRD020")
            lot_preset = st.selectbox(
                "Lot size", list(LOT_SIZE_PRESETS.keys()), index=0,
                help="Wafers ship in FOUP carriers: 25 standard, 13 thin/bonded. "
                     "Choose 'Partial Lot' to set a custom count.",
            )
            custom_wafers = st.slider(
                "Custom wafer count (used for 'Partial Lot')", 1, MAX_WAFERS, 4,
            )
            preset_value = LOT_SIZE_PRESETS[lot_preset]
            num_wafers = preset_value if preset_value is not None else custom_wafers

        # ---- Yield model ---------------------------------------------------
        with st.expander("Yield model"):
            yield_mode = st.radio(
                "Yield input", ["signature", "direct", "defect_density"],
                horizontal=True,
                format_func={"signature": "From signature",
                             "direct": "Direct yield %",
                             "defect_density": "Defect density"}.get,
                help="Direct: set wafer yield. Defect density: Y = e^(-A·D) with "
                     "A = die area in cm². Signature: whatever the pattern gives.",
            )
            target_yield_pct = st.slider("Target yield (%)", 0.0, 100.0, 90.0, 0.5)
            defect_density = st.number_input(
                "Defect density (defects/cm²)", 0.0, 20.0, 0.5, 0.05,
            )
            yield_variation_pct = st.slider(
                "Per-wafer variation (±%)", 0.0, 50.0, 0.0, 5.0,
                help="Each wafer jitters its defect density (or direct yield "
                     "target) by a random amount inside this band, so the lot "
                     "shows realistic wafer-to-wafer yield spread.",
            )

        # ---- Insertions & bins ------------------------------------------------
        with st.expander("Test insertions & bins"):
            num_insertions = st.radio(
                "Insertions", [1, 2, 3], horizontal=True,
                format_func=lambda n: ["CP1", "CP1 + CP2", "CP1 + CP2 + CP3"][n - 1],
                help="CP2/CP3 keep 90–99.9% of the previous insertion's passers; "
                     "only prior passers can pass a retest.",
            )
            bc1, bc2 = st.columns(2)
            with bc1:
                hardbin_count = st.selectbox("Hardbins", list(HARDBIN_CHOICES))
            with bc2:
                softbin_multiplier = st.selectbox(
                    "Softbins = hardbins ×", list(SOFTBIN_MULTIPLIERS))

        # ---- Test items ------------------------------------------------------
        with st.expander("Test items"):
            test_count = st.selectbox(
                "Number of test items", list(TEST_COUNT_CHOICES),
                help="Orders of magnitude only (spec: 100 or 1000, never 307).",
            )
            parametric_pct = st.slider(
                "Parametric share (%)", 0, 100, 50, 10,
                help="Rest are pass/fail items reporting 0 or 1.",
            )
            value_shape = st.selectbox(
                "Parametric value shape", list(VALUE_SHAPES.keys()),
                help="uniform: RNG 0–1 · exponential: 10^RNG · quantized: 0.2 "
                     "steps · signed: -1..+1 · scientific: X.XXe±YY · constant: one value.",
            )
            nc1, nc2 = st.columns(2)
            with nc1:
                naming_style = st.selectbox(
                    "Test name style", list(NAMING_STYLES),
                    help="simple: PARAM_0001 · obnoxious: long names differing only "
                         "at the end (UI checker) · chunked: gibberish 8-char chunks.",
                )
            with nc2:
                name_length = st.selectbox("Verbose name length", list(VERBOSE_LENGTHS))
            include_test_data = st.checkbox(
                "Write per-test results (PTRs in STDF + per-test CSV)",
                value=False,
                help="One record per die per test item — files get big fast; "
                     "a size estimate is checked before generating.",
            )

        # ---- Timing, multi-site, lots -----------------------------------------
        with st.expander("Test time, multi-site & lot sequence"):
            seconds_per_touchdown = st.slider(
                "Test time (seconds per touchdown)", 1.0, 600.0, 1.0, 1.0,
                help="Drives per-die test time and wafer start/finish timestamps.",
            )
            multi_site = st.checkbox(
                "Multi-site (parallelism from gross die per wafer)", value=True,
                help="<200 GDPW: 1 site · 200–399: 2 · 400–799: 4 · 800–1599: 8 · 1600+: 16",
            )
            site_pattern = st.selectbox(
                "Site layout pattern", list(SITE_PATTERNS), index=2,
            )
            s2s_enabled = st.checkbox(
                "Site-to-site yield loss (one weak site)", value=False,
                help="Healthy setups have all S2S factors > 95%; this drags one "
                     "random site down so the loss is visible per site.",
            )
            st.divider()
            num_lots = st.number_input(
                "Number of lots", 1, 60, 1, 1,
                help="More than 1 generates a time sequence of fab lots "
                     "(FYYWWSSSS IDs) for trend-chart demos.",
            )
            lot_cadence = st.selectbox("Lot cadence", list(LOT_CADENCES.keys()), index=1)
            auto_lot_id = st.checkbox(
                "Auto fab lot number (FYYWWSSSS)", value=False,
                help="Fab letter + year + work week + sequential number, "
                     "replacing the Lot ID field above.",
            )

        # ---- Advanced geometry ----------------------------------------------
        with st.expander("Grid offset & stepping field"):
            oc1, oc2 = st.columns(2)
            with oc1:
                x_offset = st.number_input("X offset (mm)", -10.0, 10.0, 0.0, 0.5)
            with oc2:
                y_offset = st.number_input("Y offset (mm)", -10.0, 10.0, 0.0, 0.5)
            auto_reticle = st.checkbox(
                "Auto stepping field from die size", value=True,
                help="Packs as many dies as fit into a 26×33 mm scanner field "
                     "(spec: the tool must autogenerate the stepping field).",
            )
            rc1, rc2 = st.columns(2)
            with rc1:
                dies_per_reticle_x = st.number_input(
                    "Dies per reticle (X, manual)", 1, 10, 2, 1)
                reticle_fail_die_x = st.number_input(
                    "Repeater die column (0-based)", 0, 9, 0, 1)
            with rc2:
                dies_per_reticle_y = st.number_input(
                    "Dies per reticle (Y, manual)", 1, 10, 2, 1)
                reticle_fail_die_y = st.number_input(
                    "Repeater die row (0-based)", 0, 9, 0, 1)
            repeater_fail_rate = st.slider(
                "Repeater fail rate (%)", 10, 100, 100, 10,
                help="100% = hard repeater (always fails). Lower = soft repeater.",
            )
            stripe_fail_rate = st.slider(
                "Striping fail rate (%)", 10, 100, 100, 10,
                help="Hardness of the Striping signatures (lens-tilt stripes).",
            )

        generate_btn = st.form_submit_button("Generate", type="primary", use_container_width=True)

    if generate_btn:
        # Guard against an empty multiselect — fall back to a sensible default.
        if not signature_types:
            signature_types = ["Edge Ring"]

        # Spec validation: die aspect ratio must stay between 1:2 and 2:1.
        ok, msg = validate_die_size(die_width, die_height)
        if not ok:
            st.error(msg)
            return

        # Size guardrail before generating per-test data.
        if include_test_data:
            # Rough GDPW estimate: usable area / die area.
            import math
            usable_r = diameter / 2 - edge_exclusion
            est_dies = int(math.pi * usable_r ** 2 / (die_width * die_height))
            est_records = estimate_result_count(test_count, est_dies,
                                                num_wafers * int(num_lots))
            if est_records > 5_000_000:
                st.error(
                    f"Per-test export would contain ~{est_records:,} records. "
                    "Reduce the test count, wafer count or lots (cap: 5,000,000)."
                )
                return
            if est_records > 1_000_000:
                st.warning(f"Large export: ~{est_records:,} per-test records.")

        req = WaferGenRequest(
            diameter=float(diameter),
            edge_type=auto_edge_type(float(diameter)),
            edge_exclusion=float(edge_exclusion),
            die_width=float(die_width),
            die_height=float(die_height),
            x_offset=float(x_offset),
            y_offset=float(y_offset),
            street_width=float(street_width),
            edge_orientation=edge_orientation,
            auto_reticle=bool(auto_reticle),
            dies_per_reticle_x=int(dies_per_reticle_x),
            dies_per_reticle_y=int(dies_per_reticle_y),
            reticle_fail_die_x=int(reticle_fail_die_x),
            reticle_fail_die_y=int(reticle_fail_die_y),
            repeater_fail_rate=repeater_fail_rate / 100.0,
            stripe_fail_rate=stripe_fail_rate / 100.0,
            lot_id=lot_id,
            program=program,
            num_wafers=int(num_wafers),
            num_lots=int(num_lots),
            lot_cadence=lot_cadence,
            auto_lot_id=bool(auto_lot_id),
            yield_mode=yield_mode,
            target_yield_pct=float(target_yield_pct),
            defect_density=float(defect_density),
            yield_variation_pct=float(yield_variation_pct),
            num_insertions=int(num_insertions),
            hardbin_count=int(hardbin_count),
            softbin_multiplier=int(softbin_multiplier),
            test_count=int(test_count),
            parametric_pct=int(parametric_pct),
            value_shape=value_shape,
            naming_style=naming_style,
            name_length=int(name_length),
            include_test_data=bool(include_test_data),
            seconds_per_touchdown=float(seconds_per_touchdown),
            multi_site=bool(multi_site),
            site_pattern=site_pattern,
            s2s_enabled=bool(s2s_enabled),
            s2s_healthy=not bool(s2s_enabled),
            signatures=signature_types,
        )

        with st.spinner("Generating…"):
            payload = _run_generation(req)
        st.session_state["manual_result"] = payload
        # Clear any stale download cache for this entry
        st.session_state.pop("_dl_cache_manual", None)

    if "manual_result" in st.session_state:
        p = st.session_state["manual_result"]
        st.divider()
        _render_result(
            p["result"], p["sig"], key_prefix="manual",
            preview_bytes=p.get("preview_bytes"),
            precomputed=p.get("precomputed"),
        )


def _render_stories_page() -> None:
    """Story 1 ECID matching / FT traceability demos."""
    from assembly import STORY1_SCENARIOS
    from story1_presets import (
        SCENARIO_LABELS, GDBN_SCENARIO_LABELS, MULTIDIE_SCENARIO_LABELS,
        ENCODING_MODE_LABELS, REPRESENTATION_LABELS,
        apply_scenario_to_request, apply_gdbn_scenario_to_request,
        apply_multidie_scenario_to_request,
    )

    st.markdown("## Stories — ECID Matching (Story 1)")
    st.caption(
        "Generate CP + Final Test data with ECID traceability scenarios "
        "from the Yield Stories doc. Downloads include FT units and a "
        "ground-truth match table (do not join on blank ECID)."
    )

    family = st.radio(
        "Story family",
        ["1:1 / Sweeper / Assembly errors", "Low yield at FT (GDBN)", "Multi-die products"],
        horizontal=True,
        help=(
            "1:1/Sweeper/Assembly errors = spec 1.d-f. "
            "GDBN = spec 1.g (low yield at FT caused by CP clusters). "
            "Multi-die = spec 1.c (Case B: multiple die -> one FT product)."
        ),
    )

    with st.expander("ECID encoding (spec 1.b)", expanded=False):
        ec1, ec2 = st.columns(2)
        with ec1:
            mode_label = st.selectbox(
                "ECID value", list(ENCODING_MODE_LABELS.values()), index=0,
                key="story1_ecid_mode",
            )
            ecid_mode = [k for k, v in ENCODING_MODE_LABELS.items() if v == mode_label][0]
        with ec2:
            repr_label = st.selectbox(
                "ECID representation", list(REPRESENTATION_LABELS.values()), index=0,
                key="story1_ecid_repr",
            )
            ecid_representation = [
                k for k, v in REPRESENTATION_LABELS.items() if v == repr_label][0]

    if family == "Low yield at FT (GDBN)":
        _render_gdbn_form(GDBN_SCENARIO_LABELS, apply_gdbn_scenario_to_request,
                          ecid_mode, ecid_representation)
        return
    if family == "Multi-die products":
        _render_multidie_form(MULTIDIE_SCENARIO_LABELS, apply_multidie_scenario_to_request,
                              ecid_mode, ecid_representation)
        return

    labels = [SCENARIO_LABELS.get(k, k) for k in STORY1_SCENARIOS]
    keys = list(STORY1_SCENARIOS.keys())

    with st.form("story1_form"):
        scenario_label = st.selectbox("Scenario", labels, index=0)
        scenario_id = keys[labels.index(scenario_label)]

        c1, c2 = st.columns(2)
        with c1:
            num_wafers = st.slider("Wafers per CP lot", 1, 25, 5)
            blank_pct = st.slider(
                "Blank ECID % (detail / wrecks)", 0.0, 5.0, 1.5, 0.1,
                help="Assembly wrecks for detail scenarios.",
            )
            valid_mix = st.slider(
                "Mispick valid-ECID mix %", 0.0, 100.0, 50.0, 5.0,
                help="For wrong-bin / wrong-XY: share keeping a valid ECID.",
            )
        with c2:
            mispick_fail = st.slider(
                "Mispick FT fail %", 0.0, 100.0, 100.0, 5.0,
                help="Simple = 100%. Subtle FT default = 80%.",
            )
            baseline_ft = st.slider(
                "Baseline FT fallout % (correct picks)", 0.0, 20.0, 3.0, 0.5,
            )
            num_lots = st.number_input(
                "CP lots (sweeper uses ≥2)", 1, 10, 1, 1,
            )

        gen = st.form_submit_button(
            "Generate Story 1 data", type="primary", use_container_width=True,
        )

    if gen:
        req = WaferGenRequest(
            num_wafers=int(num_wafers),
            num_lots=int(num_lots),
            signatures=["Edge Ring"],
            test_count=0,
            multi_site=False,
        )
        apply_scenario_to_request(req, scenario_id)
        req.ecid_mode = ecid_mode
        req.ecid_representation = ecid_representation
        req.blank_ecid_pct = blank_pct / 100.0
        req.valid_ecid_mix = valid_mix / 100.0
        req.mispick_ft_fail_pct = mispick_fail / 100.0
        req.baseline_ft_fallout = baseline_ft / 100.0
        if scenario_id.endswith("_simple"):
            req.blank_ecid_pct = 0.0
        if scenario_id.endswith("_detail"):
            req.blank_ecid_pct = blank_pct / 100.0
        if "subtle_ft" in scenario_id and mispick_fail >= 99.0:
            req.mispick_ft_fail_pct = 0.8

        with st.spinner("Generating Story 1 CP + FT…"):
            payload = _run_generation(req)
        st.session_state["stories_result"] = payload
        st.session_state.pop("_dl_cache_stories", None)

    if "stories_result" in st.session_state:
        p = st.session_state["stories_result"]
        st.divider()
        _render_result(
            p["result"], p["sig"], key_prefix="stories",
            preview_bytes=p.get("preview_bytes"),
            precomputed=p.get("precomputed"),
        )


def _render_gdbn_form(labels: dict, apply_fn, ecid_mode: str, ecid_representation: str) -> None:
    """Story 1g: low yield at FT caused by CP clusters (GDBN)."""
    st.caption(
        "Dramatic = CP is clean, but FT-only fallout traces a spatial pattern "
        "(default: donut) once mapped back onto CP (X, Y) via ECID. "
        "Good-die-bad-neighborhood = CP has a real scratch, and passers next "
        "to it get a default 50% chance of failing FT anyway."
    )
    keys = list(labels.keys())
    display = [labels[k] for k in keys]

    with st.form("story1_gdbn_form"):
        scenario_label = st.selectbox("GDBN scenario", display, index=0)
        scenario_id = keys[display.index(scenario_label)]

        c1, c2 = st.columns(2)
        with c1:
            num_wafers = st.slider("Wafers per CP lot", 1, 25, 5, key="gdbn_wafers")
            growth = st.slider(
                "Neighbourhood growth (dies)", 0, 3, 1, 1,
                help="Good-die-bad-neighborhood only: dilate the CP fail region "
                     "by this many dies in every dimension.",
            )
        with c2:
            fail_pct = st.slider(
                "Spatial hazard FT fail %", 0.0, 100.0, 50.0, 5.0,
                help="Good-die-bad-neighborhood default = 50%. Dramatic case "
                     "uses the donut's own fail rate (~87%) regardless of this slider.",
            )
            baseline_ft = st.slider(
                "Baseline FT fallout % (unaffected die)", 0.0, 20.0, 3.0, 0.5,
                key="gdbn_baseline",
            )

        gen = st.form_submit_button(
            "Generate GDBN data", type="primary", use_container_width=True,
        )

    if gen:
        req = WaferGenRequest(
            num_wafers=int(num_wafers),
            test_count=0,
            multi_site=False,
        )
        apply_fn(req, scenario_id)
        req.ecid_mode = ecid_mode
        req.ecid_representation = ecid_representation
        req.gdbn_growth = int(growth)
        req.gdbn_fail_pct = fail_pct / 100.0
        req.baseline_ft_fallout = baseline_ft / 100.0

        with st.spinner("Generating GDBN CP + FT…"):
            payload = _run_generation(req)
        st.session_state["stories_result"] = payload
        st.session_state.pop("_dl_cache_stories", None)

    if "stories_result" in st.session_state:
        p = st.session_state["stories_result"]
        st.divider()
        _render_result(
            p["result"], p["sig"], key_prefix="stories",
            preview_bytes=p.get("preview_bytes"),
            precomputed=p.get("precomputed"),
        )


def _render_multidie_form(labels: dict, apply_fn, ecid_mode: str, ecid_representation: str) -> None:
    """Story 1c: multi-die product traceability (Case B of the 2x2 matrix)."""
    from multidie import COMPONENT_ROLES

    st.caption(
        "Packages 3 known-good die (logic / memory / rf, each a different "
        "size) into ONE FT product. B.1 = every component traceable back to "
        "CP. B.2 = the RF component never burns an ECID, so 1 of 3 stays "
        "permanently untraceable — even on products that pass FT."
    )
    st.table(pd.DataFrame(COMPONENT_ROLES).rename(columns={
        "role": "Component", "die_width": "Die W (mm)", "die_height": "Die H (mm)",
        "cp_yield": "Nominal CP yield",
    }))

    keys = list(labels.keys())
    display = [labels[k] for k in keys]

    with st.form("story1_multidie_form"):
        scenario_label = st.selectbox("Multi-die scenario", display, index=0)
        scenario_id = keys[display.index(scenario_label)]

        c1, c2 = st.columns(2)
        with c1:
            num_products = st.slider("Products to build", 10, 2000, 200, 10)
        with c2:
            baseline_ft = st.slider(
                "Per-component baseline FT fallout %", 0.0, 20.0, 5.0, 0.5,
                key="multidie_baseline",
            )

        gen = st.form_submit_button(
            "Generate multi-die data", type="primary", use_container_width=True,
        )

    if gen:
        req = WaferGenRequest(
            num_wafers=5,
            test_count=0,
            multi_site=False,
        )
        apply_fn(req, scenario_id)
        req.ecid_mode = ecid_mode
        req.ecid_representation = ecid_representation
        req.num_multidie_products = int(num_products)
        req.baseline_ft_fallout = baseline_ft / 100.0

        with st.spinner("Generating multi-die products…"):
            payload = _run_generation(req)
        st.session_state["stories_result"] = payload
        st.session_state.pop("_dl_cache_stories", None)

    if "stories_result" in st.session_state:
        p = st.session_state["stories_result"]
        st.divider()
        _render_result(
            p["result"], p["sig"], key_prefix="stories",
            preview_bytes=p.get("preview_bytes"),
            precomputed=p.get("precomputed"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

_init_session_state()
_render_sidebar()

if st.session_state["page"] == "manual":
    _render_manual_page()
elif st.session_state["page"] == "stories":
    _render_stories_page()
else:
    _render_chat_page()
