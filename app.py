"""
Wafer Map Bot — Streamlit Web App
LLM-driven chatbot that generates synthetic wafer maps for pre-sales demos.

Primary view: conversational chat with inline maps and downloads.
Sidebar: Azure settings, manual configuration, clear conversation.
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

from geometry import WaferConfig, compute_die_grid
from renderer import render_wafermaps, figure_to_bytes, IMAGE_FORMATS
from signatures import SIGNATURE_NAMES, BIN_DEFINITIONS, apply_signature
from llm_agent import parse_user_request, WaferGenRequest, request_to_config, MAX_WAFERS
from stdf_writer import write_stdf

EXAMPLE_PROMPTS = [
    "Give me 6 wafers on a 300 mm wafer with edge ring failures — lot ID DEMO_A01",
    "5 wafers showing center cluster defects, 8×8 mm dies, 200 mm wafer",
    "Generate 4 wafers with a reticle-systematic pattern for product HBN_PRD020",
    "I need 8 wafers showing a low-yield scenario with scratches — 10×15 mm dies",
    "Create 3 wafers with mixed edge ring and center cluster failures",
    "Show me a bull's-eye pattern on 10 wafers with 5 mm dies",
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


def _build_df(all_wafers, titles, lot_id, program):
    """Build the master CSV DataFrame from generated wafers."""
    rows = []
    now_str = datetime.now().strftime("%m/%d/%Y %H:%M")
    for w_idx, (wafer, title) in enumerate(zip(all_wafers, titles)):
        for dieX, dieY, cx, cy, bin_num in wafer:
            info = BIN_DEFINITIONS.get(bin_num, {})
            rows.append({
                "Program":     program,
                "Lot":         lot_id,
                "Wafer":       title,
                "WaferNumber": w_idx + 1,
                "start_time":  now_str,
                "rework_flag": 0,
                "Bin":         bin_num,
                "dieX":        dieX,
                "dieY":        dieY,
                "BinName":     info.get("name", f"HARDBIN{bin_num}"),
                "BinState":    info.get("state", "F"),
                "BinDesc":     info.get("description", ""),
            })
    return pd.DataFrame(rows)


def _compute_yields(all_wafers):
    yields = []
    for wafer in all_wafers:
        total = len(wafer)
        passed = sum(1 for d in wafer if BIN_DEFINITIONS.get(d[4], {}).get("state") == "P")
        yields.append(passed / total * 100 if total else 0.0)
    return yields


def _generate_wafers(config, signature, num_wafers, lot_id, program):
    dies = compute_die_grid(config)
    all_wafers, titles = [], []
    for w in range(num_wafers):
        seed = w * 137 + hash(signature) % 10000
        all_wafers.append(apply_signature(dies, signature, config, seed=seed))
        titles.append(f"{lot_id}_{w + 1:02d}")
    df = _build_df(all_wafers, titles, lot_id, program)
    return all_wafers, titles, df


def _config_summary(config: WaferConfig, sig_name: str = "") -> str:
    """Short human-readable config line for captions and chat replies."""
    parts = [
        f"{int(config.diameter)} mm",
        f"{config.die_width}×{config.die_height} mm dies",
        (f"notch {config.notch_orientation}" if config.edge_type == "notch" else "flat"),
    ]
    if config.street_width > 0:
        if config.street_width < 0.1:
            parts.append(f"{config.street_width * 1000:.0f} µm street")
        else:
            parts.append(f"{config.street_width} mm street")
    if sig_name == "Reticle Pattern" or (
        config.dies_per_reticle_x != 2 or config.dies_per_reticle_y != 2
    ):
        parts.append(
            f"reticle {config.dies_per_reticle_x}×{config.dies_per_reticle_y} "
            f"({config.reticle_width_mm:.1f}×{config.reticle_height_mm:.1f} mm)"
        )
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


def _render_downloads(all_wafers, config, titles, df, lot_id, key_prefix, precomputed=None):
    """Compact download row for chat or manual results.

    Uses a per-entry, per-format session-state cache so that images and ZIPs are
    never recomputed on Streamlit re-renders — only when a new image format is
    selected for the first time.
    """
    img_fmt = st.selectbox(
        "Image format",
        IMAGE_FORMATS,
        format_func=str.upper,
        key=f"{key_prefix}_img_fmt",
    )

    # One cache dict per chat/manual entry, persists across re-renders
    cache_key = f"_dl_cache_{key_prefix}"
    if cache_key not in st.session_state:
        # Seed with any bytes already computed during generation
        st.session_state[cache_key] = dict(precomputed) if precomputed else {}
    cache = st.session_state[cache_key]

    # Ensure CSV and STDF are cached (format-agnostic)
    if "csv" not in cache:
        cache["csv"] = df.to_csv(index=False).encode("utf-8")
    if "stdf" not in cache:
        program = df["Program"].iloc[0] if not df.empty else "DEMO"
        cache["stdf"] = write_stdf(lot_id, program, titles, all_wafers)

    # Ensure grid bytes for the selected format are cached
    fmt_cache = cache.get(img_fmt, {})
    if "grid" not in fmt_cache:
        with st.spinner(f"Rendering {img_fmt.upper()} grid…"):
            fig_grid = render_wafermaps(all_wafers, config, titles=titles)
            fmt_cache["grid"] = figure_to_bytes(fig_grid, fmt=img_fmt)
            plt.close(fig_grid)
        cache[img_fmt] = fmt_cache

    # Ensure ZIP bytes for the selected format are cached (deferred until needed)
    if "zip" not in fmt_cache:
        with st.spinner(f"Building {img_fmt.upper()} ZIP ({len(all_wafers)} wafers)…"):
            fmt_cache["zip"] = _build_zip_bytes(all_wafers, titles, config, fmt=img_fmt)
        cache[img_fmt] = fmt_cache

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.download_button(
            "CSV",
            data=cache["csv"],
            file_name=f"{lot_id}_wafer_data.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_csv",
        )
    with r1c2:
        st.download_button(
            "STDF",
            data=cache["stdf"],
            file_name=f"{lot_id}_wafer_data.stdf",
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


def _render_chat_result(all_wafers, config, titles, df, lot_id, sig_name, key_prefix,
                        preview_bytes=None, precomputed=None):
    """Inline wafer maps and downloads inside an assistant chat message."""
    yields = _compute_yields(all_wafers)
    total_dies = len(all_wafers[0]) if all_wafers else 0
    avg_yield = sum(yields) / len(yields) if yields else 0

    st.caption(
        f"{lot_id} · {sig_name} · {len(all_wafers)} wafers · "
        f"{total_dies} dies/wafer · {avg_yield:.1f}% avg yield · "
        f"{_config_summary(config, sig_name)}"
    )

    if preview_bytes:
        st.image(preview_bytes, use_container_width=True)
    else:
        fig = render_wafermaps(all_wafers, config, titles=titles)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    _render_downloads(all_wafers, config, titles, df, lot_id, key_prefix, precomputed=precomputed)

    with st.expander("Yield summary & CSV preview"):
        summary_rows = []
        for title, wafer, yld in zip(titles, all_wafers, yields):
            passed = sum(1 for d in wafer if BIN_DEFINITIONS.get(d[4], {}).get("state") == "P")
            summary_rows.append({
                "Wafer ID": title,
                "Total Dies": len(wafer),
                "Pass": passed,
                "Fail": len(wafer) - passed,
                "Yield (%)": round(yld, 2),
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)


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
    )

    config = request_to_config(req)

    all_wafers, titles, df = _generate_wafers(
        config, req.signature, req.num_wafers, req.lot_id, req.program,
    )

    preview_fig = render_wafermaps(all_wafers, config, titles=titles)
    preview_bytes = figure_to_bytes(preview_fig, fmt="png")
    plt.close(preview_fig)

    # Pre-compute format-agnostic bytes and seed the PNG grid cache with the
    # preview render (already done above — zero extra cost).
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    stdf_bytes = write_stdf(req.lot_id, req.program, titles, all_wafers)
    precomputed = {
        "png": {"grid": preview_bytes},  # ZIP deferred until user requests it
        "csv": csv_bytes,
        "stdf": stdf_bytes,
    }

    method = "Azure GPT" if req.used_llm else "keyword parser"
    content = (
        f"{req.explanation}\n\n"
        f"*{method}* · **{req.signature}** · {req.num_wafers} wafers · "
        f"Lot {req.lot_id} · {_config_summary(config, req.signature)}"
    )

    return {
        "role": "assistant",
        "content": content,
        "result": {
            "all_wafers": all_wafers,
            "config": config,
            "titles": titles,
            "df": df,
            "lot_id": req.lot_id,
            "sig": req.signature,
            "preview_bytes": preview_bytes,
            "precomputed": precomputed,
        },
    }


def _handle_user_message(user_input: str) -> None:
    """Append user message, generate response, update chat history."""
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.spinner("Generating wafer maps…"):
        assistant_msg = _process_chat_request(user_input)
    st.session_state["chat_history"].append(assistant_msg)


def _render_signature_help() -> None:
    with st.expander("What can I ask for? (29 spatial signatures)"):
        st.caption(
            "Describe any pattern in plain English — e.g. \"edge ring failures\", "
            "\"scratches\", \"reticle pattern\". You can also specify street width "
            "(e.g. \"0.1 mm street\") and reticle layout (e.g. \"3×3 reticle\")."
        )
        for name in SIGNATURE_NAMES:
            bn = _SIG_BIN.get(name, 5)
            info = BIN_DEFINITIONS.get(bn, {})
            st.markdown(f"**{name}** — {info.get('description', '')}")


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
            if msg.get("result"):
                r = msg["result"]
                _render_chat_result(
                    r["all_wafers"], r["config"], r["titles"], r["df"],
                    r["lot_id"], r["sig"], key_prefix=f"chat_{idx}",
                    preview_bytes=r.get("preview_bytes"),
                    precomputed=r.get("precomputed"),
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
        c1, c2 = st.columns(2)
        with c1:
            diameter = st.number_input("Wafer diameter (mm)", 50.0, 450.0, 300.0, 1.0)
            edge_type = st.radio("Edge type", ["notch", "flat"], horizontal=True)
            notch_orientation = st.radio(
                "Notch orientation",
                ["down", "up", "left", "right"],
                horizontal=True,
                disabled=(edge_type == "flat"),
                help="Which direction the notch points (ignored for flat edge)",
            )
            edge_exclusion = st.slider("Edge exclusion (mm)", 1.0, 5.0, 3.0, 0.5)
            signature_type = st.selectbox("Signature", SIGNATURE_NAMES)
        with c2:
            die_width = st.number_input("Die width (mm)", 1.0, 30.0, 10.0, 0.5)
            die_height = st.number_input("Die height (mm)", 1.0, 30.0, 10.0, 0.5)
            street_width = st.number_input("Street width (mm)", 0.0, 5.0, 0.0, 0.05)
            lot_id = st.text_input("Lot ID", "LOT_001")
            program = st.text_input("Program", "HBN_PRD020")
            num_wafers = st.slider("Number of wafers", 1, MAX_WAFERS, 4)

        with st.expander("Grid offset"):
            x_offset = st.number_input("X offset (mm)", -10.0, 10.0, 0.0, 0.5)
            y_offset = st.number_input("Y offset (mm)", -10.0, 10.0, 0.0, 0.5)

        with st.expander("Reticle layout"):
            rc1, rc2 = st.columns(2)
            with rc1:
                dies_per_reticle_x = st.number_input(
                    "Dies per reticle (X)", 1, 6, 2, 1,
                )
                reticle_fail_die_x = st.number_input(
                    "Fail die column (0-based)", 0, 5, 0, 1,
                )
            with rc2:
                dies_per_reticle_y = st.number_input(
                    "Dies per reticle (Y)", 1, 6, 2, 1,
                )
                reticle_fail_die_y = st.number_input(
                    "Fail die row (0-based)", 0, 5, 0, 1,
                )
            st.caption(
                "Reticle layout applies to the Reticle Pattern signature and defines "
                "field size for systematic stepping."
            )

        generate_btn = st.form_submit_button("Generate", type="primary", use_container_width=True)

    if generate_btn:
        dpr_x = int(dies_per_reticle_x)
        dpr_y = int(dies_per_reticle_y)
        config = WaferConfig(
            diameter=float(diameter),
            edge_type=edge_type,
            edge_exclusion=float(edge_exclusion),
            die_width=float(die_width),
            die_height=float(die_height),
            x_offset=float(x_offset),
            y_offset=float(y_offset),
            street_width=float(street_width),
            dies_per_reticle_x=dpr_x,
            dies_per_reticle_y=dpr_y,
            reticle_fail_die_x=int(reticle_fail_die_x) % dpr_x,
            reticle_fail_die_y=int(reticle_fail_die_y) % dpr_y,
            notch_orientation=notch_orientation if edge_type == "notch" else "down",
        )
        with st.spinner("Generating…"):
            all_wafers, titles, df = _generate_wafers(
                config, signature_type, num_wafers, lot_id, program,
            )
            preview_fig = render_wafermaps(all_wafers, config, titles=titles)
            preview_bytes = figure_to_bytes(preview_fig, fmt="png")
            plt.close(preview_fig)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            stdf_bytes = write_stdf(lot_id, program, titles, all_wafers)
        st.session_state["manual_result"] = {
            "all_wafers": all_wafers,
            "config": config,
            "titles": titles,
            "df": df,
            "lot_id": lot_id,
            "sig": signature_type,
            "preview_bytes": preview_bytes,
            "precomputed": {
                "png": {"grid": preview_bytes},
                "csv": csv_bytes,
                "stdf": stdf_bytes,
            },
        }
        # Clear any stale download cache for this entry
        st.session_state.pop("_dl_cache_manual", None)

    if "manual_result" in st.session_state:
        r = st.session_state["manual_result"]
        st.divider()
        _render_chat_result(
            r["all_wafers"], r["config"], r["titles"], r["df"],
            r["lot_id"], r["sig"], key_prefix="manual",
            preview_bytes=r.get("preview_bytes"),
            precomputed=r.get("precomputed"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

_init_session_state()
_render_sidebar()

if st.session_state["page"] == "manual":
    _render_manual_page()
else:
    _render_chat_page()
