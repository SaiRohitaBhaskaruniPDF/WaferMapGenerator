"""
Wafer Map Generator — Streamlit Web App
LLM-driven bot that generates synthetic wafer maps for pre-sales demos.

Tabs:
  1. Manual Configuration  — classic form-based generation
  2. AI Chat Assistant     — natural language → wafer maps via LLM
  3. About                 — signature reference card
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
from renderer import render_wafermaps
from signatures import SIGNATURE_NAMES, BIN_DEFINITIONS, apply_signature
from llm_agent import parse_user_request, WaferGenRequest
from stdf_writer import write_stdf

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wafer Map Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0F0F1A; color: #E0E0E0; }
    [data-testid="stSidebar"]          { background-color: #16213E; }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }
    .stTabs [data-baseweb="tab-list"]  { background-color: #16213E; border-radius: 8px; }
    .stTabs [data-baseweb="tab"]       { color: #AAAACC; }
    .stTabs [aria-selected="true"]     { color: #4A90D9 !important; border-bottom: 2px solid #4A90D9; }
    .stButton > button {
        background-color: #0F3460; color: white;
        border: 1px solid #4A90D9; border-radius: 6px; font-weight: bold;
    }
    .stButton > button:hover { background-color: #4A90D9; }
    .chat-msg-user {
        background: #1E3A5F; border-radius: 12px 12px 4px 12px;
        padding: 10px 14px; margin: 6px 0 6px 40px; color: #D0E8FF;
    }
    .chat-msg-bot {
        background: #1A2E1A; border-radius: 12px 12px 12px 4px;
        padding: 10px 14px; margin: 6px 40px 6px 0; color: #CCFFCC;
    }
    .sig-card {
        background: #1A1A2E; border-radius: 8px; padding: 10px 14px;
        border-left: 4px solid #4A90D9; margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🔬 Wafer Map Generator")
st.markdown("Synthetic spatial-signature wafer maps for pre-sales demos and pipeline testing.")
st.divider()

tab_manual, tab_ai, tab_about = st.tabs(
    ["⚙️ Manual Configuration", "🤖 AI Chat Assistant", "📖 Signature Reference"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

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
        total  = len(wafer)
        passed = sum(1 for d in wafer if BIN_DEFINITIONS.get(d[4], {}).get("state") == "P")
        yields.append(passed / total * 100 if total else 0.0)
    return yields


def _render_results(all_wafers, config, titles, df, lot_id, sig_name, key_prefix=""):
    """Render wafer images, stats, and download buttons."""
    yields      = _compute_yields(all_wafers)
    total_dies  = len(all_wafers[0]) if all_wafers else 0
    avg_yield   = sum(yields) / len(yields) if yields else 0

    st.markdown(f"### Results — `{lot_id}`  |  Pattern: *{sig_name}*")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wafers Generated", len(all_wafers))
    c2.metric("Dies per Wafer",   total_dies)
    c3.metric("Average Yield",    f"{avg_yield:.1f}%")
    c4.metric("Wafer Diameter",   f"{int(config.diameter)} mm")
    st.divider()

    with st.spinner("Rendering wafer maps…"):
        fig = render_wafermaps(all_wafers, config, titles=titles)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()

    with st.expander("📊 Per-Wafer Yield Summary", expanded=True):
        rows = []
        for title, wafer, yld in zip(titles, all_wafers, yields):
            passed = sum(1 for d in wafer if BIN_DEFINITIONS.get(d[4], {}).get("state") == "P")
            rows.append({
                "Wafer ID": title,
                "Total Dies": len(wafer),
                "Pass": passed,
                "Fail": len(wafer) - passed,
                "Yield (%)": round(yld, 2),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("📋 CSV Data Preview (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 📥 Download Outputs")

    dl1, dl2, dl3, dl4 = st.columns(4)

    # CSV
    with dl1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ CSV",
            data=csv_bytes,
            file_name=f"{lot_id}_wafer_data.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_dl_csv",
        )

    # PNG (grid)
    with dl2:
        fig2 = render_wafermaps(all_wafers, config, titles=titles)
        buf  = io.BytesIO()
        fig2.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                     facecolor=fig2.get_facecolor())
        buf.seek(0)
        plt.close(fig2)
        st.download_button(
            "⬇️ PNG (grid)",
            data=buf,
            file_name=f"{lot_id}_wafermap.png",
            mime="image/png",
            use_container_width=True,
            key=f"{key_prefix}_dl_png",
        )

    # ZIP of individual PNGs
    with dl3:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx, (wafer, title) in enumerate(zip(all_wafers, titles)):
                fig_w = render_wafermaps([wafer], config, titles=[title])
                img_buf = io.BytesIO()
                fig_w.savefig(img_buf, format="png", dpi=150,
                              bbox_inches="tight", facecolor=fig_w.get_facecolor())
                plt.close(fig_w)
                zf.writestr(f"{title}.png", img_buf.getvalue())
        zip_buf.seek(0)
        st.download_button(
            "⬇️ ZIP (per wafer)",
            data=zip_buf,
            file_name=f"{lot_id}_individual_maps.zip",
            mime="application/zip",
            use_container_width=True,
            key=f"{key_prefix}_dl_zip",
        )

    # STDF
    with dl4:
        stdf_bytes = write_stdf(lot_id, df["Program"].iloc[0] if not df.empty else "DEMO",
                                titles, all_wafers)
        st.download_button(
            "⬇️ STDF",
            data=stdf_bytes,
            file_name=f"{lot_id}_wafer_data.stdf",
            mime="application/octet-stream",
            use_container_width=True,
            key=f"{key_prefix}_dl_stdf",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Manual Configuration
# ═══════════════════════════════════════════════════════════════════════════════

with tab_manual:
    left_col, right_col = st.columns([1, 3])

    with left_col:
        st.markdown("#### ⚙️ Parameters")

        st.markdown("**Wafer**")
        diameter      = st.selectbox("Diameter (mm)", [150, 200, 300], index=2, key="m_diam")
        edge_type     = st.radio("Edge Type", ["notch", "flat"], horizontal=True, key="m_edge")
        edge_exclusion = st.slider("Edge Exclusion (mm)", 1.0, 5.0, 3.0, 0.5, key="m_ee")

        st.markdown("**Die Size**")
        c_dw, c_dh = st.columns(2)
        die_width  = c_dw.number_input("W (mm)", 1.0, 30.0, 10.0, 0.5, key="m_dw")
        die_height = c_dh.number_input("H (mm)", 1.0, 30.0, 10.0, 0.5, key="m_dh")

        with st.expander("Grid Offset"):
            x_offset = st.number_input("X Offset (mm)", -10.0, 10.0, 0.0, 0.5, key="m_xo")
            y_offset = st.number_input("Y Offset (mm)", -10.0, 10.0, 0.0, 0.5, key="m_yo")

        st.markdown("**Signature**")
        signature_type = st.selectbox("Pattern", SIGNATURE_NAMES, key="m_sig")

        st.markdown("**Lot**")
        lot_id     = st.text_input("Lot ID",  "LOT_001", key="m_lot")
        program    = st.text_input("Program", "HBN_PRD020", key="m_prog")
        num_wafers = st.slider("# Wafers", 1, 25, 4, key="m_nw")

        generate_btn = st.button("🚀 Generate", type="primary",
                                  use_container_width=True, key="m_gen")

    with right_col:
        if generate_btn:
            config = WaferConfig(
                diameter=float(diameter),
                edge_type=edge_type,
                edge_exclusion=float(edge_exclusion),
                die_width=float(die_width),
                die_height=float(die_height),
                x_offset=float(x_offset),
                y_offset=float(y_offset),
            )
            with st.spinner("Computing die grid…"):
                dies       = compute_die_grid(config)
                all_wafers = []
                titles     = []
                for w in range(num_wafers):
                    seed = w * 137 + hash(signature_type) % 10000
                    all_wafers.append(apply_signature(dies, signature_type, config, seed=seed))
                    titles.append(f"{lot_id}_{w + 1:02d}")

            df = _build_df(all_wafers, titles, lot_id, program)
            st.session_state["manual_all_wafers"] = all_wafers
            st.session_state["manual_config"] = config
            st.session_state["manual_titles"] = titles
            st.session_state["manual_df"] = df
            st.session_state["manual_lot_id"] = lot_id
            st.session_state["manual_sig"] = signature_type

        if "manual_all_wafers" in st.session_state:
            _render_results(
                st.session_state["manual_all_wafers"],
                st.session_state["manual_config"],
                st.session_state["manual_titles"],
                st.session_state["manual_df"],
                st.session_state["manual_lot_id"],
                st.session_state["manual_sig"],
                key_prefix="man",
            )
        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#666;">
                <div style="font-size:56px;">🔬</div>
                <h3 style="color:#999;">Configure parameters on the left and click Generate</h3>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — AI Chat Assistant
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ai:
    st.markdown("#### 🤖 Describe the wafer maps you need in plain English")
    st.markdown(
        "The AI agent will interpret your request and generate the appropriate wafer maps. "
        "No wafer expertise required — just describe what you want."
    )

    # Azure OpenAI settings (session-only, never stored to disk)
    with st.expander("🔑 Azure OpenAI (GPT-4.1) — optional in UI if env vars are set", expanded=False):
        st.caption(
            "Set credentials here for this session, or use environment variables: "
            "`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`."
        )
        azure_endpoint_input = st.text_input(
            "Azure endpoint",
            value=st.session_state.get("ai_azure_endpoint", os.environ.get("AZURE_OPENAI_ENDPOINT", "")),
            placeholder="https://your-resource.openai.azure.com/",
            key="ai_azure_endpoint_field",
        )
        azure_key_input = st.text_input(
            "Azure API key",
            type="password",
            value=st.session_state.get("ai_api_key", ""),
            placeholder="Leave blank to use AZURE_OPENAI_API_KEY env var",
            key="ai_key_field",
        )
        azure_deployment_input = st.text_input(
            "Deployment name",
            value=st.session_state.get(
                "ai_azure_deployment",
                os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"),
            ),
            help="The deployment name you created in Azure AI Foundry / Azure OpenAI Studio — not always the same as the model label.",
            key="ai_azure_deployment_field",
        )
        if azure_endpoint_input:
            st.session_state["ai_azure_endpoint"] = azure_endpoint_input.strip()
        if azure_key_input:
            st.session_state["ai_api_key"] = azure_key_input
        if azure_deployment_input:
            st.session_state["ai_azure_deployment"] = azure_deployment_input.strip()

        if st.session_state.get("ai_azure_endpoint") and (
            st.session_state.get("ai_api_key") or os.environ.get("AZURE_OPENAI_API_KEY")
        ):
            st.success("Azure OpenAI configured for this session.")

    # Chat history
    if "ai_chat" not in st.session_state:
        st.session_state["ai_chat"] = []

    # Example prompts
    st.markdown("**Try an example:**")
    examples = [
        "Give me 6 wafers on a 300 mm wafer with edge ring failures — lot ID DEMO_A01",
        "5 wafers showing center cluster defects, 8×8 mm dies, 200 mm wafer",
        "Generate 4 wafers with a reticle-systematic pattern for product HBN_PRD020",
        "I need 8 wafers showing a low-yield scenario with scratches — 10×15 mm dies",
        "Create 3 wafers with mixed edge ring and center cluster failures",
        "Show me a bull's-eye pattern on 10 wafers with 5 mm dies",
    ]
    ex_cols = st.columns(3)
    for i, ex in enumerate(examples):
        if ex_cols[i % 3].button(ex[:55] + "…", key=f"ex_{i}", use_container_width=True):
            st.session_state["ai_prefill"] = ex

    st.divider()

    # Chat display
    for msg in st.session_state["ai_chat"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-msg-user">👤 {msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-msg-bot">🤖 {msg["content"]}</div>',
                        unsafe_allow_html=True)

    # Input row
    prompt_default = st.session_state.pop("ai_prefill", "")
    with st.form(key="ai_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your request",
            value=prompt_default,
            placeholder="e.g.  Generate 5 wafers with edge ring failures on a 300 mm wafer…",
            height=80,
            label_visibility="collapsed",
        )
        col_send, col_clear = st.columns([4, 1])
        send_btn  = col_send.form_submit_button("Send ↩", type="primary", use_container_width=True)
        clear_btn = col_clear.form_submit_button("Clear", use_container_width=True)

    if clear_btn:
        st.session_state["ai_chat"] = []
        for k in list(st.session_state.keys()):
            if k.startswith("ai_result"):
                del st.session_state[k]
        st.rerun()

    if send_btn and user_input.strip():
        st.session_state["ai_chat"].append({"role": "user", "content": user_input})

        api_key = st.session_state.get("ai_api_key", "")
        azure_endpoint = st.session_state.get("ai_azure_endpoint", "")
        azure_deployment = st.session_state.get("ai_azure_deployment", "")
        with st.spinner("🤖 Thinking…"):
            req: WaferGenRequest = parse_user_request(
                user_input,
                api_key=api_key or None,
                azure_endpoint=azure_endpoint or None,
                azure_deployment=azure_deployment or None,
            )

        # Build wafer maps from the parsed request
        config = WaferConfig(
            diameter       = req.diameter,
            edge_type      = req.edge_type,
            edge_exclusion = req.edge_exclusion,
            die_width      = req.die_width,
            die_height     = req.die_height,
            x_offset       = req.x_offset,
            y_offset       = req.y_offset,
        )

        with st.spinner("Computing die grid…"):
            dies       = compute_die_grid(config)
            all_wafers = []
            titles     = []
            for w in range(req.num_wafers):
                seed = w * 137 + hash(req.signature) % 10000
                all_wafers.append(apply_signature(dies, req.signature, config, seed=seed))
                titles.append(f"{req.lot_id}_{w + 1:02d}")

        df = _build_df(all_wafers, titles, req.lot_id, req.program)

        # Bot response message
        method_tag = "🧠 Azure GPT-4.1" if req.used_llm else "🔡 keyword parser"
        bot_msg = (
            f"({method_tag})  {req.explanation}\n\n"
            f"**Signature:** {req.signature} | "
            f"**Wafers:** {req.num_wafers} | "
            f"**Diameter:** {int(req.diameter)} mm | "
            f"**Die:** {req.die_width}×{req.die_height} mm | "
            f"**Lot:** {req.lot_id}"
        )
        st.session_state["ai_chat"].append({"role": "bot", "content": bot_msg})

        # Store result for display after rerun
        key = f"ai_result_{len(st.session_state['ai_chat'])}"
        st.session_state[key] = {
            "all_wafers": all_wafers, "config": config,
            "titles": titles, "df": df,
            "lot_id": req.lot_id, "sig": req.signature,
        }
        st.rerun()

    # Display the most recent AI-generated result
    result_keys = sorted(
        [k for k in st.session_state if k.startswith("ai_result_")],
        key=lambda k: int(k.split("_")[-1]),
    )
    if result_keys:
        latest = st.session_state[result_keys[-1]]
        st.divider()
        _render_results(
            latest["all_wafers"], latest["config"],
            latest["titles"],     latest["df"],
            latest["lot_id"],     latest["sig"],
            key_prefix=f"ai_{result_keys[-1]}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Signature Reference
# ═══════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.markdown("#### 📖 Available Spatial Signatures")
    st.markdown(
        "The bot can generate any of the following **29 signatures**. "
        "Use the exact name in manual mode, or describe it naturally in AI Chat."
    )

    sig_data = []
    # Map signature names to representative bins for colour
    _sig_bin = {
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

    for name in SIGNATURE_NAMES:
        bn   = _sig_bin.get(name, 5)
        info = BIN_DEFINITIONS.get(bn, {})
        sig_data.append({
            "Signature": name,
            "Bin": bn,
            "Bin Name": info.get("name", ""),
            "Description": info.get("description", ""),
        })

    cols = st.columns(2)
    for i, row in enumerate(sig_data):
        bn   = row["Bin"]
        info = BIN_DEFINITIONS.get(bn, {})
        color = info.get("color", "#888")
        with cols[i % 2]:
            st.markdown(
                f'<div class="sig-card">'
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background:{color};border-radius:3px;margin-right:8px;"></span>'
                f'<strong>{row["Signature"]}</strong><br>'
                f'<span style="color:#888;font-size:0.85em;">{row["Description"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### 🗂️ Output Formats")
    st.markdown("""
| Format | Contents |
|--------|----------|
| **CSV** | Program, Lot, Wafer, WaferNumber, start_time, rework_flag, Bin, dieX, dieY, BinName, BinState, BinDesc |
| **PNG (grid)** | All wafers in a single composite image |
| **ZIP (per wafer)** | One PNG per wafer, named by Wafer ID |
| **STDF** | Full STDF v4 binary: FAR, MIR, WIR/WRR per wafer, PIR/PRR per die, HBR/SBR summary, MRR |
""")

    st.divider()
    st.markdown("#### 🔌 Exensio Integration")
    st.markdown("""
Generated STDF files are formatted for direct ingestion into **Exensio**:
- HardBin / SoftBin summary records included
- Die X/Y coordinates in PRR records (standard wafer map display)
- Lot ID and Program name carried in MIR
- One WIR→WRR block per wafer (standard lot structure)

**Workflow:**  Configure → Generate → Download STDF → Ingest in Exensio → Demo Gallery
""")
