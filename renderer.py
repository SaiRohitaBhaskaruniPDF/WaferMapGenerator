"""
Wafermap renderer: draws wafer images using matplotlib.
Produces a dark-themed grid of colored die squares on a circular wafer.

Takes the binned dies and produces wafer map images using matplotlib.





"""
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from typing import List, Optional

from geometry import WaferConfig
from signatures import BIN_DEFINITIONS, DieResult

# Dark theme colors
BG_COLOR = "#1A1A2E"       # page background
WAFER_COLOR = "#2D2D44"    # wafer disc fill
OUTLINE_COLOR = "#AAAAAA"  # wafer edge
EXCLUSION_COLOR = "#555566" # edge exclusion ring dashes
GRID_COLOR = "#111122"     # thin lines between dies

"""
1. Draws the wafer disc(a circle)
2. Draws the dashed edge-exclusion ring
3. 

"""
def _draw_single_wafer(ax, dies_with_bins: List[DieResult], config: WaferConfig,
                       title: str = "") -> float:
    """
    Draw one wafermap onto a given matplotlib Axes.
    Returns the yield percentage.
    """
    radius = config.diameter / 2.0
    dw = config.die_width
    dh = config.die_height

    ax.set_facecolor(WAFER_COLOR)

    # Wafer disc background
    ax.add_patch(Circle((0, 0), radius, color=WAFER_COLOR, zorder=1))

    # Edge exclusion dashed ring
    ax.add_patch(Circle(
        (0, 0), radius - config.edge_exclusion,
        fill=False, edgecolor=EXCLUSION_COLOR,
        linestyle="--", linewidth=0.6, zorder=2
    ))

    # Draw each die as a colored rectangle
    for dieX, dieY, cx, cy, bin_num in dies_with_bins:
        color = BIN_DEFINITIONS.get(bin_num, BIN_DEFINITIONS[1])["color"]
        rect = patches.Rectangle(
            (cx - dw / 2, cy - dh / 2), dw, dh,
            linewidth=0.3,
            edgecolor=GRID_COLOR,
            facecolor=color,
            zorder=3,
        )
        ax.add_patch(rect)

    # Notch: small triangle at the bottom of the wafer
    if config.edge_type == "notch":
        notch_size = radius * 0.035
        ax.add_patch(patches.Wedge(
            (0, -radius), notch_size, 60, 120,
            color=BG_COLOR, zorder=4
        ))
    # Flat: cut the bottom of the circle with a dark rectangle
    elif config.edge_type == "flat":
        flat_y = -(radius * 0.82)
        ax.add_patch(patches.Rectangle(
            (-radius * 1.1, -radius * 1.1),
            radius * 2.2, (radius * 1.1 + flat_y),
            color=BG_COLOR, zorder=4
        ))

    # Wafer outline circle
    ax.add_patch(Circle(
        (0, 0), radius,
        fill=False, edgecolor=OUTLINE_COLOR, linewidth=1.2, zorder=5
    ))

    # Yield calculation
    total = len(dies_with_bins)
    passed = sum(1 for d in dies_with_bins if BIN_DEFINITIONS.get(d[4], {}).get("state") == "P")
    yld = (passed / total * 100) if total > 0 else 0.0

    ax.set_xlim(-radius * 1.12, radius * 1.12)
    ax.set_ylim(-radius * 1.12, radius * 1.12)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(
            f"{title}\nYield: {yld:.1f}%  ({passed}/{total})",
            color="white", fontsize=8, pad=5, fontweight="bold"
        )

    return yld


def render_single_wafer_svg(dies_with_bins: List[DieResult], config: WaferConfig,
                            title: str = "") -> str:
    """
    Render a single wafermap and return SVG markup as a string.
    Useful for embedding in HTML or saving as .svg.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    _draw_single_wafer(ax, dies_with_bins, config, title=title)

    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_wafermaps(all_wafers: List[List[DieResult]], config: WaferConfig,
                     titles: Optional[List[str]] = None) -> plt.Figure:
    """
    Render one or more wafermaps in a grid layout.
    Returns a matplotlib Figure ready to display or save.
    """
    n = len(all_wafers)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        fig.patch.set_facecolor(BG_COLOR)
        ax.text(0.5, 0.5, "No wafers to display", ha="center", va="center",
                color="white", transform=ax.transAxes)
        return fig

    # Grid layout: max 3 columns
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig_w = 5.5 * ncols
    fig_h = 5.8 * nrows + 1.2  # extra height for legend row

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG_COLOR)

    # Create grid spec: wafer rows + 1 legend row at bottom
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrows + 1, ncols, figure=fig,
                  height_ratios=[5.8] * nrows + [1.0],
                  hspace=0.35, wspace=0.1)

    for idx in range(n):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(BG_COLOR)
        title = titles[idx] if titles else f"Wafer {idx + 1}"
        _draw_single_wafer(ax, all_wafers[idx], config, title=title)

    # Hide unused subplot slots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_visible(False)

    # Legend row — show all bin colors used across all wafers
    used_bins = sorted(set(d[4] for wafer in all_wafers for d in wafer))
    legend_ax = fig.add_subplot(gs[nrows, :])
    legend_ax.set_facecolor(BG_COLOR)
    legend_ax.axis("off")

    x_start = 0.02
    spacing = min(0.18, 0.9 / max(len(used_bins), 1))
    for i, bin_num in enumerate(used_bins):
        info = BIN_DEFINITIONS.get(bin_num, {"name": f"BIN{bin_num}", "state": "?", "color": "#888888"})
        xpos = x_start + i * spacing
        legend_ax.add_patch(patches.FancyBboxPatch(
            (xpos, 0.25), 0.03, 0.5,
            boxstyle="round,pad=0.01",
            facecolor=info["color"], edgecolor="none",
            transform=legend_ax.transAxes, clip_on=False
        ))
        legend_ax.text(
            xpos + 0.04, 0.5,
            f"Bin {bin_num} — {info['name']} ({info['state']})",
            color="white", fontsize=8, va="center",
            transform=legend_ax.transAxes
        )

    return fig
