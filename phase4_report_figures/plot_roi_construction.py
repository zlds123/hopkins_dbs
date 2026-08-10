import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_roi_construction.png")

BLUE = "#2b6cb0"
ORANGE = "#c9a227"
GREEN = "#3d8b6b"
GRAY = "#6b7280"
PURPLE = "#7952b3"

REGIONS = ["Frontal_L", "Frontal_R", "Central_L", "Central_R", "Temporal_L", "Temporal_R", "Parietal_L", "Other"]
REGION_XY = {
    "Frontal_L": (-0.62, 0.55), "Frontal_R": (0.62, 0.55),
    "Central_L": (-0.35, 0.15), "Central_R": (0.35, 0.15),
    "Temporal_L": (-0.68, -0.35), "Temporal_R": (0.68, -0.35),
    "Parietal_L": (-0.2, -0.55), "Other": (0.35, -0.62),
}
REGION_COLOR = {
    "Frontal_L": "#f2c14e", "Frontal_R": "#f2c14e", "Central_L": "#e07a5f", "Central_R": "#e07a5f",
    "Temporal_L": "#81b29a", "Temporal_R": "#81b29a", "Parietal_L": "#9d8dcf", "Other": "#c2c2c2",
}

fig, ax = plt.subplots(figsize=(10.6, 6.0))
ax.set_xlim(0, 10.6)
ax.set_ylim(0, 6.0)
ax.axis("off")

rng = np.random.default_rng(7)


def arrow(x0, y0, x1, y1, color=GRAY, lw=1.4):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, lw=lw, color=color, zorder=2)
    ax.add_patch(a)


def brain_outline(cx, cy, r=0.95, fc="none", ec="black", lw=1.3, alpha=1.0):
    e = Ellipse((cx, cy), r * 2.15, r * 1.9, fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=1)
    ax.add_patch(e)
    return e


def draw_regions(cx, cy, r=0.95, alpha=0.35):
    for reg in REGIONS:
        rx, ry = REGION_XY[reg]
        e = Ellipse((cx + rx * r * 0.62, cy + ry * r * 0.62), r * 0.62, r * 0.5,
                   fc=REGION_COLOR[reg], ec="none", alpha=alpha, zorder=1)
        ax.add_patch(e)


def electrodes_for(cx, cy, r, seed, n, color):
    rng2 = np.random.default_rng(seed)
    pts = []
    tries = 0
    while len(pts) < n and tries < 4000:
        tries += 1
        x = rng2.uniform(-0.85, 0.85)
        y = rng2.uniform(-0.8, 0.8)
        if (x / 1.0) ** 2 + (y / 0.88) ** 2 <= 1.0:
            pts.append((cx + x * r, cy + y * r))
    px = [p[0] for p in pts]
    py = [p[1] for p in pts]
    ax.scatter(px, py, s=14, color=color, ec="white", lw=0.4, zorder=4)
    return pts


# ============================================================================
# Column 1: two participants, electrodes at real anatomical (MNI) coordinates
# ============================================================================
ax.text(1.3, 5.65, "Electrodes at real\n(MNI) coordinates", ha="center", fontsize=9.3, weight="bold")

brain_outline(1.3, 4.1, r=0.85, ec=BLUE, lw=1.4)
electrodes_for(1.3, 4.1, 0.85, seed=1, n=22, color=BLUE)
ax.text(1.3, 3.05, "Participant A", ha="center", fontsize=8.6, color=BLUE, weight="bold")

brain_outline(1.3, 1.55, r=0.85, ec=ORANGE, lw=1.4)
electrodes_for(1.3, 1.55, 0.85, seed=2, n=16, color=ORANGE)
ax.text(1.3, 0.5, "Participant B", ha="center", fontsize=8.6, color=ORANGE, weight="bold")

ax.text(1.3, 5.3, "different montage,\ndifferent coordinates", ha="center", fontsize=7.6,
       style="italic", color=GRAY)

arrow(2.35, 4.1, 3.05, 3.4, color=BLUE)
arrow(2.35, 1.55, 3.05, 2.2, color=ORANGE)

# ============================================================================
# Column 2: map each electrode onto a SHARED set of anatomical regions
# ============================================================================
ax.text(4.1, 5.65, "Mapped onto the SAME\nfixed region set (atlas)", ha="center", fontsize=9.3, weight="bold")

brain_outline(4.1, 2.85, r=1.05, ec="black", lw=1.5)
draw_regions(4.1, 2.85, r=1.05, alpha=0.45)
for reg in REGIONS:
    rx, ry = REGION_XY[reg]
    lx, ly = 4.1 + rx * 1.05 * 0.62, 2.85 + ry * 1.05 * 0.62
    ax.text(lx, ly, reg.replace("_", "\n"), ha="center", va="center", fontsize=6.0,
           color="black", zorder=5, weight="bold")
ax.text(4.1, 1.35, "17 regions \u00d7 hemisphere\n(shared vocabulary,\nevery participant)",
       ha="center", fontsize=7.8, style="italic", color=GRAY)

arrow(5.35, 2.85, 6.05, 2.85, color=GREEN, lw=1.6)

# ============================================================================
# Column 3: average band power within each region; impute missing regions
# ============================================================================
box_x, box_y, box_w, box_h = 6.15, 1.9, 2.15, 1.9
b = FancyBboxPatch((box_x, box_y), box_w, box_h, boxstyle="round,pad=0.04,rounding_size=0.06",
                   fc="#eaf3ee", ec=GREEN, lw=1.3, zorder=2)
ax.add_patch(b)
ax.text(box_x + box_w / 2, box_y + box_h / 2,
       "Average band power\nwithin each region\n(across that participant's\nelectrodes in it)\n\n"
       "Region with NO coverage\n\u2192 imputed from the\ntraining-pool mean",
       ha="center", va="center", fontsize=7.9, color="#1f5c42", weight="bold", linespacing=1.4)

arrow(8.3, 2.85, 9.0, 2.85, color=PURPLE, lw=1.6)

# ============================================================================
# Column 4: fixed-size feature vector, same shape for every participant
# ============================================================================
ax.text(9.75, 5.65, "Fixed-size feature,\nevery participant", ha="center", fontsize=9.3, weight="bold")


def region_band_grid(cx, cy, seed, missing=None):
    missing = missing or set()
    rng3 = np.random.default_rng(seed)
    nreg, nband = 17, 5
    w, h = 1.15, 2.1
    cellw, cellh = w / nband, h / nreg
    x0, y0 = cx - w / 2, cy - h / 2
    for i in range(nreg):
        for j in range(nband):
            if i in missing:
                fc = "#dddddd"
            else:
                v = rng3.uniform(0.2, 0.95)
                fc = plt.cm.viridis(v)
            ax.add_patch(plt.Rectangle((x0 + j * cellw, y0 + i * cellh), cellw, cellh,
                                       fc=fc, ec="white", lw=0.3, zorder=3))
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fc="none", ec="black", lw=1.1, zorder=4))


region_band_grid(9.75, 3.6, seed=11, missing={2, 9})
ax.text(9.75, 2.35, "Participant A\n85 features\n(17 regions \u00d7 5 bands)", ha="center", fontsize=7.6, color=BLUE)

region_band_grid(9.75, 1.55, seed=12, missing={0, 4, 8, 13})
ax.text(9.75, 0.35, "Participant B\n85 features\n(17 regions \u00d7 5 bands)", ha="center", fontsize=7.6, color=ORANGE)

# ---- footer with the 3-way empirical comparison ---------------------------- #
ax.text(5.3, 0.15,
       "Zero-shot cross-patient transfer, binary target:   "
       "electrode identity (no anatomical anchor) AUC = 0.52   |   "
       "anatomical ROI aggregation (this figure) AUC = 0.80   |   "
       "channel-agnostic (no spatial info) AUC = 0.77",
       ha="center", va="center", fontsize=8.3, color="black",
       bbox=dict(boxstyle="round,pad=0.4", fc="#fbf6e8", ec="#c9a227", lw=1))

fig.tight_layout()
fig.savefig(OUT, dpi=200)
print("wrote", OUT)
