import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_feature_construction.png")

BLUE = "#2b6cb0"
GREEN = "#3d8b6b"
ORANGE = "#c9a227"
GRAY = "#6b7280"
LIGHT = "#eef2f7"

fig, ax = plt.subplots(figsize=(10.2, 5.6))
ax.set_xlim(0, 10.2)
ax.set_ylim(0, 5.6)
ax.axis("off")

rng = np.random.default_rng(3)


def box(x, y, w, h, text, fc=LIGHT, ec=GRAY, fontsize=9, weight="normal", tcolor="black"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.06",
                       fc=fc, ec=ec, lw=1.2, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
           weight=weight, color=tcolor, zorder=3, linespacing=1.35)
    return b


def arrow(x0, y0, x1, y1, color=GRAY):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                        lw=1.4, color=color, zorder=2)
    ax.add_patch(a)


def electrode_grid(cx, cy, n, w=0.9, h=1.1, color=BLUE, label=""):
    """Small dot-grid icon representing one participant's electrode montage."""
    ncols = int(np.ceil(np.sqrt(n * w / h)))
    nrows = int(np.ceil(n / ncols))
    xs = np.linspace(cx - w / 2, cx + w / 2, ncols)
    ys = np.linspace(cy - h / 2, cy + h / 2, nrows)
    pts = [(x, y) for y in ys for x in xs][:n]
    px = [p[0] + rng.uniform(-0.02, 0.02) for p in pts]
    py = [p[1] + rng.uniform(-0.02, 0.02) for p in pts]
    ax.scatter(px, py, s=9, color=color, zorder=3)
    ax.add_patch(mpatches.FancyBboxPatch((cx - w / 2 - 0.12, cy - h / 2 - 0.12), w + 0.24, h + 0.24,
                                         boxstyle="round,pad=0.02,rounding_size=0.08",
                                         fc="none", ec=color, lw=1.1, ls="--", zorder=2))
    ax.text(cx, cy - h / 2 - 0.30, label, ha="center", va="top", fontsize=8.3, color=color)


# ---- Column 1: two participants, different montages ------------------------ #
ax.text(1.05, 5.3, "Two participants,\ndifferent montages", ha="center", fontsize=9.3,
       weight="bold", color="black")
electrode_grid(1.05, 4.15, 85, color=BLUE, label="Participant A\n85 electrodes")
electrode_grid(1.05, 1.55, 63, color=ORANGE, label="Participant B\n63 electrodes")

arrow(1.75, 4.15, 2.45, 3.35, color=BLUE)
arrow(1.75, 1.55, 2.45, 2.15, color=ORANGE)

# ---- Column 2: per-channel band-power cube --------------------------------- #
box(2.5, 2.55, 2.05, 1.55,
   "Band-pass filter into\n5 bands (\u03b8 \u03b1 \u03b2 low-\u03b3 high-\u03b3)\n"
   "+ Hilbert envelope\naveraged in 10-s windows",
   fc="#ffffff", ec=GRAY, fontsize=8.6)
ax.text(3.5, 2.35, "\u2192 array: windows \u00d7 channels \u00d7 5 bands", ha="center",
       fontsize=8, style="italic", color=GRAY)
ax.text(3.5, 2.10, "(channel count differs by participant)", ha="center",
       fontsize=7.6, style="italic", color=GRAY)

arrow(4.6, 3.3, 5.35, 3.3, color=GRAY)

# ---- Column 3: collapse channel axis --------------------------------------- #
box(5.4, 2.4, 2.35, 1.85,
   "Collapse the channel axis:\n7 summary statistics per band\n"
   "(mean, SD, 10th/25th/50th/\n75th/90th percentile)\n"
   "computed ACROSS all channels",
   fc="#eaf3ee", ec=GREEN, fontsize=8.4, weight="bold", tcolor="#1f5c42")

arrow(7.8, 3.3, 8.5, 3.3, color=GREEN)

# ---- Column 4: identical 35-dim feature vector for both participants ------- #
ax.text(9.35, 5.3, "Same-shape feature,\nboth participants", ha="center", fontsize=9.3,
       weight="bold", color="black")


def feat_vector(cx, cy, color, label):
    n = 35
    w, h = 0.55, 1.7
    vals = rng.uniform(0.15, 1.0, n)
    xs = np.linspace(cx - w / 2, cx + w / 2, n)
    for x, v in zip(xs, vals):
        ax.plot([x, x], [cy - h / 2, cy - h / 2 + v * h], color=color, lw=1.1, alpha=0.85)
    ax.add_patch(mpatches.FancyBboxPatch((cx - w / 2 - 0.15, cy - h / 2 - 0.08), w + 0.3, h + 0.16,
                                         boxstyle="round,pad=0.02,rounding_size=0.06",
                                         fc="none", ec=color, lw=1.1, zorder=2))
    ax.text(cx, cy - h / 2 - 0.26, label, ha="center", va="top", fontsize=8.0, color=color)


feat_vector(9.35, 4.05, BLUE, "Participant A\n35 features")
feat_vector(9.35, 1.75, ORANGE, "Participant B\n35 features")

# ---- footer note ------------------------------------------------------------ #
ax.text(5.1, 0.25,
       "35-dimensional feature (5 bands \u00d7 7 statistics) \u2014 same layout regardless of a "
       "participant's electrode count or location; the LDA weight vector is likewise 35-dimensional\n"
       "and does not correspond to any single electrode. This is what makes the model applicable, "
       "unmodified, to a participant it was never trained on.",
       ha="center", va="center", fontsize=8.1, color="black",
       bbox=dict(boxstyle="round,pad=0.4", fc="#fbf6e8", ec="#c9a227", lw=1))

fig.tight_layout()
fig.savefig(OUT, dpi=200)
print("wrote", OUT)
