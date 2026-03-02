"""
plot_quadtree.py
─────────────────────────────────────────────────────────────────────────────
Reads output/nodes.csv, output/edges.csv, and output/quadtree.csv.
Plots the final graph layout with QuadTree bounding boxes overlaid,
coloured by cell depth (inferred from cell size).

Output: figures/07_quadtree_overlay.pdf

Requirements: pip install matplotlib pandas numpy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

NODES_CSV    = Path("output/nodes.csv")
EDGES_CSV    = Path("output/edges.csv")
QUADTREE_CSV = Path("output/quadtree.csv")
OUTPUT_DIR   = Path("figures")
OUTPUT_FILE  = OUTPUT_DIR / "07_quadtree_overlay.pdf"

plt.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 11,
    "axes.titlesize" : 13,
    "figure.dpi"     : 150,
    "savefig.dpi"    : 300,
    "savefig.bbox"   : "tight",
})

NODE_SIZE    = 30
NODE_COLOUR  = "#2b6cb0"
NODE_EDGE_C  = "white"
EDGE_COLOUR  = "#90cdf4"
EDGE_ALPHA   = 0.5
EDGE_WIDTH   = 0.6
BG_COLOUR    = "#0d1117"   # dark background for contrast

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data ...")
nodes = pd.read_csv(NODES_CSV)
edges = pd.read_csv(EDGES_CSV)
qt    = pd.read_csv(QUADTREE_CSV)

print(f"  Nodes      : {len(nodes)}")
print(f"  Edges      : {len(edges)}")
print(f"  QT cells   : {len(qt)}")

# ── Infer depth from cell size ────────────────────────────────────────────────
# The root cell has the largest halfW. Each subdivision halves both dimensions.
# depth ≈ log2(root_halfW / cell_halfW)
# We normalise cell size to [0, 1] and map it to a colormap so that:
#   large cells (deep in the tree = coarse) → one colour
#   small cells (dense regions = fine)      → another colour

qt["cell_size"] = 2.0 * qt[["half_w", "half_h"]].max(axis=1)
max_size = qt["cell_size"].max()
min_size = qt["cell_size"].min()
# Normalise: 0=smallest (finest subdivision), 1=largest (coarsest)
qt["norm_size"] = (qt["cell_size"] - min_size) / (max_size - min_size + 1e-9)

# ── Figure ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG_COLOUR)
ax.set_facecolor(BG_COLOUR)

# Compute axis limits with margin
margin = 40.0
xmin = nodes["x"].min() - margin;  xmax = nodes["x"].max() + margin
ymin = nodes["y"].min() - margin;  ymax = nodes["y"].max() + margin
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")
ax.set_xticks([]);  ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)

# ── Draw QuadTree cells ───────────────────────────────────────────────────────
# Sorted large-to-small so smaller cells draw on top (more visible)
qt_sorted = qt.sort_values("cell_size", ascending=False)

# Use a perceptually uniform colormap
cmap = plt.cm.YlOrRd   # yellow (coarse) → red (fine)

for _, row in qt_sorted.iterrows():
    cx, cy = row["center_x"], row["center_y"]
    hw, hh = row["half_w"],   row["half_h"]
    t      = row["norm_size"]   # 1 = coarsest, 0 = finest

    # Colour: coarse cells = dim blue, fine cells = bright teal
    # This makes the adaptive refinement visible
    alpha  = 0.08 + 0.25 * (1.0 - t)   # fine cells more opaque
    colour = cmap(1.0 - t)

    rect = patches.Rectangle(
        (cx - hw, cy - hh), hw * 2, hh * 2,
        linewidth=0.4,
        edgecolor=(*colour[:3], min(alpha * 2.5, 0.7)),
        facecolor=(*colour[:3], alpha * 0.3),
        zorder=1
    )
    ax.add_patch(rect)

# ── Draw edges ────────────────────────────────────────────────────────────────

node_pos = nodes.set_index("node_id")[["x", "y"]]
xs, ys = [], []
for _, e in edges.iterrows():
    s, t = int(e["source"]), int(e["target"])
    if s in node_pos.index and t in node_pos.index:
        xs += [node_pos.at[s, "x"], node_pos.at[t, "x"], None]
        ys += [node_pos.at[s, "y"], node_pos.at[t, "y"], None]

if xs:
    ax.plot(xs, ys, color=EDGE_COLOUR, lw=EDGE_WIDTH,
            alpha=EDGE_ALPHA, zorder=2, solid_capstyle="round")

# ── Draw nodes ────────────────────────────────────────────────────────────────

ax.scatter(nodes["x"], nodes["y"],
           s=NODE_SIZE, c=NODE_COLOUR,
           edgecolors=NODE_EDGE_C, linewidths=0.5, zorder=3)

# ── Colorbar legend ───────────────────────────────────────────────────────────

sm = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=mcolors.Normalize(vmin=0, vmax=1)
)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
cbar.set_label("Cell coarseness  (fine → coarse)", color="white", fontsize=10)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
cbar.outline.set_edgecolor("white")

# ── Title and annotation ──────────────────────────────────────────────────────

ax.set_title(
    f"Barnes-Hut QuadTree Overlay  —  {len(qt)} cells  |  "
    f"{len(nodes)} nodes  |  {len(edges)} edges",
    color="white", pad=12
)
ax.text(0.01, 0.01,
        "Cell colour indicates spatial subdivision depth.\n"
        "Dense node regions produce smaller, finer cells.",
        transform=ax.transAxes, fontsize=8.5,
        color="#a0aec0", va="bottom")

# ── Save ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)
plt.tight_layout()
fig.savefig(OUTPUT_FILE, facecolor=BG_COLOUR)
print(f"\n  Saved: {OUTPUT_FILE}")
plt.close(fig)
