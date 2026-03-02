"""
animate.py
─────────────────────────────────────────────────────────────────────────────
Side-by-side animation: BruteForce vs Barnes-Hut layout convergence.

THE ROOT CAUSE OF "TINY NODES":
  Bounds were computed from ALL frames including iteration 0, where nodes
  are randomly scattered across the full 1920x1080 frame. The converged
  layout occupies a much smaller region. Using all-frame bounds makes
  the axis enormous and nodes appear as dots.

FIX:
  Compute axis limits from the LAST frame only (converged positions).
  This ensures the viewport is tight around where nodes actually end up.

Output: animation.mp4 (ffmpeg) or animation.gif (Pillow fallback)
Requirements: pip install matplotlib pandas numpy; winget install ffmpeg
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

FRAMES_CSV = Path("output/animation_frames.csv")
EDGES_CSV  = Path("output/edges.csv")
OUTPUT_MP4 = Path("animation.mp4")
OUTPUT_GIF = Path("animation.gif")

FPS        = 24
DPI        = 120
MARGIN     = 80.0   # padding around the final converged layout

FIGURE_W   = 16
FIGURE_H   = 6

NODE_SIZE   = 80
NODE_COLOUR = "#63b3ed"
EDGE_COLOUR = "#4a7fa5"
EDGE_ALPHA  = 0.5
EDGE_WIDTH  = 0.8
BG_COLOUR   = "#1a202c"
NODE_EDGE_C = "#ebf8ff"

plt.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 11,
    "axes.titlesize" : 13,
})

# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading data ...")
frames = pd.read_csv(FRAMES_CSV)
edges  = pd.read_csv(EDGES_CSV)

bf_frames = frames[frames["method"] == "BruteForce"].copy()
bh_frames = frames[frames["method"] == "BarnesHut"].copy()

iterations = sorted(frames["iteration"].unique())
n_frames   = len(iterations)
last_iter  = iterations[-1]

print(f"  Frames     : {n_frames}")
print(f"  Nodes      : {frames['node_id'].nunique()}")
print(f"  Edges      : {len(edges)}")
print(f"  Last iter  : {last_iter}")

# ── Bounds from FINAL frame only ──────────────────────────────────────────────
# The initial scatter fills 1920x1080, but converged layout is much tighter.
# Using final-frame bounds ensures nodes are large and clearly visible.

def final_bounds(df, method_name):
    last = df[df["iteration"] == last_iter]
    x0 = last["x"].min() - MARGIN
    x1 = last["x"].max() + MARGIN
    y0 = last["y"].min() - MARGIN
    y1 = last["y"].max() + MARGIN
    print(f"  {method_name:12s}  x=[{x0:.0f}, {x1:.0f}]  "
          f"y=[{y0:.0f}, {y1:.0f}]  "
          f"span=[{x1-x0:.0f} x {y1-y0:.0f}]")
    return x0, x1, y0, y1

print("Axis bounds (from final frame):")
bf_x0, bf_x1, bf_y0, bf_y1 = final_bounds(bf_frames, "BruteForce")
bh_x0, bh_x1, bh_y0, bh_y1 = final_bounds(bh_frames, "BarnesHut")

# ── Index all frames ──────────────────────────────────────────────────────────

def index_frames(df):
    return {it: grp.set_index("node_id")[["x", "y"]]
            for it, grp in df.groupby("iteration")}

bf_idx = index_frames(bf_frames)
bh_idx = index_frames(bh_frames)

# ── Figure ────────────────────────────────────────────────────────────────────

fig, (ax_bf, ax_bh) = plt.subplots(
    1, 2, figsize=(FIGURE_W, FIGURE_H),
    facecolor=BG_COLOUR
)
fig.suptitle("Fruchterman-Reingold  —  Layout Convergence",
             fontsize=14, fontweight="bold", color="white", y=0.98)

# ① Call tight_layout BEFORE setting axis limits
plt.tight_layout(rect=[0, 0.06, 1, 0.93])

# ② Set per-method limits from final frame, then lock autoscale
ax_bf.set_xlim(bf_x0, bf_x1);  ax_bf.set_ylim(bf_y0, bf_y1)
ax_bh.set_xlim(bh_x0, bh_x1);  ax_bh.set_ylim(bh_y0, bh_y1)
ax_bf.autoscale(False)
ax_bh.autoscale(False)

for ax, title in [
    (ax_bf, "Brute Force   $O(|V|^2)$"),
    (ax_bh, "Barnes-Hut   $O(|V|\\log|V|)$"),
]:
    ax.set_facecolor(BG_COLOUR)
    ax.set_title(title, color="white", pad=8)
    ax.set_xticks([]);  ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

iter_label = fig.text(
    0.5, 0.015, f"Iteration: {iterations[0]}",
    ha="center", fontsize=12, color="#90cdf4", fontweight="bold"
)

# ── Edge helper (batched into single plot call for speed) ─────────────────────

def draw_edges(ax, pos: pd.DataFrame):
    xs, ys = [], []
    for _, e in edges.iterrows():
        s, t = int(e["source"]), int(e["target"])
        if s in pos.index and t in pos.index:
            xs += [pos.at[s, "x"], pos.at[t, "x"], None]
            ys += [pos.at[s, "y"], pos.at[t, "y"], None]
    if xs:
        ax.plot(xs, ys, color=EDGE_COLOUR, lw=EDGE_WIDTH,
                alpha=EDGE_ALPHA, zorder=1, solid_capstyle="round")

# ── Initial scatter ───────────────────────────────────────────────────────────

def init_scatter(ax, idx):
    pos = idx[iterations[0]]
    draw_edges(ax, pos)
    return ax.scatter(
        pos["x"], pos["y"],
        s=NODE_SIZE, c=NODE_COLOUR,
        edgecolors=NODE_EDGE_C, linewidths=0.5, zorder=2
    )

sc_bf = init_scatter(ax_bf, bf_idx)
sc_bh = init_scatter(ax_bh, bh_idx)

# ── Update ────────────────────────────────────────────────────────────────────

def update(fi: int):
    it = iterations[fi]
    for ax, sc, idx in [
        (ax_bf, sc_bf, bf_idx),
        (ax_bh, sc_bh, bh_idx),
    ]:
        pos = idx.get(it)
        if pos is None:
            continue
        sc.set_offsets(pos[["x", "y"]].to_numpy())
        for ln in ax.lines[:]:
            ln.remove()
        draw_edges(ax, pos)
        ax.autoscale(False)   # re-lock after set_offsets

    iter_label.set_text(f"Iteration: {it}")
    return sc_bf, sc_bh

# ── Render ────────────────────────────────────────────────────────────────────

print(f"\nRendering {n_frames} frames at {FPS} fps ...")

anim = animation.FuncAnimation(
    fig, update,
    frames=n_frames,
    init_func=lambda: (sc_bf, sc_bh),
    interval=1000 // FPS,
    blit=False,
)

# ── Export ────────────────────────────────────────────────────────────────────

def try_mp4() -> bool:
    if "ffmpeg" not in animation.writers.list():
        return False
    w = animation.FFMpegWriter(
        fps=FPS, bitrate=2500,
        metadata={"title": "Fruchterman-Reingold Animation"}
    )
    anim.save(OUTPUT_MP4, writer=w, dpi=DPI)
    kb = OUTPUT_MP4.stat().st_size // 1024
    print(f"Saved: {OUTPUT_MP4}  ({kb} KB)")
    return True

def save_gif():
    w = animation.PillowWriter(fps=FPS)
    anim.save(OUTPUT_GIF, writer=w, dpi=DPI)
    kb = OUTPUT_GIF.stat().st_size // 1024
    print(f"Saved: {OUTPUT_GIF}  ({kb} KB)")

if not try_mp4():
    print("ffmpeg not found — saving GIF instead.")
    save_gif()

print("Done.")
