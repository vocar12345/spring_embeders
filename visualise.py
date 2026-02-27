"""
visualise.py
─────────────────────────────────────────────────────────────────────────────
Thesis figures for the Fruchterman-Reingold force-directed layout algorithm.

Produces:
  figures/01_convergence.pdf   – kinetic energy vs iteration (log scale)
  figures/02_layout.pdf        – final graph layout
  figures/03_temperature.pdf   – simulated annealing cooling curve
  figures/04_degree_dist.pdf   – degree distribution of the random graph

Requirements:
  pip install matplotlib pandas numpy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_DIR  = Path("output")
OUTPUT_DIR = Path("figures")

# Must match Config in main.cpp
COOLING_RATE = 0.95
INIT_TEMP    = 200.0
MAX_ITER     = 500

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 12,
    "legend.fontsize"  : 10,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
    "text.usetex"      : False,   # set True if LaTeX is installed
})

COLOUR_EDGE    = "#a0aec0"
COLOUR_NODE    = "#2b6cb0"
COLOUR_ACCENT  = "#e53e3e"
COLOUR_ANNEAL  = "#d69e2e"

# ── Load data ─────────────────────────────────────────────────────────────────

def load_data():
    nodes   = pd.read_csv(INPUT_DIR / "nodes.csv")
    edges   = pd.read_csv(INPUT_DIR / "edges.csv")
    metrics = pd.read_csv(INPUT_DIR / "metrics.csv")
    return nodes, edges, metrics

# ── Figure 1: Convergence curve ───────────────────────────────────────────────

def plot_convergence(metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.semilogy(metrics["iteration"], metrics["kinetic_energy"],
                color=COLOUR_ACCENT, linewidth=1.5, label="Kinetic energy $E_k$")

    # Mark the point where energy drops below 1% of its initial value
    threshold = metrics["kinetic_energy"].iloc[0] * 0.01
    crossed   = metrics[metrics["kinetic_energy"] < threshold]
    if not crossed.empty:
        conv_iter = crossed["iteration"].iloc[0]
        ax.axvline(conv_iter, color="grey", linestyle="--", linewidth=1.0,
                   label=f"1 % threshold @ iter {conv_iter}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Kinetic Energy (log scale)")
    ax.set_title("Fruchterman-Reingold — Convergence Curve")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_convergence.pdf")
    print("  ✓  figures/01_convergence.pdf")
    plt.close(fig)

# ── Figure 2: Graph layout ────────────────────────────────────────────────────

def plot_layout(nodes: pd.DataFrame, edges: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw edges first (behind nodes)
    node_pos = nodes.set_index("node_id")[["x", "y"]]
    for _, e in edges.iterrows():
        u = node_pos.loc[e["source"]]
        v = node_pos.loc[e["target"]]
        ax.plot([u.x, v.x], [u.y, v.y],
                color=COLOUR_EDGE, linewidth=0.7, alpha=0.8, zorder=1)

    # Draw nodes
    ax.scatter(nodes["x"], nodes["y"],
               s=50, color=COLOUR_NODE, edgecolors="white",
               linewidths=0.8, zorder=2)

    ax.set_title("Fruchterman-Reingold — Final Layout")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_layout.pdf")
    print("  ✓  figures/02_layout.pdf")
    plt.close(fig)

# ── Figure 3: Simulated annealing cooling curve ───────────────────────────────

def plot_temperature():
    """
    Reconstructs the theoretical cooling schedule T(t) = T_0 * alpha^t
    and plots it — no CSV needed, derived analytically from Config values.
    """
    iters = np.arange(MAX_ITER)
    T     = INIT_TEMP * (COOLING_RATE ** iters)
    T     = np.maximum(T, 1e-3)            # T_min floor

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(iters, T, color=COLOUR_ANNEAL, linewidth=1.5,
            label=rf"$T(t) = {INIT_TEMP:.0f} \times {COOLING_RATE}^{{t}}$")

    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel("Temperature $T$")
    ax.set_title("Simulated Annealing — Cooling Schedule")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_temperature.pdf")
    print("  ✓  figures/03_temperature.pdf")
    plt.close(fig)

# ── Figure 4: Degree distribution ─────────────────────────────────────────────

def plot_degree_distribution(nodes: pd.DataFrame, edges: pd.DataFrame):
    # Compute degree for each node
    degree = pd.concat([edges["source"], edges["target"]]) \
               .value_counts() \
               .reindex(nodes["node_id"], fill_value=0)

    counts = degree.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(counts.index, counts.values,
           color=COLOUR_NODE, edgecolor="white", linewidth=0.5)

    mean_deg = degree.mean()
    ax.axvline(mean_deg, color=COLOUR_ACCENT, linestyle="--", linewidth=1.2,
               label=f"Mean degree = {mean_deg:.2f}")

    ax.set_xlabel("Degree $d$")
    ax.set_ylabel("Number of vertices")
    ax.set_title("Erdős–Rényi Graph — Degree Distribution")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_degree_dist.pdf")
    print("  ✓  figures/04_degree_dist.pdf")
    plt.close(fig)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading CSV data ...")
    nodes, edges, metrics = load_data()
    print(f"  |V| = {len(nodes)}   |E| = {len(edges)}   "
          f"iterations = {len(metrics)}\n")

    print("Generating figures ...")
    plot_convergence(metrics)
    plot_layout(nodes, edges)
    plot_temperature()
    plot_degree_distribution(nodes, edges)

    print(f"\nAll figures saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()