"""
benchmark_plot.py
─────────────────────────────────────────────────────────────────────────────
Reads output/benchmark.csv and produces two publication-ready figures:

  figures/05_complexity_linear.pdf  – raw ms vs N (linear scale)
  figures/06_complexity_loglog.pdf  – log-log scale with fitted O() curves

Run after fr_benchmark has produced output/benchmark.csv.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

INPUT_FILE = Path("output/benchmark.csv")
OUTPUT_DIR = Path("figures")

# ── Style (consistent with visualise.py) ─────────────────────────────────────

plt.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 11,
    "axes.titlesize" : 13,
    "axes.labelsize" : 12,
    "legend.fontsize": 10,
    "figure.dpi"     : 150,
    "savefig.dpi"    : 300,
    "savefig.bbox"   : "tight",
})

COLOUR_BF  = "#e53e3e"   # red  – BruteForce
COLOUR_BH  = "#2b6cb0"   # blue – BarnesHut
COLOUR_FIT = "#888888"   # grey – fitted curves

# ── Fitting functions ─────────────────────────────────────────────────────────

def quadratic(N, a):
    """O(N²) model:  t = a·N²"""
    return a * N**2

def nlogn(N, a):
    """O(N log N) model:  t = a·N·log(N)"""
    return a * N * np.log(N)

# ── Load data ─────────────────────────────────────────────────────────────────

def load_benchmark() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} benchmark points from {INPUT_FILE}\n")
    print(df.to_string(index=False))
    print()
    return df

# ── Figure 5: Linear scale ────────────────────────────────────────────────────

def plot_linear(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["N"], df["BruteForce_ms"], "o-",
            color=COLOUR_BF, linewidth=1.8, markersize=5,
            label=r"Brute Force $O(|V|^2)$")
    ax.plot(df["N"], df["BarnesHut_ms"], "s-",
            color=COLOUR_BH, linewidth=1.8, markersize=5,
            label=r"Barnes-Hut $O(|V|\log|V|)$")

    ax.set_xlabel("Vertex count $|V|$")
    ax.set_ylabel("Time (ms) — 100 iterations")
    ax.set_title("Force-Directed Layout — Runtime vs Graph Size")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_complexity_linear.pdf")
    print("  ✓  figures/05_complexity_linear.pdf")
    plt.close(fig)

# ── Figure 6: Log-log with fitted curves ─────────────────────────────────────

def plot_loglog(df: pd.DataFrame):
    N  = df["N"].to_numpy(dtype=float)
    bf = df["BruteForce_ms"].to_numpy(dtype=float)
    bh = df["BarnesHut_ms"].to_numpy(dtype=float)

    # Fit theoretical models to measured data
    (a_bf,), _ = curve_fit(quadratic, N, bf)
    (a_bh,), _ = curve_fit(nlogn,    N, bh)

    N_fit  = np.linspace(N.min(), N.max(), 300)
    bf_fit = quadratic(N_fit, a_bf)
    bh_fit = nlogn(N_fit, a_bh)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Raw measurements
    ax.loglog(N, bf, "o", color=COLOUR_BF, markersize=6, label="Brute Force (measured)")
    ax.loglog(N, bh, "s", color=COLOUR_BH, markersize=6, label="Barnes-Hut (measured)")

    # Fitted curves
    ax.loglog(N_fit, bf_fit, "--", color=COLOUR_BF, linewidth=1.2,
              label=rf"Fit: $a \cdot N^2$,  $a={a_bf:.2e}$")
    ax.loglog(N_fit, bh_fit, "--", color=COLOUR_BH, linewidth=1.2,
              label=rf"Fit: $a \cdot N \log N$,  $a={a_bh:.2e}$")

    ax.set_xlabel("Vertex count $|V|$ (log scale)")
    ax.set_ylabel("Time (ms) — log scale")
    ax.set_title("Complexity Analysis — Log-Log Scale with Fitted Curves")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_complexity_loglog.pdf")
    print("  ✓  figures/06_complexity_loglog.pdf")
    plt.close(fig)

# ── Console summary ───────────────────────────────────────────────────────────

def print_speedup_table(df: pd.DataFrame):
    print("Speedup table (BruteForce / BarnesHut):")
    print(f"  {'N':>6}  {'BF (ms)':>12}  {'BH (ms)':>12}  {'Speedup':>10}")
    print("  " + "-" * 46)
    for _, row in df.iterrows():
        speedup = row["BruteForce_ms"] / row["BarnesHut_ms"]
        print(f"  {int(row['N']):>6}  "
              f"{row['BruteForce_ms']:>12.2f}  "
              f"{row['BarnesHut_ms']:>12.2f}  "
              f"{speedup:>9.1f}x")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_benchmark()
    print_speedup_table(df)
    print("\nGenerating figures ...")
    plot_linear(df)
    plot_loglog(df)

if __name__ == "__main__":
    main()
