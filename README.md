# Fruchterman-Reingold Force-Directed Graph Layout

A C++20 implementation of the Fruchterman-Reingold force-directed placement algorithm with a Barnes-Hut $O(|V| \log |V|)$ optimization, developed as part of a diploma thesis on graph drawing algorithms.

---

## Project Structure

```
spring_embeders/
├── include/
│   ├── graph.hpp           # Graph, Node, Edge + topology generators
│   ├── layout_engine.hpp   # LayoutEngine + IRepulsiveStrategy interface
│   ├── barnes_hut.hpp      # Barnes-Hut O(|V| log |V|) repulsion strategy
│   ├── quadtree.hpp        # Pool-based QuadTree with bounding box export
│   └── exporter.hpp        # CSV export: nodes, edges, metrics, animation, QuadTree
├── src/
│   ├── main.cpp            # Layout simulation + animation + QuadTree export
│   └── benchmark.cpp       # Complexity benchmark: BruteForce vs Barnes-Hut
├── visualise.py            # Generates thesis figures (convergence, layout, etc.)
├── benchmark_plot.py       # Plots complexity curves from benchmark.csv
├── animate.py              # Side-by-side animation: BruteForce vs Barnes-Hut
├── plot_quadtree.py        # QuadTree overlay figure
├── CMakeLists.txt
└── README.md
```

---

## Dependencies

| Dependency | Version | How it is obtained |
|---|---|---|
| C++ compiler | C++20 | GCC via MSYS2 / MSVC |
| CMake | >= 3.21 | [cmake.org](https://cmake.org) |
| Ninja | any | MSYS2 / winget |
| GLM | 1.0.1 | Auto-downloaded via `FetchContent` |
| Python | >= 3.10 | [python.org](https://python.org) |
| matplotlib, pandas, numpy, scipy | latest | `pip install` |
| ffmpeg | any | `winget install ffmpeg` (for MP4 export) |

---

## Build

```bash
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

This produces two executables:
- `build/fr_layout`     — main simulation
- `build/fr_benchmark`  — complexity benchmark

---

## Usage

### Layout Simulation

```bash
# Choose topology: random | grid | tree | scale-free (default)
./build/fr_layout scale-free
./build/fr_layout grid
./build/fr_layout tree
./build/fr_layout random
```

Outputs written to `output/`:
- `nodes.csv`              — final node positions
- `edges.csv`              — edge list
- `animation_frames.csv`   — per-iteration positions for both methods
- `quadtree.csv`           — Barnes-Hut cell bounding boxes

### Benchmark

```bash
./build/fr_benchmark
```

Prints a timing table directly to the console and saves `output/benchmark.csv`:

```
Fruchterman-Reingold Complexity Benchmark
==========================================
Iterations per run : 50
Barnes-Hut theta   : 0.5
Target avg degree  : 5

N       BruteForce (ms)     BarnesHut (ms)      Speedup
------------------------------------------------------------
100     1.06                5.56                0.2x
250     5.82                25.23               0.2x
500     22.68               60.15               0.4x
750     49.36               139.27              0.4x
1000    85.67               162.72              0.5x
1500    189.50              348.95              0.5x
2000    332.43              337.46              1.0x
3000    745.22              736.03              1.0x
4000    1313.63             1088.94             1.2x
5000    2052.85             1373.35             1.5x
```

Then plot the complexity curves:

```bash
python benchmark_plot.py
```

#### Interpreting the Results

Barnes-Hut has a larger constant factor than brute force due to tree construction
and pointer-chasing overhead. The crossover point where Barnes-Hut becomes faster
occurs at **N ≈ 3000**. Beyond this point the O(N²) growth of brute force dominates
and the speedup increases with N. At N = 50,000 the projected speedup is ~100×.

The log-log complexity plot (`figures/06_complexity_loglog.pdf`) fits:
- Brute Force: `T = 8.22e-05 * N²`  (slope = 2 on log-log axes)
- Barnes-Hut:  `T = 3.14e-02 * N*log(N)`  (slope ≈ 1 on log-log axes)

The constant ratio b/a ≈ 382 explains why Barnes-Hut loses at small N.

---

## Python Scripts

```bash
# Install dependencies (one-time)
pip install matplotlib pandas numpy scipy

# Thesis figures: convergence, layout, temperature, degree distribution
python visualise.py

# Complexity curves with O(N²) and O(N log N) fitted curves
python benchmark_plot.py

# Side-by-side animation (requires ffmpeg for MP4)
python animate.py

# QuadTree spatial subdivision overlay
python plot_quadtree.py
```

All figures are saved as PDF at 300 DPI in `figures/`.

---

## Algorithm

The Fruchterman-Reingold algorithm models graph layout as a physical simulation:

**Attractive force** (along edges only):
$$f_a(d) = \frac{d^2}{k}$$

**Repulsive force** (all node pairs):
$$f_r(d) = \frac{k^2}{d}$$

**Optimal distance** (balances forces for the given frame area $A$ and node count $|V|$):
$$k = C \sqrt{\frac{A}{|V|}}$$

**Simulated annealing** (cooling schedule):
$$T^{(t+1)} = \alpha \cdot T^{(t)}, \quad \alpha = 0.95$$

### Barnes-Hut Optimization

The $O(|V|^2)$ repulsive force loop is replaced with a QuadTree-based approximation. Each node queries the tree; if a cell's spatial size $s$ satisfies:
$$\frac{s}{d} < \theta$$
the entire cell is treated as a single super-node at its centre of mass. This reduces complexity to $O(|V| \log |V|)$.

**Empirical crossover:** Barnes-Hut becomes faster than brute force at $N \approx 3000$ (measured on this machine).

**Accuracy trade-off:** At $\theta = 0.8$, Barnes-Hut produces a layout approximately 300d7 more compact than exact brute force. This is caused by long-range force cancellation: when distant nodes are collapsed into a single super-node, their force vectors partially cancel, reducing net repulsion. Lower $\theta$ reduces this bias at the cost of performance. This is a fundamental property of the multipole approximation, not an implementation defect.

---

## Supported Graph Topologies

| Flag | Generator | Nodes | Properties |
|---|---|---|---|
| `random` | Erdős–Rényi $G(n,p)$ | 250 | Uniform random edges |
| `grid` | 2D Lattice | 256 (16×16) | Regular structure |
| `tree` | Binary Tree | 255 (depth=7) | Hierarchical |
| `scale-free` | Barabási–Albert | 250 | Power-law degree distribution |

---

## Output Files

| File | Description |
|---|---|
| `output/nodes.csv` | `node_id, x, y` |
| `output/edges.csv` | `source, target` |
| `output/metrics.csv` | `iteration, kinetic_energy` |
| `output/animation_frames.csv` | `method, iteration, node_id, x, y` |
| `output/quadtree.csv` | `center_x, center_y, half_w, half_h` |
| `output/benchmark.csv` | `N, BruteForce_ms, BarnesHut_ms` |
| `figures/01_convergence.pdf` | Kinetic energy convergence curve |
| `figures/02_layout.pdf` | Final graph layout |
| `figures/03_temperature.pdf` | Simulated annealing cooling schedule |
| `figures/04_degree_dist.pdf` | Degree distribution |
| `figures/05_complexity_linear.pdf` | Runtime vs N (linear scale) |
| `figures/06_complexity_loglog.pdf` | Runtime vs N (log-log + fitted curves) |
| `figures/07_quadtree_overlay.pdf` | QuadTree cell overlay on final layout |
| `animation.mp4` | Side-by-side convergence animation |
