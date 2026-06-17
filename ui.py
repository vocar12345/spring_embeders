"""
ui.py
─────────────────────────────────────────────────────────────────────────────
Spring Embedder Control Panel.

A small Tkinter desktop window for the Fruchterman-Reingold layout project.
Set every Config value, click "Run Layout", and the app:

  1. writes ui_config.txt,
  2. runs build/fr_batch.exe (lays out every graph in Input/ -> output/<name>/),
  3. renders each graph, saves output/<name>/layout.png,
  4. displays them in the window so you can flip between graphs live.

Launch by double-clicking run_ui.bat, or:  python ui.py

Requires: matplotlib, pandas  (already used by the project's plot scripts).
"""

import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Paths ─────────────────────────────────────────────────────────────────────
# When packaged with PyInstaller, sys.frozen is set and __file__ points inside a
# temporary extraction folder — so anchor on the .exe's own location instead, so
# build/fr_batch.exe, Input/ and output/ resolve next to the executable.

if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent

# CMake names the binary fr_batch.exe on Windows and fr_batch elsewhere.
EXE_NAME   = "fr_batch.exe" if os.name == "nt" else "fr_batch"
EXE        = ROOT / "build" / EXE_NAME
CONFIG_TXT = ROOT / "ui_config.txt"

# ── Styling (matches batch_visualise.py) ──────────────────────────────────────

NODE_SIZE    = 120
NODE_COLOUR  = "#2b6cb0"
NODE_EDGE_C  = "white"
EDGE_COLOUR  = "#90cdf4"
EDGE_ALPHA   = 0.6
EDGE_WIDTH   = 0.9
BG_COLOUR    = "#0d1117"

# ── Config field definitions ──────────────────────────────────────────────────
# (label, config-key, default, kind, used_by_batch)
#   kind: "float" | "int" | "str"
#   used_by_batch=False  -> animation-only field; shown but written as a comment.

FIELDS = [
    ("Frame width  (frameW)",   "frameW",      "1920",   "float", True),
    ("Frame height (frameH)",   "frameH",      "1080",   "float", True),
    ("Force const  (C)",        "C",           "1.0",    "float", True),
    ("Initial temp (initTemp)", "initTemp",    "200",    "float", True),
    ("Cooling rate (coolingRate)", "coolingRate", "0.95", "float", True),
    ("Theta        (theta)",    "theta",       "0.8",    "float", True),
    ("Max iters    (maxIter)",  "maxIter",     "300",    "int",   True),
    ("Layout seed  (layoutSeed)", "layoutSeed", "7",     "int",   True),
    ("Frame interval (frameInterval)", "frameInterval", "5", "int", False),
    ("Graph seed   (graphSeed)", "graphSeed",  "42",     "int",   False),
    ("Input dir",               "inputDir",    "Input",  "str",   True),
    ("Output dir",              "outputDir",   "output", "str",   True),
]


class ControlPanel:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Spring Embedder — Control Panel")
        root.configure(bg=BG_COLOUR)
        root.geometry("1180x760")

        self.entries: dict[str, tk.Entry] = {}
        self.graph_dirs: list[Path] = []

        self._build_left_panel()
        self._build_right_panel()

    # ── Layout: left config column ────────────────────────────────────────────
    def _build_left_panel(self):
        left = tk.Frame(self.root, bg=BG_COLOUR, padx=16, pady=14)
        left.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(left, text="Configuration", bg=BG_COLOUR, fg="white",
                 font=("Segoe UI", 13, "bold")).grid(row=0, column=0,
                 columnspan=2, sticky="w", pady=(0, 10))

        for i, (label, key, default, kind, used) in enumerate(FIELDS, start=1):
            fg = "#e2e8f0" if used else "#7d8590"
            text = label + ("" if used else "  (animation only)")
            tk.Label(left, text=text, bg=BG_COLOUR, fg=fg,
                     font=("Segoe UI", 9)).grid(row=i, column=0, sticky="w",
                     pady=3, padx=(0, 8))
            e = tk.Entry(left, width=14, font=("Consolas", 10))
            e.insert(0, default)
            e.grid(row=i, column=1, sticky="w", pady=3)
            self.entries[key] = e

        # ── Render setting: vertex size (display only, no recompile/relayout) ──
        vrow = len(FIELDS) + 1
        tk.Label(left, text="Vertex size  (render)", bg=BG_COLOUR, fg="#e2e8f0",
                 font=("Segoe UI", 9)).grid(row=vrow, column=0, sticky="w",
                 pady=(10, 3), padx=(0, 8))
        self.vertex_entry = tk.Entry(left, width=14, font=("Consolas", 10))
        self.vertex_entry.insert(0, str(NODE_SIZE))
        self.vertex_entry.grid(row=vrow, column=1, sticky="w", pady=(10, 3))
        # Enter redraws the current graph instantly — no need to re-run layout.
        self.vertex_entry.bind("<Return>", lambda _e: self.redraw_current())

        tk.Label(left, text="Press Enter to redraw with the new size.",
                 bg=BG_COLOUR, fg="#7d8590", font=("Segoe UI", 8)).grid(
                 row=vrow + 1, column=0, columnspan=2, sticky="w")

        self.run_btn = tk.Button(left, text="▶  Run Layout",
                                 command=self.on_run,
                                 bg="#238636", fg="white",
                                 activebackground="#2ea043",
                                 font=("Segoe UI", 11, "bold"),
                                 relief="flat", padx=10, pady=6, cursor="hand2")
        self.run_btn.grid(row=vrow + 2, column=0, columnspan=2,
                          sticky="we", pady=(16, 6))

        self.status = tk.Label(left, text="Ready.", bg=BG_COLOUR,
                               fg="#7d8590", font=("Segoe UI", 9),
                               wraplength=240, justify="left", anchor="w")
        self.status.grid(row=vrow + 3, column=0, columnspan=2,
                         sticky="we")

    # ── Layout: right image viewer ────────────────────────────────────────────
    def _build_right_panel(self):
        right = tk.Frame(self.root, bg=BG_COLOUR)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        bar = tk.Frame(right, bg=BG_COLOUR, padx=10, pady=10)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(bar, text="Graph:", bg=BG_COLOUR, fg="white",
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(0, 8))

        self.graph_var = tk.StringVar()
        self.combo = ttk.Combobox(bar, textvariable=self.graph_var,
                                  state="readonly", width=42)
        self.combo.pack(side=tk.LEFT)
        self.combo.bind("<<ComboboxSelected>>", lambda e: self.show_selected())

        tk.Button(bar, text="◀ Prev", command=lambda: self.step(-1),
                  relief="flat", bg="#21262d", fg="white",
                  cursor="hand2").pack(side=tk.LEFT, padx=(12, 4))
        tk.Button(bar, text="Next ▶", command=lambda: self.step(1),
                  relief="flat", bg="#21262d", fg="white",
                  cursor="hand2").pack(side=tk.LEFT)

        self.fig = Figure(figsize=(8, 6), facecolor=BG_COLOUR)
        self.ax = self.fig.add_subplot(111)
        self._blank_axes("Set parameters and click Run Layout.")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _blank_axes(self, message: str):
        self.ax.clear()
        self.ax.set_facecolor(BG_COLOUR)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        for sp in self.ax.spines.values():
            sp.set_visible(False)
        self.ax.text(0.5, 0.5, message, ha="center", va="center",
                     color="#7d8590", fontsize=12, transform=self.ax.transAxes)

    def set_status(self, text: str, colour: str = "#7d8590"):
        self.status.config(text=text, fg=colour)

    # ── Run ───────────────────────────────────────────────────────────────────
    def collect_config(self):
        """Validate fields. Returns dict on success, or None (after showing
        an error) on failure."""
        cfg = {}
        for label, key, _default, kind, _used in FIELDS:
            raw = self.entries[key].get().strip()
            if raw == "":
                messagebox.showerror("Invalid value", f"'{label}' is empty.")
                self.entries[key].focus_set()
                return None
            if kind == "str":
                cfg[key] = raw
                continue
            try:
                cfg[key] = int(raw) if kind == "int" else float(raw)
            except ValueError:
                messagebox.showerror(
                    "Invalid value",
                    f"'{label}' must be a {'whole number' if kind=='int' else 'number'}.\n"
                    f"Got: {raw!r}")
                self.entries[key].focus_set()
                return None
        return cfg

    def write_config(self, cfg: dict):
        lines = ["# Generated by ui.py — Spring Embedder Control Panel", ""]
        for _label, key, _d, _kind, used in FIELDS:
            if used:
                lines.append(f"{key}={cfg[key]}")
            else:
                lines.append(f"# {key}={cfg[key]}  (animation only, ignored by batch)")
        # newline='' + utf-8 (no BOM) so the C++ key=value parser stays happy.
        CONFIG_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def on_run(self):
        if not EXE.exists():
            messagebox.showerror(
                "Executable missing",
                f"Could not find:\n{EXE}\n\n"
                "Build it first with:\n"
                '  cmake --build build --target fr_batch')
            return
        cfg = self.collect_config()
        if cfg is None:
            return
        self.run_btn.config(state=tk.DISABLED)
        self.set_status("Running layout… this can take a moment for large graphs.",
                        "#d29922")
        threading.Thread(target=self._worker, args=(cfg,), daemon=True).start()

    def _worker(self, cfg: dict):
        try:
            self.write_config(cfg)
            result = subprocess.run(
                [str(EXE), str(CONFIG_TXT)],
                cwd=str(ROOT), capture_output=True, text=True)
            self.root.after(0, self._on_done, result, cfg)
        except Exception as exc:  # noqa: BLE001 — surface anything to the UI
            self.root.after(0, self._on_error, str(exc))

    def _on_done(self, result, cfg):
        self.run_btn.config(state=tk.NORMAL)
        if result.returncode != 0:
            self._on_error((result.stdout or "") + "\n" + (result.stderr or ""))
            return
        self.load_outputs(Path(cfg["outputDir"]))

    def _on_error(self, message: str):
        self.run_btn.config(state=tk.NORMAL)
        self.set_status("Layout failed — see dialog.", "#f85149")
        messagebox.showerror("Layout failed", message.strip()[:2000])

    # ── Load & render results ─────────────────────────────────────────────────
    def load_outputs(self, output_dir: Path):
        out = output_dir if output_dir.is_absolute() else (ROOT / output_dir)
        if not out.exists():
            self._on_error(f"Output directory not found:\n{out}")
            return

        self.graph_dirs = sorted(
            d for d in out.iterdir()
            if d.is_dir() and (d / "nodes.csv").exists()
            and (d / "edges.csv").exists())

        if not self.graph_dirs:
            self.set_status("Layout ran but produced no graphs. "
                            "Are there .txt files in Input/?", "#d29922")
            self._blank_axes("No graphs found in output.")
            self.canvas.draw()
            return

        # Save a PNG for every graph (persisted, as requested).
        for d in self.graph_dirs:
            try:
                self._render_graph_to_png(d)
            except Exception:  # noqa: BLE001 — one bad graph shouldn't stop all
                pass

        names = [d.name for d in self.graph_dirs]
        self.combo["values"] = names
        self.combo.current(0)
        self.show_selected()
        self.set_status(
            f"Done. {len(self.graph_dirs)} graph(s) laid out. "
            f"PNGs saved in '{output_dir}'.", "#3fb950")

    def step(self, delta: int):
        if not self.graph_dirs:
            return
        i = (self.combo.current() + delta) % len(self.graph_dirs)
        self.combo.current(i)
        self.show_selected()

    def _vertex_size(self) -> float:
        """Vertex marker area from the UI field; falls back to the default if
        the box holds something that isn't a positive number."""
        try:
            v = float(self.vertex_entry.get())
            return v if v > 0 else NODE_SIZE
        except (ValueError, AttributeError):
            return NODE_SIZE

    def show_selected(self):
        i = self.combo.current()
        if i < 0 or i >= len(self.graph_dirs):
            return
        self._draw(self.ax, self.graph_dirs[i])
        self.fig.tight_layout()
        self.canvas.draw()

    def redraw_current(self):
        """Re-render the currently selected graph with the current vertex size,
        without re-running the layout, and refresh its saved PNG to match."""
        i = self.combo.current()
        if i < 0 or i >= len(self.graph_dirs):
            return
        self.show_selected()
        try:
            self._render_graph_to_png(self.graph_dirs[i])
        except Exception:  # noqa: BLE001 — display already updated; PNG is a bonus
            pass
        self.set_status(f"Redrawn with vertex size {self._vertex_size():g}.",
                        "#3fb950")

    def _render_graph_to_png(self, gdir: Path):
        fig = Figure(figsize=(10, 6), facecolor=BG_COLOUR)
        ax = fig.add_subplot(111)
        self._draw(ax, gdir)
        fig.tight_layout()
        fig.savefig(gdir / "layout.png", facecolor=BG_COLOUR, dpi=150)

    def _draw(self, ax, gdir: Path):
        nodes = pd.read_csv(gdir / "nodes.csv")
        edges = pd.read_csv(gdir / "edges.csv")

        ax.clear()
        ax.set_facecolor(BG_COLOUR)
        margin = 60.0
        ax.set_xlim(nodes["x"].min() - margin, nodes["x"].max() + margin)
        ax.set_ylim(nodes["y"].min() - margin, nodes["y"].max() + margin)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        pos = nodes.set_index("node_id")[["x", "y"]]
        xs, ys = [], []
        for _, e in edges.iterrows():
            s, t = int(e["source"]), int(e["target"])
            if s in pos.index and t in pos.index:
                xs += [pos.at[s, "x"], pos.at[t, "x"], None]
                ys += [pos.at[s, "y"], pos.at[t, "y"], None]
        if xs:
            ax.plot(xs, ys, color=EDGE_COLOUR, lw=EDGE_WIDTH,
                    alpha=EDGE_ALPHA, zorder=1, solid_capstyle="round")

        ax.scatter(nodes["x"], nodes["y"], s=self._vertex_size(), c=NODE_COLOUR,
                   edgecolors=NODE_EDGE_C, linewidths=0.6, zorder=2)

        if len(nodes) <= 50:
            for _, row in nodes.iterrows():
                ax.text(row["x"], row["y"], str(int(row["node_id"])),
                        ha="center", va="center", fontsize=6,
                        color="white", fontweight="bold", zorder=3)

        ax.set_title(f"{gdir.name}    |V| = {len(nodes)}   |E| = {len(edges)}",
                     color="white", fontsize=12, pad=10)


def main():
    root = tk.Tk()
    ControlPanel(root)
    root.mainloop()


if __name__ == "__main__":
    main()
