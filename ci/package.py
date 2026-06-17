"""
ci/package.py
─────────────────────────────────────────────────────────────────────────────
Assembles a ready-to-run distribution zip for the current OS, used by the
GitHub Actions release workflow (.github/workflows/build.yml).

It expects, already built in the repo root:
  * the PyInstaller UI binary   -> dist/SpringEmbedderUI[.exe]
  * the C++ engine              -> build/fr_batch[.exe]  (or build/Release/...)

and produces:
  artifact/SpringEmbedderUI-<os>-x64/
      SpringEmbedderUI[.exe]      # the UI
      build/fr_batch[.exe]        # the engine the UI calls
      Input/...                   # sample graphs to lay out
      HOW_TO_RUN.txt
  artifact/SpringEmbedderUI-<os>-x64.zip

The folder layout matches what ui.py expects (build/fr_batch next to the UI).
"""

import os
import sys
import stat
import shutil
import zipfile
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "artifact"


def detect_label() -> str:
    if os.name == "nt":
        return "windows-x64"
    if sys.platform.startswith("linux"):
        return "linux-x64"
    if sys.platform == "darwin":
        return "macos-x64"
    return sys.platform


def find_first(*candidates: Path) -> Path:
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "None of these build outputs were found:\n  " +
        "\n  ".join(str(c) for c in candidates))


def main() -> None:
    is_win = os.name == "nt"
    label  = detect_label()

    ui_name    = "SpringEmbedderUI.exe" if is_win else "SpringEmbedderUI"
    batch_name = "fr_batch.exe" if is_win else "fr_batch"

    ui_src = find_first(ROOT / "dist" / ui_name)
    batch_src = find_first(
        ROOT / "build" / batch_name,
        ROOT / "build" / "Release" / batch_name,
        ROOT / "build" / "Release" / "fr_batch.exe",
        ROOT / "build" / "fr_batch.exe",
    )

    pkg_name = f"SpringEmbedderUI-{label}"
    staging  = ARTIFACT / pkg_name
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "build").mkdir(parents=True)

    shutil.copy2(ui_src, staging / ui_name)
    shutil.copy2(batch_src, staging / "build" / batch_name)

    # Bundle the sample input graphs so the package runs out of the box.
    if (ROOT / "Input").is_dir():
        shutil.copytree(ROOT / "Input", staging / "Input")
    else:
        (staging / "Input").mkdir()

    if is_win:
        start_steps = (
            "2. Start the app:\n"
            f"   Double-click {ui_name}\n")
        requirement = "Requires 64-bit Windows 10 or newer.\n"
    else:
        start_steps = (
            "2. Start the app from a terminal opened IN THIS FOLDER:\n"
            f"       chmod +x {ui_name} build/{batch_name}\n"
            f"       ./{ui_name}\n"
            "   (Linux has no reliable 'double-click to run' for executables,\n"
            "    so the app is launched from a terminal.)\n")
        requirement = (
            "Requires a 64-bit Linux with glibc 2.35+ (Ubuntu 22.04+, Debian 12+,\n"
            "Fedora 36+) and a graphical desktop session (the UI is a window).\n")

    (staging / "HOW_TO_RUN.txt").write_text(
        "Spring Embedder — Control Panel\n"
        "================================\n\n"
        "1. Put your adjacency-list .txt graphs in the Input/ folder.\n"
        + start_steps
        + "3. Set parameters, click Run Layout, and view the graphs.\n"
          "   PNGs are written to output/<graph>/layout.png.\n\n"
        + requirement
        + f"Keep this folder intact — the UI calls build/{batch_name}.\n",
        encoding="utf-8")

    # Build the zip, marking the two binaries executable (matters on Linux/macOS).
    exec_names = {ui_name, batch_name}
    zip_path = ARTIFACT / f"{pkg_name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_dir():
                continue
            arc = path.relative_to(ARTIFACT).as_posix()
            zi = zipfile.ZipInfo(arc)
            mode = 0o755 if path.name in exec_names else 0o644
            zi.external_attr = (mode & 0xFFFF) << 16
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, path.read_bytes())

    print(f"Packaged: {zip_path}  ({zip_path.stat().st_size/1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()
