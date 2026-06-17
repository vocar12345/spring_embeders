#!/usr/bin/env bash
# ===========================================================================
#  Spring Embedder Control Panel launcher  (Linux / macOS)
#  Run with:  ./run_ui.sh      (make it executable first: chmod +x run_ui.sh)
# ===========================================================================
set -e
cd "$(dirname "$0")"

# Build the C++ layout engine the first time, if it isn't there yet.
if [ ! -x build/fr_batch ]; then
    echo "Building fr_batch for the first time..."
    if [ ! -f build/CMakeCache.txt ]; then
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    fi
    cmake --build build --target fr_batch
fi

# Prefer python3; fall back to python.
if command -v python3 >/dev/null 2>&1; then
    exec python3 ui.py
else
    exec python ui.py
fi
