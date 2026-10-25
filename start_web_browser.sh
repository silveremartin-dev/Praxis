#!/usr/bin/env bash
# =====================================================================
# Praxis Studio Launcher for Linux / macOS
# Automatically locates Python, finds a collision-free port,
# and launches the interactive 3D Web Studio in your browser.
# =====================================================================

set -e
cd "$(dirname "$0")"

echo "[Praxis Studio] Checking Python environment..."

if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
elif [ -f "$HOME/.local/bin/python3" ]; then
    PYTHON_CMD="$HOME/.local/bin/python3"
else
    echo "[Praxis Studio] ERROR: Python 3 is not installed or not in PATH."
    exit 1
fi

exec "$PYTHON_CMD" src/studio/run.py
