#!/usr/bin/env bash
# =====================================================================
# Praxis Studio Launcher for Linux / macOS
# Starts the FastAPI backend with Uvicorn and opens the interactive
# 3D physical manipulation viewer in the default web browser.
# =====================================================================

set -e

# Change to script directory
cd "$(dirname "$0")"

echo "[Praxis Studio] Checking Python environment..."

if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[Praxis Studio] ERROR: Python 3 is not installed or not in PATH."
    exit 1
fi

# Ensure required libraries
$PYTHON_CMD -c "import fastapi, uvicorn, mujoco, trimesh" 2>/dev/null || {
    echo "[Praxis Studio] Installing required web & simulation packages..."
    $PYTHON_CMD -m pip install fastapi "uvicorn[standard]" mujoco trimesh
}

PORT=8000
URL="http://127.0.0.1:$PORT"

echo "[Praxis Studio] Launching server on $URL ..."
echo "[Praxis Studio] Press Ctrl+C to terminate."

# Open browser in background
(
    sleep 2
    if command -v xdg-open &>/dev/null; then
        xdg-open "$URL"
    elif command -v open &>/dev/null; then
        open "$URL"
    else
        echo "[Praxis Studio] Please open $URL in your web browser."
    fi
) &

# Run server
exec $PYTHON_CMD -m uvicorn src.studio.server:app --host 127.0.0.1 --port "$PORT"
