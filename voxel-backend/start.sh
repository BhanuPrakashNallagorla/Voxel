#!/usr/bin/env bash
# Startup script for the Voxel Depth API backend.
# Installs Python deps (skips if already installed) then starts uvicorn.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Voxel Depth API Startup ==="
echo "Working dir: $SCRIPT_DIR"
echo "Python: $(python3 --version)"

# Install deps
echo "--- Installing Python dependencies ---"
pip install -q -r requirements.txt

echo "--- Starting FastAPI server on port 8000 ---"
exec uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
