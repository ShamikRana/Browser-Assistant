#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "========================================"
echo " Browser Assistant (Phi-4) - Local Server"
echo "========================================"
echo

# Step 1: Activate virtual environment if present
if [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
fi

# Step 2: Resolve device. Usage: ./run_server.sh [auto|cpu|gpu]  (default: auto)
DEVICE="${1:-auto}"
DEVICE="$(echo "$DEVICE" | tr '[:upper:]' '[:lower:]')"
if [ "$DEVICE" = "cuda" ]; then DEVICE="gpu"; fi

# Step 3: Download models if they are missing
if ! python3 download_model.py --device "$DEVICE" --check-only; then
    echo "Downloading Phi-4 + embedding models ($DEVICE)..."
    python3 download_model.py --device "$DEVICE"
fi

# Step 4: Start the server
echo
echo "Starting FastAPI backend on http://127.0.0.1:5000 ..."
echo "Press Ctrl+C to stop."
echo
python3 serve.py --device "$DEVICE"
