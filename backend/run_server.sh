#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "========================================"
echo " RAG Chrome Extension - Local Server"
echo "========================================"
echo

# Step 1: Activate virtual environment
echo "Activating virtual environment..."
source ../venv/bin/activate

# Step 2: Download models (if not already downloaded)
echo "[1/2] Downloading ONNX + embedding models..."
python3 download_models.py

# Step 3: Start FastAPI server using uvicorn
echo
echo "Starting FastAPI backend on http://127.0.0.1:5000 ..."
echo "Press Ctrl+C to stop."
echo
python3 -m uvicorn server:app --host 127.0.0.1 --port 5000
