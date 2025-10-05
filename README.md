# Browser-Assistant

A lightweight local RAG-style (Retriever + Generator) assistant that extracts text from a webpage (via a Chrome extension), builds a small local vector store, and answers user questions using an ONNX-based local LLM. The project contains a FastAPI backend that runs locally and a Chrome extension UI that talks to the backend.

## Introduction
Browser Assistant is designed to let you ask questions about the content of any web page. The extension grabs the current page URL and sends it to a local FastAPI server which:

- Fetches and extracts the page text (Trafilatura).
- Builds a FAISS index from that text (langchain utilities).
- Uses a local ONNX GenAI model to generate an answer conditioned on retrieved context.
- All heavy ML artifacts are kept local (downloadable via `download_models.py`) so you can run everything offline (or on a local machine).

## Key Features
- Extracts main content from web pages using Trafilatura.
- Splits long text into chunks and builds a FAISS vector store.
- Uses ONNX GenAI model for local generation (Phi-3.5-mini-instruct ONNX).
- Small embedding model used for vectorization (gte-small).
- Simple Chrome extension UI to ask questions from the current tab.
- Simple, single-endpoint FastAPI backend: `POST /query`.

## Important model / path notes
- `download_models.py` uses `huggingface_hub.snapshot_download` to place models under `backend/models/` (or `.\models\...` paths used in `server.py`).
- `server.py` contains a `MODEL_PATH` constant that must point to the correct ONNX model snapshot folder on your disk. If you see errors like `Error opening ... genai_config.json`, ensure the `MODEL_PATH` exactly matches the downloaded snapshot path.
- The embedding model path is also set in `server.py` (`HuggingFaceEmbeddings`). Update both paths if you put models elsewhere.

## Installation
1. Clone repository:

    git clone https://github.com/ShamikRana/Browser-Assistant.git

2. Create & activate virtual environment:

### Unix / macOS

    python3 -m venv ./venv
    source ./venv/bin/activate

### Windows (PowerShell)

    python -m venv .\venv
    .\venv\Scripts\Activate.ps1

3. Install Dependencies:

    pip install -r requirements.txt

4. Download models (this will place model files under `backend/models/`):

    python download_models.py

5. Run the backend

### Windows (via provided batch file):

    cd backend
    run_server.bat

### macOS/Linux:

    cd backend
    bash run_server.sh

### or run uvicorn directly (if models are downloaded):

    cd backend
    python -m uvicorn server:app --host 127.0.0.1 --port 5000

6. When running, the server exposes a single JSON endpoint:

### POST `http://127.0.0.1:5000/query`
**Request**

    {
      "url": "https://example.com/article",
      "question": "Summarize the main claim in one sentence."
    }

**Response**

    {"answer": "Generated answer from local ONNX model..."}

## Chrome Extension (usage)
1. Open `chrome://extensions` in Chrome (or `edge://extensions` for Edge).
2. Enable Developer mode.
3. Click **Load unpacked** and select the `extension/` folder.
4. Open any page, click the extension icon, type a question, and press **Ask**.
![Extension](https://raw.githubusercontent.com/ShamikRana/Browser-Assistant/refs/heads/main/extension/icons/image.png)
5. The extension expects the backend at `http://localhost:5000`. If your server runs on a different host/port, update `manifest.json` `host_permissions` and `popup.js` accordingly.
