@echo off
cd /d "%~dp0"
echo ========================================
echo  Chrome Extension - Local Server
echo ========================================
echo.

REM Step 1: Activate virtual environment
echo Activating virtual environment...
call ..\venv\Scripts\activate

REM Step 2: Download models (if not already downloaded)
echo Downloading ONNX + embedding models...
python download_models.py

REM Step 3: Start FastAPI server using uvicorn
echo.
echo Starting FastAPI backend on http://127.0.0.1:5000 ...
echo Press CTRL+C to stop.
echo.
python -m uvicorn server:app --host 127.0.0.1 --port 5000

pause
