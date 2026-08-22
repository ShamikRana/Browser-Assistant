@echo off
cd /d "%~dp0"
echo ========================================
echo  Browser Assistant (Phi-4) - Local Server
echo ========================================
echo.

REM Step 1: Activate virtual environment if present
if exist "..\venv\Scripts\activate.bat" call ..\venv\Scripts\activate.bat

REM Step 2: Resolve device. Usage: run_server.bat [auto|cpu|gpu]  (default: auto)
set DEVICE=%1
if "%DEVICE%"=="" set DEVICE=auto
if /I "%DEVICE%"=="cuda" set DEVICE=gpu

REM Step 3: Download models if they are missing
python download_model.py --device %DEVICE% --check-only
if errorlevel 1 (
    echo Downloading Phi-4 + embedding models ^(%DEVICE%^)...
    python download_model.py --device %DEVICE%
)

REM Step 4: Start the server
echo.
echo Starting FastAPI backend on http://127.0.0.1:5000 ...
echo Press CTRL+C to stop.
echo.
python serve.py --device %DEVICE%

pause
