@echo off
echo ===================================================
echo  CyberSecureDocAI - Local Python Environment Setup
echo ===================================================

echo [1/3] Creating virtual environment (.venv)...
python -m venv .venv
call .venv\Scripts\activate

echo [2/3] Installing dependencies...
pip install -r requirements.txt

echo [3/3] Downloading offline models (Qwen2 1.5B and MiniLM embeddings)...
python download_models.py

echo.
echo ===================================================
echo Setup complete! Run 'run_local.bat' to start the app.
echo ===================================================
pause
