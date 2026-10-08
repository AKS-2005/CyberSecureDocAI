@echo off
echo ===================================================
echo  CyberSecureDocAI - Running Local Offline Server
echo ===================================================

if exist .venv\Scripts\activate (
    call .venv\Scripts\activate
)

echo Starting Flask server on http://127.0.0.1:5000 ...
python app.py
pause
