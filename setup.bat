@echo off
echo ============================================
echo   AI Traffic Violation Detection System
echo   Flipkart Gridlock 2.0 - Setup Script
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ from python.org
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing dependencies (this may take a few minutes)...
pip install --upgrade pip
pip install -r requirements.txt

echo [4/4] Downloading YOLOv8n model weights...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('Model downloaded successfully')"

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo To run the demo:
echo   venv\Scripts\activate.bat
echo   python demo.py --demo --output ./output --save-video
echo.
echo To start the dashboard:
echo   venv\Scripts\activate.bat
echo   streamlit run dashboard/app.py
echo.
echo To start the API server:
echo   venv\Scripts\activate.bat
echo   python -m api.server
echo.
pause
