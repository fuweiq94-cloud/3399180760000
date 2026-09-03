# @echo off
REM Start D3QN Snake Training Script with Virtual Environment Check

cd /d "%~dp0"

REM Check if .venv exists
if not exist ".venv\" (
    echo [ERROR] Virtual environment not found!
    echo Please run: uv venv --python 3.14
    pause
    exit /b 1
)

echo [OK] Virtual environment detected: .venv

REM Activate virtual environment and run training
echo Starting D3QN Snake AI Training...
echo.

# Navigate to src directory
cd /d "%~dp0..\src"

call ..\.venv\Scripts\activate.bat
python train.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Training failed!
    pause
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] Training completed successfully!
pause
