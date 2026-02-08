@echo off
REM ###############################################################################
REM Setup Script for Hybrid RAG System (Windows)
REM This script creates a virtual environment and installs all dependencies
REM ###############################################################################

echo ╔════════════════════════════════════════════════════════════════╗
echo ║        Hybrid RAG System - Environment Setup                  ║
echo ╔════════════════════════════════════════════════════════════════╗
echo.

REM Check Python version
echo [1/7] Checking Python version...
python --version > nul 2>&1
if errorlevel 1 (
    echo ✗ Python is not installed or not in PATH!
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✓ Python %PYTHON_VERSION% detected
echo.

REM Check if virtual environment exists
if exist "venv\" (
    echo [2/7] Virtual environment already exists.
    set /p RECREATE="Do you want to recreate it? (y/n): "
    if /i "%RECREATE%"=="y" (
        echo Removing existing virtual environment...
        rmdir /s /q venv
        goto CREATE_VENV
    ) else (
        echo Using existing virtual environment.
        goto ACTIVATE_VENV
    )
)

:CREATE_VENV
REM Create virtual environment
echo [2/7] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ✗ Failed to create virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment created
echo.

:ACTIVATE_VENV
REM Activate virtual environment
echo [3/7] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ✗ Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment activated
echo.

REM Upgrade pip
echo [4/7] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
echo ✓ pip upgraded
echo.

REM Install requirements
echo [5/7] Installing dependencies...
echo This may take 5-10 minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo ✗ Failed to install dependencies
    pause
    exit /b 1
)
echo ✓ Dependencies installed
echo.

REM Download NLTK data
echo [6/7] Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); print('NLTK data downloaded')"
echo ✓ NLTK data downloaded
echo.

REM Create necessary directories
echo [7/7] Setting up project structure...
if not exist "screenshots\" mkdir screenshots
if not exist "logs\" mkdir logs
if not exist "outputs\" mkdir outputs
echo ✓ Project structure ready
echo.

REM Success message
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║           Setup Complete! 🎉                                   ║
echo ╔════════════════════════════════════════════════════════════════╗
echo.
echo Virtual environment is ACTIVATED
echo.
echo Quick Start Commands:
echo   1. Run RAG web interface:
echo      streamlit run HybridRag.py
echo.
echo   2. Run complete evaluation:
echo      python run_evaluation.py
echo.
echo   3. Generate questions only:
echo      python question_generator.py
echo.
echo   4. Run evaluation pipeline:
echo      python evaluation_pipeline.py
echo.
echo   5. Generate reports:
echo      python report_generator.py
echo.
echo To activate the environment in future sessions:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate:
echo   deactivate
echo.
echo Note: Make sure you have 4GB+ RAM and 2GB+ free disk space
echo.
pause
