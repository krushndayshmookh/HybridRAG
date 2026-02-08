@echo off
REM Quick activation script for Windows
REM Usage: activate.bat

if exist "venv\" (
    call venv\Scripts\activate.bat
    echo ✓ Virtual environment activated
    echo.
    echo Available commands:
    echo   streamlit run HybridRag.py       - Run web interface
    echo   python run_evaluation.py         - Run complete evaluation
    echo   python question_generator.py     - Generate questions
    echo   python evaluation_pipeline.py    - Run evaluation
    echo   python report_generator.py       - Generate reports
    echo.
) else (
    echo ✗ Virtual environment not found!
    echo Please run: setup.bat
    pause
)
