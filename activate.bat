@echo off
REM Quick activation script for Windows
REM Usage: activate.bat

if exist "venv\" (
    echo Activate Virtual environment with: venv\Scripts\activate.bat
    echo.
    echo Available commands:
    echo   streamlit run HybridRag.py       - Run web interface
    echo   python run_evaluation.py         - Run complete evaluation
    echo   python question_generator.py     - Generate questions
    echo   python evaluation_pipeline.py    - Run evaluation
    echo   python report_generator.py       - Generate reports
    echo.
) else (
    echo Virtual environment not found!
    echo Please run: setup.bat
    pause
)
