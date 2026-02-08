#!/bin/bash
# Quick activation script for Unix/Mac
# Usage: source activate.sh

if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
    echo ""
    echo "Available commands:"
    echo "  streamlit run HybridRag.py       - Run web interface"
    echo "  python run_evaluation.py         - Run complete evaluation"
    echo "  python question_generator.py     - Generate questions"
    echo "  python evaluation_pipeline.py    - Run evaluation"
    echo "  python report_generator.py       - Generate reports"
    echo ""
else
    echo "✗ Virtual environment not found!"
    echo "Please run: ./setup.sh"
fi
