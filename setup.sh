#!/bin/bash

###############################################################################
# Setup Script for Hybrid RAG System
# This script creates a virtual environment and installs all dependencies
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Hybrid RAG System - Environment Setup                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}[1/7]${NC} Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 is not installed!${NC}"
    echo "Please install Python 3.8 or higher from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}✗ Python 3.8+ is required. You have Python $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}[2/7]${NC} Virtual environment already exists."
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    else
        echo "Using existing virtual environment."
        source venv/bin/activate
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
        skip_venv_creation=true
    fi
fi

# Create virtual environment
if [ "$skip_venv_creation" != true ]; then
    echo -e "${YELLOW}[2/7]${NC} Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}[3/7]${NC} Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "${YELLOW}[4/7]${NC} Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install requirements
echo -e "${YELLOW}[5/7]${NC} Installing dependencies..."
echo "This may take 5-10 minutes..."
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Download NLTK data
echo -e "${YELLOW}[6/7]${NC} Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); print('NLTK data downloaded')"
echo -e "${GREEN}✓ NLTK data downloaded${NC}"

# Create necessary directories
echo -e "${YELLOW}[7/7]${NC} Setting up project structure..."
mkdir -p screenshots
mkdir -p logs
mkdir -p outputs
echo -e "${GREEN}✓ Project structure ready${NC}"

# Success message
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           Setup Complete!                                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Virtual environment is ${GREEN}ACTIVATED${NC}"
echo ""
echo "Quick Start Commands:"
echo -e "  ${BLUE}1.${NC} Run RAG web interface:"
echo "     streamlit run HybridRag.py"
echo ""
echo -e "  ${BLUE}2.${NC} Run complete evaluation:"
echo "     python run_evaluation.py"
echo ""
echo -e "  ${BLUE}3.${NC} Generate questions only:"
echo "     python question_generator.py"
echo ""
echo -e "  ${BLUE}4.${NC} Run evaluation pipeline:"
echo "     python evaluation_pipeline.py"
echo ""
echo -e "  ${BLUE}5.${NC} Generate reports:"
echo "     python report_generator.py"
echo ""
echo "To activate the environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo -e "${YELLOW}Note:${NC} Make sure you have 4GB+ RAM and 2GB+ free disk space"
echo ""
