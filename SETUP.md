# Environment Setup Guide

## Prerequisites

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 2GB+ free space
- **OS**: Windows, macOS, or Linux

## Quick Setup (Recommended)

### For macOS/Linux

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup script
./setup.sh

# The script will:
# 1. Check Python version
# 2. Create virtual environment
# 3. Install all dependencies
# 4. Download NLTK data
# 5. Set up project structure
```

### For Windows

```cmd
# Run setup script
setup.bat

# The script will perform the same steps as above
```

## Manual Setup

If you prefer to set up manually:

### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 2: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt')"
```

### Step 5: Create Directories

```bash
mkdir -p screenshots logs outputs
```

## Activating the Environment

### Quick Activation (after initial setup)

#### On macOS/Linux

```bash
source activate.sh
```

#### On Windows

```cmd
activate.bat
```

### Manual Activation

#### macOS/Linux

```bash
source venv/bin/activate
```

#### Windows

```cmd
venv\Scripts\activate
```

## Verifying Installation

After setup, verify everything is installed correctly:

```bash
# Check Python version
python --version

# Verify key packages
python -c "import transformers; print('✓ transformers')"
python -c "import sentence_transformers; print('✓ sentence-transformers')"
python -c "import faiss; print('✓ faiss')"
python -c "import streamlit; print('✓ streamlit')"
python -c "import nltk; print('✓ nltk')"
```

## Running the System

### 1. Web Interface

```bash
streamlit run HybridRag.py
```

Opens at: <http://localhost:8501>

### 2. Complete Evaluation

```bash
python run_evaluation.py
```

### 3. Question Generation

```bash
python question_generator.py
```

### 4. Evaluation Pipeline

```bash
python evaluation_pipeline.py
```

### 5. Report Generation

```bash
python report_generator.py
```

## Troubleshooting

### Issue: Python version too old

**Solution**: Install Python 3.8+ from <https://www.python.org/>

### Issue: pip install fails

**Solution**:

```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Issue: Out of memory during evaluation

**Solution**:

- Reduce batch size in evaluation scripts
- Process questions in smaller batches
- Use CPU instead of GPU
- Close other applications

### Issue: NLTK data not found

**Solution**:

```bash
python -c "import nltk; nltk.download('punkt')"
```

### Issue: FAISS installation fails

**Solution**:

```bash
# Try CPU version
pip install faiss-cpu

# Or GPU version (if you have CUDA)
pip install faiss-gpu
```

### Issue: Streamlit won't start

**Solution**:

```bash
# Check if port is in use
streamlit run HybridRag.py --server.port 8502

# Or specify different port
streamlit run HybridRag.py --server.port 8080
```

### Issue: Torch installation fails

**Solution**:

```bash
# Install CPU-only version
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Deactivating the Environment

```bash
deactivate
```

## Updating Dependencies

To update all packages to latest versions:

```bash
pip install --upgrade -r requirements.txt
```

## Cleaning Up

To remove the virtual environment:

```bash
# Deactivate first
deactivate

# Remove directory
rm -rf venv  # macOS/Linux
rmdir /s /q venv  # Windows
```

## Environment Variables (Optional)

You can set these environment variables for customization:

```bash
# Set NLTK data path
export NLTK_DATA=./nltk_data

# Disable GPU (use CPU only)
export CUDA_VISIBLE_DEVICES=-1

# Set HuggingFace cache directory
export HF_HOME=./hf_cache

# Set Streamlit config
export STREAMLIT_SERVER_PORT=8501
```

## Development Setup

For development with additional tools:

```bash
pip install jupyter notebook ipython black flake8 pytest
```

## Docker Setup (Alternative)

If you prefer using Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "import nltk; nltk.download('punkt')"

EXPOSE 8501

CMD ["streamlit", "run", "HybridRag.py"]
```

Build and run:

```bash
docker build -t hybrid-rag .
docker run -p 8501:8501 hybrid-rag
```

## System Requirements by Task

### Web Interface (Streamlit)

- RAM: 2GB minimum
- CPU: Any modern processor
- Time: Instant startup

### Question Generation

- RAM: 4GB minimum
- CPU/GPU: GPU recommended (10x faster)
- Time: 10-15 minutes (CPU), 2-3 minutes (GPU)

### Evaluation Pipeline

- RAM: 4GB minimum
- CPU/GPU: GPU recommended
- Time: 20-30 minutes (CPU), 5-10 minutes (GPU)

### Report Generation

- RAM: 2GB minimum
- CPU: Any modern processor
- Time: 1-2 minutes

## Platform-Specific Notes

### In macOS

- May need to install Xcode Command Line Tools:

  ```bash
  xcode-select --install
  ```

- Some packages may require Homebrew dependencies

### In Linux

- May need to install system packages:

  ```bash
  sudo apt-get update
  sudo apt-get install python3-dev build-essential
  ```

### In Windows

- May need Visual C++ Build Tools for some packages
- Download from: <https://visualstudio.microsoft.com/downloads/>
- Select "Desktop development with C++"

## Getting Help

If you encounter issues not covered here:

1. Check the README.md for detailed documentation
2. Review error messages carefully
3. Check Python and package versions
4. Ensure adequate system resources (RAM, disk space)
5. Try installing packages individually to identify problematic ones

## Next Steps

After successful setup:

1. ✅ Run web interface to test basic functionality
2. ✅ Generate evaluation questions
3. ✅ Run evaluation pipeline
4. ✅ Generate reports
5. ✅ Review results and prepare submission

Good luck! 🚀
