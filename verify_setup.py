#!/usr/bin/env python3
"""
Verification Script for Hybrid RAG System
Tests that all components are properly installed and working
"""

import sys
import importlib

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✓ {package_name:30s} OK")
        return True
    except ImportError as e:
        print(f"✗ {package_name:30s} MISSING: {e}")
        return False

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (meets requirement: 3.8+)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_nltk_data():
    """Check if NLTK data is downloaded"""
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
            print("✓ NLTK punkt data            OK")
            return True
        except LookupError:
            print("✗ NLTK punkt data            MISSING (run: python -c \"import nltk; nltk.download('punkt')\")")
            return False
    except ImportError:
        print("✗ NLTK not installed")
        return False

def check_files():
    """Check if required files exist"""
    import os
    
    required_files = {
        "HybridRag.py": "Main RAG system",
        "question_generator.py": "Question generator",
        "evaluation_metrics.py": "Evaluation metrics",
        "evaluation_pipeline.py": "Evaluation pipeline",
        "report_generator.py": "Report generator",
        "requirements.txt": "Dependencies list",
        "README.md": "Documentation"
    }
    
    all_exist = True
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"✓ {filename:30s} found")
        else:
            print(f"✗ {filename:30s} MISSING - {description}")
            all_exist = False
    
    return all_exist

def main():
    """Run all verification checks"""
    
    print_header("HYBRID RAG SYSTEM - VERIFICATION")
    
    print("\n[1/4] Checking Python Version...")
    python_ok = check_python_version()
    
    print("\n[2/4] Checking Required Packages...")
    packages = [
        ("sentence-transformers", "sentence_transformers"),
        ("faiss-cpu", "faiss"),
        ("rank-bm25", "rank_bm25"),
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("beautifulsoup4", "bs4"),
        ("requests", "requests"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("rouge-score", "rouge_score"),
        ("nltk", "nltk"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("streamlit", "streamlit"),
        ("tiktoken", "tiktoken"),
        ("tqdm", "tqdm"),
    ]
    
    packages_ok = all(check_package(name, import_name) for name, import_name in packages)
    
    print("\n[3/4] Checking NLTK Data...")
    nltk_ok = check_nltk_data()
    
    print("\n[4/4] Checking Required Files...")
    files_ok = check_files()
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    all_ok = python_ok and packages_ok and nltk_ok and files_ok
    
    if all_ok:
        print("\nAll checks passed! Your environment is ready.")
        print("\nYou can now run:")
        print("  • streamlit run HybridRag.py          (Web interface)")
        print("  • python run_evaluation.py            (Complete evaluation)")
        print("  • python question_generator.py        (Generate questions)")
        print("  • python evaluation_pipeline.py       (Run evaluation)")
        print("  • python report_generator.py          (Generate reports)")
    else:
        print("\n[ERROR]  Some checks failed. Please review the errors above.")
        print("\nTo fix issues:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Download NLTK data: python -c \"import nltk; nltk.download('punkt')\"")
        print("  3. Ensure you're in the correct directory")
        print("  4. Check that all project files are present")
    
    print("\n" + "="*70 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
