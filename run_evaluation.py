#!/usr/bin/env python3
"""
Quick Start Script for Hybrid RAG Evaluation
Runs the complete evaluation pipeline in sequence
"""

import os
import sys
import time

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✓ Found {description}: {filepath}")
        return True
    else:
        print(f"✗ Missing {description}: {filepath}")
        return False

def main():
    """Run complete evaluation pipeline"""
    
    print_header("HYBRID RAG SYSTEM - AUTOMATED EVALUATION")
    print("This script will run the complete evaluation pipeline:")
    print("  1. Check prerequisites")
    print("  2. Generate evaluation questions (if needed)")
    print("  3. Run evaluation on 100 questions")
    print("  4. Perform ablation study and error analysis")
    print("  5. Generate comprehensive reports")
    print("\nThis may take 30-60 minutes depending on your hardware.")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Step 1: Check prerequisites
    print_header("Step 1: Checking Prerequisites")
    
    prerequisites = {
        "wiki_chunks.jsonl": "Preprocessed chunks",
        "fixed_urls.json": "Fixed Wikipedia URLs",
        "dense.index": "FAISS index",
        "embeddings.npy": "Dense embeddings"
    }
    
    missing = []
    for filepath, description in prerequisites.items():
        if not check_file_exists(filepath, description):
            missing.append(filepath)
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} required file(s).")
        print("Please run HybridRag.py first to set up the system:")
        print("  streamlit run HybridRag.py")
        return
    
    print("\n✓ All prerequisites satisfied!")
    
    # Step 2: Generate questions if needed
    print_header("Step 2: Question Generation")
    
    if os.path.exists("evaluation_questions.json"):
        print("✓ Found evaluation_questions.json")
        print("Skipping question generation.")
    else:
        print("Generating 100 evaluation questions...")
        try:
            from question_generator import QuestionGenerator
            qg = QuestionGenerator()
            qg.generate_questions(n_questions=100)
            print("✓ Question generation complete!")
        except Exception as e:
            print(f"✗ Error generating questions: {e}")
            return
    
    # Step 3: Run evaluation
    print_header("Step 3: Running Evaluation Pipeline")
    
    print("Evaluating RAG system on all questions...")
    print("This includes:")
    print("  - Standard metrics (MRR, ROUGE-L, NDCG@5)")
    print("  - Ablation study (dense vs sparse vs hybrid)")
    print("  - Error analysis by question type")
    
    try:
        os.system("python evaluation_pipeline.py")
        print("\n✓ Evaluation complete!")
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        return
    
    # Step 4: Generate reports
    print_header("Step 4: Generating Reports")
    
    if not os.path.exists("evaluation_results.json"):
        print("✗ evaluation_results.json not found!")
        print("Evaluation may have failed. Please check the logs.")
        return
    
    print("Generating PDF and HTML reports...")
    try:
        os.system("python report_generator.py")
        print("\n✓ Reports generated!")
    except Exception as e:
        print(f"✗ Error generating reports: {e}")
        return
    
    # Final summary
    print_header("EVALUATION COMPLETE!")
    
    print("Generated Files:")
    print("  evaluation_questions.json     - 100 Q&A pairs")
    print("  evaluation_results.json       - Detailed results")
    print("  evaluation_results.csv        - Tabular results")
    print("  evaluation_report.pdf         - Visualizations")
    print("  evaluation_report.html        - Interactive dashboard")
    
    print("\nNext Steps:")
    print("  1. Open evaluation_report.html in your browser")
    print("  2. Review evaluation_report.pdf for detailed visualizations")
    print("  3. Check evaluation_results.csv for raw data")
    
    print("\nTo run the web interface:")
    print("  streamlit run HybridRag.py")
    
    print("\n" + "="*70)
    print("Thank you for using the Hybrid RAG Evaluation System!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
