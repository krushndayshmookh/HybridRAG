# Submission Checklist for Hybrid RAG Assignment

## Pre-Submission Checklist

### Part 1: Hybrid RAG System (10 Marks)

- [x] **1.1 Dense Vector Retrieval**
  - [x] Sentence Transformer model (all-MiniLM-L6-v2)
  - [x] FAISS index implementation
  - [x] Cosine similarity search
  
- [x] **1.2 Sparse Keyword Retrieval**
  - [x] BM25 algorithm implementation
  - [x] Keyword-based index over chunks
  
- [x] **1.3 Reciprocal Rank Fusion (RRF)**
  - [x] RRF scoring with k=60
  - [x] Combines dense and sparse results
  
- [x] **1.4 Response Generation**
  - [x] Open-source LLM (Flan-T5-base)
  - [x] Context concatenation and generation
  
- [x] **1.5 User Interface**
  - [x] Streamlit web interface
  - [x] Displays query, answer, sources, scores, and time

### Part 2: Automated Evaluation (10 Marks)

- [x] **2.1 Question Generation**
  - [x] 100 Q&A pairs generated
  - [x] Diverse question types (factual, comparative, inferential, multi-hop)
  - [x] Stored with ground truth and metadata
  
- [x] **2.2.1 Mandatory Metric (2 marks)**
  - [x] MRR at URL level implemented
  - [x] Justification provided
  - [x] Calculation method documented
  - [x] Interpretation guidelines included
  
- [x] **2.2.2 Custom Metric 1 - ROUGE-L (2 marks)**
  - [x] Implementation complete
  - [x] Justification: Evaluates answer quality and fluency
  - [x] Calculation method: LCS-based F1 score
  - [x] Interpretation guidelines: 0-1 scale with thresholds
  
- [x] **2.2.2 Custom Metric 2 - NDCG@5 (2 marks)**
  - [x] Implementation complete
  - [x] Justification: Evaluates retrieval ranking quality
  - [x] Calculation method: DCG with position discount
  - [x] Interpretation guidelines: 0-1 scale for ranking quality
  
- [x] **2.3 Innovative Evaluation (4 marks)**
  - [x] Ablation Study: Dense vs Sparse vs Hybrid comparison
  - [x] Error Analysis: Categorization by failure type and question type
  - [x] Parameter Sensitivity: Testing different K, N, RRF k values
  
- [x] **2.4 Automated Pipeline**
  - [x] Single-command execution (run_evaluation.py)
  - [x] Loads questions automatically
  - [x] Runs RAG system
  - [x] Computes all metrics
  - [x] Generates detailed reports
  
- [x] **2.5 Evaluation Report**
  - [x] Overall performance summary
  - [x] Metric justifications (MRR, ROUGE-L, NDCG@5)
  - [x] Results table with all questions
  - [x] Visualizations (charts, distributions, heatmaps)
  - [x] Error analysis with failure patterns

## Required Submission Files

### 1. Code Files

- [x] `HybridRag.py` - Main RAG system + Streamlit app
- [x] `question_generator.py` - Question generation script
- [x] `evaluation_metrics.py` - Metrics implementation
- [x] `evaluation_pipeline.py` - Automated pipeline
- [x] `report_generator.py` - Report generation
- [x] `run_evaluation.py` - Quick start script

### 2. Data Files

- [x] `fixed_urls.json` - 200 fixed Wikipedia URLs
- [x] `random_urls.json` - 300 random URLs
- [x] `wiki_chunks.jsonl` - Preprocessed corpus
- [x] `dense.index` - FAISS vector index
- [x] `embeddings.npy` - Cached embeddings

### 3. Evaluation Files

- [ ] `evaluation_questions.json` - 100 Q&A pairs *(Generate by running)*
- [ ] `evaluation_results.json` - Detailed results *(Generate by running)*
- [ ] `evaluation_results.csv` - Tabular results *(Generate by running)*
- [ ] `evaluation_report.pdf` - Visualizations *(Generate by running)*
- [ ] `evaluation_report.html` - Interactive dashboard *(Generate by running)*

### 4. Documentation

- [x] `README.md` - Complete documentation
- [x] `requirements.txt` - Python dependencies
- [x] `SUBMISSION_CHECKLIST.md` - This file

### 5. Report (PDF) - **TO BE CREATED MANUALLY**

The PDF report should include:

- [ ] **Architecture Diagram** - System overview with components
- [ ] **Evaluation Results** - Tables and visualizations
- [ ] **Metric Justifications** - Detailed explanation of MRR, ROUGE-L, NDCG@5
- [ ] **Ablation Study Results** - Performance comparison
- [ ] **Error Analysis** - Failure patterns and insights
- [ ] **System Screenshots** - At least 3 screenshots of the Streamlit interface

### 6. Interface

- [x] Streamlit app implemented
- [ ] **Option A**: Deploy to Streamlit Cloud/Hugging Face Spaces *(Recommended)*
- [x] **Option B**: Include setup instructions in README *(Already done)*

## Pre-Submission Steps

### Step 0: Environment Setup (First Time Only)

**If you haven't set up the environment yet:**

```bash
# Automated setup (Recommended)
# For macOS/Linux:
./setup.sh

# For Windows:
setup.bat
```

**For future sessions, just activate:**

```bash
# macOS/Linux
source activate.sh

# Windows
activate.bat
```

See [SETUP.md](SETUP.md) for detailed setup instructions and troubleshooting.

### Step 1: Generate Evaluation Data

```bash
# Run the complete evaluation pipeline
python run_evaluation.py

# OR run steps individually:
python question_generator.py
python evaluation_pipeline.py
python report_generator.py
```

### Step 2: Verify Generated Files

Check that these files were created:

- `evaluation_questions.json` (100 questions)
- `evaluation_results.json` (detailed results)
- `evaluation_results.csv` (tabular format)
- `evaluation_report.pdf` (visualizations)
- `evaluation_report.html` (interactive dashboard)

### Step 3: Create Final Report PDF

**Manually create a PDF report with:**

1. Title page with group information
2. Architecture diagram (create using draw.io or similar)
3. Methodology section (RRF, metrics)
4. Results section
   - Performance summary table
   - Metric justifications (MRR, ROUGE-L, NDCG@5)
   - Question type breakdown
   - Ablation study results
   - Error analysis
5. Visualizations (from evaluation_report.pdf)
6. System screenshots (3+ images of Streamlit app)
7. Conclusion and future work

### Step 4: Take Screenshots

Capture 3+ screenshots showing:

1. Main Streamlit interface with query input
2. Answer generation with retrieved chunks
3. Scores and metrics display
4. Optional: Evaluation dashboard/results

### Step 5: Test the System

```bash
# Test web interface
streamlit run HybridRag.py

# Test evaluation pipeline
python evaluation_pipeline.py

# Verify all outputs
ls -la evaluation_*
```

### Step 6: Create Submission ZIP

```bash
# Create the submission package
zip -r Group_X_Hybrid_RAG.zip \
  HybridRag.py \
  question_generator.py \
  evaluation_metrics.py \
  evaluation_pipeline.py \
  report_generator.py \
  run_evaluation.py \
  README.md \
  requirements.txt \
  fixed_urls.json \
  random_urls.json \
  wiki_chunks.jsonl \
  dense.index \
  embeddings.npy \
  evaluation_questions.json \
  evaluation_results.json \
  evaluation_results.csv \
  evaluation_report.pdf \
  evaluation_report.html \
  Final_Report.pdf \
  screenshots/

# Verify ZIP contents
unzip -l Group_X_Hybrid_RAG.zip
```

## Final Verification

Before submitting, verify:

- [ ] All code files are properly commented
- [ ] README.md has installation and usage instructions
- [ ] requirements.txt includes all dependencies
- [ ] 200 fixed URLs are unique and documented
- [ ] 100 evaluation questions are generated
- [ ] All metrics are properly implemented and documented
- [ ] Ablation study shows comparative results
- [ ] Error analysis provides insights
- [ ] PDF report includes all required sections
- [ ] 3+ screenshots of the system
- [ ] ZIP file is properly named: `Group_X_Hybrid_RAG.zip`
- [ ] All files are included in the ZIP

## Submission Details

**Deadline:** February 8, 2026  
**Format:** One ZIP file per group  
**Naming:** `Group_<Number>_Hybrid_RAG.zip`

## Grading Rubric Alignment

| Component | Points | Status |
| --------- | ------ | ------ |
| Dense Retrieval | 2 | Complete |
| Sparse Retrieval | 2 | Complete |
| RRF Implementation | 2 | Complete |
| Response Generation | 2 | Complete |
| User Interface | 2 | Complete |
| **Part 1 Total** | **10** | **Complete** |
| | | |
| Question Generation | - | Complete |
| MRR Metric | 2 | Complete |
| Custom Metric 1 (ROUGE-L) | 2 | Complete |
| Custom Metric 2 (NDCG@5) | 2 | Complete |
| Innovative Evaluation | 4 | Complete |
| Automated Pipeline | - | Complete |
| Report Generation | - | Complete |
| **Part 2 Total** | **10** | **Complete** |
| | | |
| **Grand Total** | **20** | **Complete** |

## 📞 Support

If you encounter any issues:

1. Check README.md for troubleshooting
2. Review requirements.txt for dependencies
3. Ensure all prerequisites are installed
4. Check Python version compatibility (3.8+)
