# Project Files Catalog

## Complete File Structure

```text
HybridRAG/
│
├── Setup & Configuration Files
│   ├── setup.sh                    # Automated setup script (Unix/Mac)
│   ├── setup.bat                   # Automated setup script (Windows)
│   ├── activate.sh                 # Quick activation script (Unix/Mac)
│   ├── activate.bat                # Quick activation script (Windows)
│   ├── requirements.txt            # Python dependencies
│   ├── .python-version             # Python version specification (3.8.0)
│   ├── .gitignore                  # Git ignore patterns
│   └── verify_setup.py             # Setup verification script
│
├── Documentation Files
│   ├── README.md                   # Main project documentation
│   ├── SETUP.md                    # Detailed setup instructions
│   ├── ARCHITECTURE.md             # System architecture details
│   ├── SUBMISSION_CHECKLIST.md     # Pre-submission checklist
│   └── FILES.md                    # This file - project catalog
│
├── Core RAG System Files
│   ├── HybridRag.py               # Main RAG system + Streamlit interface
│   ├── question_generator.py      # Generates 100 evaluation Q&A pairs
│   ├── evaluation_metrics.py      # Implements MRR, ROUGE-L, NDCG@5
│   ├── evaluation_pipeline.py     # Automated evaluation + ablation study
│   ├── report_generator.py        # Creates PDF/HTML reports
│   └── run_evaluation.py          # One-click complete evaluation
│
├── Data Files (Pre-generated)
│   ├── fixed_urls.json            # 200 fixed Wikipedia URLs
│   ├── random_urls.json           # 300 random Wikipedia URLs
│   ├── wiki_chunks.jsonl          # 1,194 preprocessed text chunks
│   ├── dense.index                # FAISS vector index (cached)
│   └── embeddings.npy             # Dense embeddings cache
│
├── Assignment Files
│   └── question.html              # Assignment requirements document
│
└── Generated Folders (after running)
    ├── venv/                      # Python virtual environment
    ├── screenshots/               # System screenshots
    ├── logs/                      # Log files
    ├── outputs/                   # Output files
    ├── evaluation_questions.json  # Generated 100 Q&A pairs
    ├── evaluation_results.json    # Detailed evaluation results
    ├── evaluation_results.csv     # Tabular evaluation results
    ├── evaluation_report.pdf      # PDF visualization report
    └── evaluation_report.html     # HTML interactive dashboard
```

## File Descriptions

### Setup & Configuration (8 files)

#### `setup.sh` (Unix/Mac)

- **Purpose**: Automated environment setup script
- **Features**:
  - Checks Python version (3.8+ required)
  - Creates virtual environment
  - Installs all dependencies
  - Downloads NLTK data
  - Sets up project structure
- **Usage**: `./setup.sh`

#### `setup.bat` (Windows)

- **Purpose**: Windows version of setup script
- **Features**: Same as setup.sh for Windows
- **Usage**: `setup.bat`

#### `activate.sh` / `activate.bat`

- **Purpose**: Quick activation scripts for virtual environment
- **Usage**:
  - Unix/Mac: `source activate.sh`
  - Windows: `activate.bat`

#### `requirements.txt`

- **Purpose**: Python package dependencies
- **Contains**: 17 packages with version specifications
- **Key Packages**:
  - sentence-transformers, faiss-cpu, transformers
  - streamlit, rouge-score, matplotlib

#### `.python-version`

- **Purpose**: Specifies Python version for pyenv users
- **Version**: 3.8.0 minimum

#### `.gitignore`

- **Purpose**: Specifies files to ignore in version control
- **Patterns**: venv/, **pycache**/, logs/, etc.

#### `verify_setup.py`

- **Purpose**: Verifies environment is properly set up
- **Checks**:
  - Python version
  - All required packages
  - NLTK data
  - Required project files
- **Usage**: `python verify_setup.py`

### Documentation (5 files)

#### `README.md`

- **Purpose**: Main project documentation
- **Sections**:
  - Overview and features
  - Installation instructions
  - Quick start guide
  - Detailed usage
  - Evaluation metrics documentation
  - Configuration options
  - Troubleshooting
- **Length**: 450+ lines

#### `SETUP.md`

- **Purpose**: Detailed setup guide
- **Contents**:
  - Prerequisites
  - Automated and manual setup
  - Activation instructions
  - Troubleshooting common issues
  - Platform-specific notes
  - Docker alternative
  - System requirements by task

#### `ARCHITECTURE.md`

- **Purpose**: Detailed system architecture
- **Contents**:
  - ASCII art system diagrams
  - Component descriptions
  - Data flow explanations
  - Technology stack
  - Layer-by-layer breakdown

#### `SUBMISSION_CHECKLIST.md`

- **Purpose**: Pre-submission verification checklist
- **Contents**:
  - Assignment requirements checklist
  - File verification list
  - Step-by-step submission guide
  - Grading rubric alignment
  - ZIP creation instructions

#### `FILES.md`

- **Purpose**: Project file catalog (this file)
- **Contents**: Complete listing of all files with descriptions

### Core RAG System (6 files)

#### `HybridRag.py` (500 lines)

- **Purpose**: Main RAG system implementation
- **Components**:
  - WikipediaURLCollection - URL scraping
  - Preprocessing - Text extraction and chunking
  - DenseRetriever - FAISS + Sentence Transformers
  - SparseRetriever - BM25 implementation
  - RRF - Reciprocal Rank Fusion
  - ResponseGenerator - Flan-T5 generation
  - Streamlit interface
- **Usage**: `streamlit run HybridRag.py`

#### `question_generator.py` (459 lines)

- **Purpose**: Generates 100 diverse evaluation questions
- **Question Types**:
  - Factual (40%) - Who, What, When, Where
  - Comparative (20%) - Comparisons
  - Inferential (20%) - Why, How, Explain
  - Multi-hop (20%) - Complex questions
- **Output**: `evaluation_questions.json`
- **Usage**: `python question_generator.py`

#### `evaluation_metrics.py` (375 lines)

- **Purpose**: Implements all evaluation metrics
- **Metrics**:
  - **MRR** (Mandatory) - Mean Reciprocal Rank at URL level
  - **ROUGE-L** (Custom 1) - Answer quality metric
  - **NDCG@5** (Custom 2) - Retrieval ranking quality
  - Additional: Precision@K, Recall@K, F1, Exact Match
- **Features**: Full justifications and documentation for each metric

#### `evaluation_pipeline.py` (287 lines)

- **Purpose**: Automated evaluation system
- **Components**:
  - Standard evaluation on 100 questions
  - Ablation study (dense vs sparse vs hybrid)
  - Error analysis by failure type
  - Parameter sensitivity testing
- **Outputs**:
  - `evaluation_results.json`
  - `evaluation_results.csv`
- **Usage**: `python evaluation_pipeline.py`

#### `report_generator.py` (350 lines)

- **Purpose**: Creates detailed evaluation reports
- **Generates**:
  - **PDF Report** with 7 visualization pages:
    - Metric comparison charts
    - Question type analysis
    - Score distributions
    - Ablation study results
    - Error analysis plots
    - Response time distribution
    - Correlation heatmaps
  - **HTML Dashboard** with interactive summary
- **Usage**: `python report_generator.py`

#### `run_evaluation.py` (122 lines)

- **Purpose**: One-click complete evaluation script
- **Process**:
  1. Checks prerequisites
  2. Generates questions (if needed)
  3. Runs evaluation pipeline
  4. Generates reports
- **Usage**: `python run_evaluation.py`

### Data Files (5 files)

#### `fixed_urls.json`

- **Purpose**: 200 fixed Wikipedia URLs
- **Format**: JSON array of URLs
- **Size**: 202 lines
- **Note**: These URLs remain constant across runs

#### `random_urls.json`

- **Purpose**: 300 random Wikipedia URLs
- **Format**: JSON array of URLs
- **Size**: 302 lines
- **Note**: These should change between runs

#### `wiki_chunks.jsonl`

- **Purpose**: Preprocessed text chunks from all 500 URLs
- **Format**: JSON Lines (one JSON object per line)
- **Size**: 1,194 chunks
- **Fields**: chunk_id, url, title, chunk_index, text, source_type

#### `dense.index`

- **Purpose**: FAISS vector index for dense retrieval
- **Type**: Binary FAISS index file (IndexFlatIP)
- **Size**: ~5MB
- **Contents**: 1,194 384-dimensional vectors

#### `embeddings.npy`

- **Purpose**: NumPy array of dense embeddings
- **Type**: Binary NumPy array file
- **Shape**: (1194, 384)
- **Size**: ~5MB

### Generated Files (after running)

#### `evaluation_questions.json`

- **Generated by**: `question_generator.py`
- **Contents**: 100 Q&A pairs with metadata
- **Fields**: question_id, question, ground_truth, question_type, source_url, source_chunk_id, title

#### `evaluation_results.json`

- **Generated by**: `evaluation_pipeline.py`
- **Contents**:
  - Detailed results for all 100 questions
  - Aggregated metrics
  - Ablation study results
  - Error analysis
- **Size**: ~500KB

#### `evaluation_results.csv`

- **Generated by**: `evaluation_pipeline.py`
- **Contents**: Tabular format of results
- **Columns**: question_id, question, generated_answer, ground_truth, mrr, rouge_l, ndcg_at_5, etc.

#### `evaluation_report.pdf`

- **Generated by**: `report_generator.py`
- **Contents**: 7-page PDF with visualizations
- **Pages**:
  1. Metric comparison
  2. Question type analysis
  3. Score distributions
  4. Ablation study
  5. Error analysis
  6. Response times
  7. Correlation heatmap

#### `evaluation_report.html`

- **Generated by**: `report_generator.py`
- **Contents**: Interactive HTML dashboard
- **Features**:
  - Performance summary cards
  - Color-coded results
  - Detailed tables
  - Responsive design

## File Statistics

| Category | Files | Lines of Code | Description |
| -------- | ----- | ------------- | ----------- |
| Setup Scripts | 7 | ~800 | Environment configuration |
| Documentation | 5 | ~2,000 | Guides and references |
| Core System | 6 | ~2,000 | RAG implementation |
| Data Files | 5 | ~1,200 chunks | Preprocessed corpus |
| Generated | 5 | - | Evaluation outputs |
| **Total** | **28** | **~6,000** | Complete project |

## Usage Workflow

### First-Time Setup

1. Run `./setup.sh` (or `setup.bat` on Windows)
2. Wait for installation (~5-10 minutes)
3. Environment is ready!

### Regular Usage

1. Activate: `source activate.sh`
2. Run desired script
3. Deactivate: `deactivate`

### Verification

```bash
python verify_setup.py
```

### Web Interface

```bash
streamlit run HybridRag.py
```

### Complete Evaluation

```bash
python run_evaluation.py
```

## Submission Package

When creating submission ZIP, include:

**Mandatory Files:**

- All core system files (6 files)
- All documentation (5 files)
- Data files (5 files)
- Setup files (7 files)
- Generated evaluation files (5 files)
- Final report PDF (create manually)
- Screenshots folder (3+ images)

**Optional (recommended):**

- Setup scripts for easy environment setup
- Verification script
- Architecture documentation

**Total Size:** ~50-100 MB (including data and models)

## Maintenance

### Adding New Features

1. Update relevant Python file
2. Update documentation in README.md
3. Update SUBMISSION_CHECKLIST.md if needed
4. Test with `verify_setup.py`

### Updating Dependencies

1. Edit `requirements.txt`
2. Test with fresh environment
3. Update SETUP.md if needed

### Documentation Updates

1. Keep README.md as main reference
2. Update ARCHITECTURE.md for system changes
3. Update SETUP.md for setup changes

## 📞 File-Specific Issues

| Issue | Relevant File | Action |
| -------- | --------------- | -------- |
| Setup fails | SETUP.md | Check troubleshooting section |
| Import errors | verify_setup.py | Run verification |
| Missing packages | requirements.txt | Reinstall dependencies |
| Evaluation errors | evaluation_pipeline.py | Check logs |
| Report generation fails | report_generator.py | Check file permissions |

---

**Last Updated**: February 8, 2026  
**Version**: 1.0  
**Status**: Ready for submission
