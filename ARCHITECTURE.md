# Architecture Diagram for Hybrid RAG System

## System Architecture (Use this to create visual diagram)

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                         HYBRID RAG SYSTEM                                │
│                    Retrieval-Augmented Generation                         │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                          DATA COLLECTION LAYER                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────┐         ┌─────────────────┐                       │
│  │  Wikipedia API  │         │ Web Scraping    │                       │
│  │  Random URLs    │────────▶│ BeautifulSoup   │                       │
│  └─────────────────┘         └────────┬────────┘                       │
│                                        │                                 │
│  ┌─────────────────┐                  │                                 │
│  │ 200 Fixed URLs  │                  │                                 │
│  │ + 300 Random    │                  ▼                                 │
│  │ = 500 Total     │         ┌─────────────────┐                       │
│  └─────────────────┘         │ Text Extraction │                       │
│                               │ & Cleaning      │                       │
│                               └────────┬────────┘                       │
│                                        │                                 │
│                                        ▼                                 │
│                               ┌─────────────────┐                       │
│                               │ Text Chunking   │                       │
│                               │ 200-400 tokens  │                       │
│                               │ 50-token overlap│                       │
│                               └────────┬────────┘                       │
│                                        │                                 │
│                                        ▼                                 │
│                               ┌─────────────────┐                       │
│                               │ wiki_chunks.jsonl│                      │
│                               │ 1,194 chunks    │                       │
│                               └─────────────────┘                       │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                          INDEXING LAYER                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌───────────────────────────────┐    ┌───────────────────────────────┐│
│  │     DENSE INDEXING            │    │     SPARSE INDEXING           ││
│  ├───────────────────────────────┤    ├───────────────────────────────┤│
│  │                               │    │                               ││
│  │  ┌─────────────────────────┐ │    │  ┌─────────────────────────┐ ││
│  │  │ Sentence Transformer    │ │    │  │  BM25 Tokenization     │ ││
│  │  │ all-MiniLM-L6-v2        │ │    │  │  Keyword Extraction     │ ││
│  │  └──────────┬──────────────┘ │    │  └──────────┬──────────────┘ ││
│  │             │                 │    │             │                 ││
│  │             ▼                 │    │             ▼                 ││
│  │  ┌─────────────────────────┐ │    │  ┌─────────────────────────┐ ││
│  │  │ Dense Embeddings        │ │    │  │ Inverted Index          │ ││
│  │  │ 384-dimensional         │ │    │  │ Term Frequencies        │ ││
│  │  └──────────┬──────────────┘ │    │  └─────────────────────────┘ ││
│  │             │                 │    │                               ││
│  │             ▼                 │    │                               ││
│  │  ┌─────────────────────────┐ │    │                               ││
│  │  │ FAISS Index             │ │    │                               ││
│  │  │ IndexFlatIP (Cosine)    │ │    │                               ││
│  │  └─────────────────────────┘ │    │                               ││
│  └───────────────────────────────┘    └───────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                          RETRIEVAL LAYER                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│                     ┌──────────────────────┐                            │
│                     │   User Query         │                            │
│                     └──────────┬───────────┘                            │
│                                │                                         │
│              ┌─────────────────┴─────────────────┐                      │
│              │                                   │                       │
│              ▼                                   ▼                       │
│   ┌─────────────────────┐            ┌─────────────────────┐           │
│   │ Dense Retrieval     │            │ Sparse Retrieval    │           │
│   ├─────────────────────┤            ├─────────────────────┤           │
│   │ • Query Embedding   │            │ • Query Tokenization│           │
│   │ • FAISS Search      │            │ • BM25 Scoring      │           │
│   │ • Cosine Similarity │            │ • Term Matching     │           │
│   │ • Top-K Results     │            │ • Top-K Results     │           │
│   └──────────┬──────────┘            └──────────┬──────────┘           │
│              │                                   │                       │
│              │      ┌─────────────────────┐     │                       │
│              └─────▶│ Reciprocal Rank     │◀────┘                       │
│                     │ Fusion (RRF)        │                             │
│                     ├─────────────────────┤                             │
│                     │ • Merge Results     │                             │
│                     │ • RRF Scoring (k=60)│                             │
│                     │ • Re-rank by Score  │                             │
│                     │ • Select Top-N      │                             │
│                     └──────────┬──────────┘                             │
│                                │                                         │
│                                ▼                                         │
│                     ┌─────────────────────┐                             │
│                     │ Top-N Ranked Chunks │                             │
│                     │ (N=5 by default)    │                             │
│                     └─────────────────────┘                             │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                          GENERATION LAYER                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│              ┌────────────────────────────────────┐                      │
│              │  Retrieved Context + Query         │                      │
│              └────────────┬───────────────────────┘                      │
│                           │                                              │
│                           ▼                                              │
│              ┌────────────────────────────────────┐                      │
│              │  Stage 1: Chunk Summarization      │                      │
│              ├────────────────────────────────────┤                      │
│              │  • Summarize each chunk (120 tok)  │                      │
│              │  • Reduce context length           │                      │
│              │  • Maintain key information        │                      │
│              └────────────┬───────────────────────┘                      │
│                           │                                              │
│                           ▼                                              │
│              ┌────────────────────────────────────┐                      │
│              │  Google Flan-T5-base               │                      │
│              ├────────────────────────────────────┤                      │
│              │  • Sequence-to-sequence model      │                      │
│              │  • Instruction-tuned               │                      │
│              │  • GPU/CPU compatible              │                      │
│              └────────────┬───────────────────────┘                      │
│                           │                                              │
│                           ▼                                              │
│              ┌────────────────────────────────────┐                      │
│              │  Stage 2: Answer Generation        │                      │
│              ├────────────────────────────────────┤                      │
│              │  • Combine summaries + query       │                      │
│              │  • Generate final answer (150 tok) │                      │
│              │  • Concise, factual response       │                      │
│              └────────────┬───────────────────────┘                      │
│                           │                                              │
│                           ▼                                              │
│              ┌────────────────────────────────────┐                      │
│              │     Generated Answer                │                      │
│              └────────────────────────────────────┘                      │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                       USER INTERFACE LAYER                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    STREAMLIT WEB INTERFACE                        │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │                                                                    │  │
│  │  ┌─────────────────────────────────────────────────────────┐    │  │
│  │  │  Query Input: "What is quantum physics?"                │    │  │
│  │  └─────────────────────────────────────────────────────────┘    │  │
│  │                                                                    │  │
│  │  ┌─────────────────────────────────────────────────────────┐    │  │
│  │  │  Generated Answer:                                       │    │  │
│  │  │  "Quantum physics is the study of matter and energy..." │    │  │
│  │  └─────────────────────────────────────────────────────────┘    │  │
│  │                                                                    │  │
│  │  ┌─────────────────────────────────────────────────────────┐    │  │
│  │  │  Top Retrieved Chunks:                                   │    │  │
│  │  │  ┌──────────────────────────────────────────────────┐  │    │  │
│  │  │  │ 1. Chunk from "Quantum Mechanics" (Wiki)         │  │    │  │
│  │  │  │    Dense: 0.89 | Sparse: 12.3 | RRF: 0.95       │  │    │  │
│  │  │  └──────────────────────────────────────────────────┘  │    │  │
│  │  │  ┌──────────────────────────────────────────────────┐  │    │  │
│  │  │  │ 2. Chunk from "Physics" (Wiki)                   │  │    │  │
│  │  │  │    Dense: 0.85 | Sparse: 10.1 | RRF: 0.88       │  │    │  │
│  │  │  └──────────────────────────────────────────────────┘  │    │  │
│  │  │  ... (3 more chunks)                                 │    │  │
│  │  └─────────────────────────────────────────────────────────┘    │  │
│  │                                                                    │  │
│  │  Response Time: 2.35 seconds                                      │  │
│  │                                                                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                       EVALUATION LAYER                                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  QUESTION GENERATION                                              │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │  • Factual Questions (40%)     • Comparative Questions (20%)     │  │
│  │  • Inferential Questions (20%)  • Multi-hop Questions (20%)      │  │
│  │  • Total: 100 Q&A pairs with ground truth                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  EVALUATION METRICS                                               │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │  Mandatory:                                                        │  │
│  │  • MRR (Mean Reciprocal Rank) - URL-level retrieval accuracy     │  │
│  │                                                                    │  │
│  │  Custom Metrics:                                                   │  │
│  │  • ROUGE-L - Answer quality (LCS-based F1 score)                 │  │
│  │  • NDCG@5 - Retrieval ranking quality (DCG normalized)           │  │
│  │                                                                    │  │
│  │  Additional:                                                       │  │
│  │  • Precision@5, Recall@5, F1 Score, Exact Match                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  INNOVATIVE EVALUATION                                            │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │  1. Ablation Study:                                               │  │
│  │     • Dense-only retrieval                                        │  │
│  │     • Sparse-only retrieval                                       │  │
│  │     • Hybrid (RRF) retrieval                                      │  │
│  │                                                                    │  │
│  │  2. Error Analysis:                                               │  │
│  │     • Retrieval failures (low MRR)                                │  │
│  │     • Generation failures (low ROUGE-L)                           │  │
│  │     • Complete failures (both low)                                │  │
│  │     • Breakdown by question type                                  │  │
│  │                                                                    │  │
│  │  3. Parameter Sensitivity:                                        │  │
│  │     • Different K values (top-K retrieval)                        │  │
│  │     • Different N values (final chunks)                           │  │
│  │     • Different RRF k values                                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  REPORT GENERATION                                                │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │  • PDF Report: Comprehensive visualizations                       │  │
│  │    - Metric comparison charts                                     │  │
│  │    - Question type analysis                                       │  │
│  │    - Score distributions                                          │  │
│  │    - Ablation study results                                       │  │
│  │    - Error analysis plots                                         │  │
│  │    - Correlation heatmaps                                         │  │
│  │                                                                    │  │
│  │  • HTML Dashboard: Interactive summary                            │  │
│  │    - Performance metrics                                          │  │
│  │    - Detailed tables                                              │  │
│  │    - Color-coded results                                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                          TECHNOLOGY STACK                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Dense Retrieval:     sentence-transformers, faiss-cpu                   │
│  Sparse Retrieval:    rank-bm25                                          │
│  Generation:          transformers (Flan-T5), torch                      │
│  Web Scraping:        beautifulsoup4, requests                           │
│  Evaluation:          rouge-score, nltk, scikit-learn                    │
│  Visualization:       matplotlib, seaborn                                │
│  Interface:           streamlit                                          │
│  Data Processing:     pandas, numpy, tiktoken                            │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

## Key Components Description

### 1. Data Collection Layer

- **Input**: 500 Wikipedia URLs (200 fixed + 300 random)
- **Process**: Web scraping → Text extraction → Cleaning → Chunking
- **Output**: 1,194 preprocessed text chunks with metadata

### 2. Indexing Layer

- **Dense Path**: Sentence embeddings → FAISS vector index
- **Sparse Path**: BM25 tokenization → Inverted index
- **Storage**: Persistent indexes for fast retrieval

### 3. Retrieval Layer

- **Dense Retrieval**: Semantic similarity search using cosine distance
- **Sparse Retrieval**: Keyword matching using BM25 algorithm
- **RRF Fusion**: Combines both methods with reciprocal rank scoring

### 4. Generation Layer

- **Two-stage process**:
  1. Summarize retrieved chunks to fit context window
  2. Generate final answer from summaries
- **Model**: Google Flan-T5-base (instruction-tuned)

### 5. User Interface Layer

- **Streamlit web app**: Interactive Q&A interface
- **Displays**: Query, answer, sources, scores, timing

### 6. Evaluation Layer

- **Question Generation**: 100 diverse Q&A pairs
- **Metrics**: MRR, ROUGE-L, NDCG@5, and more
- **Innovation**: Ablation studies, error analysis, parameter tuning
- **Reporting**: Automated PDF and HTML report generation

## Data Flow

1. **Query** → Retrieval Layer
2. **Dense + Sparse Retrieval** → Top-K chunks each
3. **RRF Fusion** → Top-N best chunks
4. **Generation Layer** → Summarize → Generate answer
5. **User Interface** → Display results
6. **Evaluation** → Measure performance → Generate reports

## Use this diagram reference to create visual diagrams using

- Draw.io (diagrams.net)
- Lucidchart
- Microsoft Visio
- Or any other diagramming tool

### Suggested Visual Elements

- Use boxes for components
- Arrows for data flow
- Different colors for different layers
- Icons for databases, APIs, models
- Highlight the RRF fusion as a key innovation
