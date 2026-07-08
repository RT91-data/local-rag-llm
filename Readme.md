# 🤖 Local RAG LLM — AI Document Assistant

A **Retrieval Augmented Generation (RAG)** application that lets you chat with your PDF documents using hybrid search, cross-encoder reranking, 3-layer security, and Claude AI. Built with Python, LangChain, FAISS, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-red)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![Claude](https://img.shields.io/badge/Claude-Sonnet%204-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- 📄 **Multi-PDF support** — upload and query multiple documents simultaneously
- 🔍 **Hybrid retrieval** — FAISS semantic search + BM25 keyword search with similarity threshold filtering
- 🎯 **Cross-encoder reranking** — ms-marco-MiniLM-L-6-v2 rescores candidates for higher precision
- 🔄 **Query rewriting** — expands conversational follow-ups into standalone search queries
- 🛡️ **3-layer security** — index-time injection scanning, input guardrail, output validation
- 📊 **Table-aware extraction** — pdfplumber handles structured tables cell-by-cell
- 💬 **Conversation memory** — sliding window with summarisation so older context is never fully lost
- 📎 **Source citations** — every answer cites exact filename and page number
- ⚡ **Streaming responses** — token-by-token streaming via Claude API
- 💾 **Persistent index** — FAISS index saved to disk, no re-indexing on restart
- 📐 **RAGAS evaluation** — 20-question golden dataset with automated faithfulness, relevancy, precision and recall scoring

---

## 🏗️ Architecture

```
User Question
      ↓
[LAYER 2] Input Guardrail (Haiku classifier — blocks injection attempts)
      ↓
Query Rewriting (Haiku — expands follow-ups into standalone queries)
      ↓
 ┌─────────────────────────────────────────────┐
 │            Hybrid Retrieval (k=16)           │
 │     FAISS (60%) + BM25 (40%)                │
 │     Similarity threshold filter (≥0.3)       │
 │     Junk chunk filter (TOC, headers)         │
 │     CrossEncoder rerank → top 5 chunks       │
 └─────────────────────────────────────────────┘
      ↓
Context Building (chunks + source citations)
      ↓
Claude API (Sonnet/Haiku — streamed)
      ↓
[LAYER 3] Output Validation (pattern scan — blocks leakage/dangerous content)
      ↓
Answer with Page-level Citations
```

---

## 🛡️ Security Architecture

Three independent layers protect against prompt injection and data leakage:

| Layer | When | Method | Fails |
|---|---|---|---|
| **Layer 1 — Index-time scan** | PDF upload | Pattern match + Haiku semantic check | Closed (chunk removed) |
| **Layer 2 — Input guardrail** | Every query | Haiku classifier | Open (logs + allows through) |
| **Layer 3 — Output validation** | Before display | Pattern scan (no API call) | Closed (response blocked) |

**Key design principle:** Layer 1 distinguishes documents *about* injection (safe — educational) from text *actively commanding* an AI to change behaviour (unsafe — blocked). Layer 2 fails open to avoid blocking legitimate users when the scanner itself errors.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **UI** | Streamlit | Web interface |
| **PDF Parsing** | pdfplumber | Text + table extraction |
| **Embeddings** | nomic-embed-text (Ollama) | Local, free semantic embeddings |
| **Vector Store** | FAISS | Fast similarity search |
| **Keyword Search** | BM25 | Exact keyword matching |
| **Hybrid Retrieval** | LangChain EnsembleRetriever | Combines vector + keyword |
| **Reranker** | CrossEncoder ms-marco-MiniLM-L-6-v2 | Precise relevance rescoring |
| **LLM** | Claude API (Anthropic) | Answer generation, security classification, summarisation |
| **Evaluation** | RAGAS 0.4.x | Automated RAG quality metrics |
| **Framework** | LangChain | RAG pipeline orchestration |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Anthropic API key ([get one here](https://console.anthropic.com))

### 1. Clone the repository

```bash
git clone https://github.com/RT91-data/local-rag-llm.git
cd local-rag-llm
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull the embedding model

```bash
ollama pull nomic-embed-text
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your-api-key-here
```

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📖 How to Use

1. **Upload PDFs** — use the sidebar file uploader (supports multiple files)
2. **Wait for indexing** — documents are chunked, security-scanned, embedded, and saved to disk
3. **Ask questions** — type in the chat box
4. **View sources** — expand the 📎 Sources section to see chunk relevance scores and page numbers
5. **Follow-up** — ask follow-up questions; queries are automatically rewritten to be standalone
6. **Switch models** — use the sidebar to switch between Claude Sonnet (more accurate) and Haiku (faster, cheaper)
7. **Reset** — click 🚨 Wipe All Data to start fresh with new documents

---

## 📁 Project Structure

```
local-rag-llm/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── fix_vertexai_stub.py        # One-time fix for RAGAS 0.4.x + langchain-community incompatibility
├── eval/
│   ├── evaluate_rag.py         # RAGAS evaluation pipeline (headless, no Streamlit)
│   └── golden_dataset.json     # 20 curated Q&A pairs for evaluation
├── faiss_index_storage/        # Persisted FAISS index (git-ignored)
├── temp_uploads/               # Uploaded PDFs (git-ignored)
├── .env                        # API keys (not committed)
├── .gitignore
└── README.md
```

---

## 📊 RAGAS Evaluation

A full evaluation pipeline is included in `eval/`. It runs 20 curated questions against your indexed documents and scores four metrics:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? (hallucination check) |
| **Answer Relevancy** | Is the answer on-topic for the question? |
| **Context Precision** | Are the retrieved chunks actually relevant? |
| **Context Recall** | Does retrieved context contain all information needed to answer? |

**Run evaluation:**
```bash
python eval/evaluate_rag.py --questions 3   # smoke test
python eval/evaluate_rag.py                 # full 20 questions
```

Results are saved to `eval_results.csv`. The golden dataset covers the full document: security architecture, evaluation frameworks, identity management, observability, and threat vectors.

**Current baseline results (single-document index, Vibe Coding whitepaper):**

| Metric | Score |
|---|---|
| Answer Relevancy | 0.955 |
| Context Precision | 0.886 |
| Context Recall | 0.912 |

> Note: Rebuild the FAISS index with only your target document before running eval to avoid cross-document retrieval noise.

---

## 🔧 Key Technical Decisions

### Why Hybrid Retrieval?
Pure vector search misses exact keyword matches (IDs, names, codes). Pure BM25 misses semantic meaning. Combining both at 60/40 weighting captures the best of both worlds.

### Why CrossEncoder reranking?
FAISS similarity scores measure vector proximity, not actual relevance to the question. A CrossEncoder reads the question and each chunk together and assigns a true relevance score. Running it on the top 16 FAISS+BM25 candidates before cutting to 5 removes structurally similar but content-empty chunks (TOC pages, section headers) that score high on vector similarity but answer nothing.

### Why pdfplumber over PyPDFLoader?
PyPDFLoader mangles table data — rows merge, columns lose structure. pdfplumber uses geometric analysis to extract tables cell-by-cell, converting them to pipe-separated text that LLMs can read accurately.

### Why conversation summarisation over pure sliding window?
A pure sliding window drops old messages entirely. When conversations exceed 10 messages, the oldest half is summarised using Haiku and stored separately. This summary is prepended to every subsequent system prompt, so the model always has full context without unlimited token cost.

### Why Claude API over local LLMs?
Local models (llama3.1, phi3) on CPU took 5-7 minutes per response. Claude API responds in 3-5 seconds with significantly better accuracy on structured data. Embeddings remain local (free) while only the reasoning step uses the API.

---

## 💡 Improvements Roadmap

- [x] Cross-encoder reranking for higher retrieval precision
- [x] RAGAS automated quality evaluation
- [x] Query rewriting for better conversational recall
- [x] 3-layer security (injection scanning, input guardrail, output validation)
- [x] Conversation summarisation (sliding window with memory)
- [ ] Parent-document retriever pattern for multi-page enumeration questions
- [ ] Rebuild index with chunk_size=2000 and evaluate impact on recall
- [ ] Support for Word documents (.docx) and web URLs
- [ ] GitHub Actions CI/CD with automated eval regression check
- [ ] Deploy to Streamlit Cloud

---

## 📊 Performance

| Metric | Value |
|---|---|
| Avg response time | 3-5 seconds (Claude Sonnet) |
| Chunk size | 2000 chars, 400 overlap |
| FAISS retrieval k | 16 candidates |
| Reranker top-k | 5 chunks passed to Claude |
| Similarity threshold | 0.3 (L2-to-similarity converted) |
| Embedding model | nomic-embed-text (274MB, local) |
| Supported file types | PDF |

---

## 📄 License

MIT License — feel free to use and modify.

---

## 👩‍💻 Author

**Rupam Tripathi** — D365 FnO/AX consultant transitioning into AI Engineering
- GitHub: [@RT91-data](https://github.com/RT91-data)

---

*Built as part of a 6-month AI upskilling journey — 2026*