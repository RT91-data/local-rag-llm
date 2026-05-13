# 🤖 Local RAG LLM — AI Document Assistant

A production-grade **Retrieval Augmented Generation (RAG)** application that lets you chat with your PDF documents using hybrid search and Claude AI. Built with Python, LangChain, FAISS, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-red)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![Claude](https://img.shields.io/badge/Claude-Sonnet%204-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- 📄 **Multi-PDF support** — upload and query multiple documents simultaneously
- 🔍 **Hybrid Retrieval** — combines semantic vector search (FAISS) + keyword search (BM25) for maximum accuracy
- 📊 **Table-aware extraction** — pdfplumber handles structured tables in research papers and reports
- 💬 **Conversation memory** — remembers context across follow-up questions
- 📎 **Source citations** — every answer cites the exact filename and page number
- ⚡ **Streaming responses** — token-by-token streaming via Claude API
- 💾 **Persistent index** — FAISS index saved to disk, no re-indexing on restart
- 🔄 **Model selection** — switch between Claude Sonnet and Haiku in the sidebar

---

## 🏗️ Architecture

```
User Question
      ↓
 Embedding (nomic-embed-text via Ollama — local & free)
      ↓
 ┌─────────────────────────────────┐
 │     Hybrid Retrieval            │
 │  FAISS (60%) + BM25 (40%)      │
 │  Top-4 chunks retrieved         │
 └─────────────────────────────────┘
      ↓
 Context Building (chunks + source citations)
      ↓
 Claude API (Sonnet/Haiku — streamed)
      ↓
 Answer with Page-level Citations
```

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
| **LLM** | Claude API (Anthropic) | Answer generation + reasoning |
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
pip install streamlit langchain langchain-ollama langchain-community langchain-classic
pip install langchain-text-splitters faiss-cpu rank_bm25 pdfplumber python-dotenv anthropic
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
2. **Wait for indexing** — documents are chunked, embedded, and saved to disk
3. **Ask questions** — type in the chat box
4. **View sources** — expand the 📎 Sources section to see which page answered your question
5. **Follow-up** — ask follow-up questions, the app remembers conversation context
6. **Switch models** — use the sidebar to switch between Claude Sonnet (more accurate) and Haiku (faster, cheaper)
7. **Reset** — click 🚨 Wipe All Data to start fresh with new documents

---

## 📁 Project Structure

```
local-rag-llm/
├── app.py                  # Main Streamlit application
├── advanced_rag.py         # Advanced RAG pipeline (CLI version)
├── main.py                 # Entry point
├── rag_app.py              # Alternative RAG implementation
├── test_pdf.py             # PDF extraction test utility
├── create_pdf.py           # Test PDF generator
├── start_ai.bat            # Windows startup script
├── data/                   # Sample documents
├── .env                    # API keys (not committed)
├── .gitignore
├── README.md
└── INTERVIEW_PREP.md       # RAG interview Q&A guide
```

---

## 🔧 Key Technical Decisions

### Why Hybrid Retrieval?
Pure vector search misses exact keyword matches (IDs, names, codes). Pure BM25 misses semantic meaning. Combining both at 60/40 weighting captures the best of both worlds.

### Why pdfplumber over PyPDFLoader?
PyPDFLoader mangles table data — rows merge, columns lose structure. pdfplumber uses geometric analysis to extract tables cell-by-cell, converting them to pipe-separated text that LLMs can read accurately.

### Why Claude API over local LLMs?
Local models (llama3.1, phi3) on CPU took 5-7 minutes per response. Claude API responds in 3-5 seconds with significantly better accuracy on structured data. Embeddings remain local (free) while only the reasoning step uses the API.

### Why persistent FAISS index?
Re-indexing 50+ page documents on every app restart would take minutes. Saving the FAISS index to disk means instant load on restart. BM25 is rebuilt from saved PDFs (lightweight operation).

---

## 💡 Improvements Roadmap

- [ ] Upgrade to BGE-large embeddings when Python 3.14 compatibility improves
- [ ] Add reranking with cross-encoder for better retrieval precision
- [ ] Implement RAGAs for automated quality evaluation
- [ ] Add query expansion for better recall
- [ ] Support for Word documents (.docx) and web URLs
- [ ] Multi-user support with document isolation
- [ ] Deploy to Streamlit Cloud

---

## 📊 Performance

| Metric | Value |
|---|---|
| Avg response time | 3-5 seconds (Claude Sonnet) |
| Chunk size | 1000 chars, 200 overlap |
| Retrieval k | 4 chunks per query |
| Embedding model | nomic-embed-text (274MB) |
| Supported file types | PDF |
| Max document size | Limited by available RAM |

---

## 🤝 Contributing

Pull requests welcome! For major changes, please open an issue first.

---

## 📄 License

MIT License — feel free to use and modify.

---

## 👨‍💻 Author

**Rupam** — AI Engineer in progress 🚀
- GitHub: [@RT91-data](https://github.com/RT91-data)

---

*Built as part of a 9-month AI upskilling journey — May 2026*