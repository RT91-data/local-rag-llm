# 🤖 local-rag-llm — AI Document Assistant

A **production-grade Retrieval Augmented Generation (RAG)** application that lets you chat with your PDF documents using hybrid search, cross-encoder reranking, 3-layer security, semantic caching, Langfuse observability, and Claude AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-red)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![Claude](https://img.shields.io/badge/Claude-Sonnet%204.6-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- 📄 **Multi-PDF support** — upload and query multiple documents simultaneously
- 🔍 **Hybrid retrieval** — FAISS semantic search + BM25 keyword search with similarity threshold filtering
- 🎯 **Cross-encoder reranking** — ms-marco-MiniLM-L-6-v2 rescores candidates for higher precision
- ⚡ **Semantic query caching** — similar queries return in ~50ms vs ~10s, rewritten query used as canonical cache key
- 🔄 **Query rewriting** — normalises phrasing before cache lookup AND expands conversational follow-ups
- 🛡️ **3-layer security** — index-time injection scanning, input guardrail, output validation
- 📊 **Langfuse observability** — per-query tracing across 6 pipeline stages with latency, token usage, and cost
- 📐 **RAGAS evaluation** — 20 factual + 10 adversarial golden dataset with automated scoring
- 📊 **Table-aware extraction** — pdfplumber handles structured tables cell-by-cell
- 💬 **Conversation memory** — sliding window with summarisation so older context is never fully lost
- 📎 **Source citations** — every answer cites exact filename and page number
- ⚡ **Streaming responses** — token-by-token streaming via Claude API
- 💾 **Persistent index** — FAISS index saved to disk, no re-indexing on restart

---

## 🏗️ Architecture

```
User Query
    |
[Query Rewriting]  ← Haiku normalises phrasing + expands follow-ups
    |
[Semantic Cache]   ← cosine similarity check (threshold: 0.92)
    |                 HIT → return cached answer (~50ms)
    |                 MISS → continue pipeline
    |
[LAYER 2] Input Guardrail  ← Haiku classifier (fail open)
    |
 ┌──────────────────────────────────────────────────────┐
 │                Hybrid Retrieval (k=16)                │
 │     FAISS (60%) + BM25 (40%)                         │
 │     Similarity threshold filter (≥ 0.3)              │
 │     Junk chunk filter (TOC, headers)                 │
 │     CrossEncoder rerank → top 5 chunks               │
 └──────────────────────────────────────────────────────┘
    |
Context Building (chunks + source citations)
    |
Claude API (Sonnet 4.6 — streamed)
    |
[LAYER 3] Output Validation  ← pattern scan (fail closed)
    |
[Langfuse Trace]  ← logs latency, tokens, cost per stage
    |
Answer with Page-level Citations
```

---

## 🛡️ Security Architecture

Three independent layers. If one fails, the others still run.

| Layer | When | Method | Fail mode |
|---|---|---|---|
| **Layer 1 — Index-time scan** | PDF upload | Pattern match + Haiku semantic check | **Closed** — chunk removed permanently |
| **Layer 2 — Input guardrail** | Every query | Haiku classifier | **Open** — logs + allows (Claude safety is backstop) |
| **Layer 3 — Output validation** | Before display | Pattern scan, no API call | **Closed** — response blocked |

**Adversarial evaluation results: 10/10 (100% refusal rate)**

| Attack type | Result |
|---|---|
| Direct injection | ✅ Refused |
| Jailbreak (DAN) | ✅ Refused |
| Data exfiltration | ✅ Refused |
| Social engineering | ✅ Refused |
| False premise | ✅ Corrected |
| Code execution | ✅ Refused |
| Instruction override | ✅ Refused |
| False memory | ✅ Refused |
| Out-of-scope (×2) | ✅ Refused |

---

## 📊 Observability

Every query is traced in Langfuse with 6 spans:

| Span | What it tracks |
|---|---|
| `query-rewriting` | Latency, whether query was rewritten |
| `security-input-guardrail` | Latency, safe/blocked decision |
| `retrieval` | FAISS hits, BM25 hits, merged count, latency |
| `reranking` | Candidates in, chunks kept, rerank scores |
| `claude-generation` | Input/output tokens, cost USD, latency |
| `security-output-validation` | Safe/blocked decision |

**p50 latency breakdown (from Langfuse dashboard):**
- Claude generation: ~10s (80% of total)
- Security input guardrail: ~1.1s
- CrossEncoder reranking: ~0.78s
- FAISS + BM25 retrieval: ~0.62s
- Query rewriting: ~0.67s

---

## ⚡ Semantic Query Caching

Queries are normalised via rewriting before cache lookup, so similar phrasings hit the same cache entry:

- "summarise my document in 50 sentences max" → canonical form → cache HIT
- "summarise my document in 50 sentences" → same canonical form → cache HIT

Cache details:
- Storage: `query_cache.json` (project root, git-ignored)
- Similarity threshold: 0.92 cosine similarity
- Max entries: 100 (LRU eviction)
- Invalidation: automatic when new PDFs are indexed
- Speed: ~50ms on hit vs ~10s full pipeline

---

## 📐 RAGAS Evaluation

**Factual baseline (20 questions, single-document index):**

| Metric | Score |
|---|---|
| Context Recall | 0.939 |
| Context Precision | 0.889 |
| Answer Relevancy | 0.955 |

**Adversarial baseline (10 questions):**

| Metric | Score |
|---|---|
| Refusal Rate | 10/10 (100%) |

Run evaluations:
```bash
python eval/evaluate_rag.py --questions 3      # smoke test (factual)
python eval/evaluate_rag.py --adversarial-only # adversarial only
python eval/evaluate_rag.py                    # full 30 questions
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | pdfplumber (table-aware) |
| Embeddings | nomic-embed-text via Ollama (local, free) |
| Vector Store | FAISS (IndexFlatL2) |
| Keyword Search | BM25 |
| Hybrid Retrieval | LangChain EnsembleRetriever (60/40) |
| Reranker | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| LLM | Claude API (Anthropic) |
| Observability | Langfuse v2 |
| Evaluation | RAGAS |
| Backend/DB | — |
| Framework | LangChain |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Anthropic API key — [get one here](https://console.anthropic.com)
- Langfuse account — [cloud.langfuse.com](https://cloud.langfuse.com)

### 1. Clone

```bash
git clone https://github.com/RT91-data/local-rag-llm.git
cd local-rag-llm
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Pull embedding model

```bash
ollama pull nomic-embed-text
```

### 4. Set up `.env`

```
ANTHROPIC_API_KEY=sk-ant-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 5. Run

```bash
python -m streamlit run app.py
```

---

## 📁 Project Structure

```
local-rag-llm/
├── app.py                       # Main Streamlit application
├── observability.py             # Langfuse tracing module (v2 + v3 compatible)
├── semantic_cache.py            # Semantic query cache (cosine similarity)
├── fix_vertexai_stub.py         # One-time fix for Langfuse + langchain-community
├── requirements.txt
├── eval/
│   ├── evaluate_rag.py          # RAGAS + adversarial evaluation pipeline
│   ├── golden_dataset.json      # 20 factual + 10 adversarial test questions
│   └── eval_results_adversarial.csv
├── faiss_index_storage/         # Persisted FAISS index (git-ignored)
├── temp_uploads/                # Uploaded PDFs (git-ignored)
├── query_cache.json             # Semantic cache (git-ignored, runtime)
└── .env                         # API keys (not committed)
```

---

## 💡 Improvements Roadmap

- [x] Hybrid retrieval (FAISS + BM25)
- [x] Cross-encoder reranking
- [x] Query rewriting for conversational follow-ups
- [x] 3-layer security (injection scanning, input guardrail, output validation)
- [x] Conversation summarisation (sliding window with memory)
- [x] RAGAS evaluation pipeline (20 factual questions)
- [x] Langfuse observability (6-span per-query tracing)
- [x] Semantic query caching (rewritten query as canonical key)
- [x] Adversarial evaluation (10/10 refusal rate)
- [ ] Parent-document retrieval pattern (fixes multi-section answer recall)
- [ ] GitHub Actions CI/CD with automated eval regression check
- [ ] Support for Word documents (.docx) and web URLs

---

## 📊 Performance

| Metric | Value |
|---|---|
| Avg response time (cache miss) | ~10s (Claude Sonnet) |
| Avg response time (cache hit) | ~50ms |
| Chunk size | 2000 chars, 400 overlap |
| FAISS retrieval k | 16 candidates |
| Reranker top-k | 5 chunks passed to Claude |
| Similarity threshold | 0.3 |
| Cache similarity threshold | 0.92 |
| Embedding model | nomic-embed-text (274MB, local) |

---

## 📄 License

MIT License

---

## 👩‍💻 Author

**Rupam Tripathi** — D365 FnO/AX consultant + AI Engineer  
GitHub: [@RT91-data](https://github.com/RT91-data)

---

*Built as part of a 6-month AI engineering practice — 2026*