"""
RAGAS evaluation for local-rag-llm (FAISS + BM25 hybrid + Claude API).
Tested against RAGAS 0.4.x.

Prerequisites (run once):
    python -m pip install langchain-anthropic datasets

Usage:
    python eval/evaluate_rag.py --questions 3          # smoke test
    python eval/evaluate_rag.py                        # full 20 questions
    python eval/evaluate_rag.py --output results.csv   # save results
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# RAGAS IMPORT DETECTION
# Shows the ACTUAL error — not a generic "not found" message.
# RAGAS 0.4.x: EvaluationDataset / SingleTurnSample are top-level
# RAGAS 0.2–0.3.x: they lived in ragas.dataset_schema
# ─────────────────────────────────────────────────────────────────

# Step 1: Confirm RAGAS is importable at all
try:
    import ragas as _ragas_pkg
    print(f"[ragas] Found RAGAS {_ragas_pkg.__version__} at {_ragas_pkg.__file__}")
except ImportError:
    print(f"[ERROR] RAGAS not found in this Python environment.")
    print(f"        Active Python: {sys.executable}")
    print(f"        Fix: python -m pip install ragas")
    sys.exit(1)

# Step 2: Import core evaluate + metrics
try:
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    print("[ragas] Core metrics imported OK")
except ImportError as e:
    print(f"[ERROR] RAGAS metrics import failed: {e}")
    sys.exit(1)

# Step 3: Dataset schema — try 0.4.x top-level first, fall back to 0.2/0.3 submodule
RAGAS_DATASET_OK = False
try:
    # RAGAS 0.4.x style
    from ragas import EvaluationDataset, SingleTurnSample
    RAGAS_DATASET_OK = True
    RAGAS_SCHEMA_SOURCE = "ragas (0.4.x top-level)"
except ImportError:
    try:
        # RAGAS 0.2–0.3.x style
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        RAGAS_DATASET_OK = True
        RAGAS_SCHEMA_SOURCE = "ragas.dataset_schema (0.2–0.3.x)"
    except ImportError as e:
        print(f"[ragas] WARNING: EvaluationDataset not found ({e}). Falling back to datasets.Dataset.")
        RAGAS_DATASET_OK = False
        RAGAS_SCHEMA_SOURCE = "datasets.Dataset (legacy)"

# Step 4: Legacy datasets.Dataset fallback
USE_LEGACY_DATASET = not RAGAS_DATASET_OK
if USE_LEGACY_DATASET:
    try:
        from datasets import Dataset as HFDataset
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        print("[ragas] Using legacy datasets.Dataset API")
    except ImportError as e:
        print(f"[ERROR] Both new and legacy RAGAS dataset APIs unavailable: {e}")
        print(f"        Fix: python -m pip install datasets")
        sys.exit(1)

# Step 5: LangChain wrappers for RAGAS judge LLM
HAS_LC_WRAPPERS = False
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    HAS_LC_WRAPPERS = True
    print("[ragas] LangChain LLM wrappers imported OK")
except ImportError as e:
    print(f"[ragas] WARNING: LangChain wrappers unavailable ({e})")
    print(f"        This is needed to use Claude as the RAGAS judge LLM.")
    print(f"        Fix: python -m pip install langchain-anthropic")

print(f"[ragas] Dataset API: {RAGAS_SCHEMA_SOURCE}")
print(f"[ragas] LangChain wrappers: {'YES' if HAS_LC_WRAPPERS else 'NO'}")

# ─────────────────────────────────────────────────────────────────
# LANGCHAIN IMPORTS
# ─────────────────────────────────────────────────────────────────
try:
    from langchain_anthropic import ChatAnthropic
    HAS_CHAT_ANTHROPIC = True
except ImportError:
    HAS_CHAT_ANTHROPIC = False
    print("[warn] langchain-anthropic not installed — RAGAS judge will fail.")
    print("       Fix: python -m pip install langchain-anthropic")

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings

# ─────────────────────────────────────────────────────────────────
# RAG PIPELINE (headless — no Streamlit)
# Reproduces your retrieval stack from app.py:
#   FAISS threshold filter → BM25 → merge → CrossEncoder rerank → Claude
# Reads from the same INDEX_DIR and UPLOAD_DIR your Streamlit app writes to.
# ─────────────────────────────────────────────────────────────────
import pdfplumber
from anthropic import Anthropic
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

# Paths — adjust if your app lives in a different directory
RAG_ROOT     = Path(__file__).parent.parent  # local-rag-llm/
INDEX_DIR    = str(RAG_ROOT / "faiss_index_storage")
UPLOAD_DIR   = str(RAG_ROOT / "temp_uploads")
SIMILARITY_THRESHOLD = 0.3

anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

_embeddings   = None   # lazy-loaded
_vectorstore  = None   # lazy-loaded
_bm25_ret     = None   # lazy-loaded
_reranker     = None   # lazy-loaded


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return _embeddings


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def _load_pdf_with_tables(file_path: str) -> list:
    """Table-aware PDF extraction — mirrors your app.py implementation."""
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = ""
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    if table:
                        rows = [" | ".join(str(c).strip() if c else "" for c in row)
                                for row in table]
                        page_text += "\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n\n"
            text = page.extract_text()
            if text:
                page_text += text
            if page_text.strip():
                docs.append(Document(
                    page_content=page_text,
                    metadata={"source": file_path, "page": page_num + 1}
                ))
    return docs


def _load_chunks_from_disk() -> list:
    """Reload all PDF chunks from temp_uploads/ for BM25 rebuild."""
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    if os.path.exists(UPLOAD_DIR):
        for f in os.listdir(UPLOAD_DIR):
            if f.endswith(".pdf"):
                docs = _load_pdf_with_tables(os.path.join(UPLOAD_DIR, f))
                chunks.extend(splitter.split_documents(docs))
    return chunks


def _setup_rag():
    """Load FAISS index + rebuild BM25 — run once per eval session."""
    global _vectorstore, _bm25_ret

    if _vectorstore is not None:
        return

    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_DIR}. "
            "Upload a PDF in the Streamlit app first to build the index."
        )

    print("[rag] Loading FAISS index from disk...")
    _vectorstore = FAISS.load_local(
        INDEX_DIR,
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )

    print("[rag] Rebuilding BM25 from temp_uploads/...")
    chunks = _load_chunks_from_disk()
    if chunks:
        _bm25_ret = BM25Retriever.from_documents(chunks)
        _bm25_ret.k = 4
    else:
        print("[rag] WARNING: No PDFs found in temp_uploads/. BM25 disabled.")
        _bm25_ret = None


def _is_junk_chunk(content: str) -> bool:
    """
    Exclude TOC, cover pages, and header-only chunks.
    These match every query on keywords but contain no answer content.
    """
    lower = content.lower()

    if "table of contents" in lower:
        return True

    # TOC pattern: >70% of lines are short (section title + page number)
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if len(lines) >= 6:
        short = sum(1 for l in lines if len(l) < 60)
        if short / len(lines) > 0.70 and len(content) < 1500:
            return True

    return False


def _retrieve_with_threshold(query: str, k: int = 16) -> list:
    results = _vectorstore.similarity_search_with_score(query, k=k)
    filtered = []
    for doc, score in results:
        if _is_junk_chunk(doc.page_content):   # ← add this
            continue
        similarity = 1 / (1 + score)
        if similarity >= SIMILARITY_THRESHOLD:
            doc.metadata["similarity_score"] = round(similarity, 3)
            filtered.append(doc)
    return filtered


def _rerank_chunks(query: str, docs: list, top_k: int = 4) -> list:
    """CrossEncoder reranking — mirrors your app.py."""
    if not docs or len(docs) <= 1:
        return docs
    reranker = _get_reranker()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    for score, doc in scored:
        doc.metadata["rerank_score"] = round(float(score), 3)
    return [doc for _, doc in scored[:top_k]]


def query_rag(question: str) -> dict:
    """
    Full retrieval + generation without Streamlit.
    Returns {"answer": str, "contexts": list[str]}.
    """
    _setup_rag()

    # Stage 1: FAISS with threshold
    faiss_docs = _retrieve_with_threshold(question, k=16)

    # Stage 2: BM25
    bm25_docs = _bm25_ret.invoke(question) if _bm25_ret else []

    # Stage 3: Merge + deduplicate
    seen, merged = set(), []
    for doc in faiss_docs + bm25_docs:
        if _is_junk_chunk(doc.page_content):
            continue
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    # Stage 4: CrossEncoder rerank
    final_docs = _rerank_chunks(question, merged, top_k=5)

    # Stage 5: Build context string
    context_parts = []
    for i, doc in enumerate(final_docs):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page   = doc.metadata.get("page", "?")
        context_parts.append(f"[Source {i+1}: {source}, Page {page}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    # Stage 6: Generate with Claude
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="""You are a precise document assistant.
Answer using ONLY the context provided.
If the answer is not in the context, say: "I cannot find this in the provided documents."
Always cite the source filename and page number for every fact you state.""",
        messages=[{
            "role": "user",
            "content": f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION: {question}"
        }]
    )

    return {
        "answer": response.content[0].text,
        "contexts": [doc.page_content for doc in final_docs],
    }


# ─────────────────────────────────────────────────────────────────
# RAGAS JUDGE LLM + EMBEDDINGS
# ─────────────────────────────────────────────────────────────────

def _build_ragas_llm():
    """Claude sonnet as the RAGAS judge (faithfulness, relevancy scoring)."""
    if not HAS_LC_WRAPPERS or not HAS_CHAT_ANTHROPIC:
        raise RuntimeError(
            "Cannot configure RAGAS judge LLM.\n"
            "Fix: python -m pip install langchain-anthropic"
        )
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=0,
        max_tokens=1024,
    )
    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings():
    """nomic-embed-text via Ollama for answer relevancy cosine similarity."""
    if not HAS_LC_WRAPPERS:
        raise RuntimeError("LangchainEmbeddingsWrapper not available.")
    return LangchainEmbeddingsWrapper(_get_embeddings())


# ─────────────────────────────────────────────────────────────────
# RAGAS EVALUATION
# ─────────────────────────────────────────────────────────────────

def _run_rag_over_dataset(golden: list) -> list:
    """Run every golden question through the RAG pipeline."""
    results = []
    total = len(golden)
    for i, item in enumerate(golden, 1):
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i}/{total}] {q[:75]}...")
        try:
            out = query_rag(q)
            answer   = out["answer"]
            contexts = out["contexts"]
        except Exception as e:
            print(f"         WARNING: RAG failed — {e}")
            answer, contexts = "", []
        results.append({
            "question":     q,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": gt,
        })
        time.sleep(0.5)   # Claude rate limit buffer
    return results


def _evaluate_new_api(rows: list) -> pd.DataFrame:
    """RAGAS 0.2+ / 0.4.x EvaluationDataset API."""
    ragas_llm = _build_ragas_llm()
    ragas_emb = _build_ragas_embeddings()
    metrics   = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        for r in rows
    ]
    dataset = EvaluationDataset(samples=samples)
    result  = evaluate(dataset=dataset, metrics=metrics)
    return result.to_pandas()


def _evaluate_legacy_api(rows: list) -> pd.DataFrame:
    """Fallback: datasets.Dataset format (RAGAS 0.1.x compatible)."""
    ds = HFDataset.from_dict({
        "question":     [r["question"]     for r in rows],
        "answer":       [r["answer"]       for r in rows],
        "contexts":     [r["contexts"]     for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })
    result = evaluate(ds, metrics=[
        faithfulness, answer_relevancy, context_precision, context_recall
    ])
    return result.to_pandas()


# ─────────────────────────────────────────────────────────────────
# RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────

METRIC_NOTES = {
    "faithfulness":       "Hallucination check — answer grounded in retrieved context?",
    "answer_relevancy":   "Is the answer on-topic for the question?",
    "context_precision":  "Are retrieved chunks actually relevant to the query?",
    "context_recall":     "Does retrieved context contain everything needed to answer?",
}


def _print_summary(df: pd.DataFrame):
    metric_cols = [c for c in df.columns if c.lower() in METRIC_NOTES]
    if not metric_cols:
        # RAGAS 0.4.x may capitalise column names differently
        metric_cols = [c for c in df.columns if c not in
                       ("question", "answer", "contexts", "ground_truth", "user_input",
                        "response", "retrieved_contexts", "reference")]

    print("\n" + "=" * 65)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 65)
    for m in metric_cols:
        avg  = df[m].mean()
        note = METRIC_NOTES.get(m.lower(), "")
        bar  = "█" * int(avg * 20)
        print(f"  {m:<28} {avg:.4f}  {bar}")
        if note:
            print(f"  {'':28} ↳ {note}")

    print("\n" + "-" * 65)
    print("Low-scoring questions (any metric < 0.6):")
    mask = (df[metric_cols] < 0.6).any(axis=1)
    low  = df[mask][["question"] + metric_cols] if "question" in df.columns else df[mask][metric_cols]
    if low.empty:
        print("  ✅ All questions scored ≥ 0.6 on all metrics.")
    else:
        for _, row in low.iterrows():
            q = row.get("question", row.get("user_input", ""))
            print(f"\n  Q: {str(q)[:80]}")
            for m in metric_cols:
                flag = " ⚠" if row[m] < 0.6 else ""
                print(f"     {m:<28} {row[m]:.4f}{flag}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAGAS eval for local-rag-llm")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "golden_dataset.json"),
        help="Path to golden_dataset.json",
    )
    parser.add_argument(
        "--output",
        default="eval_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Run only first N questions (smoke test)",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set in .env or environment.")
        sys.exit(1)

    print(f"\nLoading golden dataset: {args.dataset}")
    with open(args.dataset) as f:
        golden = json.load(f)

    if args.questions:
        golden = golden[: args.questions]
        print(f"Smoke-test mode: {args.questions}/{len(json.load(open(args.dataset)))} questions")
    else:
        print(f"Full run: {len(golden)} questions")

    print("\nStep 1/3 — Running RAG pipeline over golden questions...")
    rows = _run_rag_over_dataset(golden)

    print("\nStep 2/3 — Running RAGAS evaluation...")
    if not USE_LEGACY_DATASET and RAGAS_DATASET_OK:
        df = _evaluate_new_api(rows)
    else:
        df = _evaluate_legacy_api(rows)

    print("\nStep 3/3 — Results:")
    _print_summary(df)

    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n✅ Full results saved to: {args.output}")


if __name__ == "__main__":
    main()