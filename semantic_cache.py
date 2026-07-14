"""
semantic_cache.py — Semantic query caching for local-rag-llm

How it works:
  1. Incoming query is embedded with nomic-embed-text
  2. Cosine similarity checked against all cached query embeddings
  3. If similarity >= THRESHOLD: return cached answer (~50ms)
  4. If miss: run full pipeline, cache the result for next time

Cache file: query_cache.json (project root, git-ignored)
Max entries: 100 (oldest evicted when full)
Threshold: 0.92 cosine similarity (high enough to avoid false positives)

Invalidation: cache is wiped when FAISS index changes
(new PDFs uploaded = new index = cache stale)
"""

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

CACHE_FILE      = "query_cache.json"
MAX_ENTRIES     = 100
SIM_THRESHOLD   = 0.92
INDEX_DIR       = "faiss_index_storage"


# ─── COSINE SIMILARITY ────────────────────────────────────────────

def _cosine_sim(a: list, b: list) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ─── FAISS INDEX TIMESTAMP ────────────────────────────────────────

def _index_mtime() -> float:
    """Return modification time of FAISS index directory. 0 if not found."""
    index_path = Path(INDEX_DIR) / "index.faiss"
    try:
        return index_path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


# ─── CACHE LOAD / SAVE ────────────────────────────────────────────

def _load_cache() -> dict:
    """
    Load cache from disk. Returns empty cache if file missing or stale.
    Cache structure:
      {
        "index_mtime": float,
        "entries": [
          {
            "query": str,
            "embedding": list[float],
            "answer": str,
            "sources": list[dict],
            "timestamp": str,
            "hits": int
          },
          ...
        ]
      }
    """
    if not os.path.exists(CACHE_FILE):
        return {"index_mtime": _index_mtime(), "entries": []}

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)

        # Invalidate if FAISS index has changed
        current_mtime = _index_mtime()
        if cache.get("index_mtime", 0) != current_mtime:
            print(f"[cache] FAISS index changed — cache invalidated "
                  f"({len(cache.get('entries', []))} entries cleared)")
            return {"index_mtime": current_mtime, "entries": []}

        return cache

    except (json.JSONDecodeError, KeyError) as e:
        print(f"[cache] Cache file corrupt, resetting: {e}")
        return {"index_mtime": _index_mtime(), "entries": []}


def _save_cache(cache: dict):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"[cache] Save failed: {e}")


# ─── PUBLIC API ───────────────────────────────────────────────────

def get_cached_answer(query: str, embeddings_model) -> dict | None:
    """
    Look up query in semantic cache.

    Args:
        query:            user query string
        embeddings_model: OllamaEmbeddings instance (already cached in app)

    Returns:
        dict with keys {answer, sources, cached_query, similarity, hits}
        or None if cache miss
    """
    t0    = time.time()
    cache = _load_cache()

    if not cache["entries"]:
        return None

    # Embed the incoming query
    try:
        q_embedding = embeddings_model.embed_query(query)
    except Exception as e:
        print(f"[cache] Embed failed: {e}")
        return None

    # Find best match
    best_sim   = 0.0
    best_entry = None

    for entry in cache["entries"]:
        sim = _cosine_sim(q_embedding, entry["embedding"])
        if sim > best_sim:
            best_sim   = sim
            best_entry = entry

    if best_sim >= SIM_THRESHOLD and best_entry:
        # Cache hit — update hit count
        best_entry["hits"] = best_entry.get("hits", 0) + 1
        _save_cache(cache)

        latency_ms = round((time.time() - t0) * 1000, 1)
        print(f"[cache] HIT  sim={best_sim:.4f}  latency={latency_ms}ms  "
              f"hits={best_entry['hits']}  "
              f"query='{best_entry['query'][:60]}'")

        return {
            "answer":        best_entry["answer"],
            "sources":       best_entry.get("sources", []),
            "cached_query":  best_entry["query"],
            "similarity":    round(best_sim, 4),
            "hits":          best_entry["hits"],
        }

    latency_ms = round((time.time() - t0) * 1000, 1)
    print(f"[cache] MISS best_sim={best_sim:.4f}  latency={latency_ms}ms")
    return None


def cache_answer(query: str, answer: str, sources: list,
                 embeddings_model) -> bool:
    """
    Store a query-answer pair in the semantic cache.

    Args:
        query:            user query string
        answer:           LLM-generated answer
        sources:          list of doc metadata dicts from retrieved chunks
        embeddings_model: OllamaEmbeddings instance

    Returns:
        True if cached successfully, False otherwise
    """
    # Don't cache refusal answers
    if "cannot find this in the provided documents" in answer.lower():
        print("[cache] Skipping cache for refusal answer")
        return False

    # Don't cache very short answers (likely errors)
    if len(answer.strip()) < 50:
        print("[cache] Skipping cache for short answer")
        return False

    try:
        q_embedding = embeddings_model.embed_query(query)
    except Exception as e:
        print(f"[cache] Embed for storage failed: {e}")
        return False

    cache = _load_cache()

    # Clean sources for JSON serialisation (remove non-serialisable items)
    clean_sources = []
    for s in sources:
        try:
            clean_sources.append({
                "source":       str(s.get("source", "")),
                "page":         str(s.get("page", "")),
                "rerank_score": float(s.get("rerank_score", 0)),
            })
        except Exception:
            pass

    entry = {
        "query":     query,
        "embedding": q_embedding,
        "answer":    answer,
        "sources":   clean_sources,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hits":      0,
    }

    cache["entries"].append(entry)

    # Evict oldest if over limit
    if len(cache["entries"]) > MAX_ENTRIES:
        removed = cache["entries"].pop(0)
        print(f"[cache] Evicted oldest entry: '{removed['query'][:50]}'")

    _save_cache(cache)
    print(f"[cache] STORED  total_entries={len(cache['entries'])}  "
          f"query='{query[:60]}'")
    return True


def get_cache_stats() -> dict:
    """Return cache statistics for display in Streamlit sidebar."""
    cache = _load_cache()
    entries = cache.get("entries", [])
    total_hits = sum(e.get("hits", 0) for e in entries)
    return {
        "total_entries": len(entries),
        "total_hits":    total_hits,
        "max_entries":   MAX_ENTRIES,
        "threshold":     SIM_THRESHOLD,
    }


def clear_cache():
    """Wipe the cache file. Called when user uploads new PDFs."""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            print("[cache] Cache cleared manually")
    except Exception as e:
        print(f"[cache] Clear failed: {e}")