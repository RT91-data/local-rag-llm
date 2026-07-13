"""
observability.py — Langfuse tracing for local-rag-llm
Compatible with Langfuse v2 AND v3.

Drop this file in C:\\AIlearning\\local-rag\\ alongside app.py.

.env must contain:
  LANGFUSE_PUBLIC_KEY=pk-lf-...
  LANGFUSE_SECRET_KEY=sk-lf-...
  LANGFUSE_HOST=https://cloud.langfuse.com
"""

import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# Claude Sonnet 4.6 pricing (per million tokens)
INPUT_COST_PER_M  = 3.00
OUTPUT_COST_PER_M = 15.00

_client      = None   # Langfuse client singleton
_api_version = None   # 'v2', 'v3_decorator', or None


def _ts(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def get_langfuse():
    """
    Initialise Langfuse client. Detects v2 vs v3 automatically.
    Returns the client (or None if setup failed).
    """
    global _client, _api_version
    if _client is not None:
        return _client

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host       = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        print("[observability] LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — tracing disabled")
        return None

    try:
        from langfuse import Langfuse
        import langfuse as lf_module
        major = int(lf_module.__version__.split(".")[0])
        print(f"[observability] Langfuse {lf_module.__version__} detected")

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

        if major >= 3 or not hasattr(client, "trace"):
            # v3 — use decorator-based context manager
            _api_version = "v3_decorator"
            _client = client
            print(f"[observability] Using Langfuse v3 decorator API — host: {host}")
        else:
            # v2 — use low-level trace/span/generation API
            _api_version = "v2"
            _client = client
            print(f"[observability] Using Langfuse v2 low-level API — host: {host}")

        return _client

    except ImportError:
        print("[observability] langfuse package not installed — run: pip install langfuse")
        return None
    except Exception as e:
        print(f"[observability] init failed: {e}")
        return None


# ─── v2 IMPLEMENTATION ───────────────────────────────────────────

def _v2_start_trace(client, query: str, session_id: str):
    return client.trace(
        name="rag-query",
        input={"query": query},
        session_id=session_id,
        tags=["local-rag-llm"],
    )


def _v2_log_span(trace, name, input_data, output_data, t0, t1, metadata=None):
    latency_ms = round((t1 - t0) * 1000, 1)
    trace.span(
        name=name,
        input=input_data,
        output=output_data,
        metadata={**(metadata or {}), "latency_ms": latency_ms},
        start_time=_ts(t0),
        end_time=_ts(t1),
    )


def _v2_log_generation(trace, query, context, answer,
                        input_tokens, output_tokens, t0, t1, model):
    latency_ms = round((t1 - t0) * 1000, 1)
    cost = round(
        (input_tokens  / 1_000_000 * INPUT_COST_PER_M) +
        (output_tokens / 1_000_000 * OUTPUT_COST_PER_M), 6
    )
    trace.generation(
        name="claude-generation",
        model=model,
        input=[{"role": "user", "content": f"Q: {query}\nCTX: {context[:300]}..."}],
        output=answer,
        usage={"input": input_tokens, "output": output_tokens, "unit": "TOKENS"},
        metadata={"latency_ms": latency_ms, "cost_usd": cost},
        start_time=_ts(t0),
        end_time=_ts(t1),
    )


def _v2_end_trace(client, trace, answer, sources, query, rewritten_query,
                   total_start, chunks_retrieved, chunks_after_rerank,
                   junk_filtered, input_tokens, output_tokens):
    total_ms = round((time.time() - total_start) * 1000, 1)
    cost = round(
        (input_tokens  / 1_000_000 * INPUT_COST_PER_M) +
        (output_tokens / 1_000_000 * OUTPUT_COST_PER_M), 6
    )
    trace.update(
        output={"answer": answer[:500]},
        metadata={
            "total_latency_ms":    total_ms,
            "query_rewritten":     query != rewritten_query,
            "chunks_retrieved":    chunks_retrieved,
            "junk_filtered":       junk_filtered,
            "chunks_after_rerank": chunks_after_rerank,
            "input_tokens":        input_tokens,
            "output_tokens":       output_tokens,
            "total_cost_usd":      cost,
            "sources": [
                {"file": s.get("source",""), "page": s.get("page",""),
                 "rerank_score": s.get("rerank_score","")}
                for s in sources
            ],
        },
    )
    client.flush()


# ─── v3 IMPLEMENTATION (decorator-based) ─────────────────────────
# In v3, we collect events in a list and log them as a single
# batch trace using the score / event API as a fallback.

class _V3TraceCollector:
    """Collects pipeline events for Langfuse v3 logging."""

    def __init__(self, client, query: str, session_id: str):
        self.client    = client
        self.query     = query
        self.session_id = session_id
        self.events    = []
        self.start     = time.time()

    def add_span(self, name, input_data, output_data, t0, t1, metadata=None):
        self.events.append({
            "type": "span", "name": name,
            "input": input_data, "output": output_data,
            "latency_ms": round((t1 - t0) * 1000, 1),
            **(metadata or {}),
        })

    def add_generation(self, query, context, answer,
                       input_tokens, output_tokens, t0, t1, model):
        self.events.append({
            "type": "generation", "name": "claude-generation",
            "model": model,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "latency_ms": round((t1 - t0) * 1000, 1),
            "cost_usd": round(
                (input_tokens / 1_000_000 * INPUT_COST_PER_M) +
                (output_tokens / 1_000_000 * OUTPUT_COST_PER_M), 6
            ),
        })

    def finalise(self, answer, sources, rewritten_query,
                 chunks_retrieved, chunks_after_rerank,
                 junk_filtered, input_tokens, output_tokens):
        total_ms = round((time.time() - self.start) * 1000, 1)
        # v3: use the event() method if available, else log as score
        try:
            from langfuse.decorators import observe, langfuse_context

            @observe()
            def _log():
                langfuse_context.update_current_trace(
                    name="rag-query",
                    input={"query": self.query},
                    output={"answer": answer[:300]},
                    session_id=self.session_id,
                    tags=["local-rag-llm"],
                    metadata={
                        "total_latency_ms":    total_ms,
                        "chunks_retrieved":    chunks_retrieved,
                        "chunks_after_rerank": chunks_after_rerank,
                        "junk_filtered":       junk_filtered,
                        "input_tokens":        input_tokens,
                        "output_tokens":       output_tokens,
                        "events":              self.events,
                    },
                )
            _log()
            print(f"[observability] v3 trace logged via decorator — {total_ms}ms")
        except Exception as e:
            print(f"[observability] v3 finalise failed: {e}")


# ─── PUBLIC API (called from app.py) ─────────────────────────────

def start_trace(query: str, session_id: str = None):
    """Start a pipeline trace. Returns (trace_obj, start_time)."""
    client = get_langfuse()
    start  = time.time()
    if client is None:
        return None, start

    try:
        if _api_version == "v2":
            return _v2_start_trace(client, query, session_id), start
        else:
            return _V3TraceCollector(client, query, session_id), start
    except Exception as e:
        print(f"[observability] start_trace failed: {e}")
        return None, start


def log_span(trace, name: str, input_data: dict, output_data: dict,
             start_time: float, end_time: float, metadata: dict = None):
    if trace is None:
        return
    try:
        if _api_version == "v2":
            _v2_log_span(trace, name, input_data, output_data,
                         start_time, end_time, metadata)
        elif isinstance(trace, _V3TraceCollector):
            trace.add_span(name, input_data, output_data,
                           start_time, end_time, metadata)
    except Exception as e:
        print(f"[observability] log_span({name}) failed: {e}")


def log_generation(trace, query: str, context: str, answer: str,
                   input_tokens: int, output_tokens: int,
                   start_time: float, end_time: float,
                   model: str = "claude-sonnet-4-6"):
    if trace is None:
        return
    try:
        if _api_version == "v2":
            _v2_log_generation(trace, query, context, answer,
                               input_tokens, output_tokens,
                               start_time, end_time, model)
        elif isinstance(trace, _V3TraceCollector):
            trace.add_generation(query, context, answer,
                                 input_tokens, output_tokens,
                                 start_time, end_time, model)
    except Exception as e:
        print(f"[observability] log_generation failed: {e}")


def end_trace(trace, answer: str, sources: list,
              query: str, rewritten_query: str, total_start: float,
              chunks_retrieved: int, chunks_after_rerank: int,
              junk_filtered: int, input_tokens: int, output_tokens: int):
    if trace is None:
        return
    try:
        if _api_version == "v2":
            _v2_end_trace(
                _client, trace, answer, sources, query, rewritten_query,
                total_start, chunks_retrieved, chunks_after_rerank,
                junk_filtered, input_tokens, output_tokens,
            )
        elif isinstance(trace, _V3TraceCollector):
            trace.finalise(
                answer, sources, rewritten_query,
                chunks_retrieved, chunks_after_rerank,
                junk_filtered, input_tokens, output_tokens,
            )
        total_ms = round((time.time() - total_start) * 1000, 1)
        print(f"[observability] trace complete — {total_ms}ms | "
              f"chunks:{chunks_after_rerank} | "
              f"tokens:{input_tokens}+{output_tokens}")
        # Force flush — send queued events to Langfuse immediately
        if _client is not None:
            try:
                _client.flush()
            except Exception:
                pass
    except Exception as e:
        print(f"[observability] end_trace failed: {e}")