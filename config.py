"""
config.py — Central Hyperparameter Registry
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY A CENTRAL CONFIG FILE?
    In a multi-file agentic system, the same value (e.g., max_iterations) may be
    referenced in the graph builder, the evaluator node, AND the benchmark script.
    If you hard-code "3" in each file, changing it later means hunting through every
    file. A single Config class acts as the single source of truth — change once,
    update everywhere.

DESIGN PATTERN:
    We use a frozen dataclass (frozen=True) so that no node can accidentally mutate
    the config at runtime. All settings are read-only after instantiation.
"""

from dataclasses import dataclass, field  # dataclass reduces boilerplate; field() allows complex defaults
from typing import Optional                 # Optional[str] = str | None, used for keys that may be absent


@dataclass(frozen=True)  # frozen=True makes all fields immutable after __init__, preventing accidental overwrites
class Config:
    """
    Central hyperparameter registry for the entire Research Agent system.

    Every tunable knob lives here. Importing this from any module gives
    consistent, validated settings. The class is deliberately frozen so that
    node functions cannot silently mutate settings mid-run.

    Usage:
        from config import Config
        cfg = Config()           # all defaults
        cfg = Config(max_iterations=5)  # override one value
    """

    # ─── LLM SETTINGS ────────────────────────────────────────────────────────
    # model_name: Which OpenAI chat model to call for planning, evaluation, and
    # synthesis. GPT-4o is the best price/performance tradeoff as of 2025.
    # Increasing to gpt-4-turbo gives marginally better reasoning but costs 3×
    # more. Decreasing to gpt-3.5-turbo cuts cost but degrades synthesis quality.
    llm_model_name: str = "gpt-4o"

    # temperature: Controls randomness in LLM outputs. 0.0 = fully deterministic
    # (always picks the highest-probability token), 1.0 = highly creative/random.
    # We use 0.1 because research agents need CONSISTENT, factual responses.
    # A planner at temperature=0.9 might invent wild sub-queries; at 0.1 it stays
    # focused. Synthesis uses a slightly higher temperature (overridden per-node).
    llm_temperature: float = 0.1

    # max_tokens: Upper bound on LLM response length. 4096 tokens ≈ 3,000 words —
    # enough for a detailed Markdown report. Increasing to 8192 costs more per call.
    # Decreasing below 1024 risks truncated reports where citations get cut off.
    llm_max_tokens: int = 4096

    # synthesis_temperature: Used only in the synthesizer node, which benefits from
    # slightly more creative prose. Higher than planning temperature but still low
    # enough to avoid hallucination. 0.3 is the "sweet spot" for report writing.
    synthesis_temperature: float = 0.3

    # ─── EMBEDDING SETTINGS ──────────────────────────────────────────────────
    # embedding_model_name: OpenAI's text-embedding-3-small produces 1536-dim
    # vectors. It's faster and cheaper than text-embedding-ada-002 and better at
    # semantic similarity tasks. Used when OPENAI_API_KEY is present.
    embedding_model_name: str = "text-embedding-3-small"

    # vector_dimension: Must match the output size of the chosen embedding model.
    # text-embedding-3-small → 1536. all-MiniLM-L6-v2 → 384.
    # ChromaDB uses this when creating the collection's index structure.
    # IMPORTANT: Changing this after data is stored requires re-indexing everything.
    vector_dimension: int = 1536

    # local_embedding_model: HuggingFace model used when no OpenAI key is set.
    # all-MiniLM-L6-v2 is only 22 MB, runs on CPU in ~50ms/batch, and achieves
    # decent semantic similarity on English text. A larger model (e.g., all-mpnet-
    # base-v2) improves recall precision but takes ~200ms/batch and is 420 MB.
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # local_vector_dimension: Output size of all-MiniLM-L6-v2. Must match
    # the model above. If you switch to all-mpnet-base-v2, change this to 768.
    local_vector_dimension: int = 384

    # ─── MEMORY / VECTOR DB SETTINGS ─────────────────────────────────────────
    # db_path: Filesystem location where ChromaDB writes its SQLite + HNSW files.
    # Relative paths work fine for local dev. In production, point to a mounted
    # persistent volume (e.g., /mnt/data/chroma_db) so memory survives container
    # restarts. The directory is created automatically if it doesn't exist.
    db_path: str = "./data/chroma_db"

    # collection_name: ChromaDB organizes documents into named collections (like
    # tables in SQL). Using a descriptive name lets you run multiple agent types
    # in the same DB without cross-contaminating their memories.
    collection_name: str = "research_memories"

    # top_k_recall: How many past memory documents to retrieve when the agent
    # starts a new query. 3 is the sweet spot: enough context to avoid re-doing
    # prior work, but not so many that the LLM's context window fills with old
    # research before it even sees the current query. Increase to 5-7 for very
    # broad domains where more context helps. Decrease to 1-2 for narrow domains.
    top_k_recall: int = 3

    # similarity_threshold: Minimum cosine similarity score (0.0–1.0) a memory
    # must score to be considered "relevant". Cosine similarity measures the angle
    # between two embedding vectors: 1.0 = identical meaning, 0.0 = unrelated.
    # 0.7 filters out loosely related memories. Lower to 0.5 to be more inclusive
    # (more noise). Raise to 0.85 for stricter relevance (less recall).
    similarity_threshold: float = 0.7

    # ─── SEARCH SETTINGS ─────────────────────────────────────────────────────
    # max_search_results: Number of URLs Tavily/DuckDuckGo returns per query.
    # 5 gives a good coverage/latency tradeoff. More results → better coverage
    # but more scraping time and higher token costs. For production, consider 3.
    max_search_results: int = 5

    # search_depth: Tavily-specific parameter. "basic" returns quick snippets;
    # "advanced" triggers Tavily's deeper re-ranking and content extraction pipeline,
    # producing higher-quality results at the cost of ~2× latency. Use "basic"
    # for fast iteration and "advanced" for production quality.
    search_depth: str = "advanced"

    # ─── WEB SCRAPER SETTINGS ────────────────────────────────────────────────
    # timeout_seconds: Max seconds to wait for an HTTP response before giving up.
    # 10s is generous enough for most sites but prevents the agent from hanging
    # on a slow server. Lower to 5 for speed, raise to 20 for academic/gov sites.
    timeout_seconds: int = 10

    # max_char_limit: After cleaning, truncate page text to this many characters
    # before passing to the LLM. 8000 chars ≈ ~2000 tokens, leaving plenty of
    # room in the 128k-token GPT-4o context window for the full state payload.
    # Reduce to 4000 if you're hitting rate-limit errors on token counts.
    max_char_limit: int = 8000

    # ─── GRAPH EXECUTION SETTINGS ────────────────────────────────────────────
    # max_iterations: The circuit-breaker for the search loop. If the evaluator
    # keeps scoring content as insufficient, the agent loops back to search.
    # Without this cap, a pathological query could loop forever, burning API
    # credits. 3 iterations = 3 full search+scrape passes before forced synthesis.
    # Increase to 5 for exhaustive research; decrease to 1 for fast demos.
    max_iterations: int = 3

    # sufficiency_threshold: The evaluation score (0.0–1.0) at which the agent
    # decides "I have enough information to write a good report." If the evaluator
    # LLM scores the collected content above this, we skip to synthesis early.
    # Must match the conditional edge logic in graph_builder.py.
    sufficiency_threshold: float = 0.7

    # ─── STREAMLIT DASHBOARD SETTINGS ────────────────────────────────────────
    # refresh_interval_ms: How often Streamlit auto-refreshes the live execution
    # tree during agent runs. 500ms = 2 updates/second, smooth but not CPU-heavy.
    # Lower for snappier UX; raising above 2000ms makes the progress feel laggy.
    refresh_interval_ms: int = 500

    # ─── DATA PATHS ──────────────────────────────────────────────────────────
    # sample_docs_path: Where download_sample_data.py writes its HTML/text files.
    sample_docs_path: str = "./data/sample_documents"

    # benchmark_output_path: Where benchmark.py writes its JSON results file.
    benchmark_output_path: str = "./data/benchmark_results.json"


# Module-level singleton — import this instead of constructing a new Config()
# in every file. Prevents subtle bugs where two files use different defaults.
cfg = Config()
