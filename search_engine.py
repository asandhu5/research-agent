"""
search_engine.py — Tavily + DuckDuckGo Search Wrapper
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY TWO SEARCH BACKENDS?

    Agentic pipelines fail catastrophically if their only external dependency
    (a paid API) is unavailable. We implement a two-tier strategy:

    Tier 1 — Tavily Search API:
        Purpose-built for LLM agents. Returns pre-cleaned, relevance-ranked
        snippets with URLs, reducing the amount of raw scraping needed.
        Requires TAVILY_API_KEY. Produces better result quality.

    Tier 2 — DuckDuckGo Search (duckduckgo-search library):
        Free, no API key, scrapes DDG's HTML search results.
        Slightly lower quality but always available during development.
        Activates automatically if Tavily fails or no key is present.

STANDARDIZED OUTPUT CONTRACT:
    Both backends normalize results to the same format:
        [{"title": str, "url": str, "snippet": str}, ...]
    This means the rest of the pipeline never needs to know which backend ran.
"""

import os                                          # reading API keys from environment
from typing import List, Dict, Any, Optional       # type hints
from dotenv import load_dotenv                     # loads .env into os.environ

from state import SourceDocument                   # validated result schema
from config import cfg                             # hyperparameters

load_dotenv()  # ensure environment variables are loaded before checking for keys


class SearchEngine:
    """
    Unified search interface supporting Tavily (primary) and DuckDuckGo (fallback).

    Design Philosophy:
        The caller (search_and_scrape_node) always calls `search(query)` and gets
        back a list of SourceDocument objects. It never needs to know or care which
        backend ran. This is the "Strategy Pattern" — swappable implementations
        behind a single interface.

    Attributes:
        _tavily_key: The Tavily API key (may be empty if not set).
        _tavily_client: Initialized Tavily client (None if no key).
    """

    def __init__(self):
        """
        Initialize search backends based on available API keys.

        We check for keys at construction time (not per-search) to avoid
        repeated environment lookups and to give clear startup diagnostics.
        """
        self._tavily_key = os.environ.get("TAVILY_API_KEY", "")  # empty = falsy

        if self._tavily_key:
            # Tavily key is present — initialize the official Tavily client.
            # Importing here (not at module top) keeps the module importable even
            # if tavily-python is somehow not installed (graceful degradation).
            try:
                from tavily import TavilyClient  # official Tavily Python SDK
                self._tavily_client = TavilyClient(api_key=self._tavily_key)
                print("[SearchEngine] Tavily API client initialized (primary search).")
            except ImportError:
                # Package not installed — fall back gracefully
                self._tavily_client = None
                print("[SearchEngine] WARNING: tavily-python not installed. Falling back to DuckDuckGo.")
        else:
            self._tavily_client = None
            print("[SearchEngine] No TAVILY_API_KEY found — DuckDuckGo fallback active.")

    def search(
        self,
        query: str,
        max_results: int = cfg.max_search_results,
    ) -> List[SourceDocument]:
        """
        Execute a web search and return standardized SourceDocument results.

        Strategy:
            1. Try Tavily if client is initialized.
            2. If Tavily raises an exception or returns empty results, fall back
               to DuckDuckGo automatically.

        Args:
            query: The search query string (1 of the planner's sub-queries).
            max_results: Maximum number of results to return.

        Returns:
            List of SourceDocument objects with title, url, snippet fields.
            Returns empty list if both backends fail (never raises an exception).
        """
        print(f"\n[SearchEngine] Searching: '{query[:80]}'")

        # ── ATTEMPT 1: Tavily ────────────────────────────────────────────────
        if self._tavily_client:
            results = self._search_tavily(query, max_results)
            if results:
                # Tavily returned usable results — return them directly
                print(f"[SearchEngine] Tavily returned {len(results)} results.")
                return results
            else:
                # Tavily returned empty (rare but possible — try fallback)
                print("[SearchEngine] Tavily returned empty results — trying DuckDuckGo fallback.")

        # ── ATTEMPT 2: DuckDuckGo Fallback ──────────────────────────────────
        results = self._search_duckduckgo(query, max_results)
        print(f"[SearchEngine] DuckDuckGo returned {len(results)} results.")
        return results

    def _search_tavily(
        self,
        query: str,
        max_results: int,
    ) -> List[SourceDocument]:
        """
        Execute a Tavily search request.

        Tavily's "advanced" depth triggers a deeper pipeline on their servers:
        - Re-ranks results by relevance to the query
        - Extracts the most relevant passage from each page
        - Filters out low-quality domains (spam, SEO-only pages)

        Args:
            query: Search query string.
            max_results: Max number of results to request from Tavily.

        Returns:
            List of SourceDocument objects, or empty list if search fails.
        """
        try:
            response = self._tavily_client.search(
                query=query,
                search_depth=cfg.search_depth,  # "advanced" for best quality
                max_results=max_results,
                include_answer=False,            # we don't use Tavily's synthesized answer
                include_raw_content=False,       # we do our own scraping for full content
            )

            # Tavily returns: {"results": [{"title", "url", "content", "score"}, ...]}
            raw_results = response.get("results", [])  # safe .get() in case key is missing

            if not raw_results:
                return []  # empty results — signal to caller to try fallback

            documents = []
            for item in raw_results:
                # Validate each result through the SourceDocument Pydantic model.
                # This catches malformed URLs or missing fields gracefully.
                try:
                    doc = SourceDocument(
                        title=item.get("title", "Untitled"),     # Tavily always provides title
                        url=item.get("url", ""),                  # URL is required; Pydantic will validate
                        snippet=item.get("content", "")[:500],    # Tavily "content" is a rich snippet; truncate
                    )
                    documents.append(doc)
                except Exception as e:
                    # One malformed result should not stop the others from processing
                    print(f"[SearchEngine] Skipping malformed Tavily result: {e}")
                    continue

            return documents

        except Exception as e:
            # Network error, rate limit, or invalid API key — log and signal fallback
            print(f"[SearchEngine] Tavily search failed: {type(e).__name__}: {e}")
            return []  # empty list tells caller to try DuckDuckGo

    def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
    ) -> List[SourceDocument]:
        """
        Execute a DuckDuckGo search using the duckduckgo-search library.

        How duckduckgo-search works:
            It sends requests to DuckDuckGo's non-API search endpoints and
            parses the HTML response. No API key required, but results are
            rate-limited by IP (roughly 20 requests/minute before soft blocks).
            The library handles retries and rate-limit backoff automatically.

        Args:
            query: Search query string.
            max_results: Max number of results to request.

        Returns:
            List of SourceDocument objects, or empty list if search fails.
        """
        try:
            from duckduckgo_search import DDGS  # DDG search library

            documents = []

            # DDGS() is a context manager that manages the underlying HTTP session.
            # Using it as a context manager ensures the session is closed properly.
            with DDGS() as ddgs:
                # ddgs.text() is a generator that yields result dicts one-by-one.
                # We use a list comprehension with a limit to avoid fetching more
                # results than we need (each result is a separate DDG page request).
                results = list(ddgs.text(
                    query,
                    max_results=max_results,  # stop after this many results
                ))

            for item in results:
                try:
                    doc = SourceDocument(
                        # DuckDuckGo returns keys: title, href, body
                        title=item.get("title", "Untitled"),
                        url=item.get("href", ""),        # DDG uses "href" not "url"
                        snippet=item.get("body", ""),    # DDG "body" is the result snippet
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"[SearchEngine] Skipping malformed DuckDuckGo result: {e}")
                    continue

            return documents

        except ImportError:
            print("[SearchEngine] duckduckgo-search not installed. No search results available.")
            return []
        except Exception as e:
            # Rate limiting, network errors, or HTML parsing failures
            print(f"[SearchEngine] DuckDuckGo search failed: {type(e).__name__}: {e}")
            return []  # never raise — return empty list so the pipeline can continue


def multi_search(
    queries: List[str],
    max_results_per_query: int = cfg.max_search_results,
) -> List[SourceDocument]:
    """
    Run multiple search queries and return deduplicated combined results.

    WHY DEDUPLICATE?
        The planner generates 3–5 related sub-queries. The same URL might appear
        in search results for multiple sub-queries (e.g., a comprehensive blog post
        about RAG). Deduplication prevents the scraper from fetching the same URL
        multiple times and the report from citing it as if it were multiple sources.

    Args:
        queries: List of sub-query strings from the planner node.
        max_results_per_query: Max results per individual query.

    Returns:
        Deduplicated list of SourceDocument objects across all queries.
    """
    engine = SearchEngine()  # create once, reuse across all queries in this batch
    seen_urls: set = set()   # track URLs we've already seen (O(1) lookup)
    all_results: List[SourceDocument] = []

    for i, query in enumerate(queries):
        print(f"[SearchEngine] Sub-query {i+1}/{len(queries)}: '{query[:60]}'")
        results = engine.search(query, max_results=max_results_per_query)

        for doc in results:
            if doc.url not in seen_urls:    # only add if URL is new
                seen_urls.add(doc.url)       # mark URL as seen
                all_results.append(doc)      # add to aggregated results
            else:
                print(f"[SearchEngine] Skipping duplicate URL: {doc.url[:60]}")

    print(f"[SearchEngine] Multi-search complete: {len(all_results)} unique URLs found.")
    return all_results
