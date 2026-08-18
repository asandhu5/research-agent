"""
tests/test_search.py — Unit Tests for the Search Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT WE'RE TESTING:
    1. Tavily primary path — correct output structure when Tavily returns results.
    2. DuckDuckGo fallback — DuckDuckGo is used when Tavily key is absent.
    3. Automatic fallback — when Tavily fails (exception), DuckDuckGo kicks in.
    4. Output standardization — both backends produce SourceDocument objects.
    5. Deduplication — multi_search() removes duplicate URLs.
    6. Empty results handling — search failure returns [] not None.

KEY TESTING TECHNIQUE — ENVIRONMENT VARIABLE MOCKING:
    SearchEngine.__init__ checks os.environ.get("TAVILY_API_KEY") at construction.
    To test the "no Tavily key" path, we temporarily remove the key from the
    environment using unittest.mock.patch.dict(os.environ, ..., clear=True).
    This is safer than manually setting/restoring os.environ.
"""

import os                                          # for checking env variable state
import pytest                                      # test framework
from unittest.mock import patch, MagicMock         # for mocking API calls and HTTP requests

from search_engine import SearchEngine, multi_search
from state import SourceDocument


# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA
# These simulate the response structures of Tavily and DuckDuckGo APIs.
# Using known mock data makes tests deterministic and independent of real APIs.
# ─────────────────────────────────────────────────────────────────────────────

TAVILY_MOCK_RESPONSE = {
    "results": [
        {
            "title": "CRAG: Corrective Retrieval Augmented Generation",
            "url": "https://arxiv.org/abs/2401.15884",
            "content": "CRAG introduces a lightweight evaluator to assess retrieved document quality. "
                       "When quality is deemed insufficient, CRAG falls back to web search for "
                       "supplementary information before generation.",
            "score": 0.92,
        },
        {
            "title": "Self-RAG: Self-Reflective Retrieval Augmented Generation",
            "url": "https://arxiv.org/abs/2310.11511",
            "content": "Self-RAG trains LLMs with special reflection tokens to selectively retrieve "
                       "and critique their own outputs, improving factual accuracy.",
            "score": 0.89,
        },
        {
            "title": "RAG Survey: Hallucination Mitigation Techniques",
            "url": "https://example.com/rag-hallucination-survey",
            "content": "A comprehensive survey of techniques to reduce hallucinations in "
                       "retrieval-augmented generation systems.",
            "score": 0.85,
        },
    ]
}

DDG_MOCK_RESULTS = [
    {
        "title": "DuckDuckGo Result: GraphRAG Overview",
        "href": "https://microsoft.github.io/graphrag/",  # DDG uses "href" not "url"
        "body": "GraphRAG uses knowledge graphs to structure relationships between entities, "
                "enabling global queries that span multiple documents.",
    },
    {
        "title": "DuckDuckGo Result: Knowledge Graph RAG",
        "href": "https://example.com/knowledge-graph-rag",
        "body": "Using knowledge graphs in RAG systems improves coherence by explicitly "
                "representing entity relationships.",
    },
]


class TestTavilySearch:
    """Tests for the Tavily search backend."""

    def test_tavily_returns_source_documents_when_key_present(self):
        """
        WHAT: When TAVILY_API_KEY is set and Tavily returns results, verify output structure.

        WHY: The rest of the pipeline (scraper, evaluator) depends on receiving
        SourceDocument objects with title, url, and snippet fields. If the adapter
        produces raw dicts or missing keys, downstream code crashes.

        HOW: We inject a fake TAVILY_API_KEY into the environment and mock the
        TavilyClient to return our pre-defined mock response.
        """
        with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test-key-12345"}, clear=False):
            with patch("search_engine.TavilyClient") as mock_tavily_class:
                mock_client = MagicMock()
                mock_client.search.return_value = TAVILY_MOCK_RESPONSE
                mock_tavily_class.return_value = mock_client

                engine = SearchEngine()
                results = engine.search("hallucination mitigation in RAG", max_results=3)

        assert len(results) > 0, "Tavily should return at least one result"
        assert all(isinstance(r, SourceDocument) for r in results), (
            "All results must be SourceDocument instances, not raw dicts"
        )
        for result in results:
            assert result.url.startswith("http"), (
                f"URL '{result.url}' should start with http/https"
            )
            assert result.title, "Each result must have a non-empty title"

    def test_tavily_returns_correct_number_of_results(self):
        """
        WHAT: Verify that the returned list has at most max_results items.

        WHY: We request at most `max_results` to control API costs and scraping time.
        The adapter must not return more than requested.
        """
        with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test-key"}, clear=False):
            with patch("search_engine.TavilyClient") as mock_tavily_class:
                mock_client = MagicMock()
                mock_client.search.return_value = TAVILY_MOCK_RESPONSE
                mock_tavily_class.return_value = mock_client

                engine = SearchEngine()
                results = engine.search("test query", max_results=2)

        assert len(results) <= 2, f"Should return at most 2 results when max_results=2, got {len(results)}"

    def test_tavily_falls_back_to_ddg_on_empty_results(self):
        """
        WHAT: When Tavily returns empty results, DuckDuckGo should be tried.

        WHY: Tavily can return empty results for queries with no indexed pages,
        very recent events, or niche topics. The agent should never stall —
        it should seamlessly try the fallback.
        """
        with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test-key"}, clear=False):
            with patch("search_engine.TavilyClient") as mock_tavily_class:
                mock_client = MagicMock()
                mock_client.search.return_value = {"results": []}  # empty Tavily results
                mock_tavily_class.return_value = mock_client

                with patch("search_engine.DDGS") as mock_ddgs_class:
                    mock_ddgs = MagicMock()
                    mock_ddgs.__enter__.return_value = mock_ddgs
                    mock_ddgs.__exit__.return_value = None
                    mock_ddgs.text.return_value = DDG_MOCK_RESULTS
                    mock_ddgs_class.return_value = mock_ddgs

                    engine = SearchEngine()
                    results = engine.search("niche query with no tavily results")

        assert len(results) > 0, (
            "When Tavily returns empty results, DuckDuckGo fallback should provide results"
        )


class TestDuckDuckGoFallback:
    """Tests for the DuckDuckGo fallback backend."""

    def test_ddg_used_when_no_tavily_key(self):
        """
        WHAT: When TAVILY_API_KEY is absent, DuckDuckGo should be used automatically.

        WHY: This is the most important fallback test. An ML engineer without a
        Tavily subscription should still be able to run the agent. The search engine
        must detect the missing key at construction time and route all calls to DDG.

        HOW: We use patch.dict with an environment that has no TAVILY_API_KEY,
        then mock DDGS to return our pre-defined results.
        """
        env_without_tavily = {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}

        with patch.dict(os.environ, env_without_tavily, clear=True):
            with patch("search_engine.DDGS") as mock_ddgs_class:
                mock_ddgs = MagicMock()
                mock_ddgs.__enter__.return_value = mock_ddgs
                mock_ddgs.__exit__.return_value = None
                mock_ddgs.text.return_value = DDG_MOCK_RESULTS
                mock_ddgs_class.return_value = mock_ddgs

                engine = SearchEngine()

                assert engine._tavily_client is None, (
                    "When TAVILY_API_KEY is not set, _tavily_client should be None"
                )

                results = engine.search("GraphRAG knowledge graphs")

        assert len(results) > 0, "DuckDuckGo fallback should return results when Tavily key is absent"
        assert all(isinstance(r, SourceDocument) for r in results), (
            "DuckDuckGo results must also be SourceDocument instances"
        )

    def test_ddg_output_uses_href_as_url(self):
        """
        WHAT: Verify that DuckDuckGo's 'href' field is correctly mapped to SourceDocument.url.

        WHY: DuckDuckGo uses the key "href" for URLs (not "url" like Tavily does).
        The adapter must translate this. If not translated, all DDG results would
        have empty URLs and no pages could be scraped.
        """
        env_without_tavily = {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}

        with patch.dict(os.environ, env_without_tavily, clear=True):
            with patch("search_engine.DDGS") as mock_ddgs_class:
                mock_ddgs = MagicMock()
                mock_ddgs.__enter__.return_value = mock_ddgs
                mock_ddgs.__exit__.return_value = None
                mock_ddgs.text.return_value = DDG_MOCK_RESULTS
                mock_ddgs_class.return_value = mock_ddgs

                engine = SearchEngine()
                results = engine.search("test query")

        assert len(results) > 0, "Should have results from DDG mock"
        # The DDG mock has href="https://microsoft.github.io/graphrag/"
        # It should appear as the url field in the SourceDocument
        urls = [r.url for r in results]
        assert "https://microsoft.github.io/graphrag/" in urls, (
            "DuckDuckGo 'href' field should be mapped to SourceDocument.url"
        )

    def test_ddg_failure_returns_empty_list(self):
        """
        WHAT: When DuckDuckGo raises an exception, search() should return [] not raise.

        WHY: The agent must never crash due to a search failure. An empty list tells
        the graph to proceed with zero results (likely triggering another iteration
        or a fallback report), rather than propagating an exception that kills the process.
        """
        env_without_tavily = {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}

        with patch.dict(os.environ, env_without_tavily, clear=True):
            with patch("search_engine.DDGS", side_effect=Exception("DDG rate limited")):
                engine = SearchEngine()
                results = engine.search("test query that causes DDG exception")

        assert results == [], (
            "When DuckDuckGo raises an exception, search() should return an empty list"
        )
        assert isinstance(results, list), "Return type should always be a list"


class TestMultiSearch:
    """Tests for the multi_search() batch helper function."""

    def test_multi_search_deduplicates_urls(self):
        """
        WHAT: When the same URL appears in multiple sub-query results, it should appear
        only once in the combined output.

        WHY: Without deduplication, the same page would be scraped multiple times,
        wasting HTTP calls, and the LLM might cite the same URL as [1] and [5] and [9],
        making the report look suspicious.
        """
        duplicate_url = "https://arxiv.org/abs/2310.11511"

        mock_results_q1 = [
            SourceDocument(title="Self-RAG", url=duplicate_url, snippet="snippet 1"),
            SourceDocument(title="CRAG", url="https://arxiv.org/abs/2401.15884", snippet="snippet 2"),
        ]
        mock_results_q2 = [
            SourceDocument(title="Self-RAG Duplicate", url=duplicate_url, snippet="same url, different search"),
            SourceDocument(title="GraphRAG", url="https://microsoft.github.io/graphrag/", snippet="snippet 3"),
        ]

        call_count = 0
        def mock_search_side_effect(query, max_results):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_results_q1
            return mock_results_q2

        with patch.object(SearchEngine, "search", side_effect=mock_search_side_effect):
            results = multi_search(["query one", "query two"])

        urls = [r.url for r in results]
        assert len(set(urls)) == len(urls), (
            "multi_search() should return unique URLs only. "
            f"Found duplicates: {[u for u in urls if urls.count(u) > 1]}"
        )
        assert duplicate_url in urls, "The duplicate URL should appear exactly once"
        assert urls.count(duplicate_url) == 1, f"'{duplicate_url}' should appear exactly once, not {urls.count(duplicate_url)} times"

    def test_multi_search_combines_all_query_results(self):
        """
        WHAT: Results from all sub-queries should be combined into one list.

        WHY: multi_search() is designed to aggregate across multiple queries.
        If it only returned results from the first query, the planner's sub-queries
        for different aspects of the topic would be wasted.
        """
        with patch.dict(os.environ, {}, clear=False):
            with patch("search_engine.SearchEngine.search") as mock_search:
                mock_search.side_effect = [
                    [SourceDocument(title=f"Result A{i}", url=f"https://a{i}.com", snippet="") for i in range(3)],
                    [SourceDocument(title=f"Result B{i}", url=f"https://b{i}.com", snippet="") for i in range(3)],
                ]
                results = multi_search(["query A", "query B"])

        assert len(results) == 6, (
            f"Expected 6 combined results from 2 queries × 3 results each, got {len(results)}"
        )
