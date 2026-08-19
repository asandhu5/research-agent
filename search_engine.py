import os                                          # reading API keys from environment
from typing import List, Dict, Any, Optional       # type hints
from dotenv import load_dotenv                     # loads .env into os.environ

from state import SourceDocument                   # validated result schema
from config import cfg                             # hyperparameters

load_dotenv()


class SearchEngine:

    def __init__(self):

        self._tavily_key = os.environ.get("TAVILY_API_KEY", "")  # empty = falsy

        if self._tavily_key:
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

        print(f"\n[SearchEngine] Searching: '{query[:80]}'")

        if self._tavily_client:
            results = self._search_tavily(query, max_results)
            if results:
                print(f"[SearchEngine] Tavily returned {len(results)} results.")
                return results
            else:
                print("[SearchEngine] Tavily returned empty results — trying DuckDuckGo fallback.")

        results = self._search_duckduckgo(query, max_results)
        print(f"[SearchEngine] DuckDuckGo returned {len(results)} results.")
        return results

    def _search_tavily(
        self,
        query: str,
        max_results: int,
    ) -> List[SourceDocument]:

        try:
            response = self._tavily_client.search(
                query=query,
                search_depth=cfg.search_depth,  # "advanced" for best quality
                max_results=max_results,
                include_answer=False,            # we don't use Tavily's synthesized answer
                include_raw_content=False,       # we do our own scraping for full content
            )
            raw_results = response.get("results", [])
            if not raw_results:
                return []
            documents = []
            for item in raw_results:
                try:
                    doc = SourceDocument(
                        title=item.get("title", "Untitled"),     # Tavily always provides title
                        url=item.get("url", ""),                  # URL is required; Pydantic will validate
                        snippet=item.get("content", "")[:500],    # Tavily "content" is a rich snippet; truncate
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"[SearchEngine] Skipping malformed Tavily result: {e}")
                    continue

            return documents

        except Exception as e:
            print(f"[SearchEngine] Tavily search failed: {type(e).__name__}: {e}")
            return []

    def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
    ) -> List[SourceDocument]:

        try:
            from duckduckgo_search import DDGS
            documents = []
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    max_results=max_results,
                ))
            for item in results:
                try:
                    doc = SourceDocument(
                        title=item.get("title", "Untitled"),
                        url=item.get("href", ""),
                        snippet=item.get("body", ""),
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
            print(f"[SearchEngine] DuckDuckGo search failed: {type(e).__name__}: {e}")
            return []


def multi_search(
    queries: List[str],
    max_results_per_query: int = cfg.max_search_results,
) -> List[SourceDocument]:

    engine = SearchEngine()
    seen_urls: set = set()
    all_results: List[SourceDocument] = []

    for i, query in enumerate(queries):
        print(f"[SearchEngine] Sub-query {i+1}/{len(queries)}: '{query[:60]}'")
        results = engine.search(query, max_results=max_results_per_query)

        for doc in results:
            if doc.url not in seen_urls:
                seen_urls.add(doc.url)
                all_results.append(doc)
            else:
                print(f"[SearchEngine] Skipping duplicate URL: {doc.url[:60]}")

    print(f"[SearchEngine] Multi-search complete: {len(all_results)} unique URLs found.")
    return all_results
