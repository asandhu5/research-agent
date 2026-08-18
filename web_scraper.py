"""
web_scraper.py — Robust HTML Parsing, Cleaning, and Text Chunking
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY SCRAPE AT ALL?

    Search APIs (Tavily, DuckDuckGo) return short snippets — typically 200–500
    characters. That's not enough for the agent to deeply understand a topic
    and synthesize a detailed report. Full-page scraping gives us 2,000–8,000
    characters of clean content per URL, enabling richer synthesis.

WEB SCRAPING EDGE CASES WE HANDLE:
    1. HTTP errors (404, 500, 403): Return empty ScrapedPage with success=False.
    2. Timeouts: Some servers are slow; we cut the connection after N seconds.
    3. Redirects: requests follows redirects automatically (up to 5 by default).
    4. Encoding issues: Some pages use non-UTF-8 encodings; we detect and handle.
    5. JavaScript-rendered content: BeautifulSoup only parses static HTML, so
       JavaScript-only pages (SPAs) will appear empty. We handle this gracefully.
    6. Anti-scraping blocks (403/429): We use a realistic User-Agent header to
       reduce bot detection. We never bypass CAPTCHA.
    7. Very large pages: We truncate to max_char_limit to stay within LLM context.
    8. Binary/media responses: We check Content-Type before parsing.
"""

import re                                          # regular expressions for whitespace cleaning
from typing import List, Optional                  # type hints
from urllib.parse import urljoin, urlparse         # URL utilities for relative→absolute URL fixing

import requests                                    # HTTP client for fetching web pages
from bs4 import BeautifulSoup                      # HTML parser
from requests.exceptions import (
    Timeout,          # raised when request exceeds timeout_seconds
    ConnectionError,  # raised when DNS lookup or TCP connection fails
    HTTPError,        # raised when response.raise_for_status() detects 4xx/5xx
    TooManyRedirects, # raised when redirect chain exceeds the limit
)

from config import cfg                             # hyperparameters (timeout, max_char_limit)
from state import ScrapedPage                      # validated output schema


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Realistic User-Agent string mimicking a Chrome browser on macOS.
# Many servers return 403 Forbidden for requests with default Python User-Agents
# like "python-requests/2.31.0". Using a browser UA reduces blocking rates.
# NOTE: This is NOT circumventing authentication — it's standard practice for
# public web scraping of publicly accessible content.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# HTTP headers sent with every request.
# Accept-Language: Requests English content (avoids locale-specific redirects).
# Accept: Signals we want HTML, not JSON or XML.
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",  # allows compressed responses (faster)
    "Connection": "keep-alive",              # reuse TCP connection for multiple requests
    "DNT": "1",                              # Do Not Track header (polite, often ignored)
}

# HTML tags to remove entirely (including their children) before extracting text.
# These tags contain non-content: scripts, styles, navigation, ads, footers.
TAGS_TO_REMOVE = [
    "script",     # JavaScript code — pure noise
    "style",      # CSS rules — pure noise
    "nav",        # Site navigation links — not research content
    "footer",     # Footer links, copyright notices — not research content
    "header",     # Site header/logo area — not research content
    "aside",      # Sidebars, "related articles" widgets — usually low value
    "advertisement",  # Ad container elements
    "noscript",   # Fallback content for no-JS browsers — often empty or duplicate
    "iframe",     # Embedded frames — cannot be parsed
    "form",       # HTML forms — not research content
    "button",     # Buttons — not research content
]


class WebScraper:
    """
    HTTP fetcher + HTML cleaner that extracts meaningful text from web pages.

    Design:
        The scraper is intentionally stateless — every call to scrape_url()
        is independent. This makes it safe to parallelize scraping calls
        across multiple URLs without shared-state race conditions.
    """

    def scrape_url(
        self,
        url: str,
        timeout: int = cfg.timeout_seconds,
        max_chars: int = cfg.max_char_limit,
    ) -> ScrapedPage:
        """
        Fetch a URL and return its cleaned text content.

        Pipeline:
            fetch_html() → parse_and_clean() → truncate → ScrapedPage

        Args:
            url: The full URL to scrape (must start with http/https).
            timeout: Seconds before abandoning the HTTP request.
            max_chars: Maximum characters to return (excess is truncated).

        Returns:
            ScrapedPage with success=True and populated content, or
            ScrapedPage with success=False and error message if anything fails.
        """
        print(f"[WebScraper] Scraping: {url[:80]}")

        # ── STEP 1: Validate URL ─────────────────────────────────────────────
        parsed = urlparse(url)
        if not parsed.scheme in ("http", "https"):
            # Non-HTTP URLs (ftp://, file://, etc.) are not scrapable
            return ScrapedPage(
                url=url,
                success=False,
                error=f"Unsupported URL scheme: '{parsed.scheme}'. Only http/https supported.",
            )

        # ── STEP 2: Fetch raw HTML ───────────────────────────────────────────
        raw_html, fetch_error = self._fetch_html(url, timeout)
        if fetch_error:
            # fetch_html returned an error string — return a failed ScrapedPage
            return ScrapedPage(url=url, success=False, error=fetch_error)

        # ── STEP 3: Parse and clean HTML ─────────────────────────────────────
        title, clean_text = self._parse_and_clean(raw_html, url)

        if not clean_text:
            # Parsing produced empty text — likely a JS-only SPA or empty page
            return ScrapedPage(
                url=url,
                title=title,
                success=False,
                error="Page contained no extractable text content (possibly JavaScript-rendered SPA).",
            )

        # ── STEP 4: Truncate to token budget ─────────────────────────────────
        # We truncate at a word boundary to avoid cutting mid-sentence,
        # which can confuse the LLM evaluator.
        if len(clean_text) > max_chars:
            # Find the last space before the character limit (word-boundary truncation)
            truncation_point = clean_text.rfind(" ", 0, max_chars)
            if truncation_point == -1:
                truncation_point = max_chars  # no space found, hard truncate
            clean_text = clean_text[:truncation_point] + "... [truncated]"

        print(f"[WebScraper] ✓ Scraped {len(clean_text)} chars from: {title[:50] or url[:50]}")

        return ScrapedPage(
            url=url,
            title=title,
            content=clean_text,
            success=True,
        )

    def _fetch_html(self, url: str, timeout: int) -> tuple[Optional[str], Optional[str]]:
        """
        Download raw HTML string from a URL.

        EDGE CASES HANDLED:
            - Timeout: Server is slow or hung — return error string.
            - ConnectionError: DNS failure, network unreachable — return error.
            - HTTPError: 404 Not Found, 403 Forbidden, 500 Server Error.
            - TooManyRedirects: Redirect loops (rare but happens with misconfigured sites).
            - Non-HTML content: PDF, ZIP, images — we skip these.

        Args:
            url: URL to fetch.
            timeout: Seconds before aborting.

        Returns:
            (html_string, None) on success, or (None, error_message) on failure.
        """
        try:
            response = requests.get(
                url,
                headers=DEFAULT_HEADERS,   # use realistic browser headers
                timeout=timeout,           # bail out after N seconds
                allow_redirects=True,      # follow HTTP redirects (301, 302, 307)
                stream=False,              # download the full response body immediately
            )

            # raise_for_status() raises an HTTPError for 4xx and 5xx status codes.
            # Without this, requests silently returns the error page as "content".
            response.raise_for_status()

            # Content-Type check: we can only parse text/html.
            # PDF, ZIP, image, JSON etc. should be skipped gracefully.
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "text/plain" not in content_type:
                # Not HTML — return a descriptive error, not an exception
                return None, f"Non-HTML content type: '{content_type}'. Skipping."

            # Detect encoding from Content-Type header or HTML meta tags.
            # response.apparent_encoding uses chardet to detect encoding from byte patterns.
            # Fallback to UTF-8 if detection fails.
            if response.encoding and response.encoding.lower() != "utf-8":
                # requests may mis-detect encoding; apparent_encoding is usually better
                response.encoding = response.apparent_encoding or "utf-8"

            return response.text, None  # success — return HTML string and no error

        except Timeout:
            # Server didn't respond within timeout_seconds
            return None, f"Request timed out after {timeout}s. Server may be overloaded."

        except ConnectionError as e:
            # DNS resolution failed, connection refused, or network unreachable
            return None, f"Connection error: {e}. URL may be invalid or server is down."

        except HTTPError as e:
            # 4xx or 5xx HTTP status code
            status = e.response.status_code if e.response is not None else "unknown"
            if status == 403:
                return None, f"HTTP 403 Forbidden — server is blocking scraping."
            elif status == 404:
                return None, f"HTTP 404 Not Found — page no longer exists."
            elif status == 429:
                return None, f"HTTP 429 Too Many Requests — rate limited by server."
            else:
                return None, f"HTTP {status} error: {e}"

        except TooManyRedirects:
            # Redirect loop — the site keeps redirecting us in circles
            return None, "Too many redirects — server is in a redirect loop."

        except Exception as e:
            # Catch-all for unexpected errors (SSL cert issues, encoding errors, etc.)
            return None, f"Unexpected error fetching {url}: {type(e).__name__}: {e}"

    def _parse_and_clean(self, html: str, url: str) -> tuple[str, str]:
        """
        Parse raw HTML and extract clean, readable text.

        CLEANING PIPELINE:
            1. Parse HTML with BeautifulSoup (lxml backend for speed).
            2. Remove all noise tags (script, style, nav, footer, etc.).
            3. Extract all remaining text nodes.
            4. Collapse multiple whitespace/newlines into single spaces.
            5. Remove lines that are suspiciously short (navigation link fragments).

        Args:
            html: Raw HTML string from the HTTP response.
            url: Used for relative-to-absolute URL resolution (not used here but
                 available for future enhancement like image extraction).

        Returns:
            (page_title, cleaned_text) — both are strings.
        """
        # Parse the HTML with lxml as the backend.
        # lxml is faster than html.parser and more lenient about malformed HTML.
        # "html.parser" is the Python built-in fallback if lxml is not installed.
        try:
            soup = BeautifulSoup(html, "lxml")  # fastest, most tolerant parser
        except Exception:
            try:
                soup = BeautifulSoup(html, "html.parser")  # built-in fallback
            except Exception as e:
                return "", ""  # unparseable HTML — return empty strings

        # ── Extract page title ────────────────────────────────────────────────
        title_tag = soup.find("title")  # <title>Page Title Here</title>
        title = title_tag.get_text(strip=True) if title_tag else ""  # empty string if no title

        # ── Remove noise tags ────────────────────────────────────────────────
        # soup.find_all() returns all instances; we iterate and call .decompose()
        # which removes the tag AND all its children from the parse tree.
        for tag_name in TAGS_TO_REMOVE:
            for tag in soup.find_all(tag_name):
                tag.decompose()  # mutates the parse tree in-place — removes tag entirely

        # Also remove elements with class/id names commonly associated with ads/popups
        # These are heuristic patterns — won't catch everything but reduces noise significantly
        noise_patterns = ["cookie", "popup", "modal", "banner", "advertisement", "sidebar"]
        for element in soup.find_all(class_=True):
            # element["class"] is a list of CSS class names
            element_classes = " ".join(element.get("class", [])).lower()
            if any(pattern in element_classes for pattern in noise_patterns):
                element.decompose()  # remove noise elements

        # ── Extract main content ─────────────────────────────────────────────
        # Prefer the <main>, <article>, or <div id="content"> elements if they exist,
        # as these are semantically marked as the page's primary content.
        # If none found, fall back to the entire <body>.
        main_content = (
            soup.find("main")                         or   # HTML5 main content landmark
            soup.find("article")                      or   # blog posts, news articles
            soup.find(id=re.compile(r"content|main|article", re.I))  or  # common content IDs
            soup.find("body")                         or   # full body as last resort
            soup                                           # entire parsed tree if no body
        )

        # get_text() extracts all text nodes, using "\n" as separator between tags.
        # strip=True removes leading/trailing whitespace from each text chunk.
        raw_text = main_content.get_text(separator="\n", strip=True)

        # ── Normalize whitespace ─────────────────────────────────────────────
        # Step A: Split into lines for line-level cleaning
        lines = raw_text.split("\n")

        # Step B: Remove lines that are too short to be meaningful content.
        # Lines < 20 characters are usually navigation fragments ("Home | About | Contact")
        # or stray UI elements. We keep them only if they're not pure punctuation/symbols.
        meaningful_lines = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) < 20 and not re.match(r'^[A-Z][\w\s]{10,}$', stripped):
                # Short line — skip unless it looks like a short heading
                if len(stripped) > 5:  # keep very short headings like "Results"
                    meaningful_lines.append(stripped)
                continue  # skip blank lines and pure whitespace
            if stripped:
                meaningful_lines.append(stripped)

        # Step C: Join with spaces and collapse multiple consecutive whitespace
        clean_text = " ".join(meaningful_lines)
        clean_text = re.sub(r'\s+', ' ', clean_text)  # collapse multiple spaces to one
        clean_text = clean_text.strip()                # remove leading/trailing whitespace

        return title, clean_text


def scrape_urls_batch(
    urls: List[str],
    timeout: int = cfg.timeout_seconds,
    max_chars: int = cfg.max_char_limit,
) -> List[ScrapedPage]:
    """
    Scrape a list of URLs and return their cleaned content.

    WHY SEQUENTIAL (NOT PARALLEL)?
        Parallel scraping with threading or asyncio is faster but:
        1. Raises the risk of hitting rate limits on the same domain simultaneously.
        2. Requires careful thread-safety around the shared output list.
        3. Makes error tracing harder in a learning environment.
        Sequential scraping is clear, debuggable, and fast enough for 3-5 URLs.
        For production with 20+ URLs, switch to concurrent.futures.ThreadPoolExecutor.

    Args:
        urls: List of URL strings to scrape.
        timeout: HTTP timeout per URL.
        max_chars: Text truncation limit per page.

    Returns:
        List of ScrapedPage objects in the same order as the input URLs.
    """
    scraper = WebScraper()  # create once, reuse across all URLs
    results: List[ScrapedPage] = []

    for i, url in enumerate(urls):
        print(f"[WebScraper] Processing URL {i+1}/{len(urls)}")
        page = scraper.scrape_url(url, timeout=timeout, max_chars=max_chars)
        results.append(page)

        if not page.success:
            # Log failures but continue — one bad URL shouldn't stop the batch
            print(f"[WebScraper] Failed: {page.error[:100]}")

    successful = sum(1 for p in results if p.success)  # count successful scrapes
    print(f"[WebScraper] Batch complete: {successful}/{len(urls)} pages scraped successfully.")
    return results
