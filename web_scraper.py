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

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",  # allows compressed responses (faster)
    "Connection": "keep-alive",              # reuse TCP connection for multiple requests
    "DNT": "1",                              # Do Not Track header (polite, often ignored)
}

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
    def scrape_url(
        self,
        url: str,
        timeout: int = cfg.timeout_seconds,
        max_chars: int = cfg.max_char_limit,
    ) -> ScrapedPage:
        print(f"[WebScraper] Scraping: {url[:80]}")
        parsed = urlparse(url)
        if not parsed.scheme in ("http", "https"):
            return ScrapedPage(
                url=url,
                success=False,
                error=f"Unsupported URL scheme: '{parsed.scheme}'. Only http/https supported.",
            )

        raw_html, fetch_error = self._fetch_html(url, timeout)
        if fetch_error:
            return ScrapedPage(url=url, success=False, error=fetch_error)

        title, clean_text = self._parse_and_clean(raw_html, url)

        if not clean_text:
            return ScrapedPage(
                url=url,
                title=title,
                success=False,
                error="Page contained no extractable text content (possibly JavaScript-rendered SPA).",
            )
        if len(clean_text) > max_chars:
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
        try:
            response = requests.get(
                url,
                headers=DEFAULT_HEADERS,   # use realistic browser headers
                timeout=timeout,           # bail out after N seconds
                allow_redirects=True,      # follow HTTP redirects (301, 302, 307)
                stream=False,              # download the full response body immediately
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "text/plain" not in content_type:
                return None, f"Non-HTML content type: '{content_type}'. Skipping."
            if response.encoding and response.encoding.lower() != "utf-8":
                response.encoding = response.apparent_encoding or "utf-8"
            return response.text, None
        except Timeout:
            return None, f"Request timed out after {timeout}s. Server may be overloaded."

        except ConnectionError as e:
            return None, f"Connection error: {e}. URL may be invalid or server is down."

        except HTTPError as e:
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
            return None, "Too many redirects — server is in a redirect loop."

        except Exception as e:
            return None, f"Unexpected error fetching {url}: {type(e).__name__}: {e}"

    def _parse_and_clean(self, html: str, url: str) -> tuple[str, str]:
        try:
            soup = BeautifulSoup(html, "lxml")  # fastest, most tolerant parser
        except Exception:
            try:
                soup = BeautifulSoup(html, "html.parser")  # built-in fallback
            except Exception as e:
                return "", ""
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        for tag_name in TAGS_TO_REMOVE:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        noise_patterns = ["cookie", "popup", "modal", "banner", "advertisement", "sidebar"]
        for element in soup.find_all(class_=True):
            element_classes = " ".join(element.get("class", [])).lower()
            if any(pattern in element_classes for pattern in noise_patterns):
                element.decompose()
        main_content = (
            soup.find("main")                         or   # HTML5 main content landmark
            soup.find("article")                      or   # blog posts, news articles
            soup.find(id=re.compile(r"content|main|article", re.I))  or  # common content IDs
            soup.find("body")                         or   # full body as last resort
            soup                                           # entire parsed tree if no body
        )
        raw_text = main_content.get_text(separator="\n", strip=True)
        lines = raw_text.split("\n")

        meaningful_lines = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) < 20 and not re.match(r'^[A-Z][\w\s]{10,}$', stripped):
                if len(stripped) > 5:
                    meaningful_lines.append(stripped)
                continue
            if stripped:
                meaningful_lines.append(stripped)

        clean_text = " ".join(meaningful_lines)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = clean_text.strip()

        return title, clean_text

def scrape_urls_batch(
    urls: List[str],
    timeout: int = cfg.timeout_seconds,
    max_chars: int = cfg.max_char_limit,
) -> List[ScrapedPage]:

    scraper = WebScraper()
    results: List[ScrapedPage] = []

    for i, url in enumerate(urls):
        print(f"[WebScraper] Processing URL {i+1}/{len(urls)}")
        page = scraper.scrape_url(url, timeout=timeout, max_chars=max_chars)
        results.append(page)
        if not page.success:
            print(f"[WebScraper] Failed: {page.error[:100]}")

    successful = sum(1 for p in results if p.success)
    print(f"[WebScraper] Batch complete: {successful}/{len(urls)} pages scraped successfully.")
    return results
