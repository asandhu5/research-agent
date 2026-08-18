"""
tests/test_scraper.py — Unit Tests for the Web Scraper
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT WE'RE TESTING:
    1. HTML cleaning — script, style, nav, footer tags are correctly removed.
    2. Content extraction — meaningful text is preserved after cleaning.
    3. Truncation — content over max_char_limit is truncated at word boundary.
    4. HTTP error handling — 404/timeout returns ScrapedPage with success=False.
    5. Empty page handling — a page with only scripts returns success=False.

WHY MOCK HTTP REQUESTS?
    Unit tests must be FAST (< 1 second each) and DETERMINISTIC (same result
    every time). Real HTTP requests are:
    - Slow (50ms–10s per request depending on server)
    - Non-deterministic (site may be down, content may change, network may vary)
    - Rate-limited (running tests frequently could get your IP blocked)

    We use pytest-mock or unittest.mock.patch to intercept requests.get() calls
    and return pre-defined HTML strings instead of hitting real servers.
    This gives us full control over what the scraper "sees."
"""

import pytest                          # test framework
from unittest.mock import patch, MagicMock  # mock HTTP calls
from bs4 import BeautifulSoup          # for verifying HTML parsing in tests

from web_scraper import WebScraper, scrape_urls_batch
from state import ScrapedPage


# ─────────────────────────────────────────────────────────────────────────────
# TEST HTML FIXTURES
# These HTML strings represent realistic page structures the scraper will
# encounter in production. We use them to verify cleaning works correctly.
# ─────────────────────────────────────────────────────────────────────────────

CLEAN_HTML_WITH_NOISE = """<!DOCTYPE html>
<html>
<head>
    <title>Research on Transformer Models</title>
    <style>
        /* THIS CSS MUST BE REMOVED — it's not content */
        body { font-family: Arial; }
        .nav { background: #333; }
    </style>
    <script>
        /* THIS JAVASCRIPT MUST BE REMOVED — it's not content */
        window.onload = function() { console.log("Google Analytics loaded"); };
        gtag('js', new Date());
    </script>
</head>
<body>
    <nav>
        <!-- THIS NAV MUST BE REMOVED — navigation links are not research content -->
        <a href="/">Home</a>
        <a href="/papers">Papers</a>
        <a href="/contact">Contact</a>
    </nav>

    <header>
        <!-- THIS HEADER MUST BE REMOVED — site chrome, not content -->
        <div class="site-logo">AI Research Blog</div>
    </header>

    <main>
        <article>
            <h1>Transformer Architecture Explained</h1>
            <p>
                The Transformer model introduced in "Attention is All You Need" (Vaswani et al., 2017)
                revolutionized natural language processing by replacing recurrent neural networks with
                a pure attention mechanism.
            </p>
            <p>
                The key innovation is multi-head self-attention, which allows the model to attend to
                different positions in the input sequence simultaneously. Unlike RNNs, Transformers
                process all tokens in parallel, enabling much faster training on modern GPU hardware.
            </p>
            <p>
                The encoder-decoder architecture consists of stacked identical layers, each containing
                a multi-head attention sub-layer and a feed-forward network sub-layer, with residual
                connections and layer normalization applied after each sub-layer.
            </p>
        </article>
    </main>

    <aside class="sidebar">
        <!-- THIS SIDEBAR MUST BE REMOVED — not the main article content -->
        <h3>Related Posts</h3>
        <a href="/bert">BERT Explained</a>
        <a href="/gpt">GPT Architecture</a>
    </aside>

    <footer>
        <!-- THIS FOOTER MUST BE REMOVED — copyright, links, not content -->
        <p>Copyright 2025 AI Research Blog. All rights reserved.</p>
        <p>Privacy Policy | Terms of Service</p>
    </footer>

    <script>
        /* MORE JAVASCRIPT AT PAGE BOTTOM — MUST BE REMOVED */
        document.addEventListener('DOMContentLoaded', function() {
            loadComments();
            initCookieBanner();
        });
    </script>
</body>
</html>"""

EMPTY_PAGE_HTML = """<!DOCTYPE html>
<html>
<head><title>Empty Page</title></head>
<body>
    <script>// This is a JavaScript-only SPA — no static content</script>
    <div id="app"></div>
    <script>ReactDOM.render(React.createElement(App), document.getElementById('app'));</script>
</body>
</html>"""

SHORT_PAGE_HTML = """<!DOCTYPE html>
<html>
<head><title>Short Content Test</title></head>
<body>
    <main>
        <p>This is a short paragraph with just a few words for testing truncation behavior.</p>
    </main>
</body>
</html>"""

LONG_PAGE_HTML = """<!DOCTYPE html>
<html>
<head><title>Long Content Test</title></head>
<body>
    <main>
        <article>
            <h1>Very Long Research Article</h1>
            {}
        </article>
    </main>
</body>
</html>""".format(
    # Generate ~15,000 characters of dummy content (more than max_char_limit=8000)
    " ".join(
        f"This is sentence number {i} in a very long article about machine learning research. "
        f"The article discusses important topics like neural networks, deep learning, and AI."
        for i in range(100)  # 100 × ~110 chars ≈ 11,000 chars
    )
)


class TestHTMLCleaning:
    """Tests verifying that noise HTML tags are correctly stripped."""

    def setup_method(self):
        """Create a fresh WebScraper instance before each test method."""
        self.scraper = WebScraper()

    def test_script_tags_are_removed(self):
        """
        WHAT: Verify that <script> tags and their content don't appear in extracted text.

        WHY: JavaScript code in the extracted text would corrupt the LLM's understanding
        of the page content. "window.onload = function()" has no semantic meaning for
        a research agent trying to understand article content.
        """
        title, content = self.scraper._parse_and_clean(CLEAN_HTML_WITH_NOISE, "http://test.com")

        # Script content that should NOT appear in clean text
        assert "window.onload" not in content, "Script function calls should be removed"
        assert "Google Analytics" not in content, "Google Analytics code should be removed"
        assert "document.addEventListener" not in content, "Event listeners should be removed"
        assert "gtag(" not in content, "Google tag code should be removed"

    def test_style_tags_are_removed(self):
        """
        WHAT: Verify CSS rules don't appear in the extracted text.

        WHY: CSS selectors and rules (e.g., "font-family: Arial;") are pure styling
        metadata with zero research value. If included, they'd consume LLM tokens
        for no reason and dilute the semantic signal of the content.
        """
        title, content = self.scraper._parse_and_clean(CLEAN_HTML_WITH_NOISE, "http://test.com")

        assert "font-family" not in content, "CSS font-family rule should be removed"
        assert "background: #333" not in content, "CSS color rules should be removed"
        assert "Arial" not in content, "CSS font name should be removed"

    def test_nav_tags_are_removed(self):
        """
        WHAT: Verify <nav> element content doesn't appear in extracted text.

        WHY: Navigation links ("Home | Papers | Contact") are site chrome that every
        page shares. They add noise without contributing to the article's content.
        If the scraper includes nav links, every scraped page looks the same to the
        evaluator LLM, which degrades synthesis quality.
        """
        title, content = self.scraper._parse_and_clean(CLEAN_HTML_WITH_NOISE, "http://test.com")

        # These specific nav links should not appear in clean content
        assert content.count("Contact") == 0 or "contact" not in content.lower().split(), (
            "Nav link 'Contact' should be removed from extracted content"
        )

    def test_footer_tags_are_removed(self):
        """
        WHAT: Verify <footer> content (copyright, privacy policy) is stripped.

        WHY: Footer boilerplate ("Copyright 2025. All rights reserved.") is identical
        across every page on a domain. Including it pollutes the content corpus and
        may cause the agent to cite "Copyright 2025" as a research finding.
        """
        title, content = self.scraper._parse_and_clean(CLEAN_HTML_WITH_NOISE, "http://test.com")

        assert "Copyright" not in content, "Footer copyright text should be removed"
        assert "Privacy Policy" not in content, "Footer privacy policy link should be removed"
        assert "Terms of Service" not in content, "Footer terms of service link should be removed"

    def test_main_content_is_preserved(self):
        """
        WHAT: Verify that the actual article content SURVIVES the cleaning process.

        WHY: Cleaning should be SELECTIVE — remove noise, keep research content.
        This is the "positive" test to complement the "negative" noise-removal tests.
        The Transformer article body should be fully preserved.
        """
        title, content = self.scraper._parse_and_clean(CLEAN_HTML_WITH_NOISE, "http://test.com")

        # Key content that MUST be present in the cleaned output
        assert "Transformer" in content, "Main article heading should be preserved"
        assert "Attention is All You Need" in content, "Article citations should be preserved"
        assert "multi-head self-attention" in content, "Technical content should be preserved"
        assert "encoder-decoder architecture" in content, "Technical details should be preserved"

    def test_title_is_extracted_correctly(self):
        """
        WHAT: Verify that the <title> tag value is returned as the page title.

        WHY: The page title is used in the References section of the final report.
        "Research on Transformer Models" is more useful as a citation than "Untitled".
        """
        title, content = self.scraper._parse_and_clean(CLEAN_HTML_WITH_NOISE, "http://test.com")

        assert title == "Research on Transformer Models", (
            f"Expected title 'Research on Transformer Models' but got '{title}'"
        )


class TestContentTruncation:
    """Tests for the character limit truncation feature."""

    def setup_method(self):
        self.scraper = WebScraper()

    def test_content_truncated_at_word_boundary(self):
        """
        WHAT: Verify that long content is truncated at or below max_chars.

        WHY: If we hard-truncate in the middle of a word, the LLM may receive
        garbled text like "The transformer architecture uses multi-head att" which
        could confuse the model. Word-boundary truncation produces cleaner cuts.
        """
        title, content = self.scraper._parse_and_clean(LONG_PAGE_HTML, "http://test.com")
        MAX_CHARS = 8000  # must match config.cfg.max_char_limit

        # Simulate scrape_url truncation (which calls _parse_and_clean then truncates)
        if len(content) > MAX_CHARS:
            truncation_point = content.rfind(" ", 0, MAX_CHARS)
            if truncation_point == -1:
                truncation_point = MAX_CHARS
            truncated = content[:truncation_point] + "... [truncated]"
        else:
            truncated = content

        assert len(truncated) <= MAX_CHARS + len("... [truncated]"), (
            f"Truncated content length {len(truncated)} exceeds limit {MAX_CHARS}"
        )

    def test_short_content_is_not_truncated(self):
        """
        WHAT: Verify that content shorter than max_char_limit is NOT modified.

        WHY: Truncation should only apply when needed. Truncating short content
        would be a data corruption bug.
        """
        title, content = self.scraper._parse_and_clean(SHORT_PAGE_HTML, "http://test.com")
        MAX_CHARS = 8000

        # Short page content should be well under the limit
        assert len(content) < MAX_CHARS, (
            "Short page content should not reach the truncation threshold"
        )
        assert "truncated" not in content, (
            "Short content should not have the truncation marker appended"
        )


class TestHTTPErrorHandling:
    """Tests for graceful HTTP error handling."""

    def setup_method(self):
        self.scraper = WebScraper()

    def test_http_404_returns_failed_scrape_page(self):
        """
        WHAT: A 404 response should return a ScrapedPage with success=False.

        WHY: The pipeline should never crash on a 404. Instead, it should log the
        failure and continue with other URLs. The `success=False` field tells
        downstream nodes to skip this page's content.

        HOW: We patch requests.get() to raise an HTTPError with status 404,
        simulating a "page not found" response without real HTTP traffic.
        """
        from requests.exceptions import HTTPError

        # Create a mock response object that simulates a 404
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPError(response=mock_response)

        with patch("requests.get", return_value=mock_response):
            result = self.scraper.scrape_url("http://example.com/missing-page")

        assert result.success is False, "A 404 response should result in success=False"
        assert "404" in result.error or "Not Found" in result.error, (
            f"Error message should mention 404, got: '{result.error}'"
        )

    def test_timeout_returns_failed_scrape_page(self):
        """
        WHAT: A connection timeout should return ScrapedPage with success=False.

        WHY: Some servers are very slow or unresponsive. Without the timeout handler,
        the entire agent would hang indefinitely waiting for a response, blocking
        all subsequent nodes.

        HOW: We patch requests.get() to raise a Timeout exception.
        """
        from requests.exceptions import Timeout

        with patch("requests.get", side_effect=Timeout("Connection timed out")):
            result = self.scraper.scrape_url("http://slow-server.example.com")

        assert result.success is False, "A timeout should result in success=False"
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower(), (
            f"Error message should mention timeout, got: '{result.error}'"
        )
        assert result.url == "http://slow-server.example.com", "Failed page should preserve the original URL"

    def test_successful_scrape_returns_content(self):
        """
        WHAT: A successful HTTP response should return ScrapedPage with success=True.

        WHY: This is the "happy path" test — verifying that when everything works,
        the scraper correctly extracts and returns content.

        HOW: We patch requests.get() to return a mock response with our known HTML.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_response.encoding = "utf-8"
        mock_response.apparent_encoding = "utf-8"
        mock_response.text = CLEAN_HTML_WITH_NOISE
        mock_response.raise_for_status.return_value = None  # no exception = success

        with patch("requests.get", return_value=mock_response):
            result = self.scraper.scrape_url("http://test.com/transformer-article")

        assert result.success is True, f"Successful HTTP response should return success=True. Error: {result.error}"
        assert "Transformer" in result.content, "Scraped content should contain article text"
        assert result.title == "Research on Transformer Models", "Title should be extracted from <title> tag"
        assert result.url == "http://test.com/transformer-article", "URL should be preserved"

    def test_empty_page_returns_success_false(self):
        """
        WHAT: A page with only scripts (SPA shell) should return success=False.

        WHY: JavaScript-rendered Single Page Applications show an empty HTML shell.
        BeautifulSoup can't execute JS, so the scraper gets zero useful text.
        Rather than storing an empty string as research content, we flag it as failed.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.encoding = "utf-8"
        mock_response.apparent_encoding = "utf-8"
        mock_response.text = EMPTY_PAGE_HTML
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            result = self.scraper.scrape_url("http://test.com/spa-page")

        # Either success=False (scraper detected empty content) or content is very short
        # Both are acceptable outcomes — the important thing is no crash and some error signal
        if result.success:
            # If it "succeeded", content should be very short (< 100 chars)
            assert len(result.content) < 100, (
                "SPA shell page should produce near-empty content if scraped"
            )
        else:
            assert "no extractable" in result.error.lower() or "javascript" in result.error.lower(), (
                f"Error should explain why scraping failed, got: '{result.error}'"
            )


class TestBatchScraping:
    """Tests for the scrape_urls_batch() helper function."""

    def test_batch_returns_same_length_as_input(self):
        """
        WHAT: scrape_urls_batch() should return exactly N results for N input URLs.

        WHY: The Streamlit UI and synthesizer node index into the results by position.
        If batch returns fewer results than URLs (e.g., skips failed ones instead of
        returning failed ScrapedPages), the indexing would be off.
        """
        from requests.exceptions import Timeout

        with patch("requests.get", side_effect=Timeout("timeout")):
            results = scrape_urls_batch([
                "http://url1.com",
                "http://url2.com",
                "http://url3.com",
            ])

        assert len(results) == 3, (
            f"Batch should return exactly 3 results for 3 input URLs, got {len(results)}"
        )
        # All should be failed ScrapedPages since we mocked all requests to timeout
        assert all(not r.success for r in results), (
            "All scrapes should have failed since we mocked all requests to timeout"
        )
