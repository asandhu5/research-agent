"""
state.py — State Machine Schema & Typed Contracts
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY DOES STATE MATTER IN AGENTIC SYSTEMS?

    In a traditional program, data flows implicitly through a call stack.
    In a LangGraph state machine, every node receives the ENTIRE current state
    as input and returns a PARTIAL update. LangGraph merges the partial update
    into the canonical state object before passing it to the next node.

    This pattern (called "reducer-style state") means:
    1. Each node is a pure-ish function: input state → output update.
    2. You can inspect the full state at any point for debugging or telemetry.
    3. Conditional routing decisions are made by reading state fields, not by
       passing return values between functions.

WHY TypedDict (not a plain dict)?
    TypedDict gives Python's type-checker (mypy, Pyright) and your IDE
    autocomplete visibility into what keys exist. A plain dict["recall_memories"]
    silently returns None for a typo; TypedDict raises an error at edit-time.

WHY IMMUTABLE FLOW?
    Each node should ONLY mutate its own fields and leave others untouched.
    Nodes that overwrite unrelated fields create debugging nightmares —
    a node late in the pipeline would silently undo work done earlier.
    The convention: every node returns only the keys it explicitly computed.
"""

from typing import TypedDict, List, Optional, Dict, Any  # typing tools for strict schemas
from pydantic import BaseModel, Field, field_validator    # pydantic for sub-object validation


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SUB-MODELS
# These are nested structures that live inside ResearchState. We use Pydantic
# here (not TypedDict) because Pydantic can validate field values at assignment
# time, not just at parse time. E.g., a SourceDocument can enforce that `url`
# actually starts with "http".
# ─────────────────────────────────────────────────────────────────────────────

class SourceDocument(BaseModel):
    """
    Represents a single web source discovered during the search phase.

    Purpose:
        Gives us a validated, consistent structure for every search result.
        Without this, different search backends (Tavily vs DuckDuckGo) might
        return slightly different key names, causing KeyError crashes downstream.

    Fields:
        title   : Human-readable page title (used in the final report's citation list).
        url     : Full URL string (used to hyperlink citations in the Markdown report).
        snippet : Short excerpt from the page (used by the evaluator as a preview
                  before committing to full scraping).
    """

    title: str = Field(description="The webpage's title or headline")  # required, no default
    url: str = Field(description="Full canonical URL")                   # required, no default
    snippet: str = Field(default="", description="Short content preview") # optional, defaults to empty string

    @field_validator("url")  # pydantic v2 field validator — runs on every assignment
    @classmethod
    def url_must_be_valid(cls, v: str) -> str:
        """Ensures the URL is at least superficially valid before storage."""
        if not v.startswith(("http://", "https://")):
            # Gracefully coerce rather than crash — some search APIs return
            # protocol-relative URLs like "//example.com/page"
            return "https:" + v if v.startswith("//") else v
        return v  # pass-through if already valid


class ScrapedPage(BaseModel):
    """
    Represents the cleaned text content of one scraped URL.

    Purpose:
        After the web scraper fetches and cleans a page, it wraps the result
        in this model so downstream nodes always get the same keys regardless
        of whether scraping succeeded or failed.

    Fields:
        url     : The URL that was scraped.
        title   : Page title extracted from <title> tag.
        content : Cleaned, truncated body text (HTML tags removed).
        success : Whether scraping succeeded (False = HTTP error / timeout).
        error   : Error message if success=False, empty string otherwise.
    """

    url: str = Field(description="The URL that was scraped")
    title: str = Field(default="", description="HTML page title")
    content: str = Field(default="", description="Cleaned page body text")
    success: bool = Field(default=True, description="Whether scraping succeeded")
    error: str = Field(default="", description="Error message if scraping failed")


class MemoryRecord(BaseModel):
    """
    Represents a single memory retrieved from ChromaDB.

    Purpose:
        When the recall_memory_node fetches past research, it wraps each hit
        in this model. The `similarity_score` field lets nodes decide whether
        a memory is relevant enough to use or should be ignored.

    Fields:
        topic           : Short label for the memory (e.g., "RAG Hallucinations 2025").
        summary         : The stored research summary text.
        similarity_score: Cosine similarity between the current query and this memory.
                          1.0 = identical meaning, 0.0 = completely unrelated.
        metadata        : Arbitrary key-value pairs stored alongside the embedding
                          (e.g., timestamp, source_urls, iteration_count).
    """

    topic: str = Field(description="Short descriptive label for this memory")
    summary: str = Field(description="The stored research insight or summary")
    similarity_score: float = Field(
        default=0.0,
        ge=0.0,  # greater-than-or-equal-to 0
        le=1.0,  # less-than-or-equal-to 1
        description="Cosine similarity to the current query (0.0–1.0)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,  # default_factory prevents the mutable-default-argument bug
        description="Arbitrary metadata (timestamps, source URLs, etc.)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN STATE SCHEMA
# ResearchState is the single object that flows through every node in the graph.
# LangGraph passes it in, nodes return partial dicts, and LangGraph merges those
# updates back in before calling the next node.
#
# IMMUTABILITY CONVENTION:
#   Every node receives a ResearchState and must return a plain dict containing
#   ONLY the keys it modified. LangGraph shallow-merges this dict into the state.
#   Never return keys you didn't compute — you might accidentally reset another
#   node's work to its default value.
# ─────────────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict, total=False):
    """
    The canonical state object for the Research Agent state machine.

    WHY TypedDict INSTEAD OF dataclass?
        LangGraph requires state to be a TypedDict (or a subclass of it) because
        it inspects the class's __annotations__ to know which keys are valid and
        how to merge partial updates. A dataclass would require extra adapter code.

    total=False:
        Makes all fields optional at the TypedDict level. This is required because
        at graph startup, the state is initialized with just `question`. Nodes
        progressively populate other fields. If total=True, Python would demand all
        keys be present from the start.

    Field-by-field explanation follows below.
    """

    # ── INPUT ────────────────────────────────────────────────────────────────

    question: str
    # The raw user research query. This is the only field set at graph start.
    # ALL other nodes read this to stay on-topic. It never changes after
    # initialization — nodes never overwrite it.

    # ── MEMORY RECALL ────────────────────────────────────────────────────────

    recalled_memories: List[MemoryRecord]
    # Populated by recall_memory_node. Contains past research summaries that are
    # semantically similar to `question`. The planner_node reads this to avoid
    # re-researching topics the agent has already covered in previous sessions.
    # An empty list means this is a cold-start query with no prior context.

    # ── PLANNING ────────────────────────────────────────────────────────────

    plan: List[str]
    # Populated by planner_node. A list of 3–5 concrete search sub-queries
    # derived by decomposing `question`. For example, if question = "How does
    # GraphRAG reduce hallucinations?", the plan might be:
    #   ["GraphRAG architecture explained", "GraphRAG vs standard RAG comparison",
    #    "hallucination mitigation techniques in retrieval-augmented generation"]
    # Breaking the question into sub-queries improves web search precision.

    # ── SEARCH & SCRAPE ──────────────────────────────────────────────────────

    raw_search_results: List[SourceDocument]
    # Populated (and EXTENDED, not replaced) by search_and_scrape_node.
    # Each element is a validated SourceDocument. The list grows across
    # loop iterations — if the agent loops back to search a second time,
    # new results are appended rather than overwriting prior ones.

    scraped_content: List[ScrapedPage]
    # Populated (and EXTENDED) by search_and_scrape_node alongside raw_search_results.
    # Contains the cleaned text body of each scraped URL. The evaluator reads
    # this to judge whether we have enough information.

    # ── LOOP CONTROL ─────────────────────────────────────────────────────────

    iteration: int
    # Populated by search_and_scrape_node and incremented each loop pass.
    # The conditional edge in graph_builder.py checks: if iteration >= max_iterations,
    # stop looping regardless of evaluation_score. This is the circuit-breaker.
    # Starts at 0 and is incremented to 1 on the first search pass.

    # ── EVALUATION ───────────────────────────────────────────────────────────

    evaluation_score: float
    # Populated by evaluator_node. A float between 0.0 and 1.0 representing how
    # well the scraped content answers the original question. The conditional edge
    # routes back to search if this is below sufficiency_threshold (0.7 by default).
    # 0.0 = completely inadequate. 1.0 = fully comprehensive. 0.7 = "good enough".

    evaluation_reasoning: str
    # Populated by evaluator_node. A natural-language explanation of WHY the
    # evaluator assigned the score it did. Displayed in the Streamlit UI as
    # "gap analysis" — tells the user what information is still missing.

    missing_topics: List[str]
    # Populated by evaluator_node alongside evaluation_score/evaluation_reasoning.
    # Specific topics the evaluator identified as not yet covered. Consumed by
    # planner_node when the graph loops back to it (evaluate -> planner on an
    # insufficient score): the planner uses this to generate NEW sub-queries
    # that target the actual gaps, instead of the original code's behavior of
    # re-running the exact same queries every iteration (this field used to be
    # computed by the evaluator and then discarded — never read anywhere).

    # ── OUTPUT ───────────────────────────────────────────────────────────────

    final_report: str
    # Populated by synthesizer_node. A fully formatted Markdown string containing:
    #   - Executive summary paragraph
    #   - Multiple detailed sections with inline citations [1], [2], etc.
    #   - Numbered reference list at the bottom with titles and URLs
    # This is the primary deliverable the user reads.

    sources: List[SourceDocument]
    # Populated by synthesizer_node. The deduplicated, ordered list of sources
    # actually cited in final_report. Separate from raw_search_results because
    # the synthesizer may not cite every scraped page — only the most relevant.

    # ── MEMORY PERSISTENCE ───────────────────────────────────────────────────

    memory_stored: bool
    # Set to True by memory_storing_node after successfully persisting the report
    # summary to ChromaDB. Checked in the Streamlit UI to show a "Memory saved"
    # confirmation badge. Defaults to False if the node hasn't run yet.
