"""
agent_nodes.py — LangGraph Execution Nodes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS A NODE IN LANGGRAPH?

    A node is a plain Python function with the signature:
        def node_name(state: ResearchState) -> dict

    LangGraph calls the function, passes the current full state, and expects
    back a PARTIAL dict containing only the keys this node modified.
    LangGraph then shallow-merges that dict into the canonical state.

    This means nodes are nearly pure functions:
        - They READ from the full state.
        - They WRITE only their designated output keys.
        - They never mutate the input state object directly.

HOW PROMPT ENGINEERING FORCES STRUCTURED JSON RESPONSES:

    LLMs by default respond in free-form text. For structured workflows, we need
    machine-parseable output. We achieve this by:

    1. Telling the LLM exactly what format to return in the system prompt.
    2. Using LangChain's `with_structured_output(PydanticModel)` wrapper, which
       instructs the LLM to return JSON matching a Pydantic schema and validates
       the response automatically.
    3. For simpler cases, we ask for JSON in the prompt and parse it manually
       with a fallback so a malformed response doesn't crash the pipeline.

HOW TEMPERATURE AFFECTS EACH NODE:

    - Planner (temperature=0.1): We want DETERMINISTIC sub-queries. If you run
      the same question twice, you should get consistent search terms. High
      temperature makes the planner generate wildly different sub-queries each
      run, making the system non-reproducible.

    - Evaluator (temperature=0.0): Evaluation is a factual judgment. We want
      the same content to always receive the same score. Zero temperature makes
      this completely deterministic.

    - Synthesizer (temperature=0.3): Report writing benefits from slightly more
      natural, varied prose. 0.3 adds just enough variety to avoid robotic-
      sounding sentences while keeping citations accurate.
"""

import json                                      # parsing LLM JSON responses
import re                                        # extracting JSON from markdown code blocks
from typing import List, Dict, Any, Optional     # type hints
from datetime import datetime                    # timestamping stored memories

from langchain_openai import ChatOpenAI          # LangChain's OpenAI chat wrapper
from langchain_core.messages import (
    SystemMessage,   # the "persona" message (instructions to the LLM)
    HumanMessage,    # the "user turn" message (the actual input)
)
from pydantic import BaseModel, Field            # structured output schemas
from dotenv import load_dotenv                   # API key loading

from config import cfg                           # hyperparameters
from state import ResearchState, SourceDocument, ScrapedPage, MemoryRecord
from memory_manager import MemoryManager         # vector store interface
from search_engine import multi_search           # multi-query search helper
from web_scraper import scrape_urls_batch        # batch URL scraper

load_dotenv()  # ensures OPENAI_API_KEY is in os.environ before ChatOpenAI initializes


# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT FACTORY
# We use a module-level factory function rather than a global singleton because
# LangChain's ChatOpenAI validates the API key at construction time. If the key
# isn't set yet when the module is imported, a global would crash at import.
# A factory defers construction to first call, by which time load_dotenv() has run.
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm(temperature: float = cfg.llm_temperature) -> ChatOpenAI:
    """
    Build and return a configured ChatOpenAI instance.

    We create a new instance per call because different nodes may need different
    temperatures. ChatOpenAI is stateless — creating multiple instances is cheap.

    Args:
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).

    Returns:
        A configured ChatOpenAI client ready for invocation.
    """
    return ChatOpenAI(
        model=cfg.llm_model_name,   # "gpt-4o"
        temperature=temperature,     # caller-specified creativity level
        max_tokens=cfg.llm_max_tokens, # cap response length to prevent runaway costs
    )


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS FOR STRUCTURED LLM OUTPUTS
# These schemas are passed to with_structured_output() to force the LLM to
# return JSON that matches them. LangChain automatically retries on parse errors.
# ─────────────────────────────────────────────────────────────────────────────

class PlannerOutput(BaseModel):
    """Schema for the planner node's structured response."""
    sub_queries: List[str] = Field(
        description="List of 3-5 specific web search queries to research the question",
        min_length=1,
        max_length=5,
    )
    reasoning: str = Field(
        description="Brief explanation of why these sub-queries cover the question"
    )


class EvaluatorOutput(BaseModel):
    """Schema for the evaluator node's structured response."""
    score: float = Field(
        description="Sufficiency score from 0.0 (inadequate) to 1.0 (comprehensive)",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="Explanation of what information is present and what gaps remain"
    )
    missing_topics: List[str] = Field(
        default_factory=list,
        description="List of specific topics not yet covered by the current research"
    )


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1: RECALL MEMORY
# ─────────────────────────────────────────────────────────────────────────────

def recall_memory_node(state: ResearchState) -> dict:
    """
    Query ChromaDB for past research relevant to the current question.

    WHY THIS IS THE FIRST NODE:
        Before spending API tokens on web search, we check if we've already
        researched this topic. If similar summaries exist in memory, the planner
        can leverage them to ask more targeted sub-queries or skip entire
        sub-topics we've already covered.

        This is analogous to a researcher checking their own notes before
        going to the library — saves time and builds cumulative knowledge.

    State reads:  question
    State writes: recalled_memories

    Args:
        state: Current ResearchState (only `question` is required here).

    Returns:
        Partial state dict with `recalled_memories` key populated.
    """
    question = state["question"]  # the original user query — our search anchor
    print(f"\n{'='*60}")
    print(f"[Node: recall_memory] Recalling past research for:")
    print(f"  '{question[:100]}'")
    print(f"{'='*60}")

    memory_manager = MemoryManager()  # connects to or creates the ChromaDB collection

    # Query the vector store for semantically similar past research.
    # top_k_recall=3 means at most 3 past summaries are retrieved.
    recalled = memory_manager.recall_memories(
        query=question,
        top_k=cfg.top_k_recall,
        similarity_threshold=cfg.similarity_threshold,
    )

    if recalled:
        print(f"[recall_memory] Found {len(recalled)} relevant memory/memories:")
        for m in recalled:
            print(f"  - '{m.topic}' (similarity={m.similarity_score:.2f})")
    else:
        print("[recall_memory] No relevant memories found — proceeding with cold search.")

    # Return only the keys this node computed.
    # LangGraph merges this into the full state; other keys are untouched.
    return {"recalled_memories": recalled}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2: PLANNER
# ─────────────────────────────────────────────────────────────────────────────

REPLAN_SYSTEM_PROMPT = """You are an expert research strategist refining a research plan based on
evaluator feedback from a prior search pass.

The queries already tried did not sufficiently cover the research question. Your task is to
generate 2-4 NEW search queries that specifically target the topics identified as missing —
complementing, not duplicating, what has already been searched for.

Rules:
- Do NOT repeat or closely rephrase any of the already-tried queries listed below
- Each new query must directly target one of the missing topics, not the question in general
- Each query must be specific and concrete enough for a search engine
- Prefer queries likely to return technical articles, research papers, and authoritative sources
- Return ONLY valid JSON matching the required schema"""


def planner_node(state: ResearchState) -> dict:
    """
    Decompose the user's research question into targeted web search sub-queries —
    or, when the graph has looped back here after an insufficient evaluation,
    generate a NEW set of queries that specifically target what the evaluator
    said was missing.

    WHY DECOMPOSE THE QUESTION?
        A complex question like "How does GraphRAG address RAG hallucinations?"
        maps poorly to a single search query. Search engines return best results
        for specific, concrete queries. Decomposing into:
          1. "GraphRAG architecture knowledge graph construction"
          2. "GraphRAG vs standard RAG hallucination comparison"
          3. "community summarization GraphRAG global queries"
        ... retrieves more relevant, complementary pages than one broad query.

    WHY THIS NODE IS ALSO THE REPLAN STEP:
        graph_builder.py routes evaluate -> planner (not straight back to
        search_scrape) whenever the evaluator's score is insufficient. Without
        that, search_and_scrape_node would re-run the exact same `plan` list
        it was given the first time -- Tavily/DuckDuckGo are close to
        deterministic for identical queries, so a second identical search
        mostly just rediscovers the same URLs (search_and_scrape_node already
        filters out URLs it's seen before), making the "search again" loop
        pointless. `iteration > 0` is what distinguishes a replan pass here
        from the initial one: iteration is only ever incremented by
        search_and_scrape_node, so it can only be nonzero if this node has
        already produced a plan once before and the loop came back around.

    MEMORY INTEGRATION:
        On the INITIAL pass, if recalled_memories exists, we inject them into
        the planner prompt so the LLM can avoid generating sub-queries for
        topics already covered by past research.

    State reads:  question, recalled_memories, iteration, plan (as "already tried"
                  context on a replan pass), missing_topics, evaluation_reasoning
    State writes: plan

    Args:
        state: Current ResearchState.

    Returns:
        Partial state dict with `plan` key (list of sub-query strings).
    """
    question = state["question"]
    recalled = state.get("recalled_memories", [])
    iteration = state.get("iteration", 0)
    previous_queries = state.get("plan", [])
    missing_topics = state.get("missing_topics", [])
    evaluation_reasoning = state.get("evaluation_reasoning", "")

    is_replan = iteration > 0

    if is_replan:
        print(f"\n[Node: planner] Replanning after iteration {iteration} (evaluator found gaps)...")

        tried_block = "\n".join(f"- {q}" for q in previous_queries) if previous_queries else "(none recorded)"
        missing_block = (
            "\n".join(f"- {t}" for t in missing_topics)
            if missing_topics
            else "(evaluator did not list specific gaps -- broaden coverage of angles not yet tried)"
        )

        system_prompt = REPLAN_SYSTEM_PROMPT
        human_prompt = f"""Research Question: {question}

Already-tried search queries (do not repeat or closely rephrase these):
{tried_block}

Missing topics identified by the evaluator:
{missing_block}

Evaluator's reasoning: {evaluation_reasoning}

Generate 2-4 new, targeted search queries that specifically address the missing topics above."""

        # If the replan LLM call itself fails, fall back to using the missing
        # topics directly as queries (still a real, different search from what
        # was already tried) rather than the original question again, which
        # would just repeat the first pass's own fallback.
        fallback_queries = missing_topics[:4] if missing_topics else [
            f"{question} additional sources",
            f"{question} in-depth analysis",
        ]
    else:
        print(f"\n[Node: planner] Decomposing question into sub-queries...")

        # ── BUILD MEMORY CONTEXT BLOCK ────────────────────────────────────────
        # If we have past memories, format them for injection into the prompt.
        # This tells the LLM what we already know so it can focus on gaps.
        memory_context = ""
        if recalled:
            memory_parts = []
            for mem in recalled:
                # Truncate long summaries to keep the prompt within token limits
                summary_preview = mem.summary[:400] + "..." if len(mem.summary) > 400 else mem.summary
                memory_parts.append(f"PAST RESEARCH — '{mem.topic}' (relevance={mem.similarity_score:.2f}):\n{summary_preview}")
            memory_context = "\n\n".join(memory_parts)
            memory_context = f"\n\n## EXISTING KNOWLEDGE FROM PAST RESEARCH:\n{memory_context}\n\nUse this context to avoid re-researching covered topics."

        # ── SYSTEM PROMPT ────────────────────────────────────────────────────
        # Key prompt engineering techniques used here:
        #   1. Role assignment ("expert research strategist") — improves focus.
        #   2. Explicit output format specification (JSON with exact keys).
        #   3. Constraints (3-5 queries, specific not generic).
        #   4. Context injection (past memories).
        system_prompt = """You are an expert research strategist for an autonomous web research agent.

Your task is to decompose a complex research question into 3-5 specific, concrete search queries
that can be directly entered into a search engine.

Rules:
- Each sub-query must be specific enough for a search engine (not vague like "learn about RAG")
- Cover different aspects/angles of the question for comprehensive coverage
- If past research context is provided, generate queries that fill knowledge GAPS, not repeat known info
- Prefer queries likely to return technical articles, research papers, and authoritative sources
- Return ONLY valid JSON matching the required schema"""

        human_prompt = f"""Research Question: {question}{memory_context}

Generate 3-5 specific web search sub-queries to comprehensively research this question."""

        fallback_queries = [
            question,                        # use the original question as-is
            f"{question} explained",         # request explanation
            f"{question} recent developments 2025",  # request recent info
        ]

    # ── LLM CALL WITH STRUCTURED OUTPUT ─────────────────────────────────────
    # with_structured_output(PlannerOutput) tells LangChain to:
    #   1. Add instructions to the LLM to respond in JSON matching the Pydantic schema.
    #   2. Parse the response and validate it against PlannerOutput.
    #   3. Retry once if the response doesn't parse correctly.
    llm = _get_llm(temperature=cfg.llm_temperature)  # 0.1 for deterministic planning
    structured_llm = llm.with_structured_output(PlannerOutput)  # attach output parser

    try:
        result: PlannerOutput = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])

        sub_queries = result.sub_queries
        print(f"[planner] Generated {len(sub_queries)} sub-queries:")
        for i, q in enumerate(sub_queries, 1):
            print(f"  {i}. {q}")
        print(f"[planner] Reasoning: {result.reasoning[:150]}")

    except Exception as e:
        # Structured output failed (malformed JSON, API error, etc.)
        # Fall back to a simple decomposition so the agent can continue.
        print(f"[planner] Structured output failed: {e}. Using fallback plan.")
        sub_queries = fallback_queries

    return {"plan": sub_queries}  # return only the plan key


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3: SEARCH AND SCRAPE
# ─────────────────────────────────────────────────────────────────────────────

def search_and_scrape_node(state: ResearchState) -> dict:
    """
    Execute web searches for all planned sub-queries and scrape the top results.

    WHAT HAPPENS IN THIS NODE:
        1. Takes the plan (list of sub-queries) from the previous node.
        2. Runs multi_search() which queries Tavily or DuckDuckGo for each sub-query.
        3. Deduplicates URLs so we don't scrape the same page twice.
        4. Fetches and cleans the HTML content of each discovered URL.
        5. Appends new results to the existing lists (important for loop iterations!).

    WHY APPEND INSTEAD OF REPLACE?
        The agent may loop multiple times (search → evaluate → search again).
        On loop iteration 2, we don't want to throw away the URLs from iteration 1.
        By appending, the evaluator on iteration 2 can see ALL accumulated content.

    State reads:  plan, raw_search_results (existing), scraped_content (existing), iteration
    State writes: raw_search_results, scraped_content, iteration

    Args:
        state: Current ResearchState with plan populated.

    Returns:
        Partial state dict updating raw_search_results, scraped_content, iteration.
    """
    plan = state.get("plan", [])
    existing_search = state.get("raw_search_results", [])   # prior iterations' results
    existing_scraped = state.get("scraped_content", [])      # prior iterations' scrapes
    current_iteration = state.get("iteration", 0)            # default 0 = not yet run

    new_iteration = current_iteration + 1  # increment loop counter for this pass
    print(f"\n[Node: search_and_scrape] Iteration {new_iteration} — Searching {len(plan)} sub-queries")

    # ── STEP 1: MULTI-SEARCH ─────────────────────────────────────────────────
    # Run all planned sub-queries through the search engine.
    # multi_search() deduplicates URLs WITHIN this batch.
    new_search_results = multi_search(
        queries=plan,
        max_results_per_query=cfg.max_search_results,
    )

    # Filter out URLs we already scraped in previous iterations to avoid re-scraping
    existing_urls = {page.url for page in existing_scraped}  # set for O(1) lookup
    fresh_search_results = [
        doc for doc in new_search_results
        if doc.url not in existing_urls  # only keep URLs we haven't seen before
    ]

    print(f"[search_and_scrape] {len(new_search_results)} search results, "
          f"{len(fresh_search_results)} new URLs to scrape.")

    # ── STEP 2: SCRAPE NEW URLS ──────────────────────────────────────────────
    if fresh_search_results:
        urls_to_scrape = [doc.url for doc in fresh_search_results]
        new_scraped = scrape_urls_batch(urls_to_scrape)
    else:
        new_scraped = []  # nothing new to scrape
        print("[search_and_scrape] No new URLs to scrape — all already seen in prior iterations.")

    # ── STEP 3: ACCUMULATE RESULTS ───────────────────────────────────────────
    # APPEND new results to existing (don't replace — we want cumulative content)
    combined_search = existing_search + fresh_search_results
    combined_scraped = existing_scraped + new_scraped

    # Count successful scrapes for status display
    successful_scrapes = sum(1 for p in new_scraped if p.success)
    total_chars = sum(len(p.content) for p in combined_scraped if p.success)

    print(f"[search_and_scrape] New scrapes: {successful_scrapes}/{len(new_scraped)} successful")
    print(f"[search_and_scrape] Total accumulated content: {total_chars:,} characters")

    return {
        "raw_search_results": combined_search,     # accumulated across all iterations
        "scraped_content": combined_scraped,        # accumulated across all iterations
        "iteration": new_iteration,                 # updated loop counter
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4: EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

def evaluator_node(state: ResearchState) -> dict:
    """
    Assess whether the accumulated research is sufficient to answer the question.

    HOW THE EVALUATOR WORKS:
        The LLM acts as a "research quality judge." It reads:
        1. The original research question.
        2. The titles and snippet previews of all scraped pages.
        3. The full text of the most content-rich scraped pages.

        It then rates the content from 0.0 to 1.0:
        - 0.0–0.4: Critically insufficient (major aspects of question not covered)
        - 0.4–0.7: Partially sufficient (some aspects covered but gaps remain)
        - 0.7–1.0: Sufficient (most aspects covered; can write a complete report)

        The conditional edge in graph_builder.py uses this score to decide whether
        to loop back for more searching or proceed to synthesis.

    WHY NOT JUST COUNT SOURCES?
        More sources ≠ more relevant information. 10 search results about Python
        syntax are useless if the question is about GraphRAG. The LLM evaluates
        semantic relevance, not quantity.

    State reads:  question, scraped_content, raw_search_results, iteration
    State writes: evaluation_score, evaluation_reasoning

    Args:
        state: Current ResearchState with scraped content populated.

    Returns:
        Partial state dict with evaluation_score and evaluation_reasoning.
    """
    question = state["question"]
    scraped = state.get("scraped_content", [])
    iteration = state.get("iteration", 1)

    print(f"\n[Node: evaluator] Evaluating research quality (iteration {iteration})...")

    # If no content was scraped, immediately score 0.0 to trigger another search loop
    if not scraped or not any(p.success for p in scraped):
        print("[evaluator] No scraped content available — scoring 0.0 (no information).")
        return {
            "evaluation_score": 0.0,
            "evaluation_reasoning": "No web content was successfully scraped. All requests failed.",
            "missing_topics": [],
        }

    # ── BUILD CONTENT SUMMARY FOR LLM ────────────────────────────────────────
    # We don't pass ALL scraped text to the evaluator (that would be 40,000+ chars).
    # Instead, we build a structured summary: titles + first 500 chars of each page.
    # This fits in a reasonable context window and still gives the LLM enough to judge.

    content_summary_parts = []
    for i, page in enumerate(scraped, 1):
        if not page.success or not page.content:
            continue  # skip failed scrapes
        # Show title and first 500 characters of cleaned content
        preview = page.content[:500].strip()
        content_summary_parts.append(
            f"[Source {i}] {page.title or page.url[:60]}\n{preview}..."
        )

    content_summary = "\n\n".join(content_summary_parts)

    # Truncate the overall content summary to ~6000 chars to stay within token budget
    if len(content_summary) > 6000:
        content_summary = content_summary[:6000] + "\n... [additional sources truncated for evaluation]"

    # ── SYSTEM PROMPT ────────────────────────────────────────────────────────
    system_prompt = """You are a rigorous research quality evaluator for an autonomous web research agent.

Your job is to assess whether the collected web research is sufficient to answer the research question
comprehensively and accurately.

Scoring Guide:
- 0.0-0.3: Critically insufficient — core concepts not covered, misleading or irrelevant sources
- 0.3-0.5: Marginally insufficient — some relevant info but major aspects missing
- 0.5-0.7: Partially sufficient — covers 50-70% of what's needed, identifiable gaps remain
- 0.7-0.85: Sufficient — covers most aspects, minor gaps but could write a complete report
- 0.85-1.0: Highly sufficient — comprehensive coverage, multiple perspectives, recent sources

Be strict: only score above 0.7 if the CORE question can be meaningfully answered."""

    human_prompt = f"""Research Question: {question}

Iteration: {iteration} of {cfg.max_iterations} maximum

Collected Research (titles and previews):
{content_summary}

Evaluate the sufficiency of this research and return a structured assessment."""

    # ── LLM CALL ─────────────────────────────────────────────────────────────
    llm = _get_llm(temperature=0.0)  # temperature=0.0 for fully deterministic evaluation
    structured_llm = llm.with_structured_output(EvaluatorOutput)

    try:
        result: EvaluatorOutput = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])

        score = result.score
        reasoning = result.reasoning
        missing = result.missing_topics

        print(f"[evaluator] Score: {score:.2f} | "
              f"{'✓ Sufficient' if score >= cfg.sufficiency_threshold else '✗ Insufficient'}")
        print(f"[evaluator] Reasoning: {reasoning[:200]}")
        if missing:
            print(f"[evaluator] Missing topics: {', '.join(missing[:3])}")

    except Exception as e:
        # Evaluation failed — use a safe fallback score that depends on iteration
        # If we're on the last iteration, assume we have enough (force synthesis)
        fallback_score = 0.8 if iteration >= cfg.max_iterations else 0.4
        print(f"[evaluator] Structured evaluation failed: {e}. Using fallback score: {fallback_score}")
        score = fallback_score
        reasoning = f"Evaluation model call failed ({type(e).__name__}). Proceeding with available content."
        missing = []

    return {
        "evaluation_score": score,
        "evaluation_reasoning": reasoning,
        # Read by planner_node the next time the graph loops back to it, so a
        # low score actually changes what gets searched for next instead of
        # the identical queries being re-run (missing_topics used to be
        # computed here and then never read anywhere).
        "missing_topics": missing,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5: SYNTHESIZER
# ─────────────────────────────────────────────────────────────────────────────

def synthesizer_node(state: ResearchState) -> dict:
    """
    Generate a comprehensive, cited Markdown research report.

    WHAT A GOOD SYNTHESIS LOOKS LIKE:
        The synthesizer acts as a technical writer with access to all scraped sources.
        It produces a structured Markdown report with:
        - An executive summary (3–5 sentences) suitable for a non-expert reader.
        - Multiple topical sections, each covering one angle of the question.
        - Inline citations in [1], [2] format referencing the numbered source list.
        - A References section with titles and hyperlinked URLs.

    PROMPT ENGINEERING TECHNIQUES USED:
        1. "Role + Task" framing — "You are a senior research analyst..."
        2. Output format specification — exact Markdown structure with examples.
        3. Source injection — numbered source list so the LLM can cite by number.
        4. Explicit citation instruction — "you MUST cite at least one source per claim."

    WHY HIGHER TEMPERATURE FOR SYNTHESIS?
        Report writing benefits from natural prose variation. Temperature 0.3 makes
        the text readable and non-robotic while staying accurate. Planning and
        evaluation use lower temperature because those require consistent outputs.

    State reads:  question, scraped_content, raw_search_results, recalled_memories
    State writes: final_report, sources

    Args:
        state: Current ResearchState with all scraped content accumulated.

    Returns:
        Partial state dict with final_report (Markdown string) and sources list.
    """
    question = state["question"]
    scraped = state.get("scraped_content", [])
    search_results = state.get("raw_search_results", [])
    recalled = state.get("recalled_memories", [])

    print(f"\n[Node: synthesizer] Generating research report...")

    # ── BUILD NUMBERED SOURCE LIST ────────────────────────────────────────────
    # Collect successful scrapes and build a numbered citation index.
    # The LLM will reference these as [1], [2], [3] in the report.
    sources_for_report = []
    content_blocks = []

    for i, page in enumerate(scraped, 1):
        if page.success and page.content:
            sources_for_report.append(SourceDocument(
                title=page.title or f"Source {i}",
                url=page.url,
                snippet=page.content[:200],  # short snippet for the references section
            ))
            # Full content block for the LLM to synthesize from
            content_block = (
                f"[SOURCE {i}] Title: {page.title or 'Untitled'}\n"
                f"URL: {page.url}\n"
                f"Content:\n{page.content[:2000]}\n"  # limit per-source chars
            )
            content_blocks.append(content_block)

    if not content_blocks:
        # No content to synthesize — produce a minimal report explaining the failure
        print("[synthesizer] No successful scrapes — producing fallback report.")
        fallback_report = (
            f"# Research Report: {question}\n\n"
            "**Note:** The agent was unable to retrieve web content for this query. "
            "This may be due to network restrictions, rate limits, or the absence of API keys.\n\n"
            "## Recommendations\n"
            "- Check your internet connection and API keys in `.env`.\n"
            "- Try running `python download_sample_data.py` to test with local data.\n"
        )
        return {"final_report": fallback_report, "sources": []}

    # ── INJECT PRIOR MEMORY CONTEXT ──────────────────────────────────────────
    memory_section = ""
    if recalled:
        mem_parts = [
            f"PRIOR RESEARCH — '{m.topic}':\n{m.summary[:600]}"
            for m in recalled
        ]
        memory_section = "\n\n## PRIOR RESEARCH CONTEXT (from memory):\n" + "\n\n".join(mem_parts)

    # Join all content blocks into one large context block for the LLM
    all_content = "\n\n" + "─"*40 + "\n\n".join(content_blocks)

    # ── SYSTEM PROMPT ────────────────────────────────────────────────────────
    system_prompt = f"""You are a senior research analyst and technical writer.

Write a comprehensive, well-structured research report in Markdown format based on the provided sources.

MANDATORY STRUCTURE:
# [Report Title Based on Question]

## Executive Summary
(3-5 sentences covering the most important findings)

## [Section 1 — First Major Topic]
(Detailed analysis with inline citations like [1], [2])

## [Section 2 — Second Major Topic]
(Continue analysis)

## [Additional Sections as needed]

## Key Takeaways
- Bullet points summarizing actionable insights

## References
1. [Source Title](url)
2. [Source Title](url)
...

CRITICAL RULES:
- Cite sources by number in square brackets [1] for every factual claim
- Use ALL provided sources — don't ignore any source that contains relevant information
- The References section MUST list all numbered sources with their actual URLs
- Write in professional, clear technical prose
- If prior research context is provided, integrate it naturally (cite it as [Prior Research])
- Do NOT invent or hallucinate information not present in the sources"""

    human_prompt = (
        f"Research Question: {question}\n\n"
        f"Available Sources ({len(content_blocks)} pages):\n"
        f"{all_content}"
        f"{memory_section}"
    )

    # ── LLM CALL ─────────────────────────────────────────────────────────────
    llm = _get_llm(temperature=cfg.synthesis_temperature)  # 0.3 for natural prose

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        report = response.content  # extract the text content from the AIMessage

        if not report or len(report) < 200:
            # Response is suspiciously short — likely truncated or empty
            raise ValueError(f"Report too short ({len(report)} chars). May be truncated.")

        print(f"[synthesizer] ✓ Report generated: {len(report):,} characters, "
              f"{len(sources_for_report)} cited sources.")

    except Exception as e:
        print(f"[synthesizer] Report generation failed: {e}")
        # Fallback: build a minimal report manually from the snippets
        report = _build_fallback_report(question, sources_for_report, recalled)

    return {
        "final_report": report,
        "sources": sources_for_report,
    }


def _build_fallback_report(
    question: str,
    sources: List[SourceDocument],
    recalled: List[MemoryRecord],
) -> str:
    """
    Build a minimal Markdown report when the LLM call fails.

    This ensures the pipeline always produces SOME output rather than crashing.
    The report won't be as polished as LLM-synthesized output, but it's better
    than an empty string that would break downstream nodes.
    """
    lines = [
        f"# Research Report: {question}",
        "",
        "## Executive Summary",
        "This report was generated using cached source snippets after the synthesis model encountered an error.",
        "",
        "## Sources and Findings",
    ]
    for i, source in enumerate(sources, 1):
        lines.append(f"\n### [{i}] {source.title}")
        lines.append(f"**URL**: {source.url}")
        lines.append(f"{source.snippet}")

    if recalled:
        lines.append("\n## Prior Research Context")
        for mem in recalled:
            lines.append(f"\n### {mem.topic}")
            lines.append(mem.summary[:500])

    lines.append("\n## References")
    for i, source in enumerate(sources, 1):
        lines.append(f"{i}. [{source.title}]({source.url})")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6: MEMORY STORING
# ─────────────────────────────────────────────────────────────────────────────

def memory_storing_node(state: ResearchState) -> dict:
    """
    Summarize research findings and persist them to ChromaDB.

    WHY STORE A SUMMARY INSTEAD OF THE FULL REPORT?
        The full Markdown report can be 3,000–5,000 words. Storing that many
        tokens per memory entry would:
        1. Consume more disk space in the vector store.
        2. Overwhelm the LLM context when multiple memories are recalled simultaneously.
        3. Slow down embedding computation (longer text = more tokens = slower API call).

        A 200–300 word summary captures the key insights while remaining compact
        enough to inject multiple memories into future prompts without budget issues.

    State reads:  question, final_report, sources, iteration
    State writes: memory_stored

    Args:
        state: Current ResearchState with final_report populated.

    Returns:
        Partial state dict with memory_stored=True on success.
    """
    question = state["question"]
    report = state.get("final_report", "")
    sources = state.get("sources", [])
    iteration = state.get("iteration", 1)

    print(f"\n[Node: memory_storing] Persisting research to long-term memory...")

    if not report:
        print("[memory_storing] No report to store — skipping.")
        return {"memory_stored": False}

    # ── GENERATE A COMPACT SUMMARY ────────────────────────────────────────────
    # We ask the LLM to distill the full report into a 150–200 word summary.
    # This summary is what gets embedded and stored in ChromaDB.
    summary_prompt = (
        f"Summarize the following research report in 150-200 words. "
        f"Focus on the most important findings and conclusions. "
        f"Write as a self-contained paragraph that could be read without the full report.\n\n"
        f"Report (first 3000 chars):\n{report[:3000]}"
    )

    try:
        llm = _get_llm(temperature=0.1)  # low temperature for accurate summarization
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary = response.content.strip()

        if len(summary) < 50:
            # Suspiciously short summary — use the report's first paragraph instead
            first_para = report.split("\n\n")[1] if "\n\n" in report else report[:300]
            summary = first_para.strip()

    except Exception as e:
        print(f"[memory_storing] Summary generation failed: {e}. Using first 300 chars as summary.")
        # Use the report's opening paragraph as a fallback summary
        paragraphs = [p for p in report.split("\n\n") if len(p) > 50]
        summary = paragraphs[0][:300] if paragraphs else report[:300]

    # ── BUILD METADATA ────────────────────────────────────────────────────────
    metadata = {
        "question": question[:200],                    # truncated to fit metadata limits
        "source_count": len(sources),                  # how many sources were cited
        "iteration_count": iteration,                  # how many search loops were needed
        "research_date": datetime.now().isoformat(),   # ISO timestamp
        "source_urls": [s.url for s in sources[:5]],  # first 5 URLs for reference
    }

    # ── STORE IN CHROMADB ─────────────────────────────────────────────────────
    memory_manager = MemoryManager()
    doc_id = memory_manager.store_memory(
        topic=question[:100],  # use question as the topic label (truncated)
        summary=summary,
        metadata=metadata,
    )

    if doc_id:
        print(f"[memory_storing] ✓ Research stored in memory (id={doc_id[:8]}...)")
        print(f"[memory_storing] Summary ({len(summary)} chars): {summary[:150]}...")
        return {"memory_stored": True}
    else:
        print("[memory_storing] Memory storage skipped (duplicate or error).")
        return {"memory_stored": False}
