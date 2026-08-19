

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

load_dotenv()


def _get_llm(temperature: float = cfg.llm_temperature) -> ChatOpenAI:
    return ChatOpenAI(
        model=cfg.llm_model_name,
        temperature=temperature,
        max_tokens=cfg.llm_max_tokens,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

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


def recall_memory_node(state: ResearchState) -> dict:

    question = state["question"]
    print(f"\n{'='*60}")
    print(f"[Node: recall_memory] Recalling past research for:")
    print(f"  '{question[:100]}'")
    print(f"{'='*60}")

    memory_manager = MemoryManager()
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

    return {"recalled_memories": recalled}


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


def search_and_scrape_node(state: ResearchState) -> dict:

    plan = state.get("plan", [])
    existing_search = state.get("raw_search_results", [])   # prior iterations' results
    existing_scraped = state.get("scraped_content", [])      # prior iterations' scrapes
    current_iteration = state.get("iteration", 0)            # default 0 = not yet run

    new_iteration = current_iteration + 1  # increment loop counter for this pass
    print(f"\n[Node: search_and_scrape] Iteration {new_iteration} — Searching {len(plan)} sub-queries")

    new_search_results = multi_search(
        queries=plan,
        max_results_per_query=cfg.max_search_results,
    )

    existing_urls = {page.url for page in existing_scraped}  # set for O(1) lookup
    fresh_search_results = [
        doc for doc in new_search_results
        if doc.url not in existing_urls  # only keep URLs we haven't seen before
    ]

    print(f"[search_and_scrape] {len(new_search_results)} search results, "
          f"{len(fresh_search_results)} new URLs to scrape.")

    if fresh_search_results:
        urls_to_scrape = [doc.url for doc in fresh_search_results]
        new_scraped = scrape_urls_batch(urls_to_scrape)
    else:
        new_scraped = []  # nothing new to scrape
        print("[search_and_scrape] No new URLs to scrape — all already seen in prior iterations.")

    combined_search = existing_search + fresh_search_results
    combined_scraped = existing_scraped + new_scraped


    successful_scrapes = sum(1 for p in new_scraped if p.success)
    total_chars = sum(len(p.content) for p in combined_scraped if p.success)

    print(f"[search_and_scrape] New scrapes: {successful_scrapes}/{len(new_scraped)} successful")
    print(f"[search_and_scrape] Total accumulated content: {total_chars:,} characters")

    return {
        "raw_search_results": combined_search,     # accumulated across all iterations
        "scraped_content": combined_scraped,        # accumulated across all iterations
        "iteration": new_iteration,                 # updated loop counter
    }



def evaluator_node(state: ResearchState) -> dict:

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

    content_summary_parts = []
    for i, page in enumerate(scraped, 1):
        if not page.success or not page.content:
            continue  # skip failed scrapes
        preview = page.content[:500].strip()
        content_summary_parts.append(
            f"[Source {i}] {page.title or page.url[:60]}\n{preview}..."
        )

    content_summary = "\n\n".join(content_summary_parts)


    if len(content_summary) > 6000:
        content_summary = content_summary[:6000] + "\n... [additional sources truncated for evaluation]"

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

        fallback_score = 0.8 if iteration >= cfg.max_iterations else 0.4
        print(f"[evaluator] Structured evaluation failed: {e}. Using fallback score: {fallback_score}")
        score = fallback_score
        reasoning = f"Evaluation model call failed ({type(e).__name__}). Proceeding with available content."
        missing = []

    return {
        "evaluation_score": score,
        "evaluation_reasoning": reasoning,
        "missing_topics": missing,
    }


def synthesizer_node(state: ResearchState) -> dict:

    question = state["question"]
    scraped = state.get("scraped_content", [])
    search_results = state.get("raw_search_results", [])
    recalled = state.get("recalled_memories", [])

    print(f"\n[Node: synthesizer] Generating research report...")

    sources_for_report = []
    content_blocks = []

    for i, page in enumerate(scraped, 1):
        if page.success and page.content:
            sources_for_report.append(SourceDocument(
                title=page.title or f"Source {i}",
                url=page.url,
                snippet=page.content[:200],  # short snippet for the references section
            ))
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
        paragraphs = [p for p in report.split("\n\n") if len(p) > 50]
        summary = paragraphs[0][:300] if paragraphs else report[:300]

    metadata = {
        "question": question[:200],                    # truncated to fit metadata limits
        "source_count": len(sources),                  # how many sources were cited
        "iteration_count": iteration,                  # how many search loops were needed
        "research_date": datetime.now().isoformat(),   # ISO timestamp
        "source_urls": [s.url for s in sources[:5]],  # first 5 URLs for reference
    }

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
