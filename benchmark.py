"""
benchmark.py — Evaluation & Performance Metrics Suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Systematically measure the research agent's performance across standardized
    queries and report latency, recall rate, citation density, and iteration counts.

METRICS EXPLAINED:

    1. Total Latency (seconds):
       Wall-clock time from graph.invoke() start to final state return.
       Includes LLM call latency, web scraping time, and embedding calls.
       Target: < 60s for a single query with 3 iterations.

    2. Memory Recall Hit Rate:
       Percentage of queries where recalled_memories was non-empty.
       High rate = agent effectively leverages past work.
       Formula: queries_with_recalls / total_queries × 100

    3. Search Request Count:
       Total number of web search requests made per query.
       = len(plan) × iterations = 3-5 sub-queries × 1-3 iterations.
       Lower is better (memory recall reduces search needs).

    4. Citation Ratio:
       Number of citation markers [1], [2], etc. per 500 words in the final report.
       Measures how well the synthesizer grounds claims in sources.
       Formula: citation_count / (word_count / 500)
       Target: > 2.0 citations per 500 words for a well-cited report.

    5. Evaluation Score:
       The final evaluation_score returned by the evaluator node.
       Measures how comprehensively the research answered the question.
       Target: > 0.7 for a production-quality report.
"""

import json         # writing benchmark results to JSON file
import time         # measuring wall-clock latency
import re           # counting citation markers in reports
import os           # checking for API keys
from typing import List, Dict, Any  # type hints
from datetime import datetime       # timestamping benchmark runs
from pathlib import Path            # cross-platform file paths

from dotenv import load_dotenv
load_dotenv()

from config import cfg              # benchmark output path
from memory_manager import MemoryManager  # for checking memory state


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK QUERY SET
# Three queries covering different domains and memory-leveraging scenarios.
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_QUERIES = [
    {
        "id": "query_1",
        "question": "What are the top techniques for reducing hallucinations in RAG systems in 2025?",
        "description": "Primary research query — tests cold-start search + synthesis pipeline",
        "expected_topics": ["CRAG", "Self-RAG", "faithfulness", "RAGAS", "retrieval quality"],
    },
    {
        "id": "query_2",
        "question": "How does GraphRAG use knowledge graphs to address RAG hallucination problems?",
        "description": "Memory continuity test — should recall prior RAG hallucination research",
        "expected_topics": ["knowledge graph", "entity", "community", "global queries"],
    },
    {
        "id": "query_3",
        "question": "What are the hardware requirements for fine-tuning LLMs with QLoRA on consumer GPUs?",
        "description": "Technical deep-dive — tests domain-specific retrieval and synthesis",
        "expected_topics": ["VRAM", "RTX", "4-bit", "LoRA", "memory", "quantization"],
    },
]


def count_citations(report_text: str) -> int:
    """
    Count the number of inline citation markers in a Markdown report.

    Citation markers follow the pattern [1], [2], [3], etc.
    We use a regex that matches opening bracket, one or more digits, closing bracket.

    Args:
        report_text: The full Markdown report string.

    Returns:
        Integer count of citation markers found.
    """
    # Pattern: \[ matches literal [, \d+ matches one or more digits, \] matches literal ]
    citations = re.findall(r'\[\d+\]', report_text)
    return len(citations)  # count all matches


def count_words(text: str) -> int:
    """
    Count words in a text string.

    Uses a simple whitespace split for word counting.
    For a Markdown report, this includes headers and citation markers as "words",
    which is intentional — we want total word count for the citation ratio formula.

    Args:
        text: Any string.

    Returns:
        Integer word count.
    """
    return len(text.split())  # split on any whitespace


def calculate_citation_ratio(report_text: str) -> float:
    """
    Calculate citations per 500 words in the report.

    WHY 500-WORD NORMALIZATION?
        Raw citation counts aren't comparable across reports of different lengths.
        A 100-word report with 2 citations and a 1000-word report with 2 citations
        have very different citation densities. Per-500-word normalizes for length.

    Formula: citation_ratio = citation_count / (word_count / 500)

    Args:
        report_text: Full Markdown report.

    Returns:
        Float citations-per-500-words ratio. Returns 0.0 for empty reports.
    """
    if not report_text:
        return 0.0

    words = count_words(report_text)
    citations = count_citations(report_text)

    if words == 0:
        return 0.0

    # Normalize: citations per 500-word block
    ratio = citations / (words / 500)
    return round(ratio, 2)  # round to 2 decimal places for display


def run_single_benchmark(query_config: dict) -> dict:
    """
    Run a single benchmark query and collect all metrics.

    This function runs the full agent pipeline for one query and captures
    timing, memory usage, and output quality metrics.

    Args:
        query_config: One entry from BENCHMARK_QUERIES.

    Returns:
        A metrics dict with all measured values.
    """
    question = query_config["question"]
    query_id = query_config["id"]

    print(f"\n{'─' * 60}")
    print(f"Benchmark Query: {query_id}")
    print(f"Question: {question[:80]}")
    print(f"{'─' * 60}")

    # Check if OpenAI API key is available
    has_openai = bool(os.environ.get("OPENAI_API_KEY", ""))

    if not has_openai:
        # Without API key, we can't run the full pipeline.
        # Return a mock result with a warning.
        print(f"[Benchmark] WARNING: No OPENAI_API_KEY found. Running mock benchmark.")
        return {
            "query_id": query_id,
            "question": question,
            "status": "skipped_no_api_key",
            "latency_seconds": 0.0,
            "memory_recall_hit": False,
            "recalled_memory_count": 0,
            "search_request_count": 0,
            "iteration_count": 0,
            "evaluation_score": 0.0,
            "citation_count": 0,
            "word_count": 0,
            "citation_ratio": 0.0,
            "report_length_chars": 0,
            "note": "Skipped — OPENAI_API_KEY not set. Run with API key for real benchmarks.",
        }

    # Import graph runner here (not at top) to avoid import errors if dependencies
    # are partially installed during testing without API keys.
    from graph_builder import run_research

    # Record start time for latency measurement
    start_time = time.time()

    try:
        # Run the full agent pipeline
        final_state = run_research(question)
        elapsed = time.time() - start_time  # wall-clock latency in seconds

        # ── EXTRACT METRICS ───────────────────────────────────────────────────
        recalled = final_state.get("recalled_memories", [])
        report = final_state.get("final_report", "")
        iteration = final_state.get("iteration", 0)
        eval_score = final_state.get("evaluation_score", 0.0)
        plan = final_state.get("plan", [])
        search_results = final_state.get("raw_search_results", [])

        # Memory recall hit = at least one memory was recalled above threshold
        memory_hit = len(recalled) > 0

        # Total search requests = number of unique search calls executed
        # Approximation: len(plan) × iteration (each iteration runs all sub-queries)
        # Actual count may be less due to deduplication in multi_search()
        search_count = len(search_results)  # deduplicated search results as proxy

        # Citation analysis
        citations = count_citations(report)
        words = count_words(report)
        ratio = calculate_citation_ratio(report)

        metrics = {
            "query_id": query_id,
            "question": question,
            "status": "completed",
            "latency_seconds": round(elapsed, 2),
            "memory_recall_hit": memory_hit,
            "recalled_memory_count": len(recalled),
            "search_request_count": search_count,
            "iteration_count": iteration,
            "evaluation_score": round(eval_score, 3),
            "citation_count": citations,
            "word_count": words,
            "citation_ratio": ratio,
            "report_length_chars": len(report),
        }

        print(f"[Benchmark] ✓ Completed in {elapsed:.1f}s")
        print(f"  Memory recall: {'Yes' if memory_hit else 'No'} ({len(recalled)} memories)")
        print(f"  Iterations: {iteration}")
        print(f"  Eval score: {eval_score:.2f}")
        print(f"  Citations: {citations} ({ratio:.1f} per 500 words)")
        print(f"  Report: {len(report):,} chars / {words:,} words")

        return metrics

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Benchmark] ✗ Failed after {elapsed:.1f}s: {e}")
        return {
            "query_id": query_id,
            "question": question,
            "status": "failed",
            "error": str(e),
            "latency_seconds": round(elapsed, 2),
            "memory_recall_hit": False,
            "recalled_memory_count": 0,
            "search_request_count": 0,
            "iteration_count": 0,
            "evaluation_score": 0.0,
            "citation_count": 0,
            "word_count": 0,
            "citation_ratio": 0.0,
            "report_length_chars": 0,
        }


def compute_aggregate_metrics(results: List[dict]) -> dict:
    """
    Compute summary statistics across all benchmark runs.

    Aggregates individual run metrics into averages and rates for the
    summary table. Only includes completed runs in averages.

    Args:
        results: List of metric dicts from run_single_benchmark().

    Returns:
        Dict with aggregate/summary statistics.
    """
    completed = [r for r in results if r["status"] == "completed"]

    if not completed:
        return {
            "total_queries": len(results),
            "completed_queries": 0,
            "note": "No queries completed. Check API keys and network.",
        }

    avg_latency = sum(r["latency_seconds"] for r in completed) / len(completed)
    recall_rate = sum(1 for r in completed if r["memory_recall_hit"]) / len(completed) * 100
    avg_eval_score = sum(r["evaluation_score"] for r in completed) / len(completed)
    avg_citation_ratio = sum(r["citation_ratio"] for r in completed) / len(completed)
    avg_iterations = sum(r["iteration_count"] for r in completed) / len(completed)
    total_search_requests = sum(r["search_request_count"] for r in results)

    return {
        "total_queries": len(results),
        "completed_queries": len(completed),
        "avg_latency_seconds": round(avg_latency, 2),
        "memory_recall_hit_rate_percent": round(recall_rate, 1),
        "avg_evaluation_score": round(avg_eval_score, 3),
        "avg_citation_ratio_per_500_words": round(avg_citation_ratio, 2),
        "avg_iterations_per_query": round(avg_iterations, 1),
        "total_search_requests": total_search_requests,
    }


def generate_markdown_table(results: List[dict], aggregate: dict) -> str:
    """
    Generate a formatted Markdown evaluation report with results table.

    This is written to stdout and optionally to a file. The table uses
    GitHub-flavored Markdown syntax for nice rendering in README files.

    Args:
        results: Individual run results.
        aggregate: Aggregate statistics from compute_aggregate_metrics().

    Returns:
        Formatted Markdown string.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# Research Agent Benchmark Results",
        f"",
        f"**Run Date**: {now}",
        f"**LLM Model**: {cfg.llm_model_name}",
        f"**Max Iterations**: {cfg.max_iterations}",
        f"**Sufficiency Threshold**: {cfg.sufficiency_threshold}",
        f"",
        f"## Per-Query Results",
        f"",
        f"| Query ID | Status | Latency (s) | Memory Hit | Iterations | Eval Score | Citations/500w | Report Words |",
        f"|----------|--------|-------------|------------|------------|------------|----------------|--------------|",
    ]

    for r in results:
        mem_hit = "✓" if r.get("memory_recall_hit") else "✗"
        status_icon = "✓" if r["status"] == "completed" else ("⚠" if r["status"] == "skipped_no_api_key" else "✗")
        lines.append(
            f"| {r['query_id']} "
            f"| {status_icon} {r['status'][:12]} "
            f"| {r.get('latency_seconds', 0):.1f}s "
            f"| {mem_hit} "
            f"| {r.get('iteration_count', 0)} "
            f"| {r.get('evaluation_score', 0):.2f} "
            f"| {r.get('citation_ratio', 0):.1f} "
            f"| {r.get('word_count', 0):,} |"
        )

    lines += [
        f"",
        f"## Aggregate Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Queries | {aggregate.get('total_queries', 0)} |",
        f"| Completed Queries | {aggregate.get('completed_queries', 0)} |",
        f"| Avg Latency | {aggregate.get('avg_latency_seconds', 0):.1f}s |",
        f"| Memory Recall Hit Rate | {aggregate.get('memory_recall_hit_rate_percent', 0):.1f}% |",
        f"| Avg Evaluation Score | {aggregate.get('avg_evaluation_score', 0):.3f} |",
        f"| Avg Citations / 500 Words | {aggregate.get('avg_citation_ratio_per_500_words', 0):.2f} |",
        f"| Avg Iterations / Query | {aggregate.get('avg_iterations_per_query', 0):.1f} |",
        f"| Total Search Requests | {aggregate.get('total_search_requests', 0)} |",
        f"",
        f"## Interpretation Guide",
        f"",
        f"- **Latency**: Target < 60s per query. High latency usually means more search iterations.",
        f"- **Memory Hit Rate**: Higher = better. 0% means memory is empty (run download_sample_data.py).",
        f"- **Evaluation Score**: Target > 0.70. Below 0.70 means the agent looped to find more content.",
        f"- **Citations / 500 words**: Target > 2.0. Low ratio = synthesizer not grounding claims in sources.",
        f"- **Iterations**: 1 = efficient (found enough content in one search). 3 = maxed out (hard limit).",
    ]

    return "\n".join(lines)


def main():
    """
    Run the full benchmark suite and save results.

    Execution flow:
    1. Print system info (API key availability, ChromaDB state).
    2. Run each benchmark query sequentially.
    3. Compute aggregate metrics.
    4. Print Markdown table to stdout.
    5. Save JSON results to ./data/benchmark_results.json.
    """
    print("=" * 70)
    print("Research Agent Benchmark Suite")
    print("=" * 70)

    # Print system diagnostics
    has_openai = bool(os.environ.get("OPENAI_API_KEY", ""))
    has_tavily = bool(os.environ.get("TAVILY_API_KEY", ""))

    print(f"\nSystem Status:")
    print(f"  OPENAI_API_KEY:  {'✓ Set' if has_openai else '✗ Not set (benchmarks will be skipped)'}")
    print(f"  TAVILY_API_KEY:  {'✓ Set' if has_tavily else '✗ Not set (DuckDuckGo fallback active)'}")

    # Check memory state
    try:
        memory = MemoryManager()
        print(f"  ChromaDB:        ✓ {memory.count} memories stored")
    except Exception as e:
        print(f"  ChromaDB:        ✗ Error: {e}")

    # ── RUN BENCHMARKS ────────────────────────────────────────────────────────
    print(f"\nRunning {len(BENCHMARK_QUERIES)} benchmark queries...")
    results = []

    for query_config in BENCHMARK_QUERIES:
        result = run_single_benchmark(query_config)
        results.append(result)

    # ── COMPUTE AGGREGATES ────────────────────────────────────────────────────
    aggregate = compute_aggregate_metrics(results)

    # ── GENERATE AND PRINT MARKDOWN TABLE ─────────────────────────────────────
    markdown_report = generate_markdown_table(results, aggregate)
    print(f"\n{'=' * 70}")
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(markdown_report)

    # ── SAVE JSON RESULTS ─────────────────────────────────────────────────────
    output_path = Path(cfg.benchmark_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark_output = {
        "benchmark_run_timestamp": datetime.now().isoformat(),
        "system_config": {
            "llm_model": cfg.llm_model_name,
            "max_iterations": cfg.max_iterations,
            "sufficiency_threshold": cfg.sufficiency_threshold,
            "has_openai_key": has_openai,
            "has_tavily_key": has_tavily,
        },
        "per_query_results": results,
        "aggregate_metrics": aggregate,
    }

    output_path.write_text(
        json.dumps(benchmark_output, indent=2),
        encoding="utf-8",
    )

    print(f"\n✓ Raw results saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
