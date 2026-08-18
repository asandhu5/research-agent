"""
tests/test_graph_flow.py — Integration Tests for the Full LangGraph State Machine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT WE'RE TESTING:
    1. Full graph traversal — the state machine runs from START to END.
    2. Final state validation — `final_report` is populated after the run.
    3. Iteration counter — the `iteration` field reflects how many search loops ran.
    4. Routing correctness — high eval score → single iteration (no re-loop).
    5. Loop behavior — low eval score on first iteration → second iteration.
    6. Circuit-breaker — agent stops at max_iterations even without high score.
    7. Replanning — a low score actually changes what gets searched for next,
       using the evaluator's missing_topics, instead of the loop re-running
       the exact same queries.

WHY INTEGRATION TESTS ON TOP OF UNIT TESTS?
    Unit tests (test_evaluator.py, test_scraper.py etc.) verify individual nodes
    in isolation. But the graph's VALUE is in the TRANSITIONS — does a low evaluator
    score actually trigger another search? Does the final state have all required keys?

    Integration tests verify that the nodes compose correctly when wired together
    through LangGraph's state machine.

HOW WE MOCK THE ENTIRE PIPELINE:
    We mock at the "seam" between the graph and external services:
    - MemoryManager.recall_memories → returns empty list (no prior memories)
    - SearchEngine.search → returns pre-defined SourceDocument list
    - requests.get (inside WebScraper) → returns pre-defined HTML
    - ChatOpenAI.invoke → returns pre-defined LLM responses for each node

    This gives us a fully runnable graph with zero API calls.

    Structured-output calls are dispatched by SCHEMA (PlannerOutput vs
    EvaluatorOutput), not by call order/count (see `structured_llm_router`
    below). The planner node now runs twice on any run that loops -- once for
    the initial plan, once per replan -- so a single shared side_effect list
    indexed by call position would silently misalign the moment the number of
    loop iterations changed, which is exactly what would happen with the
    "one shared list" approach this file used before the replanning loop
    existed.
"""

import itertools
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from agent_nodes import EvaluatorOutput, PlannerOutput
from state import SourceDocument


# ─────────────────────────────────────────────────────────────────────────────
# MOCK RESPONSE FACTORIES
# Centralized factories for mock LLM responses. Changing these affects all tests
# that use them, making it easy to update test assumptions in one place.
# ─────────────────────────────────────────────────────────────────────────────

def make_mock_planner_output(sub_queries: List[str] = None):
    """Create a mock PlannerOutput object."""
    mock_output = MagicMock()
    mock_output.sub_queries = sub_queries or [
        "RAG hallucination mitigation CRAG technique",
        "Self-RAG reflection tokens mechanism",
        "RAGAS evaluation metrics faithfulness",
    ]
    mock_output.reasoning = "These queries cover the key techniques for RAG hallucination mitigation."
    return mock_output


def make_mock_evaluator_output(score: float = 0.85, reasoning: str = "Sufficient.", missing_topics: List[str] = None):
    """Create a mock EvaluatorOutput object with specified score."""
    mock_output = MagicMock()
    mock_output.score = score
    mock_output.reasoning = reasoning
    mock_output.missing_topics = missing_topics or []
    return mock_output


def make_mock_llm_response(content: str = "# Mock Research Report\n\nThis is a test report [1].\n\n## References\n1. [Test Source](http://test.com)"):
    """Create a mock AIMessage response (used by synthesizer and memory_storing nodes)."""
    mock_response = MagicMock()
    mock_response.content = content
    return mock_response


def structured_llm_router(planner_effect, evaluator_effect):
    """Return a `with_structured_output` side_effect that dispatches on which
    Pydantic schema was actually requested, plus the two per-schema mocks so
    a test can assert on call_count / call_args_list separately for planner
    calls vs evaluator calls.

    `planner_effect` / `evaluator_effect` are anything valid as a MagicMock
    `side_effect` (a list -- raises StopIteration if over-consumed, which is
    a deliberately loud failure if a test's call-count assumption is wrong --
    or an infinite iterator like itertools.repeat(...) for tests that don't
    care how many times that schema gets requested).
    """
    planner_llm = MagicMock()
    planner_llm.invoke.side_effect = planner_effect
    evaluator_llm = MagicMock()
    evaluator_llm.invoke.side_effect = evaluator_effect

    def router(schema, **kwargs):
        if schema is PlannerOutput:
            return planner_llm
        if schema is EvaluatorOutput:
            return evaluator_llm
        raise AssertionError(f"Unexpected structured-output schema requested: {schema!r}")

    return router, planner_llm, evaluator_llm


MOCK_SEARCH_RESULTS = [
    SourceDocument(
        title="CRAG: Corrective RAG",
        url="https://arxiv.org/abs/2401.15884",
        snippet="CRAG uses a quality evaluator to correct poor retrievals."
    ),
    SourceDocument(
        title="Self-RAG Framework",
        url="https://arxiv.org/abs/2310.11511",
        snippet="Self-RAG uses reflection tokens for adaptive retrieval."
    ),
]

MOCK_SCRAPED_HTML = """<html>
<head><title>CRAG Research Paper</title></head>
<body>
<main>
<h1>Corrective Retrieval Augmented Generation</h1>
<p>CRAG proposes a lightweight evaluator to assess retrieved document quality.
When quality is insufficient, CRAG uses web search as a fallback.
This approach significantly reduces hallucinations in knowledge-intensive QA tasks.</p>
</main>
</body>
</html>"""


def make_initial_state(question: str) -> dict:
    return {
        "question": question,
        "recalled_memories": [], "plan": [],
        "raw_search_results": [], "scraped_content": [],
        "iteration": 0, "evaluation_score": 0.0,
        "evaluation_reasoning": "", "missing_topics": [],
        "final_report": "", "sources": [], "memory_stored": False,
    }


class TestGraphFullExecution:
    """Integration tests for the complete graph execution flow."""

    def test_graph_produces_final_report(self):
        """
        WHAT: After a full graph run, final_state["final_report"] should be non-empty.

        WHY: The final_report is the primary deliverable of the agent. If the graph
        terminates without a report, something in the synthesizer or routing logic
        is broken. This test is the most important "smoke test" for the system.

        HOW: We mock all external calls and run graph.invoke() end-to-end,
        then assert on the final state's final_report key.
        """
        mock_report_content = (
            "# Research Report: RAG Hallucination Mitigation\n\n"
            "## Executive Summary\n"
            "This report examines RAG hallucination mitigation techniques [1][2].\n\n"
            "## Techniques\n"
            "CRAG addresses retrieval quality through corrective mechanisms [1].\n"
            "Self-RAG uses reflection tokens to adaptively retrieve information [2].\n\n"
            "## References\n"
            "1. [CRAG Paper](https://arxiv.org/abs/2401.15884)\n"
            "2. [Self-RAG Paper](https://arxiv.org/abs/2310.11511)\n"
        )

        with patch("agent_nodes.MemoryManager") as mock_mm_class, \
             patch("agent_nodes.multi_search", return_value=MOCK_SEARCH_RESULTS), \
             patch("web_scraper.requests.get") as mock_get, \
             patch("agent_nodes._get_llm") as mock_get_llm:

            mock_mm = MagicMock()
            mock_mm.recall_memories.return_value = []
            mock_mm.store_memory.return_value = "test-uuid-001"
            mock_mm_class.return_value = mock_mm

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.encoding = "utf-8"
            mock_response.apparent_encoding = "utf-8"
            mock_response.text = MOCK_SCRAPED_HTML
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # score=0.88 on the first pass is above threshold -- no replanning here.
            router, planner_llm, evaluator_llm = structured_llm_router(
                planner_effect=itertools.repeat(make_mock_planner_output()),
                evaluator_effect=itertools.repeat(make_mock_evaluator_output(score=0.88)),
            )
            mock_llm = MagicMock()
            mock_llm.with_structured_output.side_effect = router
            mock_llm.invoke.return_value = make_mock_llm_response(mock_report_content)
            mock_get_llm.return_value = mock_llm

            from graph_builder import build_research_graph
            graph = build_research_graph(use_checkpointer=False)
            final_state = graph.invoke(make_initial_state(
                "What are the best techniques for reducing hallucinations in RAG?"
            ))

        assert "final_report" in final_state, "Final state must contain 'final_report' key"
        assert final_state["final_report"], "final_report must be a non-empty string"
        assert len(final_state["final_report"]) > 100, (
            f"final_report should be at least 100 chars, got {len(final_state.get('final_report', ''))}"
        )

    def test_iteration_counter_is_set_after_search(self):
        """
        WHAT: After the graph runs, `iteration` should be >= 1.

        WHY: The iteration counter tracks how many search loops were executed.
        It MUST be at least 1 because search_and_scrape_node always runs at least once.
        If iteration is 0, it means search_and_scrape_node never ran, which indicates
        a critical routing failure.
        """
        with patch("agent_nodes.MemoryManager") as mock_mm_class, \
             patch("agent_nodes.multi_search", return_value=MOCK_SEARCH_RESULTS), \
             patch("web_scraper.requests.get") as mock_get, \
             patch("agent_nodes._get_llm") as mock_get_llm:

            mock_mm = MagicMock()
            mock_mm.recall_memories.return_value = []
            mock_mm.store_memory.return_value = "uuid-123"
            mock_mm_class.return_value = mock_mm

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.encoding = "utf-8"
            mock_response.apparent_encoding = "utf-8"
            mock_response.text = MOCK_SCRAPED_HTML
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            router, planner_llm, evaluator_llm = structured_llm_router(
                planner_effect=itertools.repeat(make_mock_planner_output()),
                evaluator_effect=itertools.repeat(make_mock_evaluator_output(score=0.9)),  # high → single iteration
            )
            mock_llm = MagicMock()
            mock_llm.with_structured_output.side_effect = router
            mock_llm.invoke.return_value = make_mock_llm_response()
            mock_get_llm.return_value = mock_llm

            from graph_builder import build_research_graph
            graph = build_research_graph()
            final_state = graph.invoke(make_initial_state("Test question"))

        iteration = final_state.get("iteration", 0)
        assert iteration >= 1, (
            f"iteration should be at least 1 after the graph runs, got {iteration}"
        )

    def test_high_eval_score_prevents_extra_iterations(self):
        """
        WHAT: When evaluator returns score >= 0.7 on first pass, agent should NOT loop.

        WHY: The conditional edge should route to "synthesize" when score >= threshold.
        If it loops anyway, we're burning extra API calls for no reason.
        We verify this both by checking multi_search was only called once AND
        that the planner itself only ran once (no replan should have happened).
        """
        iteration_tracker = []  # capture iteration value each time search runs

        def tracking_multi_search(queries, max_results_per_query=None):
            """Wraps multi_search to track how many times it's called."""
            iteration_tracker.append(len(iteration_tracker) + 1)
            return MOCK_SEARCH_RESULTS

        with patch("agent_nodes.MemoryManager") as mock_mm_class, \
             patch("agent_nodes.multi_search", side_effect=tracking_multi_search), \
             patch("web_scraper.requests.get") as mock_get, \
             patch("agent_nodes._get_llm") as mock_get_llm:

            mock_mm = MagicMock()
            mock_mm.recall_memories.return_value = []
            mock_mm.store_memory.return_value = "uuid-456"
            mock_mm_class.return_value = mock_mm

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.encoding = "utf-8"
            mock_response.apparent_encoding = "utf-8"
            mock_response.text = MOCK_SCRAPED_HTML
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            router, planner_llm, evaluator_llm = structured_llm_router(
                planner_effect=[make_mock_planner_output()],  # exactly one call expected
                evaluator_effect=[make_mock_evaluator_output(score=0.95)],  # very high score → no loop
            )
            mock_llm = MagicMock()
            mock_llm.with_structured_output.side_effect = router
            mock_llm.invoke.return_value = make_mock_llm_response()
            mock_get_llm.return_value = mock_llm

            from graph_builder import build_research_graph
            graph = build_research_graph()
            final_state = graph.invoke(make_initial_state("Test question with high-quality content"))

        assert len(iteration_tracker) == 1, (
            f"With a high evaluation score (0.95), the graph should only search once. "
            f"But multi_search was called {len(iteration_tracker)} time(s)."
        )
        assert final_state.get("iteration", 0) == 1, (
            f"iteration should be 1 (single search pass), got {final_state.get('iteration', 0)}"
        )
        assert planner_llm.invoke.call_count == 1, (
            "planner_node should not run a second (replan) time when the first score is already sufficient"
        )

    def test_circuit_breaker_stops_at_max_iterations(self):
        """
        WHAT: Even with a consistently low evaluator score, the graph stops at max_iterations.

        WHY: This is the most critical safety test. Without the circuit-breaker, a query
        that can never be satisfied (e.g., real-time data not indexed anywhere) would
        loop forever, consuming infinite API credits.

        HOW: We configure the evaluator mock to always return score=0.3 (below threshold).
        We then verify the graph terminates and iteration equals max_iterations, AND
        that the planner ran exactly max_iterations times (1 initial + (max_iterations-1)
        replans) -- not more, proving the circuit-breaker cuts off the replan loop too,
        not just the search loop.
        """
        from config import cfg  # import actual config for max_iterations value

        with patch("agent_nodes.MemoryManager") as mock_mm_class, \
             patch("agent_nodes.multi_search", return_value=MOCK_SEARCH_RESULTS), \
             patch("web_scraper.requests.get") as mock_get, \
             patch("agent_nodes._get_llm") as mock_get_llm:

            mock_mm = MagicMock()
            mock_mm.recall_memories.return_value = []
            mock_mm.store_memory.return_value = "uuid-789"
            mock_mm_class.return_value = mock_mm

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.encoding = "utf-8"
            mock_response.apparent_encoding = "utf-8"
            mock_response.text = MOCK_SCRAPED_HTML
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            router, planner_llm, evaluator_llm = structured_llm_router(
                planner_effect=itertools.repeat(make_mock_planner_output()),
                evaluator_effect=itertools.repeat(make_mock_evaluator_output(
                    score=0.3, reasoning="Always insufficient (circuit-breaker test)", missing_topics=["everything"]
                )),
            )
            mock_llm = MagicMock()
            mock_llm.with_structured_output.side_effect = router
            mock_llm.invoke.return_value = make_mock_llm_response()
            mock_get_llm.return_value = mock_llm

            from graph_builder import build_research_graph
            graph = build_research_graph()
            final_state = graph.invoke(make_initial_state("Question that can never be satisfied (circuit-breaker test)"))

        final_iteration = final_state.get("iteration", 0)
        assert final_iteration <= cfg.max_iterations, (
            f"The circuit-breaker should stop the graph at max_iterations={cfg.max_iterations}. "
            f"But iteration reached {final_iteration}. This means the loop didn't terminate!"
        )
        assert final_iteration > 0, "At least one search iteration must have run"
        assert final_state.get("final_report"), (
            "Graph must produce a final_report even when circuit-breaker fires (low eval score)"
        )
        assert evaluator_llm.invoke.call_count == cfg.max_iterations, (
            f"evaluator should run exactly max_iterations={cfg.max_iterations} times, "
            f"got {evaluator_llm.invoke.call_count}"
        )
        assert planner_llm.invoke.call_count == cfg.max_iterations, (
            f"planner should run once initially plus once per replan, totaling "
            f"max_iterations={cfg.max_iterations} calls (the circuit-breaker fires on the "
            f"evaluate step of the final iteration, so there's no replan call after that one), "
            f"got {planner_llm.invoke.call_count}"
        )

    def test_low_score_triggers_replan_targeting_missing_topics(self):
        """
        WHAT: A low first-pass score with specific missing_topics should cause the
        SECOND planner call to (a) receive those missing_topics and the first
        plan's queries in its prompt, and (b) produce a genuinely different set
        of search queries -- which search_and_scrape_node then actually searches
        for, not a repeat of the first pass's queries.

        WHY: This is the actual behavior recommendation #2 asks for: the original
        code computed missing_topics in evaluator_node and then never read them
        anywhere, so a second iteration re-ran the identical queries from the
        first plan. This test would have failed against that code (the second
        planner call would never have happened at all -- the loop went straight
        back to search_scrape) and fails now if the replan prompt stops
        including the gap-analysis context.
        """
        searched_queries: List[List[str]] = []

        def tracking_multi_search(queries, max_results_per_query=None):
            searched_queries.append(list(queries))
            return MOCK_SEARCH_RESULTS

        initial_queries = ["broad query about the topic", "another broad query"]
        replan_queries = ["deep dive into missing topic A", "deep dive into missing topic B"]

        with patch("agent_nodes.MemoryManager") as mock_mm_class, \
             patch("agent_nodes.multi_search", side_effect=tracking_multi_search), \
             patch("web_scraper.requests.get") as mock_get, \
             patch("agent_nodes._get_llm") as mock_get_llm:

            mock_mm = MagicMock()
            mock_mm.recall_memories.return_value = []
            mock_mm.store_memory.return_value = "uuid-replan"
            mock_mm_class.return_value = mock_mm

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.encoding = "utf-8"
            mock_response.apparent_encoding = "utf-8"
            mock_response.text = MOCK_SCRAPED_HTML
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            router, planner_llm, evaluator_llm = structured_llm_router(
                planner_effect=[
                    make_mock_planner_output(initial_queries),
                    make_mock_planner_output(replan_queries),
                ],
                evaluator_effect=[
                    make_mock_evaluator_output(
                        score=0.4, reasoning="Missing key technical details",
                        missing_topics=["missing topic A", "missing topic B"],
                    ),
                    make_mock_evaluator_output(score=0.9, reasoning="Now sufficient"),
                ],
            )
            mock_llm = MagicMock()
            mock_llm.with_structured_output.side_effect = router
            mock_llm.invoke.return_value = make_mock_llm_response()
            mock_get_llm.return_value = mock_llm

            from graph_builder import build_research_graph
            graph = build_research_graph()
            final_state = graph.invoke(make_initial_state("A question with an initially incomplete answer"))

        # The planner ran twice: once for the initial plan, once to replan.
        assert planner_llm.invoke.call_count == 2

        # The replan call's prompt must actually contain the evaluator's
        # feedback -- this is the literal "wiring" the fix is about.
        replan_call_messages = planner_llm.invoke.call_args_list[1].args[0]
        replan_human_message = next(m for m in replan_call_messages if hasattr(m, "content") and "Research Question" in m.content)
        assert "missing topic A" in replan_human_message.content
        assert "missing topic B" in replan_human_message.content
        assert "broad query about the topic" in replan_human_message.content  # already-tried context

        # search_and_scrape_node actually searched the SECOND plan's queries,
        # not a repeat of the first -- two distinct search passes with two
        # distinct query sets, not the same queries twice.
        assert len(searched_queries) == 2
        assert searched_queries[0] == initial_queries
        assert searched_queries[1] == replan_queries
        assert searched_queries[0] != searched_queries[1]

        # The final state's `plan` reflects the most recent (replan) queries.
        # missing_topics reflects the LAST evaluation (the one that scored
        # 0.9 and found nothing missing), not the first, since each
        # evaluator_node call overwrites the field with its own result.
        assert final_state.get("plan") == replan_queries
        assert final_state.get("missing_topics") == []
