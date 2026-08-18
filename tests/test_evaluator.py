"""
tests/test_evaluator.py — Unit Tests for the Evaluator Node
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT WE'RE TESTING:
    1. Poor context → low score (< 0.5): When scraped content is completely irrelevant
       to the question, the evaluator should return a low sufficiency score.
    2. Comprehensive context → high score (>= 0.7): When content directly and fully
       answers the question, the evaluator should return a high score.
    3. No content → score 0.0: The node should handle empty scraped_content gracefully.
    4. Score range validity: Evaluator never returns a score outside [0.0, 1.0].

WHY MOCK THE LLM?
    The evaluator node calls the OpenAI API. Unit tests must:
    - Not require an API key (CI environments don't have one)
    - Run fast (not wait for API response latency)
    - Be deterministic (same input = same score, not random based on GPT sampling)

    We mock ChatOpenAI.invoke() to return pre-defined EvaluatorOutput JSON,
    simulating what a well-functioning evaluator would produce.

    This is called "mock-based testing" or "test doubles" — we replace the real
    dependency with a controllable substitute for the purpose of isolation.
"""

import pytest                               # test framework
from unittest.mock import patch, MagicMock  # LLM mocking
from typing import List                      # type hints

from state import ResearchState, ScrapedPage, MemoryRecord


# ─────────────────────────────────────────────────────────────────────────────
# TEST STATE BUILDERS
# Helper functions that construct synthetic ResearchState dicts for testing.
# This keeps individual test methods clean and focused on assertions.
# ─────────────────────────────────────────────────────────────────────────────

def build_state_with_poor_content() -> dict:
    """
    Build a ResearchState where scraped content is completely irrelevant to the question.

    Scenario: Question is about RAG hallucinations, but all scraped pages are about
    cooking recipes. An ideal evaluator should score this very low (< 0.4).

    Returns:
        A dict matching ResearchState structure with irrelevant content.
    """
    return {
        "question": "What are the best techniques for reducing hallucinations in RAG systems?",
        "plan": ["RAG hallucination mitigation techniques"],
        "recalled_memories": [],
        "raw_search_results": [],
        "scraped_content": [
            ScrapedPage(
                url="https://cookingblog.com/pasta-recipes",
                title="Best Pasta Recipes for Beginners",
                content=(
                    "Learn how to make delicious pasta dishes at home. Start with simple "
                    "spaghetti carbonara using eggs, pecorino cheese, guanciale, and black pepper. "
                    "The key to authentic carbonara is to add the egg mixture off the heat to avoid "
                    "scrambling. For a creamier result, reserve some pasta water. Boil pasta al dente "
                    "for the best texture. Italian cooking is all about high-quality ingredients "
                    "and proper technique. Buon appetito!"
                ),
                success=True,
            ),
            ScrapedPage(
                url="https://gardeningsite.com/tomato-growing",
                title="How to Grow Perfect Tomatoes",
                content=(
                    "Growing tomatoes requires well-draining soil, plenty of sunlight (at least 8 hours), "
                    "and consistent watering. Plant seedlings after the last frost date. Use stakes or "
                    "cages to support heavy fruit-bearing branches. Fertilize every two weeks with a "
                    "balanced NPK fertilizer. Watch for common pests like aphids and tomato hornworms. "
                    "Heirloom varieties offer superior flavor but are more disease-prone than hybrids."
                ),
                success=True,
            ),
        ],
        "iteration": 1,
        "evaluation_score": 0.0,
        "evaluation_reasoning": "",
        "final_report": "",
        "sources": [],
        "memory_stored": False,
    }


def build_state_with_comprehensive_content() -> dict:
    """
    Build a ResearchState where scraped content comprehensively answers the question.

    Scenario: Question about RAG hallucinations, and content directly covers CRAG,
    Self-RAG, RAGAS, and other key techniques. An ideal evaluator should score this
    >= 0.7 to trigger synthesis rather than another search loop.

    Returns:
        A dict matching ResearchState structure with highly relevant content.
    """
    return {
        "question": "What are the best techniques for reducing hallucinations in RAG systems?",
        "plan": [
            "RAG hallucination mitigation techniques 2025",
            "Self-RAG reflection token mechanism",
            "CRAG corrective retrieval augmented generation",
        ],
        "recalled_memories": [],
        "raw_search_results": [],
        "scraped_content": [
            ScrapedPage(
                url="https://arxiv.org/abs/2401.15884",
                title="CRAG: Corrective Retrieval Augmented Generation",
                content=(
                    "CRAG (Corrective Retrieval Augmented Generation) introduces a lightweight evaluator "
                    "trained to assess the quality of retrieved documents before generation. When the "
                    "evaluator judges retrieved context as low-quality or irrelevant, CRAG activates "
                    "a web search fallback to acquire supplementary information. Results show CRAG "
                    "significantly reduces hallucinations on knowledge-intensive QA benchmarks including "
                    "PopQA and PubHealth. The corrective mechanism operates as a plug-in module compatible "
                    "with any RAG pipeline without fine-tuning the base LLM."
                ),
                success=True,
            ),
            ScrapedPage(
                url="https://arxiv.org/abs/2310.11511",
                title="Self-RAG: Self-Reflective Retrieval Augmented Generation",
                content=(
                    "Self-RAG trains language models to adaptively retrieve passages and generate "
                    "text with special reflection tokens: ISREL (is retrieved passage relevant?), "
                    "ISSUP (does the generation support the retrieved passage?), and ISUSE (is the "
                    "response useful?). By training on a curated dataset with these tokens, Self-RAG "
                    "achieves state-of-the-art performance on six diverse tasks while reducing "
                    "unnecessary retrievals by 25%. The reflection mechanism enables the model to "
                    "self-critique and correct factual errors mid-generation."
                ),
                success=True,
            ),
            ScrapedPage(
                url="https://docs.ragas.io/hallucination-metrics",
                title="RAGAS: RAG Assessment Framework",
                content=(
                    "RAGAS provides automated evaluation metrics for RAG pipelines without requiring "
                    "human annotation. Key metrics: Faithfulness (is generated content supported by "
                    "retrieved context?), Answer Relevancy (does the answer address the question?), "
                    "Context Recall (are the relevant documents retrieved?), and Context Precision "
                    "(are retrieved documents actually relevant?). These four metrics form a comprehensive "
                    "evaluation suite for detecting hallucinations and retrieval failures in production "
                    "RAG systems."
                ),
                success=True,
            ),
        ],
        "iteration": 1,
        "evaluation_score": 0.0,
        "evaluation_reasoning": "",
        "final_report": "",
        "sources": [],
        "memory_stored": False,
    }


def build_state_with_no_content() -> dict:
    """
    Build a ResearchState where no content was scraped at all.

    This simulates a complete scraping failure (e.g., all URLs returned 404/timeout).
    The evaluator should return 0.0 to trigger another search iteration.

    Returns:
        A dict with an empty scraped_content list.
    """
    return {
        "question": "What are the best techniques for reducing hallucinations in RAG systems?",
        "plan": ["RAG hallucination mitigation"],
        "recalled_memories": [],
        "raw_search_results": [],
        "scraped_content": [],  # intentionally empty
        "iteration": 1,
        "evaluation_score": 0.0,
        "evaluation_reasoning": "",
        "final_report": "",
        "sources": [],
        "memory_stored": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASSES
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluatorWithPoorContext:
    """Tests for the evaluator's behavior on irrelevant/insufficient content."""

    def test_poor_content_produces_low_score(self):
        """
        WHAT: Supply cooking/gardening content for a RAG question → expect score < 0.5.

        WHY: The evaluator must correctly identify when scraped content doesn't address
        the research question. If it scores irrelevant content as 0.8+, the agent would
        skip additional searches and synthesize a report with no useful information.

        HOW: We mock the LLM to return a low score (0.2) — representing what a well-
        calibrated evaluator would produce for completely off-topic content.
        We then verify that the node correctly uses this score.
        """
        mock_evaluator_output = MagicMock()
        mock_evaluator_output.score = 0.2  # low score = irrelevant content
        mock_evaluator_output.reasoning = (
            "The retrieved content discusses cooking and gardening, which is completely "
            "unrelated to the research question about RAG hallucination mitigation techniques. "
            "None of the core topics (Self-RAG, CRAG, faithfulness, RAGAS) are covered."
        )
        mock_evaluator_output.missing_topics = [
            "RAG hallucination techniques",
            "CRAG corrective retrieval",
            "Self-RAG reflection tokens",
            "RAGAS evaluation metrics",
        ]

        with patch("agent_nodes._get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_structured_llm = MagicMock()
            mock_structured_llm.invoke.return_value = mock_evaluator_output
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_get_llm.return_value = mock_llm

            from agent_nodes import evaluator_node
            result = evaluator_node(build_state_with_poor_content())

        assert "evaluation_score" in result, "evaluator_node must return 'evaluation_score' key"
        score = result["evaluation_score"]

        assert score < 0.5, (
            f"Poor/irrelevant content should produce a low evaluation score (< 0.5), "
            f"but got {score:.2f}. The agent would incorrectly proceed to synthesis."
        )

    def test_poor_content_reasoning_is_returned(self):
        """
        WHAT: Verify that the evaluator's reasoning text is returned in the state update.

        WHY: The reasoning text is displayed in the Streamlit UI as "gap analysis."
        Without it, users can't understand WHY the agent is doing another search loop.
        """
        mock_evaluator_output = MagicMock()
        mock_evaluator_output.score = 0.15
        mock_evaluator_output.reasoning = "Content is completely off-topic — cooking and gardening only."
        mock_evaluator_output.missing_topics = ["RAG techniques"]

        with patch("agent_nodes._get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_structured_llm = MagicMock()
            mock_structured_llm.invoke.return_value = mock_evaluator_output
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_get_llm.return_value = mock_llm

            from agent_nodes import evaluator_node
            result = evaluator_node(build_state_with_poor_content())

        assert "evaluation_reasoning" in result, "evaluator_node must return 'evaluation_reasoning' key"
        assert len(result["evaluation_reasoning"]) > 0, "Reasoning should be a non-empty string"


class TestEvaluatorWithComprehensiveContext:
    """Tests for the evaluator's behavior on highly relevant content."""

    def test_comprehensive_content_produces_high_score(self):
        """
        WHAT: Supply relevant CRAG/Self-RAG/RAGAS content for a RAG question → score >= 0.7.

        WHY: When the agent has found high-quality, relevant content, it should
        recognize this and proceed directly to synthesis without wasting another
        search iteration. A score >= 0.7 triggers the "synthesize" route.

        HOW: Mock the LLM to return 0.88 — representing a well-calibrated evaluator
        finding comprehensive coverage of the research question.
        """
        mock_evaluator_output = MagicMock()
        mock_evaluator_output.score = 0.88  # high score = comprehensive content
        mock_evaluator_output.reasoning = (
            "The retrieved content comprehensively covers RAG hallucination mitigation techniques. "
            "CRAG's corrective mechanism, Self-RAG's reflection tokens, and RAGAS evaluation "
            "metrics are all covered with technical depth. The content provides actionable "
            "techniques, benchmark results, and implementation details."
        )
        mock_evaluator_output.missing_topics = []  # no missing topics

        with patch("agent_nodes._get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_structured_llm = MagicMock()
            mock_structured_llm.invoke.return_value = mock_evaluator_output
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_get_llm.return_value = mock_llm

            from agent_nodes import evaluator_node
            result = evaluator_node(build_state_with_comprehensive_content())

        score = result["evaluation_score"]
        assert score >= 0.7, (
            f"Comprehensive, relevant content should produce a high evaluation score (>= 0.7), "
            f"but got {score:.2f}. The agent would incorrectly loop back to search."
        )

    def test_score_is_within_valid_range(self):
        """
        WHAT: Evaluator scores must always be within [0.0, 1.0].

        WHY: The conditional edge in graph_builder.py compares the score to 0.7.
        If the score is negative or > 1.0 (due to a conversion bug), routing logic
        would behave incorrectly — potentially looping forever or never looping.
        """
        mock_evaluator_output = MagicMock()
        mock_evaluator_output.score = 0.92
        mock_evaluator_output.reasoning = "Highly relevant content."
        mock_evaluator_output.missing_topics = []

        with patch("agent_nodes._get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_structured_llm = MagicMock()
            mock_structured_llm.invoke.return_value = mock_evaluator_output
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_get_llm.return_value = mock_llm

            from agent_nodes import evaluator_node
            result = evaluator_node(build_state_with_comprehensive_content())

        score = result["evaluation_score"]
        assert 0.0 <= score <= 1.0, (
            f"Evaluation score must be in [0.0, 1.0] range, got {score}"
        )


class TestEvaluatorEdgeCases:
    """Tests for edge cases and error handling in the evaluator."""

    def test_no_scraped_content_returns_zero_score(self):
        """
        WHAT: When scraped_content is empty, evaluator should return score=0.0.

        WHY: The evaluator has a guard clause: "if no content, score 0.0 immediately."
        This prevents sending an empty content summary to the LLM (which would waste
        tokens and might produce a nonsensical score like 0.5 for emptiness).

        HOW: We do NOT mock the LLM here — we test that the node's guard clause
        handles empty content BEFORE calling the LLM.
        """
        from agent_nodes import evaluator_node
        result = evaluator_node(build_state_with_no_content())

        assert result["evaluation_score"] == 0.0, (
            f"Empty scraped_content should immediately return score=0.0, "
            f"but got {result['evaluation_score']}"
        )
        assert "evaluation_reasoning" in result, "Should always return reasoning even for empty state"

    def test_evaluator_returns_required_keys(self):
        """
        WHAT: evaluator_node must always return both 'evaluation_score' AND 'evaluation_reasoning'.

        WHY: LangGraph will raise a KeyError if a node returns unexpected keys or
        misses expected keys. The graph schema defines which keys each node can update.
        Both keys are used downstream (score for routing, reasoning for UI display).
        """
        mock_evaluator_output = MagicMock()
        mock_evaluator_output.score = 0.65
        mock_evaluator_output.reasoning = "Partially sufficient."
        mock_evaluator_output.missing_topics = []

        with patch("agent_nodes._get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_structured_llm = MagicMock()
            mock_structured_llm.invoke.return_value = mock_evaluator_output
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_get_llm.return_value = mock_llm

            from agent_nodes import evaluator_node
            result = evaluator_node(build_state_with_comprehensive_content())

        assert "evaluation_score" in result, "Must return 'evaluation_score' key"
        assert "evaluation_reasoning" in result, "Must return 'evaluation_reasoning' key"
        assert isinstance(result["evaluation_score"], float), "Score must be a float"
        assert isinstance(result["evaluation_reasoning"], str), "Reasoning must be a string"

    def test_evaluator_handles_llm_failure_gracefully(self):
        """
        WHAT: If the LLM call raises an exception, evaluator should return a fallback score.

        WHY: The LLM API can fail due to rate limits, network errors, or invalid keys.
        The evaluator must handle this gracefully — using a fallback score based on
        the current iteration — so the graph can continue rather than crashing.
        """
        with patch("agent_nodes._get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_structured_llm = MagicMock()
            mock_structured_llm.invoke.side_effect = Exception("OpenAI API rate limit exceeded")
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_get_llm.return_value = mock_llm

            from agent_nodes import evaluator_node
            result = evaluator_node(build_state_with_comprehensive_content())

        assert "evaluation_score" in result, "Must return score even on LLM failure"
        score = result["evaluation_score"]
        assert 0.0 <= score <= 1.0, f"Fallback score must be in [0.0, 1.0], got {score}"
        assert "evaluation_reasoning" in result, "Must return reasoning even on LLM failure"
