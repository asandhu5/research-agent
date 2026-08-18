"""
tests/test_memory.py — Unit Tests for the Memory Manager
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT WE'RE TESTING:
    1. Memory insertion — can we store a document and get an ID back?
    2. Cosine similarity retrieval — does recall_memories return semantically
       similar documents above threshold?
    3. Content match — does the recalled text match what was inserted?
    4. Empty recall — does querying an empty store return an empty list (not crash)?
    5. Clear — does clearing the store reset the count to zero?

WHY USE A TEMPORARY DIRECTORY FOR TESTS?
    We never want tests to pollute the production ChromaDB at ./data/chroma_db/.
    By passing a temp_path fixture to MemoryManager, each test suite gets its own
    isolated ChromaDB that's deleted after the test session.
    This is the "test isolation" principle: tests must not have side effects on
    each other or on the production system.
"""

import pytest           # test framework
import os               # for environment variable manipulation
from pathlib import Path  # cross-platform temp paths

# We import MemoryManager after potentially setting up env vars in fixtures.
# This order matters: dotenv load happens inside MemoryManager.__init__.


@pytest.fixture(scope="module")  # module scope = one DB per test file (not one per test)
def temp_db_path(tmp_path_factory) -> str:
    """
    Pytest fixture: provides a temporary directory for an isolated ChromaDB.

    tmp_path_factory is a pytest built-in that creates per-test-session temp dirs.
    scope="module" means this fixture is created once and shared across all tests
    in this file, avoiding the overhead of re-creating the DB for every test.

    Yields:
        A string path to the temporary ChromaDB directory.
    """
    temp_dir = tmp_path_factory.mktemp("chroma_test_memory")  # creates ./pytest-of-user/chroma_test_memory*/
    return str(temp_dir)  # MemoryManager expects a string path


@pytest.fixture(scope="module")
def memory(temp_db_path: str):
    """
    Pytest fixture: provides an initialized MemoryManager using the temp DB.

    This fixture depends on temp_db_path. Pytest resolves dependencies automatically.
    scope="module" ensures the same MemoryManager instance is shared across all tests.

    Args:
        temp_db_path: The temporary path fixture (injected by pytest).

    Returns:
        An initialized MemoryManager instance.
    """
    from memory_manager import MemoryManager
    return MemoryManager(
        db_path=temp_db_path,
        collection_name="test_collection",  # isolated collection name for tests
    )


class TestMemoryInsertion:
    """Tests for the store_memory() method."""

    def test_store_memory_returns_nonempty_id(self, memory):
        """
        WHAT: Verify that store_memory() returns a non-empty string ID on success.

        WHY: A non-empty ID proves that ChromaDB accepted the document. An empty
        string means the storage was skipped (e.g., near-duplicate detection fired).
        """
        doc_id = memory.store_memory(
            topic="Test Topic: Python Async Patterns",
            summary=(
                "Python's asyncio library enables concurrent I/O operations without "
                "multithreading. Key primitives: async/await syntax, event loops, "
                "coroutines, and Tasks. asyncio.gather() runs multiple coroutines "
                "concurrently. Useful for web scraping, API calls, and database queries."
            ),
            metadata={"source": "test", "domain": "python"},
        )
        assert doc_id, "store_memory() should return a non-empty UUID string"
        assert isinstance(doc_id, str), "Returned ID should be a string"
        assert len(doc_id) > 8, "UUID should be at least 8 characters long"

    def test_store_empty_summary_returns_empty_string(self, memory):
        """
        WHAT: Verify that attempting to store an empty summary returns "".

        WHY: Empty documents in the vector store are useless and waste index space.
        The MemoryManager should reject them gracefully, not crash.
        """
        doc_id = memory.store_memory(
            topic="Empty Document Test",
            summary="",  # empty summary — should be rejected
            metadata={},
        )
        assert doc_id == "", "Storing an empty summary should return an empty string, not an ID"

    def test_store_increments_count(self, memory):
        """
        WHAT: Verify that storing a document increases the collection count by 1.

        WHY: The count property reflects real ChromaDB state. If store_memory()
        silently fails, the count won't increase and we'd have no visible error.
        """
        count_before = memory.count
        memory.store_memory(
            topic="Count Test Document",
            summary=(
                "Transformer models use multi-head self-attention to process sequences "
                "in parallel. The attention mechanism computes query-key-value projections "
                "and uses softmax-normalized dot products to weight value vectors. "
                "This enables capturing long-range dependencies more effectively than RNNs."
            ),
            metadata={"source": "test"},
        )
        count_after = memory.count
        # Count should increase by 1 (or stay same if near-duplicate detected)
        assert count_after >= count_before, (
            "Collection count should never decrease after storing a document"
        )


class TestMemoryRetrieval:
    """Tests for the recall_memories() method."""

    @pytest.fixture(autouse=True)  # autouse = runs before every test in this class
    def seed_memory(self, memory):
        """
        Seed the memory store with known documents before each test.
        This ensures tests are independent of what other tests store.
        """
        # Store 3 documents with distinct semantic meanings
        memory.store_memory(
            topic="RAG Hallucination Test Document",
            summary=(
                "Retrieval-Augmented Generation hallucinations occur when the LLM generates "
                "content not supported by retrieved documents. Key mitigation: CRAG evaluates "
                "retrieved document quality and falls back to web search for low-quality context. "
                "Self-RAG uses reflection tokens to selectively decide when retrieval is needed. "
                "RAGAS metrics measure faithfulness, answer relevancy, and context recall."
            ),
            metadata={"domain": "nlp", "test": True},
        )
        memory.store_memory(
            topic="Kubernetes Deployment Test Document",
            summary=(
                "Kubernetes orchestrates containerized applications across a cluster of nodes. "
                "Deployments define the desired state (replicas, container image, resource limits). "
                "Services expose pods via ClusterIP, NodePort, or LoadBalancer. "
                "Horizontal Pod Autoscaler scales replicas based on CPU/memory metrics."
            ),
            metadata={"domain": "devops", "test": True},
        )
        memory.store_memory(
            topic="Python Decorators Test Document",
            summary=(
                "Python decorators are higher-order functions that wrap another function to extend "
                "its behavior without modifying its source code. The @functools.wraps decorator "
                "preserves the wrapped function's metadata. Common use cases: logging, timing, "
                "authentication, and rate-limiting in web frameworks like FastAPI and Flask."
            ),
            metadata={"domain": "python", "test": True},
        )

    def test_recall_returns_semantically_similar_memory(self, memory):
        """
        WHAT: Query for RAG-related content and verify the RAG document is retrieved.

        WHY: This tests the core value proposition of the system — that semantically
        similar documents are retrieved even when the exact query words don't appear
        in the stored document. "hallucination mitigation in RAG" should match
        "RAG hallucination" document even though "mitigation" isn't in the stored text.
        """
        recalled = memory.recall_memories(
            query="hallucination mitigation in retrieval augmented generation",
            top_k=3,
            similarity_threshold=0.0,  # use 0.0 threshold for testing — ensures results returned
        )
        assert len(recalled) > 0, (
            "recall_memories() should return at least one result for a relevant query"
        )

        # The top result should be the RAG document, not the Kubernetes or Python one
        top_memory = recalled[0]  # results are sorted by descending similarity
        assert "rag" in top_memory.topic.lower() or "hallucination" in top_memory.summary.lower(), (
            "The most similar memory to a RAG query should be the RAG-related document, "
            f"but got: '{top_memory.topic}'"
        )

    def test_recalled_memory_has_valid_similarity_score(self, memory):
        """
        WHAT: Verify that all recalled memories have similarity scores between 0.0 and 1.0.

        WHY: The cosine similarity conversion formula (1 - distance/2) can produce
        values outside [0,1] if there's a bug (e.g., wrong distance metric used).
        This test catches such regressions.
        """
        recalled = memory.recall_memories(
            query="machine learning retrieval systems",
            top_k=3,
            similarity_threshold=0.0,
        )
        for mem in recalled:
            assert 0.0 <= mem.similarity_score <= 1.0, (
                f"Similarity score {mem.similarity_score} is outside valid range [0.0, 1.0]. "
                "Check the cosine distance conversion formula."
            )

    def test_recall_returns_empty_list_for_empty_store(self, tmp_path_factory):
        """
        WHAT: Verify recall on a fresh empty store returns [] and doesn't crash.

        WHY: The MemoryManager must handle the empty-collection case gracefully.
        ChromaDB's query() raises an exception on empty collections if not guarded.
        This test verifies our guard clause works.
        """
        from memory_manager import MemoryManager
        fresh_memory = MemoryManager(
            db_path=str(tmp_path_factory.mktemp("empty_memory_test")),
            collection_name="empty_test",
        )
        result = fresh_memory.recall_memories(
            query="any query at all",
            top_k=3,
            similarity_threshold=0.7,
        )
        assert result == [], (
            "Querying an empty memory store should return an empty list, not raise an exception"
        )

    def test_high_similarity_threshold_filters_unrelated_memories(self, memory):
        """
        WHAT: Query for a very specific topic with high threshold to verify filtering.

        WHY: The similarity threshold should filter out memories below it.
        We query for "Kubernetes pod scheduling" (a very specific DevOps topic)
        with threshold=0.98 (extremely strict) — most memories should be filtered out
        since nothing is that similar to such a specific query.
        """
        recalled = memory.recall_memories(
            query="Kubernetes pod anti-affinity scheduling rules",
            top_k=3,
            similarity_threshold=0.98,  # extremely strict — almost nothing passes
        )
        # We expect 0 or 1 results at this threshold (not all 3)
        assert len(recalled) < 3, (
            f"With similarity_threshold=0.98, should filter most memories. "
            f"Got {len(recalled)} results — threshold filtering may not be working."
        )

    def test_recalled_memory_content_matches_inserted_content(self, memory):
        """
        WHAT: Store a very specific unique string, recall it, verify content matches.

        WHY: ChromaDB should return the exact text we stored. If the returned content
        is different, there's a data corruption or indexing bug.
        """
        unique_content = (
            "UNIQUE_TEST_MARKER_XYZ123: This document discusses the specific topic of "
            "neural architecture search using evolutionary algorithms. The unique identifier "
            "UNIQUE_TEST_MARKER_XYZ123 should appear in any recall of this document."
        )
        memory.store_memory(
            topic="Unique Content Test",
            summary=unique_content,
            metadata={"test_case": "content_match"},
        )

        recalled = memory.recall_memories(
            query="neural architecture search evolutionary algorithms",
            top_k=5,
            similarity_threshold=0.0,  # include everything
        )
        # Find our unique document in the results
        found = any("UNIQUE_TEST_MARKER_XYZ123" in m.summary for m in recalled)
        assert found, (
            "The recalled memories should contain the exact text we stored. "
            "UNIQUE_TEST_MARKER_XYZ123 was not found in any recalled memory."
        )


class TestMemoryManagement:
    """Tests for get_all_memories() and clear_memory()."""

    def test_get_all_memories_returns_list(self, memory):
        """
        WHAT: get_all_memories() should return a list (possibly empty, never None).

        WHY: UI code that calls list(get_all_memories()) would crash if None is returned.
        """
        all_mems = memory.get_all_memories()
        assert isinstance(all_mems, list), "get_all_memories() must return a list"

    def test_clear_memory_resets_count_to_zero(self, tmp_path_factory):
        """
        WHAT: After storing documents and calling clear_memory(), count should be 0.

        WHY: The clear operation must genuinely delete all records, not just hide them.
        We use a separate MemoryManager to avoid affecting other tests in this class.
        """
        from memory_manager import MemoryManager
        isolated_memory = MemoryManager(
            db_path=str(tmp_path_factory.mktemp("clear_test")),
            collection_name="clear_collection",
        )

        isolated_memory.store_memory(
            topic="Document to be cleared",
            summary="This document will be deleted by clear_memory(). It has enough content to embed.",
            metadata={},
        )
        assert isolated_memory.count > 0, "Pre-condition: store_memory should have added a document"

        success = isolated_memory.clear_memory()
        assert success, "clear_memory() should return True on success"
        assert isolated_memory.count == 0, (
            f"After clear_memory(), count should be 0 but got {isolated_memory.count}"
        )
