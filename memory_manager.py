"""
memory_manager.py — Persistent Vector Store Interface
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS A VECTOR DATABASE?

    A vector database stores "embeddings" — dense numerical arrays (e.g., 1536
    floating-point numbers) that encode the semantic meaning of text. When you
    search it, you provide a query embedding and the DB finds stored embeddings
    that are geometrically close in this high-dimensional space.

    "Close" is measured by COSINE SIMILARITY:
        similarity = dot(A, B) / (|A| × |B|)

    Where A and B are two embedding vectors. The result ranges:
        1.0  → vectors point in exactly the same direction (identical meaning)
        0.0  → vectors are perpendicular (completely unrelated meaning)
       -1.0  → vectors point in opposite directions (antonyms, rarely seen in practice)

    Example:
        embed("GraphRAG reduces hallucinations") ≈ embed("RAG hallucination mitigation")
        → similarity ≈ 0.88 (very similar)

        embed("GraphRAG reduces hallucinations") ≈ embed("Python list comprehensions")
        → similarity ≈ 0.12 (unrelated)

HOW VECTOR INDICES WORK (HNSW):

    ChromaDB uses HNSW (Hierarchical Navigable Small World) graphs as its index
    structure. Instead of scanning every stored vector (O(n) linear scan), HNSW
    builds a multi-layer graph where similar vectors are connected. At query time,
    it navigates the graph in O(log n), making retrieval fast even with millions
    of stored documents.

WHY LONG-TERM MEMORY PREVENTS WASTED API CALLS:

    Without memory, every new query triggers a full web search + scrape cycle.
    If you ask about "RAG hallucinations" twice in a week, you pay for two full
    runs. With memory, the second query finds the stored summary at similarity
    0.91 and the planner can skip sub-queries already answered, saving ~$0.15
    in API costs and ~30 seconds of latency.

DUAL EMBEDDING STRATEGY:
    Primary   → OpenAI text-embedding-3-small (1536 dims, requires API key)
    Fallback  → HuggingFace all-MiniLM-L6-v2 (384 dims, fully local, free)
    The fallback ensures the agent works in offline/air-gapped environments.
"""

import os                              # for reading environment variables and file paths
import uuid                            # for generating unique memory document IDs
from typing import List, Dict, Any, Optional  # type hints for all public methods
from datetime import datetime          # for timestamping stored memories

import chromadb                        # the embedded vector database client
from chromadb.config import Settings   # ChromaDB configuration options
from dotenv import load_dotenv         # loads .env file into os.environ at import time

from config import cfg                 # central hyperparameters
from state import MemoryRecord         # validated memory schema

load_dotenv()  # reads .env file so OPENAI_API_KEY is available via os.environ


class MemoryManager:
    """
    Persistent long-term memory system backed by ChromaDB.

    Responsibilities:
        1. Initialize and connect to the ChromaDB local vector store.
        2. Choose the right embedding function (OpenAI vs local HuggingFace).
        3. Expose clean CRUD-like methods: recall, store, list, clear.

    WHY A CLASS INSTEAD OF LOOSE FUNCTIONS?
        The ChromaDB client and collection objects are expensive to initialize
        (they open SQLite files, load index structures). By wrapping them in a
        class, we initialize once and reuse the connection across all method
        calls, avoiding repeated disk I/O overhead.
    """

    def __init__(self, db_path: str = cfg.db_path, collection_name: str = cfg.collection_name):
        """
        Initialize the MemoryManager.

        Args:
            db_path: Filesystem path to the ChromaDB persistence directory.
                     Created automatically if it doesn't exist.
            collection_name: Name of the ChromaDB collection (like a table name).
                             Separate collections isolate different agents' memories.
        """
        self.db_path = db_path                    # store path for diagnostics
        self.collection_name = collection_name    # store name for diagnostics

        # Create the persistence directory if it doesn't exist yet.
        # exist_ok=True prevents a crash if the folder already exists.
        os.makedirs(db_path, exist_ok=True)
        print(f"[MemoryManager] Using ChromaDB at: {os.path.abspath(db_path)}")

        # Detect which embedding strategy to use based on API key availability.
        # We check for the key ONCE at initialization so every subsequent embed
        # call doesn't redundantly re-read the environment.
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")  # empty string = falsy
        self._using_openai = bool(self._openai_key)              # True if key is set

        # Select the embedding function.
        # ChromaDB's embedding_function parameter accepts any callable that maps
        # List[str] → List[List[float]]. Both adapters below implement this interface.
        if self._using_openai:
            self._embedding_fn = self._build_openai_embedding_fn()
            print("[MemoryManager] Using OpenAI embeddings (text-embedding-3-small)")
        else:
            self._embedding_fn = self._build_local_embedding_fn()
            print("[MemoryManager] No OPENAI_API_KEY found — using local SentenceTransformers fallback")

        # Create the ChromaDB client with persistent storage.
        # PersistentClient writes to disk automatically; no manual .persist() needed.
        self._client = chromadb.PersistentClient(
            path=db_path,  # ChromaDB creates SQLite + HNSW files here
        )

        # Get or create the collection.
        # get_or_create_collection is idempotent — safe to call every time the
        # agent starts without wiping existing data.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,  # used for ALL add() and query() calls
            metadata={"hnsw:space": "cosine"},       # use cosine similarity (not L2 Euclidean)
        )
        print(f"[MemoryManager] Collection '{collection_name}' loaded. "
              f"Records: {self._collection.count()}")

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: EMBEDDING FUNCTION BUILDERS
    # These methods return a ChromaDB-compatible embedding function object.
    # We defer the actual model loading to here so the constructor stays fast.
    # ─────────────────────────────────────────────────────────────────────────

    def _build_openai_embedding_fn(self):
        """
        Builds a ChromaDB embedding function backed by OpenAI's API.

        Why OpenAI embeddings?
            text-embedding-3-small produces high-quality 1536-dimensional vectors
            in ~50ms/request via the API. The vectors are aligned with GPT-4o's
            tokenization space, meaning the LLM's understanding of text is
            geometrically consistent with what's stored in the vector DB.

        Returns:
            A chromadb.utils.embedding_functions.OpenAIEmbeddingFunction instance.
        """
        from chromadb.utils import embedding_functions  # lazy import to keep startup fast

        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=self._openai_key,            # the key read from environment
            model_name=cfg.embedding_model_name, # "text-embedding-3-small"
        )

    def _build_local_embedding_fn(self):
        """
        Builds a ChromaDB embedding function backed by a local SentenceTransformer.

        Why SentenceTransformers?
            The all-MiniLM-L6-v2 model produces 384-dimensional vectors using a
            distilled BERT architecture. It runs entirely on CPU, requires no
            internet connection after the first download (model cached in
            ~/.cache/huggingface/), and processes ~1000 sentences/second on
            modern hardware.

        Note on dimension mismatch:
            If you have existing ChromaDB data from OpenAI (1536 dims) and then
            switch to local (384 dims), the stored embeddings and new query
            embeddings will have incompatible shapes → retrieval will fail.
            Always use the same embedding model for a given collection.

        Returns:
            A chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction.
        """
        from chromadb.utils import embedding_functions  # lazy import

        print(f"[MemoryManager] Loading local model: {cfg.local_embedding_model}")
        print("[MemoryManager] This downloads ~22 MB on first run (cached after that).")

        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=cfg.local_embedding_model  # "sentence-transformers/all-MiniLM-L6-v2"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def recall_memories(
        self,
        query: str,
        top_k: int = cfg.top_k_recall,
        similarity_threshold: float = cfg.similarity_threshold,
    ) -> List[MemoryRecord]:
        """
        Retrieve the most semantically relevant past memories for a given query.

        HOW IT WORKS STEP-BY-STEP:
            1. The query string is embedded → a 1536-dim (or 384-dim) float array.
            2. ChromaDB's HNSW index computes cosine similarity between this
               query vector and all stored document vectors.
            3. The top_k most similar documents are returned with their scores.
            4. We filter out documents below similarity_threshold (default 0.7).
            5. Results are wrapped in MemoryRecord Pydantic objects for type safety.

        WHY FILTERING BELOW THRESHOLD?
            HNSW always returns exactly top_k results, even if they're not relevant.
            If you stored memories about Python and asked about quantum physics,
            HNSW would still return the Python memories as "most similar" — just with
            a low score. The threshold prevents the agent from building on irrelevant
            prior research.

        Args:
            query: The current research question being investigated.
            top_k: Maximum number of memories to retrieve.
            similarity_threshold: Minimum cosine similarity to include a result.

        Returns:
            List of MemoryRecord objects sorted by descending similarity score.
        """
        # Guard against empty collection — HNSW query on empty collection crashes
        if self._collection.count() == 0:
            print("[MemoryManager] Memory store is empty — cold start, no prior context.")
            return []  # return empty list, not None; callers always expect a list

        print(f"[MemoryManager] Querying memory for: '{query[:80]}...' (top_k={top_k})")

        try:
            # ChromaDB query — this is where the vector similarity search happens.
            # include=["documents", "distances", "metadatas"] tells ChromaDB to
            # return the stored text, the cosine distance, and the metadata dict.
            results = self._collection.query(
                query_texts=[query],                             # list of 1 query string
                n_results=min(top_k, self._collection.count()), # can't ask for more than exists
                include=["documents", "distances", "metadatas"], # what to return per result
            )
        except Exception as e:
            # Network errors (OpenAI embeddings) or index corruption (rare) should
            # not crash the entire agent — just return empty memories.
            print(f"[MemoryManager] Memory recall failed: {e}")
            return []

        memories: List[MemoryRecord] = []  # accumulate valid memories here

        # ChromaDB returns nested lists because it supports batch queries.
        # results["documents"][0] = the document texts for query #0 (our only query)
        # results["distances"][0] = the cosine distances for query #0
        # ChromaDB returns DISTANCE (0=identical, 2=opposite), not similarity.
        # We convert: similarity = 1 - (distance / 2) to get a 0.0–1.0 score.

        docs = results["documents"][0]        # list of stored document strings
        distances = results["distances"][0]   # list of cosine distances (0.0–2.0)
        metadatas = results["metadatas"][0]   # list of metadata dicts

        for doc, dist, meta in zip(docs, distances, metadatas):
            similarity = 1.0 - (dist / 2.0)  # convert cosine distance to similarity score

            if similarity < similarity_threshold:
                # Below threshold — too dissimilar to be useful context.
                # Still log it for debugging but don't add to results.
                print(f"[MemoryManager] Skipping memory (similarity={similarity:.2f} < {similarity_threshold}): "
                      f"{meta.get('topic', 'unknown')[:40]}")
                continue  # skip to next result without adding to the list

            # Wrap in MemoryRecord for type-safe downstream consumption
            memory = MemoryRecord(
                topic=meta.get("topic", "Unknown Topic"),    # stored at write time
                summary=doc,                                  # the actual text content
                similarity_score=round(similarity, 4),        # round for cleaner display
                metadata=meta,                                # full metadata dict pass-through
            )
            memories.append(memory)
            print(f"[MemoryManager] ✓ Recalled memory (similarity={similarity:.2f}): "
                  f"{memory.topic[:60]}")

        print(f"[MemoryManager] Retrieved {len(memories)} relevant memories above threshold.")
        return memories  # sorted by descending similarity because HNSW returns in order

    def store_memory(
        self,
        topic: str,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Persist a new research insight into the vector store.

        WHAT GETS STORED:
            ChromaDB stores 3 things per document:
            1. The raw text (summary) — returned during recall for the LLM to read.
            2. The embedding vector — used for similarity search.
            3. Metadata dict — arbitrary key-value pairs for filtering and display.

        WHY DEDUPLICATE?
            If the agent runs the same query twice, it would store two nearly identical
            summaries. We check if an extremely similar document already exists
            (similarity > 0.95) and skip storing if so, preventing memory bloat.

        Args:
            topic: Short descriptive label (e.g., "RAG Hallucination Mitigations 2025").
            summary: The full research summary text to store and embed.
            metadata: Extra context like source URLs, iteration count, timestamp.

        Returns:
            The UUID string assigned to this memory entry.
        """
        if not summary or not summary.strip():
            # Don't store empty summaries — they waste index slots and confuse recall
            print("[MemoryManager] Skipping storage of empty summary.")
            return ""

        # Build the metadata dict with default fields merged with caller-provided extras
        full_metadata = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),  # ISO format: "2025-07-21T14:30:00"
            "char_length": len(summary),               # for size monitoring
        }
        if metadata:
            full_metadata.update(metadata)  # caller fields override defaults if same key

        # Check for near-duplicate before storing (similarity > 0.95 = essentially same content)
        if self._collection.count() > 0:
            try:
                existing = self._collection.query(
                    query_texts=[summary],
                    n_results=1,
                    include=["distances"],
                )
                if existing["distances"][0]:
                    top_distance = existing["distances"][0][0]
                    top_similarity = 1.0 - (top_distance / 2.0)
                    if top_similarity > 0.95:
                        # This summary is nearly identical to an existing one — skip storage
                        print(f"[MemoryManager] Near-duplicate detected (similarity={top_similarity:.3f}). "
                              f"Skipping storage to prevent memory bloat.")
                        return ""
            except Exception:
                pass  # If dedup check fails, proceed with storage anyway (better to store twice than lose data)

        doc_id = str(uuid.uuid4())  # universally unique ID — guarantees no collision across runs

        try:
            self._collection.add(
                documents=[summary],      # the text to embed and store
                metadatas=[full_metadata], # the metadata dict
                ids=[doc_id],             # unique identifier
            )
            print(f"[MemoryManager] ✓ Stored memory: '{topic[:60]}' (id={doc_id[:8]}...)")
            return doc_id
        except Exception as e:
            print(f"[MemoryManager] Failed to store memory: {e}")
            return ""  # return empty string so callers can check truthiness

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """
        Fetch every stored memory for display in the Streamlit memory explorer.

        WHY NOT RETURN MemoryRecord OBJECTS HERE?
            The memory explorer tab needs raw metadata for filtering/sorting in
            Streamlit's st.dataframe(). Returning plain dicts is simpler for UI
            code than unpacking Pydantic models.

        Returns:
            List of dicts, each with keys: id, topic, summary, timestamp, char_length.
            Returns empty list if the collection is empty.
        """
        if self._collection.count() == 0:
            return []  # early exit — don't call .get() on empty collection

        try:
            results = self._collection.get(
                include=["documents", "metadatas"]  # retrieve text + metadata, skip embeddings
            )
        except Exception as e:
            print(f"[MemoryManager] Failed to fetch all memories: {e}")
            return []

        memories = []
        for doc_id, document, metadata in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        ):
            memories.append({
                "id": doc_id,
                "topic": metadata.get("topic", "Unknown"),
                "summary": document,
                "timestamp": metadata.get("timestamp", ""),
                "char_length": metadata.get("char_length", len(document)),
                "metadata": metadata,  # full metadata for advanced display
            })

        # Sort by timestamp descending so newest memories appear first in the UI
        memories.sort(key=lambda x: x["timestamp"], reverse=True)
        return memories

    def clear_memory(self) -> bool:
        """
        Delete all stored memories and recreate the empty collection.

        WHY RECREATE INSTEAD OF DELETE-ALL?
            ChromaDB's delete_collection + get_or_create_collection is cleaner than
            attempting to delete all documents individually. Recreating also resets
            the HNSW index structure, reclaiming disk space that individual deletions
            don't always release immediately.

        Returns:
            True if cleared successfully, False if an error occurred.
        """
        try:
            self._client.delete_collection(self.collection_name)  # removes all data + index
            # Recreate with the same settings so subsequent calls still work
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"[MemoryManager] ✓ Cleared all memories in collection '{self.collection_name}'.")
            return True
        except Exception as e:
            print(f"[MemoryManager] Failed to clear memory: {e}")
            return False

    @property
    def count(self) -> int:
        """Returns the number of stored memory documents. Useful for UI badges."""
        return self._collection.count()
