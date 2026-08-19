import os                              # for reading environment variables and file paths
import uuid                            # for generating unique memory document IDs
from typing import List, Dict, Any, Optional  # type hints for all public methods
from datetime import datetime          # for timestamping stored memories

import chromadb                        # the embedded vector database client
from chromadb.config import Settings   # ChromaDB configuration options
from dotenv import load_dotenv         # loads .env file into os.environ at import time

from config import cfg                 # central hyperparameters
from state import MemoryRecord         # validated memory schema

load_dotenv()


class MemoryManager:

    def __init__(self, db_path: str = cfg.db_path, collection_name: str = cfg.collection_name):

        self.db_path = db_path
        self.collection_name = collection_name

        os.makedirs(db_path, exist_ok=True)
        print(f"[MemoryManager] Using ChromaDB at: {os.path.abspath(db_path)}")

        self._openai_key = os.environ.get("OPENAI_API_KEY", "")
        self._using_openai = bool(self._openai_key)

        if self._using_openai:
            self._embedding_fn = self._build_openai_embedding_fn()
            print("[MemoryManager] Using OpenAI embeddings (text-embedding-3-small)")
        else:
            self._embedding_fn = self._build_local_embedding_fn()
            print("[MemoryManager] No OPENAI_API_KEY found — using local SentenceTransformers fallback")

        self._client = chromadb.PersistentClient(
            path=db_path,
        )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[MemoryManager] Collection '{collection_name}' loaded. "
              f"Records: {self._collection.count()}")

    def _build_openai_embedding_fn(self):

        from chromadb.utils import embedding_functions  # lazy import to keep startup fast

        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=self._openai_key,            # the key read from environment
            model_name=cfg.embedding_model_name, # "text-embedding-3-small"
        )

    def _build_local_embedding_fn(self):
        from chromadb.utils import embedding_functions  # lazy import

        print(f"[MemoryManager] Loading local model: {cfg.local_embedding_model}")
        print("[MemoryManager] This downloads ~22 MB on first run (cached after that).")

        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=cfg.local_embedding_model
        )

    def recall_memories(
        self,
        query: str,
        top_k: int = cfg.top_k_recall,
        similarity_threshold: float = cfg.similarity_threshold,
    ) -> List[MemoryRecord]:
        if self._collection.count() == 0:
            print("[MemoryManager] Memory store is empty — cold start, no prior context.")
            return []

        print(f"[MemoryManager] Querying memory for: '{query[:80]}...' (top_k={top_k})")
        try:
            results = self._collection.query(
                query_texts=[query],                             # list of 1 query string
                n_results=min(top_k, self._collection.count()), # can't ask for more than exists
                include=["documents", "distances", "metadatas"], # what to return per result
            )
        except Exception as e:
            print(f"[MemoryManager] Memory recall failed: {e}")
            return []
        memories: List[MemoryRecord] = []


        docs = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        for doc, dist, meta in zip(docs, distances, metadatas):
            similarity = 1.0 - (dist / 2.0)

            if similarity < similarity_threshold:
                print(f"[MemoryManager] Skipping memory (similarity={similarity:.2f} < {similarity_threshold}): "
                      f"{meta.get('topic', 'unknown')[:40]}")
                continue

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
        if not summary or not summary.strip():
            # Don't store empty summaries — they waste index slots and confuse recall
            print("[MemoryManager] Skipping storage of empty summary.")
            return ""

        full_metadata = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),  # ISO format: "2025-07-21T14:30:00"
            "char_length": len(summary),               # for size monitoring
        }
        if metadata:
            full_metadata.update(metadata)

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
                        print(f"[MemoryManager] Near-duplicate detected (similarity={top_similarity:.3f}). "
                              f"Skipping storage to prevent memory bloat.")
                        return ""
            except Exception:
                pass  # If dedup check fails, proceed with storage anyway (better to store twice than lose data)

        doc_id = str(uuid.uuid4())

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
        if self._collection.count() == 0:
            return []

        try:
            results = self._collection.get(
                include=["documents", "metadatas"]
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
                "metadata": metadata,
            })


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
            self._client.delete_collection(self.collection_name)

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
