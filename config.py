from dataclasses import dataclass, field  # dataclass reduces boilerplate; field() allows complex defaults
from typing import Optional                 # Optional[str] = str | None, used for keys that may be absent

@dataclass(frozen=True)
class Config:

    llm_model_name: str = "openai-gpt"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    synthesis_temperature: float = 0.3
    embedding_model_name: str = "text-embedding-3-small"
    vector_dimension: int = 1536
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_vector_dimension: int = 384
    db_path: str = "./data/chroma_db"
    collection_name: str = "research_memories"
    top_k_recall: int = 3
    similarity_threshold: float = 0.7
    max_search_results: int = 5
    search_depth: str = "advanced"
    timeout_seconds: int = 10
    max_char_limit: int = 8000
    max_iterations: int = 3
    sufficiency_threshold: float = 0.7
    refresh_interval_ms: int = 500
    sample_docs_path: str = "./data/sample_documents"
    benchmark_output_path: str = "./data/benchmark_results.json"

cfg = Config()
