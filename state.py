from typing import TypedDict, List, Optional, Dict, Any  # typing tools for strict schemas
from pydantic import BaseModel, Field, field_validator    # pydantic for sub-object validation

class SourceDocument(BaseModel):


    title: str = Field(description="The webpage's title or headline")
    url: str = Field(description="Full canonical URL")
    snippet: str = Field(default="", description="Short content preview") # optional, defaults to empty string

    @field_validator("url")
    @classmethod
    def url_must_be_valid(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            return "https:" + v if v.startswith("//") else v
        return v

class ScrapedPage(BaseModel):

    url: str = Field(description="The URL that was scraped")
    title: str = Field(default="", description="HTML page title")
    content: str = Field(default="", description="Cleaned page body text")
    success: bool = Field(default=True, description="Whether scraping succeeded")
    error: str = Field(default="", description="Error message if scraping failed")

class MemoryRecord(BaseModel):

    topic: str = Field(description="Short descriptive label for this memory")
    summary: str = Field(description="The stored research insight or summary")
    similarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity to the current query (0.0–1.0)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (timestamps, source URLs, etc.)"
    )

class ResearchState(TypedDict, total=False):

    question: str
    recalled_memories: List[MemoryRecord]
    plan: List[str]
    raw_search_results: List[SourceDocument]
    scraped_content: List[ScrapedPage]
    iteration: int
    evaluation_score: float
    evaluation_reasoning: str
    missing_topics: List[str]
    final_report: str
    sources: List[SourceDocument]
    memory_stored: bool
