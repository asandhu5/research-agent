# Autonomous Web Research Agent with Persistent Long-Term Memory

---

## Overview

This project builds an **Autonomous Web Research Agent** that accepts any complex query, decomposes it into sub-research tasks, autonomously searches and scrapes the web, evaluates its own findings, and synthesizes a fully-cited Markdown report. The agent uses a **LangGraph state machine** — a directed cyclic graph where Python functions are "nodes" and typed rules are "edges" — enabling conditional search loops, early exits when quality is sufficient, and guaranteed termination. Research summaries are embedded into a local **ChromaDB vector database**, so future queries retrieve relevant prior findings and build cumulative knowledge over time without restarting from scratch.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     RESEARCH AGENT STATE MACHINE                            │
│                                                                             │
│  INPUT: question                                                            │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────┐                                                            │
│  │   recall    │  Queries ChromaDB for past research on similar topics      │
│  │  (Node 1)   │  using cosine similarity on embedded query text            │
│  └──────┬──────┘                                                            │
│         │ recalled_memories (List[MemoryRecord])                            │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │    plan     │  GPT-4o decomposes the question into 3–5 search sub-queries│
│  │  (Node 2)   │  informed by recalled memories to avoid re-research        │
│  └──────┬──────┘                                                            │
│         │ plan (List[str])                                                  │
│         ▼                                                                   │
│  ┌──────────────┐   ◄─────────────────────────────────────────────────┐    │
│  │ search_scrape│   Score < 0.7 AND iteration < max_iterations         │    │
│  │  (Node 3)    │   Runs Tavily / DuckDuckGo → BeautifulSoup scraper   │    │
│  └──────┬───────┘                                                      │    │
│         │ raw_search_results, scraped_content, iteration++             │    │
│         ▼                                                              │    │
│  ┌─────────────┐                                                       │    │
│  │  evaluate   │  GPT-4o scores scraped content sufficiency (0.0–1.0)  │    │
│  │  (Node 4)   │  and identifies knowledge gaps in evaluation_reasoning │    │
│  └──────┬──────┘                                                       │    │
│         │                                                              │    │
│    ┌────┴─────────────────────────────────────────┐                   │    │
│    │          route_after_evaluation()             │                   │    │
│    │  if score >= 0.7 OR iteration >= max_iter:   │                   │    │
│    │         → synthesize                          │                   │    │
│    │  else: ─────────────────────────────────────►┘                   │    │
│    └────────────┬───────────────────────────────────                  │    │
│                 │ score >= 0.7 OR circuit-breaker                     │    │
│                 ▼                                                      │    │
│         ┌─────────────┐                                                │    │
│         │ synthesize  │  GPT-4o generates full Markdown report with    │    │
│         │  (Node 5)   │  inline [1][2] citations and References section│    │
│         └──────┬──────┘                                                │    │
│                │ final_report, sources                                 │    │
│                ▼                                                       │    │
│         ┌─────────────┐                                                │    │
│         │    store    │  Summarizes report, embeds summary, persists   │    │
│         │  (Node 6)   │  to ChromaDB for future memory recall          │    │
│         └──────┬──────┘                                                │    │
│                │                                                       │    │
│                ▼                                                       │    │
│              [END]                                                     │    │
└─────────────────────────────────────────────────────────────────────────────┘

VECTOR MEMORY LAYER (ChromaDB)
┌─────────────────────────────────────────────┐
│  Embedding Models:                          │
│    Primary:  OpenAI text-embedding-3-small  │
│    Fallback: all-MiniLM-L6-v2 (local, free) │
│                                             │
│  Operations:                                │
│    recall node:  similarity search (top_k)  │
│    store node:   embed + upsert summary     │
│                                             │
│  Index: HNSW (cosine similarity)            │
│  Storage: ./data/chroma_db/ (SQLite + HNSW) │
└─────────────────────────────────────────────┘
```

---

## Installation & Environment Setup

### Prerequisites

- Python 3.10 or higher
- `pip` (Python package manager)
- Optional: OpenAI API key (required for full pipeline, not for tests)
- Optional: Tavily API key (recommended; DuckDuckGo fallback available without it)

### Step 1: Clone and Enter Project

```bash
cd "research_agent_project"
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### Step 3: Install All Dependencies

```bash
pip install -r requirements.txt

```

This installs approximately 800 MB of packages including PyTorch (for local embeddings), LangChain, ChromaDB, and Streamlit. On first run, the local embedding model (`all-MiniLM-L6-v2`, ~22 MB) is downloaded and cached in `~/.cache/huggingface/`.

### Step 4: Configure API Keys

Create a `.env` file in `research_agent_project/`:

```bash
# Required for full LLM pipeline (planning, evaluation, synthesis)
OPENAI_API_KEY=sk-your-openai-key-here

# Recommended for best search quality (DuckDuckGo fallback available without it)
TAVILY_API_KEY=tvly-your-tavily-key-here
```

**Where to get keys:**
- OpenAI: https://platform.openai.com/api-keys
- Tavily: https://app.tavily.com (free tier: 1,000 searches/month)

---

## Dataset Ingestion & Local Storage

### Seed the Vector Memory (No API Key Required)

```bash
python3 download_sample_data.py
```

This script:
1. Creates `./data/sample_documents/` with 4 HTML/text research documents on:
   - Vision Transformer (ViT) architectures
   - RAG hallucination mitigation techniques (2025)
   - Quantization with QLoRA
   - Agentic workflows with LangGraph
2. Embeds each document using the local `all-MiniLM-L6-v2` model
3. Stores all 4 summaries in ChromaDB at `./data/chroma_db/`
4. Prints a verification table with record counts and paths
5. Runs a test recall query to demonstrate similarity search

---

## Running the Agent

### Example 1 — Synthetic Memory Setup (No API Key, ~10 seconds)

```bash
python3 download_sample_data.py
pytest tests/
```

This seeds local vector storage with 4 ML research documents and runs all 28 unit tests. Zero API tokens are spent. Tests mock all LLM and HTTP calls. Expected: all tests pass in < 30 seconds.

---

### Example 2 — Initial Cold Query (Requires OPENAI_API_KEY + TAVILY_API_KEY, ~45s)

```bash
python3 main.py --query "What are the top techniques for reducing hallucinations in RAG systems in 2025/2026?"
```

**What happens:**
1. `recall_memory_node` checks ChromaDB for prior research
2. `planner_node` generates 3–5 focused sub-queries (e.g., "CRAG corrective retrieval", "Self-RAG reflection tokens", "RAGAS hallucination metrics")
3. `search_and_scrape_node` queries Tavily for each sub-query, then scrapes top URLs
4. `evaluator_node` scores content sufficiency (expect 0.7–0.9 for this well-indexed topic)
5. `synthesizer_node` writes a ~1,200-word cited Markdown report
6. `memory_storing_node` embeds and stores the summary to ChromaDB

**Verify:** Look for "Memory stored: True" in the output and check `./data/report_*.md`

---

### Example 3 — Memory Continuity Test (~30s)

```bash
python3 main.py --query "Based on our prior research on RAG hallucinations, how does GraphRAG specifically address these issues?"
```

**What to verify:**
- Output shows: `[recall_memory] ✓ Recalled memory (similarity=0.xx): RAG Hallucination Mitigations 2025`
- The planner generates sub-queries that acknowledge prior research and focus on GraphRAG specifically
- The final report references both the newly scraped GraphRAG content AND prior RAG hallucination context

This demonstrates cumulative research — the agent doesn't repeat what it already knows.

---

### Example 4 — Streamlit Dashboard Demo

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

**Tab 1 — Research Workstation:**
- Select an example query from the dropdown (or type your own)
- Click "🚀 Run Autonomous Research"
- Watch the live execution tree update as each node completes
- Read the full Markdown report with clickable source links
- Download the report as a `.md` file

**Tab 2 — Long-Term Memory Explorer:**
- View all stored ChromaDB records with topics, timestamps, and full summaries
- Search for memories by semantic query (tests the vector retrieval independently)
- Clear all memories to start fresh (irreversible — confirms via warning)

---


## Configuration Reference

All hyperparameters live in `config.py` as a frozen dataclass. Key parameters:

| Parameter | Default | Effect of Increasing | Effect of Decreasing |
|-----------|---------|---------------------|---------------------|
| `max_iterations` | 3 | More thorough, higher cost | Faster, may miss info |
| `sufficiency_threshold` | 0.7 | Harder to exit loop | Exits loop too easily |
| `top_k_recall` | 3 | More memory context | Less memory context |
| `similarity_threshold` | 0.7 | Stricter memory filtering | More memory recalled |
| `max_char_limit` | 8000 | More content per page | Faster scraping |
| `llm_temperature` | 0.1 | More creative plans | More deterministic |
| `max_search_results` | 5 | More sources per query | Fewer sources |

---
