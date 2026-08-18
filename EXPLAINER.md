# Project Explainer — Everything You Need to Know
### Autonomous Web Research Agent with Persistent Long-Term Memory

> This file is a plain-English conversation about what we built, what it needs,
> how every piece works, what the output looks like, and exactly how to test it.
> No code changes. Just understanding.

---

## TABLE OF CONTENTS

1. [The Big Picture — What Did We Actually Build?](#1-the-big-picture)
2. [What APIs and Keys Do You Need?](#2-what-apis-and-keys-do-you-need)
3. [File-by-File Explanation — What Does Each File Do?](#3-file-by-file-explanation)
4. [The Flow — Step by Step, What Happens When You Run It?](#4-the-flow-step-by-step)
5. [What Does the Output Actually Look Like?](#5-what-does-the-output-look-like)
6. [What Do You Need Installed Before Testing?](#6-what-do-you-need-installed-before-testing)
7. [How to Test It — Every Scenario Explained](#7-how-to-test-it)
8. [How to See What You Are Testing — Reading the Output](#8-how-to-read-the-test-output)
9. [What Is Still Missing or Could Break](#9-what-is-still-missing-or-could-break)
10. [Glossary — Plain English Definitions of Every Technical Term](#10-glossary)

---

## 1. The Big Picture

### What is this project, in one sentence?

It is a Python program that you give a research question to — like "What are the best
techniques to reduce hallucinations in AI systems in 2025?" — and it goes to the internet,
reads websites, decides whether it has found enough information, writes a full cited report
in Markdown format, and then saves a summary of what it learned into a local database so
that the next time you ask something similar, it remembers and doesn't start from scratch.

### Why does this matter for an ML engineer?

Three reasons:

**First — Agentic state machines.** Most ML code you write is sequential: run this, then
that, then stop. This project uses a graph where the agent can loop back on itself. After
searching the web, it evaluates its own findings and decides "I don't have enough yet,
search again." That decision loop, with a guaranteed stopping condition, is the core
pattern of modern agentic AI systems. Companies like Anthropic, OpenAI, and Google use
variations of this pattern for their autonomous agents.

**Second — Vector memory.** The agent doesn't just run and forget. It stores summaries of
completed research as mathematical vectors (lists of numbers that encode meaning). Next
time you ask something related, it finds those vectors and uses the old summaries as
context. This is how long-term memory works in AI systems — not by storing text and doing
keyword search, but by storing meaning and doing similarity search.

**Third — Production patterns.** Every piece of this code handles failures gracefully.
If OpenAI is down, it falls back to local models. If Tavily is unavailable, it falls back
to DuckDuckGo. If a website times out, it logs the error and continues. These are the
patterns that make the difference between a demo that works once and a system that runs
reliably.

---

## 2. What APIs and Keys Do You Need?

### The short answer

| Key | Required? | What happens without it | Where to get it | Free? |
|-----|-----------|------------------------|-----------------|-------|
| `OPENAI_API_KEY` | YES for full run | LLM calls fail completely | platform.openai.com/api-keys | No — ~$0.05–0.15 per query |
| `TAVILY_API_KEY` | NO | DuckDuckGo fallback activates | app.tavily.com | Yes — 1,000 searches/month free |

### The detailed answer

**OPENAI_API_KEY**

This key is used in three places:

1. In `memory_manager.py` — to generate embeddings. When you store a research summary,
   the OpenAI API converts that text into a 1,536-number vector. When you search for
   similar memories, your query is also converted to a vector and compared.

2. In `agent_nodes.py` — for the LangChain `ChatOpenAI` client that powers the Planner,
   Evaluator, and Synthesizer nodes. These are the calls that cost money.

3. The model used is `gpt-4o` as configured in `config.py`. This is OpenAI's flagship
   model as of 2025. It costs approximately $5 per million input tokens and $15 per
   million output tokens.

**A realistic cost estimate:** One full research run with 3 iterations costs roughly
$0.05–$0.20 depending on how long your question is, how many web pages get scraped,
and how long the final report is. The synthesizer node produces the longest output
and is the most expensive step.

**What if you have NO OpenAI key?**

The system is designed to handle this. In `memory_manager.py`, if `OPENAI_API_KEY`
is absent, it automatically loads a free local model called `all-MiniLM-L6-v2` from
HuggingFace. This model runs entirely on your laptop's CPU, needs no internet after
the first download (~22 MB), and is completely free. However, it only produces
384-dimensional vectors instead of 1,536-dimensional ones, which means slightly less
precise similarity matching.

The LLM nodes (planner, evaluator, synthesizer) CANNOT fall back to anything local
in the current code — they need an OpenAI key. Without one, the graph will crash at
the planner node. The tests, however, all mock the LLM calls, so `pytest tests/`
works with zero API keys.

**TAVILY_API_KEY**

Tavily is a search API purpose-built for AI agents. It returns clean, pre-ranked
snippets rather than raw HTML search results. Without it, the agent automatically
falls back to `duckduckgo-search`, which is a free Python library that scrapes
DuckDuckGo's website. DuckDuckGo quality is slightly lower and it can rate-limit
you if you make many rapid requests, but it works fine for testing and development.

### Where do you put the keys?

You create a file called `.env` in the `research_agent_project/` folder. There is
already a template called `.env.example`. You copy it and fill in your values:

```
OPENAI_API_KEY=sk-proj-...your actual key here...
TAVILY_API_KEY=tvly-...your actual key here...
```

The `python-dotenv` library (listed in `requirements.txt`) reads this file at startup
and loads the keys into the program's memory. The keys are never printed to the screen
or stored in any database — they just live in that file and in RAM while the program runs.

---

## 3. File-by-File Explanation

### `requirements.txt` — The Dependency List

This file lists every Python library the project needs, with exact version numbers pinned
(e.g., `chromadb==0.5.23`). Pinning versions means that if you run `pip install -r
requirements.txt` today or a year from now, you get the exact same libraries. Without
pinning, a newer version of a library might break something silently.

Each library entry in the file has a comment explaining what it does and why it was chosen.

### `config.py` — The Control Panel

This is a single Python file that holds every number and setting that controls how the
system behaves. Things like: how many search results to fetch per query (5), how many
past memories to recall (3), what the minimum quality score must be before writing
the report (0.7), how long to wait for a website to respond before giving up (10 seconds).

It is written as a `dataclass(frozen=True)`, which means once the program starts, nothing
can accidentally change these settings. It's like a read-only configuration object.

The reason ALL settings live here (instead of scattered through the code) is that you
only need to look in one place to understand the system's behavior, and you only need to
change one file to tune it.

### `state.py` — The Shared Memory Between Steps

This file defines what information flows between every step of the pipeline.

Think of it like a relay race baton. Every "runner" (node in the graph) picks up the
baton, reads what previous runners wrote on it, adds their own information, and passes
it forward. The `ResearchState` TypedDict defines every possible field that can be
written on that baton:

- `question` — the original query you asked (never changes)
- `plan` — the list of sub-queries the Planner generated
- `scraped_content` — the cleaned text from each scraped website
- `evaluation_score` — a number from 0 to 1 rating how good the research is
- `final_report` — the finished Markdown report
- `recalled_memories` — past research retrieved from the database
- and several more

It also defines helper models like `SourceDocument` (a validated URL + title + snippet
object) and `MemoryRecord` (a past research summary with its similarity score).

The reason this is so important: in LangGraph, you cannot just pass variables between
functions normally. Every function receives the entire state object and returns only
the specific fields it changed. The `state.py` file defines the shape of that object
so Python's type checker (and your IDE) can catch mistakes.

### `memory_manager.py` — The Long-Term Memory System

This is the brain of the persistent memory feature. It is a class called `MemoryManager`
that wraps ChromaDB (the vector database) behind a clean interface with four methods:

**`store_memory(topic, summary, metadata)`**
You give it a topic label, a text summary, and some extra metadata (like timestamp,
source URLs). It converts the summary text into a vector (using either OpenAI or local
SentenceTransformers), and stores the vector + text in ChromaDB on your hard drive.

**`recall_memories(query, top_k, similarity_threshold)`**
You give it a query string. It converts that query to a vector and finds the top_k most
similar stored vectors using ChromaDB's HNSW index. It returns the original text
summaries alongside their similarity scores, but only if their score is above the
threshold (default 0.7). Below that threshold they're considered too different to be
useful.

**`get_all_memories()`**
Returns every stored memory as a list of dictionaries. Used by the Streamlit app's
memory explorer tab to show everything in the database.

**`clear_memory()`**
Deletes the entire collection and recreates an empty one. Used by the Streamlit app's
"Clear All Memory" button.

**The dual-embedding strategy explained:**
At initialization, the class checks for `OPENAI_API_KEY`. If it's there, it uses
OpenAI's `text-embedding-3-small` model (better quality, costs a tiny amount per call,
returns 1,536-dimensional vectors). If the key is absent, it loads
`all-MiniLM-L6-v2` locally (free, slightly less accurate, returns 384-dimensional
vectors). Everything else in the class works identically either way.

One important detail: once you pick one embedding model and store data, you CANNOT
switch to the other for the same database. The stored vectors have a fixed number of
dimensions, and mixing 1,536-dim and 384-dim vectors in the same index will cause
retrieval to fail or produce nonsense results.

### `search_engine.py` — The Web Search Layer

This file contains the `SearchEngine` class and a helper function `multi_search()`.

**`SearchEngine`** has a single public method `search(query, max_results)` that returns
a list of `SourceDocument` objects. Internally it tries Tavily first, and if Tavily
fails or isn't configured, it tries DuckDuckGo. Both backends normalize their output
to the same `SourceDocument` format so the rest of the code doesn't need to know which
one ran.

**`multi_search(queries)`** takes the list of sub-queries from the Planner and runs
`SearchEngine.search()` for each one. It deduplicates results by URL — if the same
page appears in two sub-query results, it only appears once in the final list. This
prevents the same page from being scraped twice and cited multiple times in the report.

### `web_scraper.py` — The HTML Parser

This file contains the `WebScraper` class.

When given a URL, it does the following:
1. Sends an HTTP GET request with a realistic browser User-Agent header (so the server
   doesn't block it as a bot)
2. Checks the response for errors (404, 403, 500, timeout)
3. Parses the raw HTML with BeautifulSoup using the lxml parser
4. Removes `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>`, `<aside>`, `<form>`,
   `<button>`, and common ad/popup class names
5. Tries to find the `<main>` or `<article>` element (which usually contains the actual
   article content) rather than extracting the entire page
6. Extracts all text nodes, cleans up whitespace
7. Truncates to `max_char_limit` characters (default 8,000) at a word boundary

The result is a `ScrapedPage` object with `success=True/False`, the cleaned `content`
text, the `title`, and the `url`. If anything goes wrong at any step — network error,
parsing failure, empty content — it returns `success=False` with an error message
instead of crashing.

### `agent_nodes.py` — The Six Steps of the Pipeline

This is the largest and most important file. It contains six Python functions, one per
step in the research pipeline. Each function takes the current state, does its job,
and returns a dict with only the fields it modified.

**`recall_memory_node(state)`**
Reads: `state["question"]`
Writes: `recalled_memories`

Creates a `MemoryManager`, calls `recall_memories(question)`, and returns whatever
past summaries are similar enough. On the first run ever, this returns an empty list.

**`planner_node(state)`**
Reads: `state["question"]`, `state["recalled_memories"]`
Writes: `plan`

Builds a prompt for GPT-4o that includes the question and any recalled memories. Uses
LangChain's `with_structured_output(PlannerOutput)` to force the LLM to return a
JSON object matching the `PlannerOutput` Pydantic schema (a list of 3–5 sub-queries
plus a reasoning string). The structured output feature means LangChain handles JSON
parsing and retries automatically.

**`search_and_scrape_node(state)`**
Reads: `state["plan"]`, `state["raw_search_results"]` (previous iterations)
Writes: `raw_search_results`, `scraped_content`, `iteration`

Calls `multi_search(plan)` to get a deduplicated list of URLs, then calls
`scrape_urls_batch(urls)` to fetch and clean each page. APPENDS to existing results
rather than replacing them — so if this is the second search loop, the content from
the first loop is preserved and new content is added. Increments the `iteration` counter.

**`evaluator_node(state)`**
Reads: `state["question"]`, `state["scraped_content"]`, `state["iteration"]`
Writes: `evaluation_score`, `evaluation_reasoning`

Builds a content summary (title + first 500 chars of each page) and asks GPT-4o to
score it from 0.0 to 1.0. Uses `temperature=0.0` (completely deterministic — same
content always gets the same score). Has a guard clause: if `scraped_content` is empty,
immediately returns `score=0.0` without calling the LLM at all.

**`synthesizer_node(state)`**
Reads: `state["question"]`, `state["scraped_content"]`, `state["recalled_memories"]`
Writes: `final_report`, `sources`

Passes all scraped content as numbered sources ([SOURCE 1], [SOURCE 2]...) and asks
GPT-4o to write a structured Markdown report with inline citations. Uses
`temperature=0.3` to produce more natural prose. Has a fallback method
`_build_fallback_report()` that constructs a minimal report from raw snippets
if the LLM call fails.

**`memory_storing_node(state)`**
Reads: `state["question"]`, `state["final_report"]`, `state["sources"]`
Writes: `memory_stored`

First asks GPT-4o to write a 150–200 word summary of the full report (to keep memory
compact). Then calls `MemoryManager.store_memory()` to embed and persist that summary
to ChromaDB. The deduplication check inside `store_memory()` prevents storing nearly
identical summaries if you run the same query twice.

### `graph_builder.py` — The Wiring

This file assembles the six node functions into a LangGraph state machine. It:

1. Creates a `StateGraph(ResearchState)` — tells LangGraph the shape of the state
2. Adds all six nodes with `graph.add_node(name, function)` — the planning node is
   registered as `"planner"`, not `"plan"`, because `"plan"` is also a key in
   `ResearchState`, and a node name colliding with an existing state key made
   `build_research_graph()` raise `ValueError` before any node could ever run
3. Sets `recall` as the entry point
4. Adds unconditional edges: recall→planner→search_scrape→evaluate and synthesize→store→END
5. Adds the conditional edge after `evaluate`:
   - If `evaluation_score >= 0.7` OR `iteration >= max_iterations` → go to `synthesize`
   - Otherwise → go back to `planner` (not straight to `search_scrape`) — `planner_node`
     detects `iteration > 0` and replans using the evaluator's `missing_topics`, generating
     NEW queries instead of the loop re-running the identical ones from the first pass, then
     its own existing `planner→search_scrape` edge carries the new plan forward
6. Calls `graph.compile()` to validate and lock the structure

The `run_research(question)` function at the bottom is a convenience wrapper — it builds
the graph, sets up the initial state (only `question` is populated, everything else is
empty), and calls `graph.invoke()`. This is what `main.py` and `benchmark.py` call.

### `download_sample_data.py` — The Test Data Seeder

This script is entirely standalone. It does not need an OpenAI key. It:

1. Creates `./data/sample_documents/` if it doesn't exist
2. Writes four files: two HTML and two plain text, each containing a detailed (~300 word)
   research summary about ViT architecture, RAG hallucinations, QLoRA quantization,
   and LangGraph agentic patterns
3. Initializes `MemoryManager` (which will use local SentenceTransformers if no OpenAI key)
4. Calls `store_memory()` for each of the four documents
5. Verifies by calling `get_all_memories()` and printing a count
6. Runs a demo recall query to show the similarity search working
7. Writes an `ingestion_metadata.json` file recording what was stored

You run this ONCE before testing, and it gives the memory system pre-populated data
so you can immediately test memory recall without having done any real research queries yet.

### `benchmark.py` — Performance Measurement

This script runs three pre-defined research questions through the full pipeline and
measures: total latency per query, whether any memories were recalled, how many web
searches happened, the final evaluation score, and how many citations per 500 words
the final report contains. It outputs a Markdown table to the terminal and saves
a JSON file to `./data/benchmark_results.json`. This is how you compare the system
before and after tuning config parameters.

### `main.py` — The Command Line Interface

This is what you type when you want to run the agent from the terminal without the
Streamlit web interface. It:

- Parses command-line arguments (`--query`, `--output`, `--iterations`, `--no-memory`,
  `--verbose`)
- Validates that the OpenAI key is set and prints a clear error if not
- Calls `run_research(question)` from `graph_builder.py`
- Prints the full Markdown report to the terminal
- Saves the report to a `.md` file (auto-named with a timestamp if no output path given)
- Prints a summary: eval score, iterations used, sources cited, memory stored

### `app.py` — The Streamlit Web Dashboard

This is a web app you open in your browser. It has two tabs:

**Tab 1 — Research Workstation:**
An input box for your query, a "Run Autonomous Research" button, a live execution tree
that updates as each node completes (showing what was recalled, what sub-queries were
generated, how many pages were scraped, the evaluation score), and then the full
rendered Markdown report below with clickable links. There's also a download button.

**Tab 2 — Long-Term Memory Explorer:**
Shows every record stored in ChromaDB as expandable cards. Has a search box to test
memory retrieval independently (without running a full research query). Shows similarity
scores for each result. Has a "Clear All Memory" danger button.

### `tests/` — The Five Test Files

Each test file is isolated and mocks all external dependencies (API calls, HTTP requests,
file system operations). They can all run without any API keys in under 60 seconds.

- `test_memory.py` — tests ChromaDB insertion, retrieval, similarity scoring, clearing
- `test_scraper.py` — tests HTML cleaning, content preservation, HTTP error handling
- `test_search.py` — tests Tavily/DDG selection, output format, deduplication
- `test_evaluator.py` — tests low score for irrelevant content, high score for relevant content
- `test_graph_flow.py` — tests the full state machine end-to-end with all mocked dependencies

`conftest.py` adds the project root to Python's import path so all test files can
import modules like `from memory_manager import MemoryManager`.

---

## 4. The Flow — Step by Step, What Happens When You Run It?

Let's trace exactly what happens when you type:

```
python main.py --query "How does GraphRAG reduce hallucinations in RAG systems?"
```

**Step 0 — Startup**
`main.py` loads your `.env` file (so `OPENAI_API_KEY` enters the environment), checks
the key is present, prints diagnostics, and calls `run_research(question)`.

**Step 1 — Graph Build**
`graph_builder.py` creates the StateGraph, wires up all six nodes and edges, and calls
`compile()`. This takes about 0.1 seconds. No API calls yet.

**Step 2 — Initial State**
The graph starts with this state object:
```
question = "How does GraphRAG reduce hallucinations in RAG systems?"
recalled_memories = []
plan = []
raw_search_results = []
scraped_content = []
iteration = 0
evaluation_score = 0.0
...everything else = empty/zero
```

**Step 3 — recall node**
`MemoryManager` opens the ChromaDB files at `./data/chroma_db/`. It embeds the question
text into a vector using OpenAI's API (one small API call, costs ~$0.0001). It queries
ChromaDB for the top 3 most similar stored vectors. If you ran `download_sample_data.py`,
it will find the "RAG Hallucinations 2025" summary with a similarity score of maybe 0.81
(very relevant). That summary gets loaded into `recalled_memories`. If the store is empty,
`recalled_memories` stays an empty list.

**Step 4 — plan node**
GPT-4o receives the question AND the recalled memory summary. It generates something like:
```
1. "GraphRAG knowledge graph entity extraction pipeline"
2. "GraphRAG community detection hallucination reduction comparison RAG"
3. "Microsoft GraphRAG vs standard RAG benchmark results 2025"
4. "knowledge graph RAG global queries community summaries"
```
Because it saw the RAG hallucinations memory, it might skip generating sub-queries about
basic RAG hallucination causes (since we already know those) and focus on GraphRAG-specific
aspects. This is memory being useful.

**Step 5 — search_scrape node (iteration 1)**
`multi_search()` runs all four sub-queries through Tavily (or DDG). It collects maybe
15–20 unique URLs in total (5 per query, minus duplicates across queries). It then calls
`scrape_urls_batch()` on those URLs sequentially. For each URL, it downloads the HTML,
strips scripts/nav/footer, extracts the main text, and truncates to 8,000 characters.
Maybe 12 out of 15 pages succeed (the rest are 404, timeout, or JS-only SPAs). The
state now has `iteration=1`, 12 `ScrapedPage` objects, and 15–20 `SourceDocument` records.

**Step 6 — evaluate node**
GPT-4o reads a summary of all 12 pages (title + first 500 chars each). It decides how
well the collected content answers the original question. For a well-indexed topic like
GraphRAG, it will probably score 0.82 after one iteration — good enough to proceed.
It returns `evaluation_score=0.82` and a reasoning string explaining what's covered.

**Step 7 — route_after_evaluation**
0.82 >= 0.7 (sufficiency threshold), so the router returns "synthesize". No loop needed.

**Step 8 — synthesize node**
GPT-4o receives all 12 scraped page texts formatted as "[SOURCE 1] Title: ... Content: ..."
plus the recalled RAG hallucinations memory. It writes a ~1,200-word Markdown report with
an executive summary, sections on GraphRAG's architecture, how community summaries reduce
hallucinations, a comparison to standard RAG, and a references section listing all 12
sources as numbered links. Uses `temperature=0.3` for natural prose.

**Step 9 — store node**
GPT-4o writes a 180-word summary of the report. `MemoryManager` embeds that summary
and stores it in ChromaDB as "How does GraphRAG reduce hallucinations in RAG systems?"
Next time you ask anything about GraphRAG or RAG hallucinations, this session will be
recalled.

**Step 10 — END**
`main.py` receives the final state. It prints the full Markdown report to your terminal,
saves it as `./data/report_20260725_060900.md`, and prints a summary box:
```
Evaluation score:   0.82
Search iterations:  1
Sources cited:      9
Memory recalled:    1 past research sessions
Memory stored:      True
Report saved to:    /absolute/path/to/data/report_20260725_060900.md
```

Total time: approximately 35–50 seconds.

---

## 5. What Does the Output Actually Look Like?

### Terminal output during execution

When you run `python main.py --query "..."`, your terminal will show a running stream of
status messages from each node. Here is a realistic example of what you will see:

```
======================================================================
  Autonomous Web Research Agent
======================================================================
  Query: How does GraphRAG reduce hallucinations in RAG systems?

Environment Check:
  OPENAI_API_KEY:  ✓ Configured
  TAVILY_API_KEY:  ✓ Configured
  Memory store:   4 records in ChromaDB

[GraphBuilder] Initializing StateGraph with ResearchState schema...
[GraphBuilder] ✓ 6 nodes registered
[GraphBuilder] ✓ Graph compiled (no checkpointer)

============================================================
[Node: recall_memory] Recalling past research for:
  'How does GraphRAG reduce hallucinations in RAG systems?'
============================================================
[MemoryManager] Querying memory for: 'How does GraphRAG reduce hallucinations...' (top_k=3)
[MemoryManager] ✓ Recalled memory (similarity=0.81): RAG Hallucination Mitigations 2025
[MemoryManager] Retrieved 1 relevant memories above threshold.

[Node: planner] Decomposing question into sub-queries...
[planner] Generated 4 sub-queries:
  1. GraphRAG knowledge graph entity extraction architecture
  2. GraphRAG community detection global queries vs local RAG
  3. Microsoft GraphRAG hallucination reduction benchmark comparison
  4. knowledge graph structured retrieval augmented generation 2025
[planner] Reasoning: These queries cover GraphRAG's technical...

[Node: search_and_scrape] Iteration 1 — Searching 4 sub-queries
[SearchEngine] Sub-query 1/4: 'GraphRAG knowledge graph entity extraction'
[SearchEngine] Tavily returned 5 results.
[SearchEngine] Sub-query 2/4: 'GraphRAG community detection global queries...'
[SearchEngine] Tavily returned 5 results.
...
[SearchEngine] Multi-search complete: 14 unique URLs found.
[WebScraper] Processing URL 1/14
[WebScraper] ✓ Scraped 7842 chars from: Microsoft GraphRAG: Unlocking LLM...
[WebScraper] Processing URL 2/14
[WebScraper] ✓ Scraped 6201 chars from: GraphRAG vs RAG: A Comprehensive...
...
[search_and_scrape] Total accumulated content: 68,432 characters

[Node: evaluator] Evaluating research quality (iteration 1)...
[evaluator] Score: 0.82 | ✓ Sufficient
[evaluator] Reasoning: Content comprehensively covers GraphRAG's architecture...

[Router] Evaluation score: 0.82 | Iteration: 1/3
[Router] ✓ Sufficiency threshold met (0.82 >= 0.70) → synthesize

[Node: synthesizer] Generating research report...
[synthesizer] ✓ Report generated: 4,231 characters, 9 cited sources.

[Node: memory_storing] Persisting research to long-term memory...
[memory_storing] ✓ Research stored in memory (id=f3a9b2c1...)
```

### The final Markdown report

The report the synthesizer writes looks something like this (this is illustrative):

```markdown
# GraphRAG: How Knowledge Graphs Reduce Hallucinations in RAG Systems

## Executive Summary

GraphRAG (Graph Retrieval-Augmented Generation) addresses a fundamental limitation
of standard RAG systems: fragmented, disconnected context. By constructing a knowledge
graph from source documents and using community detection algorithms, GraphRAG enables
global queries that synthesize information across entire document collections [1][2].
Studies show GraphRAG reduces hallucination rates by 28–41% compared to naive RAG
on knowledge-intensive benchmarks [3].

## How GraphRAG Works: The Architecture

Standard RAG retrieves individual text chunks that may be semantically similar to
your query but logically disconnected. GraphRAG takes a fundamentally different
approach [1]:

1. **Entity Extraction**: An LLM reads all source documents and identifies named
   entities (people, concepts, organizations, techniques) [2].
2. **Relationship Mapping**: Relationships between entities are extracted and
   stored as graph edges [2].
3. **Community Detection**: The Leiden algorithm partitions the entity graph
   into communities — groups of closely related concepts [4].
4. **Community Summarization**: GPT-4o writes a summary for each community,
   capturing holistic relationships that individual chunks would miss [1][4].

## Why This Reduces Hallucinations

The core cause of RAG hallucinations is "context islands" — when retrieved chunks
don't connect to each other coherently, the LLM fills the gaps with fabrications [3].
GraphRAG eliminates this by...

## References

1. [Microsoft GraphRAG: Unlocking LLM Discovery on Narrative Private Data](https://microsoft.github.io/graphrag/)
2. [From Local to Global: A Graph RAG Approach to QA over Large Text Corpora](https://arxiv.org/abs/2404.16130)
...
```

### The memory explorer tab in Streamlit

When you open `http://localhost:8501` Tab 2, you see cards like:

```
📝 How does GraphRAG reduce hallucinations in RAG systems?
   Stored: 2026-07-25 06:15:43  |  Size: 892 characters  |  ID: f3a9b2c1...
   ▶ [click to expand and read the full summary]

📝 RAG Hallucination Mitigations 2025
   Stored: 2026-07-25 05:00:00  |  Size: 814 characters  |  ID: a1b2c3d4...
   ▶ [click to expand]
```

---

## 6. What Do You Need Installed Before Testing?

### Required software

1. **Python 3.10 or higher** — check with `python3 --version`
2. **pip** — comes with Python

### Required Python packages

You install everything with one command from inside the `research_agent_project/` folder:

```bash
pip install -r requirements.txt
```

This will take 5–15 minutes the first time because:
- PyTorch (required by SentenceTransformers) is large (~2 GB)
- Several other packages have C extensions that need compilation

On subsequent installs (if you recreate the virtual environment), it's faster because
pip caches downloaded packages.

### First-time model download

The first time you run anything that uses the local embedding model, Python automatically
downloads `all-MiniLM-L6-v2` (~22 MB) from HuggingFace's servers and caches it at
`~/.cache/huggingface/hub/`. You need internet access for this first download.
After that, it works offline.

### What you need for EACH test scenario

| Scenario | Internet? | OpenAI key? | Tavily key? | Prior setup? |
|----------|-----------|-------------|-------------|--------------|
| `pytest tests/` | No | No | No | `pip install -r requirements.txt` |
| `python download_sample_data.py` | Yes (first run only for model download) | No | No | requirements installed |
| `python main.py --query "..."` | Yes | YES | Recommended | requirements + .env file |
| `streamlit run app.py` | Yes | YES | Recommended | requirements + .env file |
| `python benchmark.py` | Yes | YES | Recommended | requirements + .env + sample data |

---

## 7. How to Test It — Every Scenario Explained

### Scenario A: Zero-cost test — does the code even work?

This is what you run first, before spending any money. It runs all 28 automated unit tests.
Zero API calls. Zero cost. Typically completes in 20–45 seconds.

**Command:**
```bash
cd research_agent_project
python download_sample_data.py
pytest tests/ -v
```

**Why run `download_sample_data.py` first?**
The `test_memory.py` tests create their own isolated temporary databases, so they don't
need the sample data. But it's good practice to run the seeder first anyway to verify
your environment is working. If `download_sample_data.py` crashes, something is wrong
with your installation before you even run tests.

**What `pytest tests/ -v` does:**
The `-v` flag means "verbose" — it prints each test name and PASSED/FAILED individually.
Without `-v`, pytest only shows a summary line like `28 passed in 18.42s`.

**What tests actually check (in plain English):**

From `test_memory.py`:
- "Can I store a document and get a UUID back?" → yes
- "Does storing an empty string return empty string and not crash?" → yes
- "Does a RAG-related query return the RAG document as the most similar?" → yes
- "Are all similarity scores between 0 and 1?" → yes
- "Does querying an empty database return [] not an exception?" → yes
- "After clearing, is the count 0?" → yes

From `test_scraper.py`:
- "Does the cleaned content NOT contain script tags?" → verified
- "Does the cleaned content NOT contain CSS rules?" → verified
- "Does the cleaned content NOT contain nav links?" → verified
- "Does the actual article text SURVIVE cleaning?" → verified
- "Does a 404 response return a failed ScrapedPage?" → verified (mocked HTTP)
- "Does a timeout return a failed ScrapedPage?" → verified (mocked HTTP)
- "Does a successful response return the article content?" → verified (mocked HTTP)

From `test_search.py`:
- "Does Tavily output get converted to SourceDocument objects?" → verified (mocked Tavily)
- "Is DuckDuckGo used when no Tavily key is set?" → verified (mocked env vars)
- "Does multi_search deduplicate URLs across queries?" → verified

From `test_evaluator.py`:
- "Does cooking content score < 0.5 for a RAG question?" → verified (mocked LLM)
- "Does comprehensive RAG content score >= 0.7?" → verified (mocked LLM)
- "Does empty scraped content immediately return 0.0?" → verified (no LLM call needed)
- "Does the evaluator return both score AND reasoning keys?" → verified

From `test_graph_flow.py`:
- "Does the full graph produce a non-empty final_report?" → verified (all mocked)
- "Is the iteration counter >= 1 after the graph runs?" → verified
- "With a high eval score, does the graph do only 1 search iteration?" → verified
- "With always-low eval score, does the graph stop at max_iterations=3?" → verified

### Scenario B: Memory seeding and basic verification

```bash
python download_sample_data.py
```

This creates the database with 4 pre-written research summaries. You're testing:
1. Can ChromaDB be created and written to?
2. Can the embedding model run?
3. Does the deduplication check work?
4. Does the similarity search return sensible results for a test query?

You will see output confirming 4 documents were stored, a demo recall showing which
stored document was most similar to the test query, and the exact file paths where
everything was saved.

### Scenario C: Full pipeline test with API keys

```bash
python main.py --query "What are the hardware requirements for fine-tuning LLMs with QLoRA?"
```

This tests the ENTIRE system end-to-end. You're verifying:
- OpenAI key is valid and LLM calls succeed
- Tavily (or DuckDuckGo) returns real search results
- Real websites can be scraped
- The evaluator scores the content (expect 0.75–0.90 for this well-indexed topic)
- The synthesizer produces a proper Markdown report
- The memory is stored (check for "Memory stored: True" in output)

After this completes, run it a SECOND time with a related query:
```bash
python main.py --query "How does QLoRA compare to full fine-tuning in terms of accuracy?"
```

This time, the recall node should find the first query's memory (similarity ~0.75+)
and the planner should generate different sub-queries that acknowledge we already know
the basic hardware requirements.

### Scenario D: Streamlit dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Test:
1. Select an example query from the dropdown and click "Use Example"
2. Click "Run Autonomous Research" — watch the execution tree expand in real-time
3. Read the report when it appears
4. Switch to Tab 2 and verify all stored memories are listed
5. Type a query into the "Search memory" box and see retrieved results with similarity scores
6. Click on any memory card to expand and read the full summary

### Scenario E: Benchmarking

```bash
python benchmark.py
```

This runs three preset queries and prints a Markdown table to your terminal. It also
saves `./data/benchmark_results.json`. You use this to:
- Verify system performance baseline
- Compare before/after config changes (e.g., does raising `max_iterations` to 5 improve scores?)
- Generate numbers to include in a portfolio/presentation

If you don't have API keys, benchmark.py will print "Skipped — OPENAI_API_KEY not set"
for each query but still produce the output file with zeroed metrics.

---

## 8. How to Read the Test Output

### What `pytest tests/ -v` output means

When all tests pass, you see something like:

```
tests/test_memory.py::TestMemoryInsertion::test_store_memory_returns_nonempty_id PASSED    [ 3%]
tests/test_memory.py::TestMemoryInsertion::test_store_empty_summary_returns_empty_string PASSED  [ 6%]
tests/test_memory.py::TestMemoryInsertion::test_store_increments_count PASSED    [ 10%]
tests/test_memory.py::TestMemoryRetrieval::test_recall_returns_semantically_similar_memory PASSED  [ 13%]
...
tests/test_graph_flow.py::TestGraphFullExecution::test_circuit_breaker_stops_at_max_iterations PASSED  [ 96%]
tests/test_graph_flow.py::TestGraphFullExecution::test_high_eval_score_prevents_extra_iterations PASSED  [100%]

========================= 28 passed in 24.36s =========================
```

Every line means: the class → the test function name → PASSED (green) or FAILED (red).
The percentage in brackets shows overall progress through the test suite.

### What a failing test looks like

If a test fails, pytest shows you:
1. The test name
2. The exact line where the assertion failed
3. The expected value vs what was actually returned
4. A stack trace showing which function caused the problem

Example of a failing test output:

```
FAILED tests/test_evaluator.py::TestEvaluatorWithPoorContext::test_poor_content_produces_low_score

=================== short test summary info ====================
FAILED tests/test_evaluator.py::TestEvaluatorWithPoorContext::test_poor_content_produces_low_score

AssertionError: Poor/irrelevant content should produce a low evaluation score (< 0.5),
but got 0.80. The agent would incorrectly proceed to synthesis.
assert 0.80 < 0.5
```

This tells you: the test expected the evaluator to return a score below 0.5 for
irrelevant content, but the mock returned 0.80 instead. This would mean either the
test's mock setup was wrong, or the evaluator node has a bug where it's not using
the mocked value correctly.

### What `download_sample_data.py` output means

At the end, you see the summary table:

```
SUMMARY
════════════════════════════
Files written:     4
Documents stored:  4 records in ChromaDB
ChromaDB path:     /Users/yourname/.../data/chroma_db
Documents path:    /Users/yourname/.../data/sample_documents

Stored memories:
  • Vision Transformer (ViT) Architecture Overview
    Timestamp: 2026-07-25T06:10:22 | Size: 743 chars
  • RAG Hallucination Mitigations 2025
    Timestamp: 2026-07-25T06:10:23 | Size: 812 chars
  • Quantization Techniques with QLoRA
    Timestamp: 2026-07-25T06:10:24 | Size: 791 chars
  • Agentic Workflows with LangGraph
    Timestamp: 2026-07-25T06:10:25 | Size: 808 chars
```

Then the demo recall:

```
DEMO: Testing Memory Recall
════════════════════════════
Query: 'How does RAG prevent hallucinations in language models?'

Top 2 recalled memories:
  1. [0.79] RAG Hallucination Mitigations 2025
     Preview: Hallucination in Retrieval-Augmented Generation (RAG) systems occurs...
  2. [0.61] Agentic Workflows with LangGraph
     Preview: LangGraph is a library for building stateful, multi-actor applications...
```

The number in brackets is the cosine similarity score. 0.79 means "very semantically
similar." 0.61 means "somewhat related but not directly about this topic." Notice the
LangGraph memory scored 0.61 (not 0.7 threshold) which means it would NOT be included
in a real recall with default settings — it's below the threshold.

### What a Streamlit run tells you

In the browser, after clicking "Run Autonomous Research," you'll see the execution tree
update live with colored expandable cards:

- Green border = completed node
- The recall card shows which memories were found (or "cold start" if none)
- The plan card shows the exact sub-queries that were generated
- The search card shows the iteration number, how many search results came back, and how
  many pages were successfully scraped (e.g., "11/14 successfully scraped")
- The evaluate card shows the score with a colored number (red < 0.5, yellow 0.5–0.7,
  green >= 0.7)
- The synthesize card shows the report word count and source count
- The store card shows "Research saved to long-term memory ✓" or the skip message

Below the execution tree, the full Markdown report renders with proper headings,
bold text, and clickable links.

---

## 9. What Is Still Missing or Could Break

### Things that work but could be improved

**The scraper cannot handle JavaScript-rendered pages.** About 30–40% of modern websites
are "Single Page Applications" where the actual content is loaded by JavaScript after
the HTML arrives. BeautifulSoup only sees the empty HTML shell. These scrapes come back
with `success=False` and the agent skips them. You'll notice some URLs in the search
results that clearly have good content but appear as empty scrapes. The fix is to add
Playwright (a headless browser) as an optional scraping backend.

**Scraping is sequential.** The agent scrapes 10–15 URLs one at a time. If each takes 2
seconds, that's 20–30 seconds just for scraping. A production system would run these
in parallel using Python's ThreadPoolExecutor, reducing that to 5–8 seconds. The current
sequential approach was chosen deliberately for this learning project because parallel
scraping is harder to debug.

**Memory never shrinks.** Every research session adds ~800 characters to ChromaDB. If you
run hundreds of queries, the database grows and recall takes slightly longer. There is no
pruning logic. For a learning project this is fine; for production you'd add a background
job that deletes old or duplicate memories.

**No streaming in the CLI.** The `main.py` command-line interface runs the graph and
waits for the final state before printing anything. The terminal appears frozen for
35–50 seconds with no feedback. The Streamlit app has live streaming (via
`graph.stream()`), but `main.py` uses `graph.invoke()` which is blocking. You could
change `main.py` to use `graph.stream()` for live terminal updates.

### Things that require API keys to actually work

The following will NOT work at all without an OpenAI key:
- `python main.py --query "..."`
- `python benchmark.py` (queries are skipped, outputs zeros)
- `streamlit run app.py` Tab 1 (the "Run Autonomous Research" button is disabled)

The following WILL work without any API keys:
- `pytest tests/` (all 28 tests pass)
- `python download_sample_data.py` (uses local SentenceTransformers)
- `streamlit run app.py` Tab 2 (the memory explorer, since data was already seeded)

### Potential breakage points to watch for

**ChromaDB version mismatch.** ChromaDB changes its API between major versions (0.4.x vs
0.5.x). The requirements.txt pins `chromadb==0.5.23`. If you update it, API method names
may change and the MemoryManager class will need updates.

**DuckDuckGo rate limiting.** If you run the benchmark or multiple queries in quick
succession, DuckDuckGo may return empty results or a 429 error. If you see "DuckDuckGo
search failed" in the terminal frequently, slow down between queries or get a Tavily key.

**LangGraph version pinned tightly.** LangGraph is under active development. Version
0.2.55 is used. Newer versions occasionally change how `add_conditional_edges()` works
or how state is merged. If you upgrade, the graph routing may need to be re-tested.

**The local embedding model requires PyTorch.** `sentence-transformers` depends on
PyTorch (~2 GB download on first install). On some machines with limited disk space or
slow internet, the install may fail or time out. If `pip install -r requirements.txt`
hangs on the PyTorch download, try installing PyTorch first with the CPU-only variant
from pytorch.org, then re-running the requirements install.

---

## 10. Glossary — Plain English Definitions

**Agent / Agentic system:** A program that can make decisions autonomously rather than
just executing predetermined steps. "Agentic" specifically means the system can choose
its own next action based on what it has observed so far.

**ChromaDB:** An open-source database that is specifically designed to store and search
embedding vectors. It runs entirely on your local machine — no server required — and
stores its data as files in a directory on your hard drive.

**Cosine similarity:** A mathematical measure (between 0.0 and 1.0) of how similar two
vectors are. Two vectors pointing in the same direction score 1.0 (same meaning). Two
vectors at 90 degrees score 0.0 (unrelated meaning). Used to compare query embeddings
against stored memory embeddings.

**Embedding / Vector embedding:** The process of converting text into a list of numbers
(a vector) that captures the text's meaning. Similar meanings produce numerically similar
vectors. "GraphRAG reduces hallucinations" and "knowledge graphs prevent LLM errors"
would produce similar vectors because they mean similar things.

**Entry point:** In LangGraph, the node where graph execution begins. We set this to the
`recall` node so past memories are always loaded first.

**Evaluation score:** A number from 0.0 to 1.0 that the evaluator LLM assigns to the
collected research content. It represents how well the content answers the original
question. Below 0.7 triggers another search loop; at or above 0.7, the agent proceeds
to write the report.

**Frozen dataclass:** A Python dataclass where all fields are read-only after the object
is created. `config.py` uses `@dataclass(frozen=True)` so no code can accidentally
change configuration values at runtime.

**HNSW (Hierarchical Navigable Small World):** The graph-based index structure ChromaDB
uses internally for fast similarity search. Instead of comparing your query vector to
every stored vector (slow), HNSW navigates a network of connected vectors to find the
most similar ones in much fewer comparisons.

**LangChain:** A Python library that provides high-level wrappers for LLM API calls,
prompt templates, output parsers, and integrations with external tools. Used in this
project mainly for `ChatOpenAI`, `SystemMessage`, `HumanMessage`, and
`with_structured_output()`.

**LangGraph:** A library built on top of LangChain for building stateful, multi-step
agent workflows as graphs. It handles the state machine logic — which node runs next,
how state updates are merged, conditional routing — so you just write the node functions.

**Mock / Mocking:** In testing, replacing a real dependency (like the OpenAI API) with
a fake version that returns predetermined values. This makes tests fast, free, and
deterministic.

**Node:** In LangGraph, a Python function that represents one step in the pipeline. Each
node receives the full state, does its work, and returns a partial dict of updated fields.

**Partial dict:** When a LangGraph node returns `{"plan": ["query1", "query2"]}`, it's
returning only the fields it changed — not the entire state object. LangGraph merges
this partial update into the full state before passing it to the next node.

**Pydantic:** A Python library for data validation. When you define a model like
`SourceDocument(BaseModel)` with a `url: str` field, Pydantic validates that the `url`
field is indeed a string every time you create a `SourceDocument`. It also powers
LangChain's `with_structured_output()` feature.

**RAG (Retrieval-Augmented Generation):** A technique where you supplement an LLM's
response by first retrieving relevant documents and giving them to the LLM as context.
Instead of relying on the LLM's training data, you "augment" its generation with fresh
retrieved information.

**Session state (Streamlit):** A dict-like object in Streamlit (`st.session_state`) that
persists across page re-renders. Since Streamlit re-runs the entire script on every user
interaction, any computed results would be lost without session state.

**Similarity threshold:** The minimum cosine similarity score (default 0.7) a stored
memory must score against the current query to be considered "relevant" and included in
the recalled memories. Memories below this threshold are ignored as too different.

**State machine:** A computational model where a system moves between defined "states"
based on rules. In this project, the states are the nodes (recall, planner, search, evaluate,
synthesize, store) and the rules are the edges (always go from plan to search; go from
evaluate to either synthesize or back to search depending on the score).

**Structured output:** When you call `llm.with_structured_output(SomePydanticModel)`,
you're telling LangChain to instruct the LLM to respond in JSON format matching that
Pydantic model's schema, and to automatically parse and validate the response. This
prevents you from manually parsing JSON from free-form LLM text.

**Tavily:** A search API built specifically for AI agents. Unlike Google/Bing APIs,
Tavily returns pre-cleaned, relevance-ranked text snippets rather than URLs to crawl
yourself. Makes search results more LLM-friendly out of the box.

**TypedDict:** A Python construct from the `typing` module that creates a dictionary
with a known, fixed set of keys and their types. Used for `ResearchState` because
LangGraph's internal framework inspects TypedDict annotations to understand the state
schema.

**Vector store / Vector database:** A database optimized for storing embedding vectors
and performing fast similarity searches against them. ChromaDB is the vector store used
in this project.

---

*This file was written as a companion to the code in `research_agent_project/`.
Nothing in this file changes any code. Its only purpose is to help you understand
what you built, what it needs, and how to run and test it confidently.*
