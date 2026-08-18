"""
download_sample_data.py — Sample Dataset Generator & ChromaDB Seeder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    This script creates a pre-populated memory store so you can test the
    long-term memory feature WITHOUT needing API keys or internet access.

    It generates 4 synthetic research documents representing past research
    on ML topics, writes them as HTML and text files, then embeds and ingests
    them into ChromaDB.

    After running this script, the agent will "remember" past research on:
    1. Vision Transformer (ViT) architectures
    2. RAG hallucination mitigation techniques (2025)
    3. Quantization with QLoRA
    4. Agentic workflows with LangGraph

RUN THIS BEFORE pytest tests/ TO SEED THE LOCAL VECTOR STORE.
"""

import os           # file system operations
import sys          # for sys.exit() on critical failures
import json         # writing metadata JSON files
from pathlib import Path  # modern path manipulation (cleaner than os.path)

# Load environment variables before importing any modules that depend on them
from dotenv import load_dotenv
load_dotenv()

# Import our own modules
from config import cfg               # paths and hyperparameters
from memory_manager import MemoryManager  # vector store interface


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE DOCUMENTS
# Each document simulates a past research session the agent completed.
# They are intentionally substantive (200-400 words each) so the embedding
# captures rich semantic information for the similarity search to work on.
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_DOCUMENTS = [
    {
        "id": "doc_vit_architecture",
        "topic": "Vision Transformer (ViT) Architecture Overview",
        "filename": "vit_architecture_overview.html",
        "format": "html",  # will be written as an HTML file
        "summary": (
            "Vision Transformers (ViT) apply the transformer self-attention mechanism "
            "directly to image patches rather than convolutions. An image is split into "
            "fixed-size patches (e.g., 16×16 pixels), each linearly embedded into a token "
            "vector. A class token is prepended, positional embeddings are added, and the "
            "sequence is passed through standard transformer encoder blocks. "
            "Key advantages over CNNs include: (1) global receptive field from the first "
            "layer due to self-attention, enabling long-range spatial dependencies; "
            "(2) no inductive biases like translation equivariance, making ViTs more "
            "data-hungry but more flexible; (3) scalability — ViT-G achieves state-of-the-art "
            "on ImageNet with 1.8B parameters. "
            "Variants include DeiT (data-efficient ViT using distillation), Swin Transformer "
            "(hierarchical patch merging for dense prediction tasks), and MAE (masked autoencoder "
            "self-supervised pretraining). "
            "Practical limitations: ViTs require large datasets (JFT-300M or ImageNet-21k) for "
            "scratch training. Transfer learning from pretrained checkpoints is recommended for "
            "smaller datasets. Inference speed is slower than EfficientNet at small scales but "
            "faster at large scales due to better hardware utilization."
        ),
        "metadata": {
            "research_date": "2025-03-15",
            "domain": "computer_vision",
            "source_urls": [
                "https://arxiv.org/abs/2010.11929",
                "https://arxiv.org/abs/2111.06377"
            ],
            "iteration_count": 2,
            "source_count": 5,
        }
    },
    {
        "id": "doc_rag_hallucination",
        "topic": "RAG Hallucination Mitigations 2025",
        "filename": "rag_hallucination_mitigations_2025.html",
        "format": "html",
        "summary": (
            "Hallucination in Retrieval-Augmented Generation (RAG) systems occurs when the "
            "language model generates plausible-sounding but factually incorrect content that "
            "is not supported by the retrieved documents. Key mitigation techniques in 2025: "
            "(1) Faithful Summarization — post-processing pipeline that verifies each claim "
            "in the generated output against the retrieved chunks using an entailment model. "
            "(2) Self-RAG — LLMs trained with reflection tokens that critique their own "
            "retrieval decisions at generation time, selectively retrieving only when needed. "
            "(3) CRAG (Corrective RAG) — evaluates retrieval quality with a lightweight "
            "classifier; if quality is low, it falls back to web search before generation. "
            "(4) Ensemble Voting — runs N retrieval passes with different query formulations "
            "and takes the majority-consensus answer. "
            "(5) Chain-of-Note — forces the model to write reading notes per document before "
            "synthesizing, improving faithfulness to sources. "
            "(6) GraphRAG — uses a knowledge graph to structure entity relationships, reducing "
            "disconnected 'islands' of retrieved context that cause coherence failures. "
            "Evaluation: RAGAS metrics (faithfulness, answer relevancy, context recall) provide "
            "automated scores for RAG pipeline quality without human annotation."
        ),
        "metadata": {
            "research_date": "2025-06-10",
            "domain": "nlp_rag",
            "source_urls": [
                "https://arxiv.org/abs/2310.11511",
                "https://arxiv.org/abs/2401.15884"
            ],
            "iteration_count": 3,
            "source_count": 8,
        }
    },
    {
        "id": "doc_qlora_quantization",
        "topic": "Quantization Techniques with QLoRA",
        "filename": "qlora_quantization_techniques.txt",
        "format": "text",  # will be written as a plain text file
        "summary": (
            "QLoRA (Quantized Low-Rank Adaptation) enables fine-tuning of large language models "
            "on consumer GPUs by combining 4-bit NormalFloat (NF4) quantization with Low-Rank "
            "Adapter (LoRA) matrices. Key concepts: "
            "(1) 4-bit NF4 Quantization — weights are stored as 4-bit integers using a "
            "quantile-based normalization scheme optimized for normally distributed weights. "
            "This reduces a 65B parameter model from 130 GB (float16) to ~35 GB. "
            "(2) Double Quantization — the quantization constants themselves are quantized "
            "a second time, saving an additional 0.4 bits per parameter. "
            "(3) LoRA Adapters — only small rank-decomposed matrices (r=16, alpha=32) are "
            "updated during fine-tuning; the quantized base model weights are frozen. "
            "Gradient computation flows through the LoRA matrices using straight-through "
            "estimators for the quantized weights. "
            "(4) Paged Optimizers — NVIDIA unified memory is used to page optimizer states "
            "to CPU RAM when GPU memory is full, preventing OOM errors during long sequences. "
            "Results: QLoRA achieves near-full fine-tuning performance while reducing GPU "
            "memory requirements by 65–70%. A 7B model can be fine-tuned on a single 24 GB "
            "RTX 3090. Best practices: use NF4 for base weights, bfloat16 for computation "
            "dtype, and apply LoRA to all attention projection layers (q, k, v, o)."
        ),
        "metadata": {
            "research_date": "2025-04-22",
            "domain": "model_optimization",
            "source_urls": [
                "https://arxiv.org/abs/2305.14314",
                "https://github.com/artidoro/qlora"
            ],
            "iteration_count": 2,
            "source_count": 4,
        }
    },
    {
        "id": "doc_agentic_langgraph",
        "topic": "Agentic Workflows with LangGraph",
        "filename": "agentic_workflows_langgraph.txt",
        "format": "text",
        "summary": (
            "LangGraph is a library for building stateful, multi-actor applications with LLMs "
            "as a directed graph of nodes (computation units) and edges (transitions). "
            "Core concepts: "
            "(1) StateGraph — the primary graph class. You define a TypedDict as the state schema "
            "and add nodes that receive the full state and return partial updates. "
            "(2) Nodes — Python functions or Runnables. Each node reads from state and writes "
            "ONLY its designated output keys back via a partial dict. "
            "(3) Conditional Edges — routing logic implemented as a Python function that returns "
            "a string key mapped to the next node. Enables loops, branching, and early exits. "
            "(4) Persistence & Checkpointing — LangGraph can attach a Checkpointer (e.g., "
            "SqliteSaver) to save state after every node, enabling resume-from-checkpoint, "
            "time-travel debugging, and human-in-the-loop approval workflows. "
            "(5) Multi-Agent Patterns — nodes can themselves be sub-graphs, enabling hierarchical "
            "agent architectures (e.g., a Supervisor agent that routes to specialized sub-agents). "
            "Common patterns: ReAct (reason + act loops), Plan-and-Execute (separate planning from "
            "action), Reflexion (self-critique loop), and Multi-Agent Debate. "
            "LangGraph vs. LangChain Agents: LangGraph gives you explicit control over the graph "
            "topology; LangChain's AgentExecutor is a fixed while-loop. Use LangGraph when your "
            "workflow requires custom routing, multi-step planning, or parallel execution."
        ),
        "metadata": {
            "research_date": "2025-07-01",
            "domain": "agentic_ai",
            "source_urls": [
                "https://langchain-ai.github.io/langgraph/",
                "https://arxiv.org/abs/2402.01680"
            ],
            "iteration_count": 1,
            "source_count": 6,
        }
    },
]


def create_sample_html(doc: dict) -> str:
    """
    Generate a realistic-looking HTML research page from document data.

    Why HTML?
        We save some documents as HTML to test the web scraper's HTML cleaning
        pipeline. When test_scraper.py loads these files, it verifies that
        BeautifulSoup correctly strips scripts and navigation tags.

    Args:
        doc: A SAMPLE_DOCUMENTS entry dict.

    Returns:
        A complete HTML string with realistic structure (nav, scripts, main content).
    """
    # We include noise elements (<script>, <nav>, <footer>) on purpose so the
    # scraper tests can verify they're correctly removed.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{doc['topic']}</title>
    <style>
        /* THIS STYLE TAG SHOULD BE REMOVED BY THE SCRAPER */
        body {{ font-family: Arial, sans-serif; }}
        nav {{ background: #333; color: white; padding: 10px; }}
    </style>
    <script>
        // THIS SCRIPT TAG SHOULD BE REMOVED BY THE SCRAPER
        console.log("Google Analytics initialized");
        window.addEventListener('load', function() {{
            gtag('event', 'page_view', {{ send_to: 'G-XXXXXXXXXX' }});
        }});
    </script>
</head>
<body>
    <nav>
        <!-- THIS NAV SHOULD BE REMOVED BY THE SCRAPER -->
        <a href="/">Home</a> |
        <a href="/research">Research</a> |
        <a href="/about">About</a> |
        <a href="/contact">Contact</a>
    </nav>

    <header>
        <!-- THIS HEADER SHOULD BE REMOVED BY THE SCRAPER -->
        <div class="site-logo">Research Agent Demo Database</div>
        <div class="advertisement">[ AD SPACE — REMOVE ME ]</div>
    </header>

    <main>
        <article>
            <h1>{doc['topic']}</h1>
            <p class="meta">Published: {doc['metadata']['research_date']} | Domain: {doc['metadata']['domain']}</p>

            <section id="summary">
                <h2>Research Summary</h2>
                <p>{doc['summary']}</p>
            </section>

            <section id="sources">
                <h2>Key Sources</h2>
                <ul>
                    {"".join(f'<li><a href="{url}">{url}</a></li>' for url in doc['metadata']['source_urls'])}
                </ul>
            </section>

            <section id="metadata">
                <h2>Research Metadata</h2>
                <p>Iteration count: {doc['metadata']['iteration_count']}</p>
                <p>Source count: {doc['metadata']['source_count']}</p>
            </section>
        </article>
    </main>

    <aside class="sidebar">
        <!-- THIS SIDEBAR SHOULD BE REMOVED BY THE SCRAPER -->
        <h3>Related Articles</h3>
        <p>See also: other research on machine learning topics.</p>
    </aside>

    <footer>
        <!-- THIS FOOTER SHOULD BE REMOVED BY THE SCRAPER -->
        <p>&copy; 2025 Research Agent Demo. All rights reserved.</p>
        <p>Privacy Policy | Terms of Service | Cookie Settings</p>
    </footer>

    <script>
        // MORE JAVASCRIPT — SHOULD BE REMOVED
        document.addEventListener('DOMContentLoaded', function() {{
            console.log("Page loaded");
        }});
    </script>
</body>
</html>"""


def create_sample_text(doc: dict) -> str:
    """
    Generate a plain-text research document from document data.

    Args:
        doc: A SAMPLE_DOCUMENTS entry dict.

    Returns:
        A formatted plain-text string with headers and content.
    """
    source_list = "\n".join(
        f"  [{i+1}] {url}"
        for i, url in enumerate(doc['metadata']['source_urls'])
    )
    return f"""RESEARCH DOCUMENT
═══════════════════════════════════════════

Topic: {doc['topic']}
Date:  {doc['metadata']['research_date']}
Domain: {doc['metadata']['domain']}
Sources Consulted: {doc['metadata']['source_count']}
Research Iterations: {doc['metadata']['iteration_count']}

───────────────────────────────────────────
RESEARCH SUMMARY
───────────────────────────────────────────

{doc['summary']}

───────────────────────────────────────────
KEY SOURCES
───────────────────────────────────────────

{source_list}

───────────────────────────────────────────
END OF DOCUMENT
───────────────────────────────────────────
"""


def main():
    """
    Main execution function. Creates files, ingests into ChromaDB, and verifies.

    Execution steps:
    1. Create output directories.
    2. Write HTML and text files to ./data/sample_documents/.
    3. Initialize MemoryManager (creates ChromaDB if it doesn't exist).
    4. Embed and store each document summary in ChromaDB.
    5. Verify storage by querying back a test query and checking counts.
    6. Print final summary with exact counts and file locations.
    """
    print("=" * 60)
    print("Research Agent — Sample Data Generator")
    print("=" * 60)

    # ── STEP 1: CREATE DIRECTORIES ────────────────────────────────────────────
    docs_path = Path(cfg.sample_docs_path)  # "./data/sample_documents"
    docs_path.mkdir(parents=True, exist_ok=True)  # create all parent directories
    print(f"\n[Step 1/5] Created document directory: {docs_path.absolute()}")

    # ── STEP 2: WRITE SAMPLE FILES ────────────────────────────────────────────
    print(f"\n[Step 2/5] Writing {len(SAMPLE_DOCUMENTS)} sample documents...")
    written_files = []

    for doc in SAMPLE_DOCUMENTS:
        file_path = docs_path / doc["filename"]  # full path to the output file

        if doc["format"] == "html":
            content = create_sample_html(doc)  # generate HTML with noise elements
            file_path.write_text(content, encoding="utf-8")  # write to disk
        else:  # plain text
            content = create_sample_text(doc)
            file_path.write_text(content, encoding="utf-8")

        written_files.append(str(file_path))
        file_size_kb = file_path.stat().st_size / 1024  # file size in kilobytes
        print(f"  ✓ {doc['filename']} ({file_size_kb:.1f} KB) — '{doc['topic']}'")

    # ── STEP 3: INITIALIZE MEMORY MANAGER ────────────────────────────────────
    print(f"\n[Step 3/5] Initializing ChromaDB memory store at: {cfg.db_path}")
    print("  (This may download the embedding model on first run...)")

    try:
        memory = MemoryManager(
            db_path=cfg.db_path,
            collection_name=cfg.collection_name,
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize MemoryManager: {e}")
        print("  Check that chromadb and sentence-transformers are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)  # exit with non-zero code to signal failure to scripts/CI

    # ── STEP 4: INGEST DOCUMENTS INTO CHROMADB ────────────────────────────────
    print(f"\n[Step 4/5] Embedding and ingesting {len(SAMPLE_DOCUMENTS)} documents into ChromaDB...")
    ingested_ids = []

    for doc in SAMPLE_DOCUMENTS:
        print(f"  Ingesting: '{doc['topic']}'")

        doc_id = memory.store_memory(
            topic=doc["topic"],
            summary=doc["summary"],
            metadata={
                **doc["metadata"],  # spread all metadata from the doc definition
                "filename": doc["filename"],
                "doc_id": doc["id"],
            },
        )

        if doc_id:
            ingested_ids.append(doc_id)
            print(f"  ✓ Stored with ChromaDB ID: {doc_id[:12]}...")
        else:
            print(f"  ⚠ Skipped (possibly already exists as near-duplicate)")

    # ── STEP 5: VERIFY AND REPORT ─────────────────────────────────────────────
    print(f"\n[Step 5/5] Verifying ChromaDB storage...")

    total_memories = memory.count
    all_memories = memory.get_all_memories()

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Files written:     {len(written_files)}")
    print(f"Documents stored:  {total_memories} records in ChromaDB")
    print(f"ChromaDB path:     {Path(cfg.db_path).absolute()}")
    print(f"Documents path:    {docs_path.absolute()}")
    print(f"")
    print(f"Stored memories:")
    for m in all_memories:
        print(f"  • {m['topic'][:65]}")
        print(f"    Timestamp: {m['timestamp']} | Size: {m['char_length']} chars")

    # ── BONUS: DEMONSTRATE MEMORY RECALL ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"DEMO: Testing Memory Recall")
    print(f"{'=' * 60}")

    test_query = "How does RAG prevent hallucinations in language models?"
    print(f"\nQuery: '{test_query}'")
    recalled = memory.recall_memories(test_query, top_k=2, similarity_threshold=0.3)

    if recalled:
        print(f"\nTop {len(recalled)} recalled memories:")
        for i, mem in enumerate(recalled, 1):
            print(f"  {i}. [{mem.similarity_score:.2f}] {mem.topic}")
            print(f"     Preview: {mem.summary[:120]}...")
    else:
        print("  No memories recalled above threshold.")
        print("  (This is expected if using the local embedding model for the first time.)")

    # ── WRITE METADATA JSON ───────────────────────────────────────────────────
    metadata_file = docs_path / "ingestion_metadata.json"
    ingestion_record = {
        "total_documents": len(SAMPLE_DOCUMENTS),
        "successfully_ingested": len(ingested_ids),
        "chroma_db_path": str(Path(cfg.db_path).absolute()),
        "documents": [
            {"id": d["id"], "topic": d["topic"], "filename": d["filename"]}
            for d in SAMPLE_DOCUMENTS
        ],
    }
    metadata_file.write_text(json.dumps(ingestion_record, indent=2), encoding="utf-8")
    print(f"\n[Done] Ingestion metadata saved to: {metadata_file}")
    print(f"\nYou can now run: pytest tests/")
    print(f"Or launch:       streamlit run app.py")
    print(f"Or query:        python main.py --query 'How do ViTs differ from CNNs?'")


if __name__ == "__main__":
    main()
