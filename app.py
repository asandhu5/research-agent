

import os           # for API key checks
import re           # for citation counting in the summary panel
import time         # for timing agent execution
from datetime import datetime  # for timestamps in execution log
from typing import Optional    # type hints

import streamlit as st         # the entire UI framework
from dotenv import load_dotenv  # load .env before anything else

load_dotenv()

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        color: #e2e8f0;
        font-size: 2rem;
        margin: 0;
    }
    .main-header p {
        color: #94a3b8;
        margin: 0.5rem 0 0 0;
    }

    /* Execution node timeline cards */
    .node-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        font-family: monospace;
        font-size: 0.85rem;
        color: #e2e8f0;
    }
    .node-card.completed { border-left-color: #22c55e; }
    .node-card.running   { border-left-color: #f59e0b; animation: pulse 1s infinite; }
    .node-card.failed    { border-left-color: #ef4444; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.6; }
    }

    /* Metric badge */
    .metric-badge {
        display: inline-block;
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        color: #60a5fa;
        margin: 0.2rem;
    }

    /* Source link cards */
    .source-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    .source-card a { color: #60a5fa; text-decoration: none; }
    .source-card a:hover { text-decoration: underline; }

    /* Memory record cards */
    .memory-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.6rem 0;
    }
    .memory-card h4 { color: #f1f5f9; margin: 0 0 0.4rem 0; font-size: 0.95rem; }
    .memory-card p  { color: #94a3b8; font-size: 0.82rem; margin: 0.2rem 0; }
    .similarity-badge {
        background: #14532d;
        color: #4ade80;
        padding: 0.15rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
    }

    /* Report viewer */
    .report-container {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.5rem 2rem;
    }

    /* Warning/info banners */
    .api-warning {
        background: #431407;
        border: 1px solid #7c2d12;
        color: #fbbf24;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

if "research_state" not in st.session_state:
    st.session_state.research_state = None   # will hold the final ResearchState dict

if "execution_log" not in st.session_state:
    st.session_state.execution_log = []      # list of (node_name, timestamp, status) tuples

if "is_running" not in st.session_state:
    st.session_state.is_running = False      # prevents double-click submission

if "last_query" not in st.session_state:
    st.session_state.last_query = ""         # tracks the most recent query for display



def check_api_keys() -> tuple[bool, bool]:

    has_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    has_tavily = bool(os.environ.get("TAVILY_API_KEY", "").strip())
    return has_openai, has_tavily

def run_agent_with_streaming(question: str) -> dict:

    from graph_builder import build_research_graph

    graph = build_research_graph(use_checkpointer=False)

    initial_state = {
        "question": question,
        "recalled_memories": [],
        "plan": [],
        "raw_search_results": [],
        "scraped_content": [],
        "iteration": 0,
        "evaluation_score": 0.0,
        "evaluation_reasoning": "",
        "missing_topics": [],
        "final_report": "",
        "sources": [],
        "memory_stored": False,
    }

    node_display_names = {
        "recall": "🧠 Recalling Long-Term Memory",
        "planner": "📋 Planning Research Sub-Queries",
        "search_scrape": "🔍 Searching Web & Scraping Pages",
        "evaluate": "⚖️  Evaluating Research Sufficiency",
        "synthesize": "✍️  Synthesizing Research Report",
        "store": "💾 Storing Findings to Memory",
    }

    accumulated_state = {**initial_state}  # track cumulative state across all nodes
    st.session_state.execution_log = []     # reset execution log for this run

    progress_container = st.empty()

    for node_update in graph.stream(initial_state):
        for node_name, node_output in node_update.items():
            timestamp = datetime.now().strftime("%H:%M:%S")
            display_name = node_display_names.get(node_name, f"⚙️  {node_name}")

            st.session_state.execution_log.append({
                "node": node_name,
                "display": display_name,
                "timestamp": timestamp,
                "status": "completed",
                "output_keys": list(node_output.keys()) if isinstance(node_output, dict) else [],
            })

            if isinstance(node_output, dict):
                accumulated_state.update(node_output)

            with progress_container.container():
                st.markdown("**Execution Progress:**")
                for log_entry in st.session_state.execution_log:
                    _render_node_card(log_entry, accumulated_state)

    return accumulated_state


def _render_node_card(log_entry: dict, state: dict) -> None:

    node = log_entry["node"]
    label = f"{log_entry['display']} — {log_entry['timestamp']}"

    with st.expander(label, expanded=False):
        if node == "recall":
            # Show recalled memories
            recalled = state.get("recalled_memories", [])
            if recalled:
                for m in recalled:
                    st.markdown(f"**{m.topic}** (similarity: {m.similarity_score:.2f})")
                    st.caption(m.summary[:200] + "...")
            else:
                st.caption("No prior memories found for this query (cold start).")

        elif node == "planner":
            plan = state.get("plan", [])
            missing = state.get("missing_topics", [])
            if state.get("iteration", 0) > 0 and missing:
                st.caption(f"Replanning to target: {', '.join(missing[:3])}")
            if plan:
                for i, q in enumerate(plan, 1):
                    st.markdown(f"**{i}.** {q}")
            else:
                st.caption("No plan generated.")

        elif node == "search_scrape":
            # Show scrape statistics
            scraped = state.get("scraped_content", [])
            search_results = state.get("raw_search_results", [])
            iteration = state.get("iteration", 0)
            st.markdown(f"**Iteration:** {iteration}")
            st.markdown(f"**Search results found:** {len(search_results)}")
            st.markdown(f"**Pages scraped:** {len(scraped)}")
            successful = sum(1 for p in scraped if p.success)
            st.markdown(f"**Successfully scraped:** {successful}/{len(scraped)}")

        elif node == "evaluate":
            # Show evaluation score and reasoning
            score = state.get("evaluation_score", 0.0)
            reasoning = state.get("evaluation_reasoning", "")
            color = "#22c55e" if score >= 0.7 else ("#f59e0b" if score >= 0.5 else "#ef4444")
            st.markdown(f"**Score:** <span style='color:{color};font-weight:bold'>{score:.2f}</span>", unsafe_allow_html=True)
            if reasoning:
                st.caption(reasoning[:400])

        elif node == "synthesize":
            # Show report preview
            report = state.get("final_report", "")
            sources = state.get("sources", [])
            st.markdown(f"**Report generated:** {len(report):,} characters")
            st.markdown(f"**Sources cited:** {len(sources)}")

        elif node == "store":
            # Show memory storage status
            stored = state.get("memory_stored", False)
            if stored:
                st.success("Research saved to long-term memory ✓")
            else:
                st.warning("Memory storage skipped (duplicate or no report).")


def render_source_list(sources: list) -> None:

    if not sources:
        st.caption("No sources available.")
        return

    for i, source in enumerate(sources, 1):
        title = getattr(source, 'title', source.get('title', 'Untitled')) if not isinstance(source, dict) else source.get('title', 'Untitled')
        url = getattr(source, 'url', source.get('url', '#')) if not isinstance(source, dict) else source.get('url', '#')
        st.markdown(
            f'<div class="source-card">'
            f'<strong>[{i}]</strong> <a href="{url}" target="_blank">{title}</a>'
            f'<br><small style="color:#64748b">{url[:80]}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("""
<div class="main-header">
    <h1>🔬 Autonomous Web Research Agent</h1>
    <p>Persistent Long-Term Memory · LangGraph State Machine · GPT-4o Synthesis</p>
</div>
""", unsafe_allow_html=True)

has_openai, has_tavily = check_api_keys()
if not has_openai:
    st.markdown(
        '<div class="api-warning">⚠️ <strong>OPENAI_API_KEY not configured.</strong> '
        'The research pipeline requires an OpenAI API key. '
        'Create a <code>.env</code> file with <code>OPENAI_API_KEY=sk-your-key</code> '
        'and restart with <code>streamlit run app.py</code>.</div>',
        unsafe_allow_html=True,
    )

tab1, tab2 = st.tabs(["🔬 Research Workstation", "🧠 Long-Term Memory Explorer"])


with tab1:
    st.markdown("### Research Query")

    example_queries = [
        "What are the top techniques for reducing hallucinations in RAG systems in 2025?",
        "How does GraphRAG use knowledge graphs to improve retrieval quality?",
        "What are the hardware requirements for fine-tuning LLMs with QLoRA?",
        "Explain the Vision Transformer architecture and its advantages over CNNs.",
        "What are the best practices for deploying LangGraph agents in production?",
    ]

    col_ex, col_btn = st.columns([4, 1])

    with col_ex:
        selected_example = st.selectbox(
            "Quick-start examples:",
            options=["(type your own question below)"] + example_queries,
            label_visibility="collapsed",
        )

    with col_btn:
        if st.button("Use Example", use_container_width=True):
            # Pre-fill the query box with the selected example
            if selected_example != "(type your own question below)":
                st.session_state["query_input"] = selected_example
                st.rerun()

    query = st.text_area(
        "Enter your research question:",
        value=st.session_state.get("query_input", ""),
        height=80,
        placeholder="e.g., What are the top techniques for reducing hallucinations in RAG systems in 2025?",
        key="query_text_area",
        label_visibility="visible",
    )

    col_run, col_clear = st.columns([3, 1])

    with col_run:
        run_button = st.button(
            "🚀 Run Autonomous Research",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_running or not has_openai,  # disable if no API key
        )

    with col_clear:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.research_state = None
            st.session_state.execution_log = []
            st.session_state.last_query = ""
            st.rerun()

    if run_button and query.strip():
        st.session_state.is_running = True
        st.session_state.last_query = query.strip()
        st.session_state.research_state = None  # clear previous results
        st.session_state.execution_log = []

        st.markdown("---")
        st.markdown("### Live Execution Tree")

        start_time = time.time()

        try:
            with st.spinner("Agent running..."):
                final_state = run_agent_with_streaming(query.strip())

            elapsed = time.time() - start_time
            st.session_state.research_state = final_state
            st.success(f"✓ Research completed in {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            st.error(f"Agent failed after {elapsed:.1f}s: {str(e)}")
            st.exception(e)
        finally:
            st.session_state.is_running = False

        st.rerun()  # re-run to display results section below

    if st.session_state.research_state:
        state = st.session_state.research_state

        report = state.get("final_report", "")
        sources = state.get("sources", [])
        eval_score = state.get("evaluation_score", 0.0)
        iteration = state.get("iteration", 0)
        recalled = state.get("recalled_memories", [])
        memory_stored = state.get("memory_stored", False)
        plan = state.get("plan", [])

        st.markdown("---")
        st.markdown(f"### Results for: _{st.session_state.last_query[:80]}_")

        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        with mcol1:
            st.metric("Eval Score", f"{eval_score:.2f}", help="LLM's sufficiency score (0–1). >0.7 = comprehensive.")
        with mcol2:
            st.metric("Iterations", iteration, help="Number of search loops executed.")
        with mcol3:
            st.metric("Sources", len(sources), help="Number of web pages cited in the report.")
        with mcol4:
            st.metric("Memory Recalled", len(recalled), help="Past research sessions used as context.")
        with mcol5:
            st.metric("Memory Saved", "✓" if memory_stored else "✗", help="Whether findings were persisted to ChromaDB.")

        reasoning = state.get("evaluation_reasoning", "")
        if reasoning:
            with st.expander("📊 Evaluation Analysis", expanded=False):
                st.markdown(reasoning)

        if st.session_state.execution_log:
            with st.expander("⚙️ Execution Log", expanded=False):
                for entry in st.session_state.execution_log:
                    st.markdown(
                        f'<div class="node-card completed">'
                        f'<strong>{entry["display"]}</strong> — {entry["timestamp"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("---")
        st.markdown("### 📄 Research Report")

        if report:
            st.markdown(report)  # renders full Markdown with headers, bold, links
        else:
            st.warning("No report was generated. This usually means no web content could be scraped.")

        if sources:
            st.markdown("---")
            st.markdown("### 🔗 Sources")
            render_source_list(sources)

        if report:
            st.download_button(
                label="⬇️ Download Report as Markdown",
                data=report.encode("utf-8"),
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=False,
            )

with tab2:
    st.markdown("### 🧠 Persistent Memory Store (ChromaDB)")
    st.caption(
        "This tab shows all research summaries stored in your local ChromaDB vector database. "
        "Run `python download_sample_data.py` to seed the store with sample memories."
    )

    @st.cache_resource  # cache the MemoryManager across re-runs (one init per server restart)
    def get_memory_manager():
        """Returns a cached MemoryManager instance."""
        from memory_manager import MemoryManager
        return MemoryManager()

    try:
        memory = get_memory_manager()
        total_count = memory.count

        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total Memories", total_count)
        with col_stat2:
            embedding_type = "OpenAI" if os.environ.get("OPENAI_API_KEY") else "Local (MiniLM)"
            st.metric("Embedding Model", embedding_type)
        with col_stat3:
            from config import cfg as app_cfg
            st.metric("DB Path", app_cfg.db_path)

        st.markdown("---")

        st.markdown("#### Test Memory Retrieval")
        mem_col1, mem_col2 = st.columns([4, 1])

        with mem_col1:
            search_query = st.text_input(
                "Search memory:",
                placeholder="e.g., How does RAG prevent hallucinations?",
                label_visibility="collapsed",
            )
        with mem_col2:
            search_button = st.button("🔍 Search", use_container_width=True)

        if search_button and search_query.strip():
            with st.spinner("Querying vector store..."):
                recalled = memory.recall_memories(
                    query=search_query,
                    top_k=5,
                    similarity_threshold=0.0,  # show all results regardless of score for explorer
                )

            if recalled:
                st.markdown(f"**{len(recalled)} results** for: _{search_query}_")
                for mem_record in recalled:
                    st.markdown(
                        f'<div class="memory-card">'
                        f'<h4>{mem_record.topic}</h4>'
                        f'<p><span class="similarity-badge">similarity: {mem_record.similarity_score:.3f}</span></p>'
                        f'<p>{mem_record.summary[:400]}{"..." if len(mem_record.summary) > 400 else ""}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No memories found. Try running `python download_sample_data.py` first.")

        st.markdown("---")
        st.markdown("#### All Stored Memories")

        all_memories = memory.get_all_memories()

        if all_memories:
            for mem_data in all_memories:
                with st.expander(f"📝 {mem_data['topic'][:80]}", expanded=False):
                    info_col, _ = st.columns([2, 1])
                    with info_col:
                        st.caption(f"**Stored:** {mem_data['timestamp'][:19]}")
                        st.caption(f"**Size:** {mem_data['char_length']} characters")
                        st.caption(f"**ID:** {mem_data['id'][:16]}...")
                    st.markdown(mem_data['summary'])

                    raw_meta = mem_data.get("metadata", {})
                    if raw_meta:
                        with st.expander("Raw Metadata"):
                            st.json(raw_meta)
        else:
            st.info(
                "Memory store is empty. Run `python download_sample_data.py` to populate "
                "with 4 sample research documents."
            )
            if st.button("Run Sample Data Seeder"):
                with st.spinner("Running download_sample_data.py..."):
                    import subprocess
                    result = subprocess.run(
                        ["python", "download_sample_data.py"],
                        capture_output=True,
                        text=True,
                    )
                if result.returncode == 0:
                    st.success("✓ Sample data ingested successfully! Reload the page to see memories.")
                else:
                    st.error(f"Script failed:\n{result.stderr[:500]}")

        st.markdown("---")

        st.markdown("#### Danger Zone")
        st.warning(
            "**Clear Memory** permanently deletes ALL stored research from ChromaDB. "
            "This cannot be undone. Re-run `download_sample_data.py` to restore sample data."
        )
        clear_col1, clear_col2 = st.columns([3, 1])
        with clear_col2:
            if st.button("🗑️ Clear All Memory", type="secondary", use_container_width=True):
                with st.spinner("Clearing memory store..."):
                    success = memory.clear_memory()
                    st.cache_resource.clear()  # bust the MemoryManager cache so count refreshes
                if success:
                    st.success("✓ All memories cleared.")
                    st.rerun()
                else:
                    st.error("Failed to clear memory. Check logs for details.")

    except Exception as e:
        st.error(f"Failed to connect to ChromaDB: {e}")
        st.code(str(e))
        st.info(
            "Make sure ChromaDB is installed: `pip install chromadb`\n"
            "And run: `python download_sample_data.py` to initialize the database."
        )


st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#475569;font-size:0.8rem;">'
    'Autonomous Research Agent · LangGraph + ChromaDB + GPT-4o · '
    'Built for ML Engineers Learning Agentic Workflows'
    '</p>',
    unsafe_allow_html=True,
)
