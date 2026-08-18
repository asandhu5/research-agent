"""
main.py — CLI Entry Point for the Research Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE:
    python main.py --query "What are the top RAG hallucination mitigation techniques?"
    python main.py --query "Explain GraphRAG" --output ./my_report.md
    python main.py --query "ViT vs CNN comparison" --no-memory

This script is the command-line interface for running a single research query.
It wraps graph_builder.run_research() and handles:
    - Argument parsing (query, output file, flags)
    - Environment variable validation
    - Clear startup diagnostics
    - Writing the final report to a Markdown file
"""

import argparse    # standard library CLI argument parser
import os          # environment variable reading
import sys         # sys.exit() for clean error exits
from pathlib import Path  # cross-platform file paths
from datetime import datetime  # for timestamped output filenames

from dotenv import load_dotenv
load_dotenv()  # load .env before importing LangChain (which reads API keys at import)


def parse_args() -> argparse.Namespace:
    """
    Define and parse command-line arguments.

    Returns:
        An argparse.Namespace object with attributes for each argument.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Autonomous Web Research Agent with Persistent Long-Term Memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --query "What is GraphRAG?"
  python main.py --query "RAG hallucinations 2025" --output report.md
  python main.py --query "QLoRA fine-tuning requirements" --no-memory --iterations 2

Setup:
  1. Copy .env.example to .env and fill in your API keys.
  2. Run: python download_sample_data.py  (seeds local vector memory)
  3. Run: python main.py --query "your question here"
        """,
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,  # query is mandatory — agent can't run without a question
        help="The research question to investigate (wrap in quotes if it contains spaces)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,  # None = auto-generate filename based on timestamp
        help="Output file path for the Markdown report (default: ./data/report_TIMESTAMP.md)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=None,  # None = use cfg.max_iterations default
        help=f"Maximum search iterations (default: {3}). Higher = more thorough but slower.",
    )

    parser.add_argument(
        "--no-memory",
        action="store_true",  # flag — present = True, absent = False
        default=False,
        help="Skip memory recall and storage (useful for testing the cold-start pipeline)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print extra debug information during execution",
    )

    return parser.parse_args()


def validate_environment() -> bool:
    """
    Check that required API keys are configured and print a diagnostic summary.

    We don't REQUIRE OpenAI key here (user might want to test with mocks),
    but we warn loudly if it's missing since the full pipeline won't work.

    Returns:
        True if minimum required keys are set, False otherwise.
    """
    has_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    has_tavily = bool(os.environ.get("TAVILY_API_KEY", "").strip())

    print("\nEnvironment Check:")
    print(f"  OPENAI_API_KEY:  {'✓ Configured' if has_openai else '✗ MISSING — LLM calls will fail'}")
    print(f"  TAVILY_API_KEY:  {'✓ Configured' if has_tavily else '⚠ Not set — DuckDuckGo fallback will be used'}")

    if not has_openai:
        print("\n[ERROR] OPENAI_API_KEY is required to run the full research pipeline.")
        print("Create a .env file with:")
        print("  OPENAI_API_KEY=sk-your-key-here")
        print("\nFor testing WITHOUT an API key, run: pytest tests/")
        return False  # signal caller to exit

    return True  # all required keys present


def save_report(report: str, output_path: str) -> str:
    """
    Save the final Markdown report to a file.

    Args:
        report: The Markdown string from the synthesizer node.
        output_path: Desired file path, or None to auto-generate.

    Returns:
        The actual file path where the report was saved.
    """
    if not output_path:
        # Auto-generate filename: report_YYYYMMDD_HHMMSS.md
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path("./data").mkdir(exist_ok=True)  # create data dir if it doesn't exist
        output_path = f"./data/report_{timestamp}.md"

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)  # ensure output directory exists
    path.write_text(report, encoding="utf-8")
    return str(path.absolute())


def main():
    """
    Main CLI execution function.

    Orchestrates: argument parsing → environment validation → graph execution
    → report display → file saving.
    """
    args = parse_args()

    print("=" * 70)
    print("  Autonomous Web Research Agent")
    print("=" * 70)
    print(f"  Query: {args.query}")

    # Override config if iteration count is specified via CLI
    if args.iterations:
        # Temporarily monkey-patch the config for this run.
        # In a production system, you'd pass this through a Config override mechanism.
        import config
        object.__setattr__(config.cfg, 'max_iterations', args.iterations)
        print(f"  Max iterations: {args.iterations} (overriding default)")

    # Validate environment before doing expensive operations
    if not validate_environment():
        sys.exit(1)  # exit with error code so shell scripts can detect failure

    # Import after environment validation to avoid import-time LLM initialization
    from graph_builder import run_research
    from memory_manager import MemoryManager

    # Show memory status
    try:
        mem = MemoryManager()
        print(f"  Memory store:   {mem.count} records in ChromaDB")
        if mem.count == 0:
            print("  ⚠ Memory is empty. Run: python download_sample_data.py")
    except Exception as e:
        print(f"  Memory store:   Error — {e}")

    print()

    # ── RUN THE RESEARCH AGENT ────────────────────────────────────────────────
    final_state = run_research(args.query)

    # ── DISPLAY RESULTS ───────────────────────────────────────────────────────
    report = final_state.get("final_report", "")
    sources = final_state.get("sources", [])
    eval_score = final_state.get("evaluation_score", 0.0)
    iteration = final_state.get("iteration", 0)
    recalled = final_state.get("recalled_memories", [])

    print("\n" + "=" * 70)
    print("RESEARCH COMPLETE — FINAL REPORT")
    print("=" * 70)
    print(report)
    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Evaluation score:    {eval_score:.2f}")
    print(f"  Search iterations:   {iteration}")
    print(f"  Sources cited:       {len(sources)}")
    print(f"  Memory recalled:     {len(recalled)} past research sessions")
    print(f"  Memory stored:       {final_state.get('memory_stored', False)}")

    # ── SAVE REPORT TO FILE ───────────────────────────────────────────────────
    if report:
        saved_path = save_report(report, args.output)
        print(f"\n  Report saved to:     {saved_path}")
    else:
        print("\n  [WARNING] No report generated. Check API keys and network.")

    print()


if __name__ == "__main__":
    main()
