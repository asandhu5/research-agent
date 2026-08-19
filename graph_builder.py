from typing import Literal                      # type for routing return values
from langgraph.graph import StateGraph, END     # core LangGraph classes
from langgraph.checkpoint.memory import MemorySaver  # in-memory checkpointing (optional)

from config import cfg                          # max_iterations and threshold constants
from state import ResearchState                 # typed state schema
from agent_nodes import (
    recall_memory_node,       # Node 1: retrieve past research from ChromaDB
    planner_node,             # Node 2: decompose question into sub-queries
    search_and_scrape_node,   # Node 3: search web + scrape URLs
    evaluator_node,           # Node 4: score content sufficiency
    synthesizer_node,         # Node 5: generate Markdown report
    memory_storing_node,      # Node 6: persist summary to ChromaDB
)

def route_after_evaluation(
    state: ResearchState,
) -> Literal["replan", "synthesize"]:

    score = state.get("evaluation_score", 0.0)
    iteration = state.get("iteration", 0)

    print(f"\n[Router] Evaluation score: {score:.2f} | Iteration: {iteration}/{cfg.max_iterations}")

    if iteration >= cfg.max_iterations:
        print(f"[Router] ✓ Max iterations ({cfg.max_iterations}) reached → proceeding to synthesize")
        return "synthesize"  # force synthesis even if content isn't great

    if score >= cfg.sufficiency_threshold:
        print(f"[Router] ✓ Sufficiency threshold met ({score:.2f} >= {cfg.sufficiency_threshold}) → synthesize")
        return "synthesize"  # early exit saves API calls

    print(f"[Router] ✗ Score {score:.2f} < {cfg.sufficiency_threshold} and "
          f"iteration {iteration} < {cfg.max_iterations} → looping back to replan")
    return "replan"  # LangGraph will call planner_node again, in replan mode

def build_research_graph(use_checkpointer: bool = False):

    graph = StateGraph(ResearchState)
    print("[GraphBuilder] Initializing StateGraph with ResearchState schema...")

    graph.add_node("recall", recall_memory_node)       # Node 1: memory recall
    graph.add_node("planner", planner_node)            # Node 2: query planning / replanning
    graph.add_node("search_scrape", search_and_scrape_node)  # Node 3: search + scrape
    graph.add_node("evaluate", evaluator_node)         # Node 4: quality evaluation
    graph.add_node("synthesize", synthesizer_node)     # Node 5: report synthesis
    graph.add_node("store", memory_storing_node)       # Node 6: memory persistence
    print("[GraphBuilder] ✓ 6 nodes registered: recall → planner → search_scrape → evaluate → synthesize → store")
    graph.set_entry_point("recall")
    print("[GraphBuilder] Entry point set to: 'recall'")
    graph.add_edge("recall", "planner")
    graph.add_edge("planner", "search_scrape")
    graph.add_edge("search_scrape", "evaluate")
    graph.add_edge("synthesize", "store")
    graph.add_edge("store", END)
    print("[GraphBuilder] ✓ Unconditional edges: recall→planner→search_scrape→evaluate | synthesize→store→END")

    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "synthesize": "synthesize",
            "replan": "planner",
        },
    )
    print("[GraphBuilder] ✓ Conditional edge: evaluate → [synthesize | replan→planner]")
    print("[GraphBuilder]   Route condition: score >= 0.7 OR iteration >= max_iterations")

    if use_checkpointer:

        checkpointer = MemorySaver()
        compiled_graph = graph.compile(checkpointer=checkpointer)
        print("[GraphBuilder] ✓ Graph compiled with MemorySaver checkpointer.")
    else:
        compiled_graph = graph.compile()
        print("[GraphBuilder] ✓ Graph compiled (no checkpointer).")

    print("\n[GraphBuilder] Final graph topology:")
    print("  recall → planner → search_scrape")
    print("              ▲              │")
    print("              │ (replan if   ▼")
    print("              │  score<0.7)  evaluate")
    print("              └──────────────┘")
    print("                        │")
    print("                        ↓ (score >= 0.7 OR max iterations)")
    print("                    synthesize → store → END")

    return compiled_graph

def run_research(question: str, stream: bool = False):

    print(f"\n{'='*70}")
    print(f"Starting Research Agent")
    print(f"Query: {question}")
    print(f"{'='*70}\n")

    graph = build_research_graph(use_checkpointer=False)

    initial_state: ResearchState = {
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

    if stream:

        return graph.stream(initial_state)  # return generator, not result
    else:
        final_state = graph.invoke(initial_state)
        print(f"\n{'='*70}")
        print(f"Research Complete!")
        print(f"Report length: {len(final_state.get('final_report', '')):,} characters")
        print(f"Sources cited: {len(final_state.get('sources', []))}")
        print(f"Total iterations: {final_state.get('iteration', 0)}")
        print(f"Memory stored: {final_state.get('memory_stored', False)}")
        print(f"{'='*70}\n")
        return final_state
