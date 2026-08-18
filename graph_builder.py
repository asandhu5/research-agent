"""
graph_builder.py — LangGraph State Machine Definition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS A STATE MACHINE?

    A state machine is a computational model where:
    1. The system is always in exactly ONE "state" (node).
    2. It transitions between states based on defined rules (edges).
    3. Some transitions are conditional — the next state depends on a value
       in the current state (e.g., evaluation_score >= threshold).
    4. Execution terminates when the machine reaches an END state.

    In traditional agentic code, you write this as:
        result = plan(question)
        while not satisfied:
            result = search(result)
            result = evaluate(result)
            if result.score >= threshold: break
        result = synthesize(result)

    LangGraph externalizes these transitions into a graph structure, which:
    - Makes the flow INSPECTABLE and VISUALIZABLE (draw the graph!).
    - Allows nodes to be tested independently without running the full graph.
    - Enables advanced features like human-in-the-loop, checkpointing, and replay.

CYCLES IN STATE MACHINES — THE SEARCH LOOP:

    Standard DAGs (directed acyclic graphs) cannot have cycles — execution flows
    in one direction from start to end. But research agents NEED cycles:
    the agent may need to search multiple times before it has enough information.

    LangGraph is built specifically for cyclic graphs. The conditional edge
    between `evaluate` and `search_scrape` creates a feedback loop:

        search_scrape → evaluate → (score < 0.7 AND iteration < max) → search_scrape
                                 → (score >= 0.7 OR iteration >= max) → synthesize

TERMINATION GUARANTEE:

    Infinite loops are catastrophic in production (unbounded API costs, hanging
    processes). We guarantee termination with TWO independent mechanisms:
    1. `iteration < max_iterations`: Hard limit on loop count. Even if the
       evaluator always scores < 0.7, we stop at max_iterations=3.
    2. `evaluation_score >= sufficiency_threshold`: Quality exit. We stop early
       as soon as the content is good enough, saving API calls.

    The conditional routing function below implements both checks.
"""

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


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING FUNCTION
# This function is called by LangGraph after the `evaluate` node finishes.
# It reads the state and returns a STRING identifying the next node to visit.
# The string must match one of the keys in the add_conditional_edges() call below.
# ─────────────────────────────────────────────────────────────────────────────

def route_after_evaluation(
    state: ResearchState,
) -> Literal["replan", "synthesize"]:
    """
    Decide whether to loop back for another round of research or proceed to synthesis.

    ROUTING LOGIC:
        Two independent conditions decide the route:

        Condition A — QUALITY EXIT (good path):
            evaluation_score >= sufficiency_threshold (0.7)
            → We have enough information. Proceed to synthesis.

        Condition B — QUANTITY EXIT (circuit-breaker):
            iteration >= max_iterations (3)
            → We've searched 3 times already. Force synthesis regardless of score.
            → This prevents infinite loops and unbounded API spending.

        If NEITHER condition is met:
            → Loop back to `planner`, not directly to `search_scrape`. The
              planner node re-runs in "replan" mode when it sees iteration > 0
              (see agent_nodes.py::planner_node), using the evaluator's
              missing_topics to generate NEW queries -- then its own existing
              unconditional planner -> search_scrape edge carries execution
              forward with that new plan. Looping straight back to
              search_scrape (the original design) reused the exact same
              `plan` list every time, so a second search pass mostly just
              rediscovered the same URLs and burned an iteration for nothing.

    WHY RETURN A STRING INSTEAD OF A NODE REFERENCE?
        LangGraph's add_conditional_edges() maps string return values to node names.
        This decouples the routing function from the graph structure — you can
        rename nodes without changing the routing logic.

    Args:
        state: Current ResearchState after evaluator_node has run.

    Returns:
        "synthesize" if we should proceed to report writing.
        "replan" if we should loop back for another, differently-targeted round.
    """
    score = state.get("evaluation_score", 0.0)        # 0.0 if evaluator hasn't run (fallback)
    iteration = state.get("iteration", 0)              # 0 if search hasn't run (fallback)

    print(f"\n[Router] Evaluation score: {score:.2f} | Iteration: {iteration}/{cfg.max_iterations}")

    # Condition B: Circuit-breaker — hard stop regardless of quality
    if iteration >= cfg.max_iterations:
        print(f"[Router] ✓ Max iterations ({cfg.max_iterations}) reached → proceeding to synthesize")
        return "synthesize"  # force synthesis even if content isn't great

    # Condition A: Quality exit — content is good enough
    if score >= cfg.sufficiency_threshold:
        print(f"[Router] ✓ Sufficiency threshold met ({score:.2f} >= {cfg.sufficiency_threshold}) → synthesize")
        return "synthesize"  # early exit saves API calls

    # Neither condition met — loop back through the planner for a new, targeted plan
    print(f"[Router] ✗ Score {score:.2f} < {cfg.sufficiency_threshold} and "
          f"iteration {iteration} < {cfg.max_iterations} → looping back to replan")
    return "replan"  # LangGraph will call planner_node again, in replan mode


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER FUNCTION
# Returns a compiled LangGraph runnable ready to invoke.
# We wrap construction in a function (rather than building at module import time)
# for two reasons:
#   1. Avoids side effects at import time (graph creation is free but adds clarity).
#   2. Allows passing a checkpointer for persistent session memory (optional).
# ─────────────────────────────────────────────────────────────────────────────

def build_research_graph(use_checkpointer: bool = False):
    """
    Construct and compile the research agent state machine.

    GRAPH STRUCTURE:
        recall → planner → search_scrape → evaluate
                    ▲                          │
                    │       ┌──────────────────┤ route_after_evaluation()
                    │       │                  │
                    └── replan             synthesize → store → END
                (planner_node runs again,
                 in replan mode, using the
                 evaluator's missing_topics)

    WHAT `compile()` DOES:
        LangGraph's compile() validates the graph structure (checks for missing
        node definitions, unreachable nodes, etc.) and returns a `CompiledGraph`
        object with `.invoke()`, `.stream()`, and `.astream()` methods.

    WHY A CHECKPOINTER IS OPTIONAL:
        MemorySaver stores graph execution state in memory, enabling:
        - Resuming a graph from any past node (replay debugging).
        - Human-in-the-loop: pause at a node, let a human review/edit, then continue.
        For production, replace MemorySaver with a database-backed checkpointer
        (e.g., SqliteSaver) so state survives process restarts.

    Args:
        use_checkpointer: If True, attach MemorySaver for session replay support.

    Returns:
        A compiled LangGraph runnable. Call `.invoke({"question": "..."})` to run.
    """
    # ── STEP 1: INITIALIZE THE GRAPH ─────────────────────────────────────────
    # StateGraph requires the state type so it can validate node input/output keys.
    # ResearchState (TypedDict) tells LangGraph which keys are legal.
    graph = StateGraph(ResearchState)
    print("[GraphBuilder] Initializing StateGraph with ResearchState schema...")

    # ── STEP 2: REGISTER NODES ───────────────────────────────────────────────
    # add_node(name, function) registers a node with a string identifier.
    # The name is what you use in add_edge() and add_conditional_edges() calls.
    # The function must accept a ResearchState and return a dict.
    # NOTE: this node is named "planner", not "plan" -- ResearchState already
    # has a *state key* called "plan" (state.py), and LangGraph's StateGraph
    # builds an internal channel for every state key with that same name.
    # A node named identically to an existing state key collides with that
    # channel: `graph.add_node("plan", planner_node)` raised
    # `ValueError: 'plan' is already being used as a state key` at
    # build_research_graph() time, before any node ever ran -- meaning the
    # graph could not be constructed at all under its own pinned dependency
    # versions. The state key stays "plan" (agent_nodes.py, tests, and the
    # UI all read/write it); only the node identifier changes.
    graph.add_node("recall", recall_memory_node)       # Node 1: memory recall
    graph.add_node("planner", planner_node)            # Node 2: query planning / replanning
    graph.add_node("search_scrape", search_and_scrape_node)  # Node 3: search + scrape
    graph.add_node("evaluate", evaluator_node)         # Node 4: quality evaluation
    graph.add_node("synthesize", synthesizer_node)     # Node 5: report synthesis
    graph.add_node("store", memory_storing_node)       # Node 6: memory persistence
    print("[GraphBuilder] ✓ 6 nodes registered: recall → planner → search_scrape → evaluate → synthesize → store")

    # ── STEP 3: SET ENTRY POINT ──────────────────────────────────────────────
    # The entry point is the FIRST node LangGraph calls when you invoke the graph.
    # We start at `recall` so memory context is loaded before any LLM call.
    graph.set_entry_point("recall")
    print("[GraphBuilder] Entry point set to: 'recall'")

    # ── STEP 4: ADD UNCONDITIONAL EDGES ─────────────────────────────────────
    # add_edge(from, to) creates a deterministic transition — `from` ALWAYS goes to `to`.
    # These edges form the "happy path" through the pipeline.
    graph.add_edge("recall", "planner")        # memory recall always leads to planning
    graph.add_edge("planner", "search_scrape") # every plan (initial OR replan) leads to a search pass
    graph.add_edge("search_scrape", "evaluate") # searching always leads to evaluation
    # NOTE: evaluate → ??? is handled by the conditional edge below. When it
    # routes to "replan", that resolves to the "planner" node above, and this
    # same planner→search_scrape edge carries the new plan forward -- no
    # separate edge is needed for the loop-back path.
    graph.add_edge("synthesize", "store")      # synthesis always leads to memory storage
    graph.add_edge("store", END)               # storage always terminates the graph
    print("[GraphBuilder] ✓ Unconditional edges: recall→planner→search_scrape→evaluate | synthesize→store→END")

    # ── STEP 5: ADD CONDITIONAL EDGE ────────────────────────────────────────
    # add_conditional_edges(source, routing_function, mapping) creates a fork:
    # - `source`: the node whose output triggers routing.
    # - `routing_function`: called with the current state; returns a string key.
    # - `mapping`: dict mapping routing_function return values to node names.
    #
    # IMPORTANT: Every possible return value of routing_function MUST appear as
    # a key in the mapping dict. LangGraph raises at compile time if any are missing.
    graph.add_conditional_edges(
        "evaluate",              # source node: routing happens AFTER evaluation
        route_after_evaluation,  # routing function: returns "synthesize" or "replan"
        {
            "synthesize": "synthesize",  # if score >= threshold or max iter → go to synthesis
            "replan": "planner",         # else → back to planner_node, in replan mode
        },
    )
    print("[GraphBuilder] ✓ Conditional edge: evaluate → [synthesize | replan→planner]")
    print("[GraphBuilder]   Route condition: score >= 0.7 OR iteration >= max_iterations")

    # ── STEP 6: COMPILE ──────────────────────────────────────────────────────
    # compile() validates the graph and returns a runnable object.
    # Validation checks:
    # - All nodes referenced in edges actually exist.
    # - Every node can be reached from the entry point.
    # - No conditional edge return value is unmapped.

    if use_checkpointer:
        # Attach an in-memory checkpointer for session replay support.
        # With a checkpointer, every node execution is saved as a checkpoint.
        # You can then call graph.get_state(config) to inspect any past state.
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
    """
    Convenience function to run a research query end-to-end.

    This is the main entry point used by main.py and benchmark.py.
    It builds the graph, sets the initial state, and invokes it.

    INITIAL STATE DESIGN:
        The graph starts with only `question` populated. Every other field
        is populated progressively as nodes execute. This "lazy initialization"
        is why we use `total=False` in the ResearchState TypedDict — we can't
        populate all fields at once because they depend on prior nodes' outputs.

    Args:
        question: The user's research query string.
        stream: If True, use graph.stream() to yield state updates in real-time.
                Used by the Streamlit app for live progress tracking.

    Returns:
        If stream=False: The final ResearchState dict after all nodes complete.
        If stream=True: A generator yielding (node_name, partial_state) tuples.
    """
    print(f"\n{'='*70}")
    print(f"Starting Research Agent")
    print(f"Query: {question}")
    print(f"{'='*70}\n")

    graph = build_research_graph(use_checkpointer=False)

    # Initial state: only the question is set. LangGraph initializes all other
    # TypedDict keys to None/empty by default (because total=False).
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
        # stream() yields a dict for each node as it completes.
        # Each dict has the form: {"node_name": {"key": value, ...}}
        # The Streamlit app uses this for live progress display.
        return graph.stream(initial_state)  # return generator, not result
    else:
        # invoke() runs the entire graph synchronously and returns the final state.
        final_state = graph.invoke(initial_state)
        print(f"\n{'='*70}")
        print(f"Research Complete!")
        print(f"Report length: {len(final_state.get('final_report', '')):,} characters")
        print(f"Sources cited: {len(final_state.get('sources', []))}")
        print(f"Total iterations: {final_state.get('iteration', 0)}")
        print(f"Memory stored: {final_state.get('memory_stored', False)}")
        print(f"{'='*70}\n")
        return final_state
