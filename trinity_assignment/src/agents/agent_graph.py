import logging
from langgraph.graph import StateGraph, START, END
# Import custom modules
from .state import AgentState
from .planner_agent import planner_node
from .retriever_agent import retriever_node
from .synthesizer_agent import synthesizer_node

# --- LOGGING CONFIGURATION ---
# Configured to show the timestamp, agent name, and the specific message.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Building Graph workflow")

def build_workflow() -> StateGraph:
    """
    Initializes and compiles the LangGraph state machine.
    
    Returns:
        Compiled LangGraph application.
    """
    logger.info("Initializing AgentState Graph...")
    workflow = StateGraph(AgentState)

    # Add processing nodes to the graph
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Define the execution flow (Edges)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()