"""LangGraph workflow of the central Fact-Checking Agent.

    START -> resolve_context --REUSE_CONTEXT--------------------------> synthesize -> END
                  |                                                       ^
                  +--> knowledge_base --> plan_research --finished--------+
                                            |     ^  ^
                       Send (parallel) ---- +     |  +-- (all calls refused: plan again)
                       knowledge_base / research_agent / web_agent
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.fact_checking_agent import FactCheckingAgent
from app.graph.nodes import (
    COMPONENT_NODES,
    PLAN,
    RESOLVE,
    SYNTHESIZE,
    Nodes,
    route_after_context,
    route_after_planning,
)
from app.graph.state import GraphState
from app.schemas.conversation import FactCheckOutput

# Types stored in checkpoints. Listing them lets the checkpoint serializer restore them
# without the "unregistered type" warning (and keeps working under strict mode).
CHECKPOINT_TYPES = [
    ("app.agents.fact_checking_agent", name)
    for name in ("ComponentResult", "ContextResolution", "PendingCall", "TurnWork")
] + [
    ("app.schemas.conversation", name)
    for name in ("ContextDecision", "ConversationState", "FactCheckOutput", "ResearchStep", "Topic", "TurnRecord")
] + [
    ("app.schemas.evidence", "Evidence"), ("app.schemas.evidence", "Stance"),
    ("app.schemas.verification", "Verdict"), ("app.schemas.verification", "VerificationResult"),
]


def make_serializer():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


def build_graph(agent: FactCheckingAgent, checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    nodes = Nodes(agent)
    graph = StateGraph(GraphState)
    graph.add_node(RESOLVE, nodes.resolve_context)
    graph.add_node(PLAN, nodes.plan_research)
    graph.add_node(SYNTHESIZE, nodes.synthesize)
    for component, node in COMPONENT_NODES.items():
        graph.add_node(node, nodes.component(component))
        graph.add_edge(node, PLAN)  # after research, the planner sees the results

    targets = [SYNTHESIZE, PLAN, *COMPONENT_NODES.values()]
    graph.add_edge(START, RESOLVE)
    graph.add_conditional_edges(RESOLVE, route_after_context, targets)
    graph.add_conditional_edges(PLAN, route_after_planning, targets)
    graph.add_edge(SYNTHESIZE, END)
    return graph.compile(checkpointer=checkpointer)


def output_from_state(values: dict) -> FactCheckOutput:
    output = values.get("output")
    if output is None:
        raise RuntimeError("The fact-checking graph finished without an output")
    return output
