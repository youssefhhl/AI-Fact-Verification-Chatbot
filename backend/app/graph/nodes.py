"""LangGraph nodes and routers. Thin wrappers around FactCheckingAgent methods:
the reasoning lives in app/agents/fact_checking_agent.py, not here."""

from typing import Any

from langgraph.types import Send

from app.agents.fact_checking_agent import (
    KNOWLEDGE_BASE,
    RESEARCH_AGENT,
    WEB_AGENT,
    FactCheckingAgent,
    PendingCall,
)
from app.graph.state import ComponentTask, GraphState
from app.schemas.conversation import ContextDecision, ConversationState, FactCheckOutput

RESOLVE, PLAN, SYNTHESIZE = "resolve_context", "plan_research", "synthesize"
COMPONENT_NODES = {KNOWLEDGE_BASE: "knowledge_base", RESEARCH_AGENT: "research_agent", WEB_AGENT: "web_agent"}


def _sends(state: GraphState) -> list[Send]:
    work, resolution = state["work"], state["resolution"]
    return [
        Send(COMPONENT_NODES[call.component], ComponentTask(
            call=call.model_dump(), claim=resolution.claim, context=work.context, turn_id=work.turn_id,
        ))
        for call in work.pending
    ]


class Nodes:
    def __init__(self, agent: FactCheckingAgent) -> None:
        self.agent = agent

    async def resolve_context(self, state: GraphState) -> dict[str, Any]:
        conversation = state.get("conversation") or ConversationState()
        message = state["message"].strip()
        resolution = await self.agent.resolve_context(message, conversation)
        work = self.agent.start_turn(resolution, conversation)
        return {"message": message, "conversation": conversation, "resolution": resolution,
                "work": work, "output": None}

    def component(self, component: str):
        async def run(task: ComponentTask) -> dict[str, Any]:
            call = PendingCall.model_validate(task["call"])
            result = await self.agent.run_component(call, task["claim"], task["context"], task["turn_id"])
            return {"results": [result]}

        run.__name__ = COMPONENT_NODES[component]
        return run

    async def plan_research(self, state: GraphState) -> dict[str, Any]:
        work = state["work"].model_copy(deep=True)
        mine = [r for r in state.get("results") or [] if r.turn_id == work.turn_id]
        self.agent.record_results(work, mine[work.consumed:])
        await self.agent.plan_research(state["resolution"], work)
        return {"work": work}

    async def synthesize(self, state: GraphState) -> dict[str, Any]:
        work = state["work"].model_copy(deep=True)
        resolution, message = state["resolution"], state["message"]
        verification, answer, gaps = await self.agent.synthesize(message, resolution, work)
        conversation = self.agent.finalize(message, resolution, state["conversation"], work,
                                           verification, answer, gaps)
        output = FactCheckOutput(
            decision=resolution.decision, claim=resolution.claim, answer=answer,
            verification=verification, research_steps=work.steps,
            limitations=work.limitations, state=conversation,
        )
        return {"work": work, "conversation": conversation, "output": output}


# --- Conditional edges --------------------------------------------------------------

def route_after_context(state: GraphState) -> str | list[Send]:
    """REUSE_CONTEXT -> answer from stored evidence; otherwise start with RAG."""
    if state["resolution"].decision == ContextDecision.REUSE_CONTEXT:
        return SYNTHESIZE
    return _sends(state) or PLAN  # no knowledge base available -> plan directly


def route_after_planning(state: GraphState) -> str | list[Send]:
    """Fan out to the components the planner chose (one, several, or both agents
    in parallel), plan again, or finish."""
    work = state["work"]
    if work.pending:
        return _sends(state)
    return SYNTHESIZE if work.finished else PLAN
