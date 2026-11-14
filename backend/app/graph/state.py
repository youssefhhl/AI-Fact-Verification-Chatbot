"""Typed LangGraph state for one conversation (one LangGraph thread).

Persistent across turns (checkpointed):
- `conversation`: the agent's memory (topics, evidence, verdicts, past turns).

Reset by the first node of every turn:
- `resolution`, `work`, `output`.

`results` is written by research branches running in parallel, so it needs a
reducer. Results carry the turn id; the reducer drops results of older turns.
"""

from typing import Annotated, TypedDict

from app.agents.fact_checking_agent import ComponentResult, ContextResolution, TurnWork
from app.schemas.conversation import ConversationState, FactCheckOutput


def collect_results(left: list[ComponentResult] | None, right: list[ComponentResult] | None) -> list[ComponentResult]:
    left, right = left or [], right or []
    if not right:
        return left
    turn_id = right[0].turn_id
    return [r for r in left if r.turn_id == turn_id] + list(right)


class GraphState(TypedDict, total=False):
    message: str  # input of the current turn
    conversation: ConversationState
    resolution: ContextResolution | None
    work: TurnWork | None
    results: Annotated[list[ComponentResult], collect_results]
    output: FactCheckOutput | None


class ComponentTask(TypedDict):
    """Payload sent (with langgraph Send) to a research node."""

    call: dict
    claim: str
    context: str
    turn_id: str
