import logging

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.state import collect_results
from app.graph.workflow import build_graph, make_serializer, output_from_state
from app.agents.fact_checking_agent import ComponentResult
from app.schemas.conversation import ContextDecision
from tests.test_fact_checking_agent import (
    CLAIM, FINISH, KB_1, PAPER, WEB, FakeLLM, call, context, make_agent, plan, synthesis,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def visited(graph, inputs, config=None) -> list[str]:
    nodes = []
    async for update in graph.astream(inputs, config, stream_mode="updates"):
        nodes.extend(update)
    return nodes


async def test_routing_new_claim_with_both_agents_in_parallel() -> None:
    llm = FakeLLM(plans=[plan(call("run_research_agent", instructions="studies please"),
                              call("run_web_agent", 1, instructions="official sources")), FINISH],
                  syntheses=[synthesis("FALSE", 0.8, [(0, "CONTRADICTS", True)])])
    agent, *_ = make_agent(llm, kb=[KB_1], research=[PAPER], web=[WEB])

    nodes = await visited(build_graph(agent), {"message": CLAIM})

    assert nodes[:3] == ["resolve_context", "knowledge_base", "plan_research"]
    assert sorted(nodes[3:5]) == ["research_agent", "web_agent"]  # same superstep
    assert nodes[5:] == ["plan_research", "synthesize"]


async def test_routing_additional_rag_then_finish() -> None:
    llm = FakeLLM(plans=[plan(call("search_knowledge_base", query="vitamin D hip fracture")), FINISH],
                  syntheses=[synthesis()])
    agent, *_ = make_agent(llm, kb=[KB_1])
    nodes = await visited(build_graph(agent), {"message": CLAIM})
    assert nodes == ["resolve_context", "knowledge_base", "plan_research", "knowledge_base",
                     "plan_research", "synthesize"]


async def test_routing_reuse_context_skips_research() -> None:
    llm = FakeLLM(contexts=[context("REUSE_CONTEXT", "explain", requires_verdict=False)],
                  plans=[FINISH], syntheses=[synthesis(), synthesis()])
    agent, *_ = make_agent(llm, kb=[KB_1])
    graph = build_graph(agent, InMemorySaver(serde=make_serializer()))
    config = {"configurable": {"thread_id": "a"}}
    await graph.ainvoke({"message": CLAIM}, config)

    nodes = await visited(graph, {"message": "Explain it"}, config)

    assert nodes == ["resolve_context", "synthesize"]


async def test_routing_without_knowledge_base_goes_to_planner() -> None:
    llm = FakeLLM(syntheses=[synthesis()])
    agent, *_ = make_agent(llm)
    agent.components.pop("knowledge_base")
    nodes = await visited(build_graph(agent), {"message": CLAIM})
    assert nodes == ["resolve_context", "plan_research", "synthesize"]


async def test_checkpoint_restores_context_and_threads_are_isolated(caplog) -> None:
    elderly = "Vitamin D prevents fractures in elderly people"
    llm = FakeLLM(contexts=[context("TARGETED_RESEARCH", elderly)],
                  syntheses=[synthesis("FALSE", 0.8, [(0, "CONTRADICTS", True)])] * 3)
    agent, order, *_ = make_agent(llm, kb=lambda q: [WEB] if "Kubernetes" in q else [KB_1])
    graph = build_graph(agent, InMemorySaver(serde=make_serializer()))
    thread_a = {"configurable": {"thread_id": "conversation-a"}}
    thread_b = {"configurable": {"thread_id": "conversation-b"}}

    with caplog.at_level(logging.WARNING):
        await graph.ainvoke({"message": CLAIM}, thread_a)
        second = output_from_state(await graph.ainvoke({"message": "What about elderly people?"}, thread_a))
        other = output_from_state(await graph.ainvoke({"message": "Kubernetes is written in Go"}, thread_b))

    # Same thread: the follow-up was resolved against the restored conversation.
    assert second.decision == ContextDecision.TARGETED_RESEARCH and second.claim == elderly
    assert len(second.state.turns) == 2 and len(second.state.topics) == 1
    assert llm.calls.count("context") == 1  # only the follow-up needed context resolution
    # Other thread: starts empty, no context resolution, no leak from thread A.
    assert other.decision == ContextDecision.NEW_VERIFICATION and len(other.state.turns) == 1
    assert "Vitamin D" not in other.state.model_dump_json()
    assert [t.claim for t in other.state.topics] == ["Kubernetes is written in Go"]
    # Checkpointed state is the agent's memory, restorable without "unregistered type" warnings.
    saved = (await graph.aget_state(thread_a)).values["conversation"]
    assert [t.message for t in saved.turns] == [CLAIM, "What about elderly people?"]
    assert "unregistered type" not in caplog.text


async def test_failed_turn_does_not_break_next_turn() -> None:
    llm = FakeLLM(syntheses=[synthesis()])
    agent, *_ = make_agent(llm, kb=[KB_1])
    graph = build_graph(agent, InMemorySaver(serde=make_serializer()))
    config = {"configurable": {"thread_id": "x"}}
    original = agent.synthesize

    async def boom(*args, **kwargs):
        raise RuntimeError("crash in the middle of a turn")

    agent.synthesize = boom
    with pytest.raises(RuntimeError):
        await graph.ainvoke({"message": CLAIM}, config)
    agent.synthesize = original
    llm.contexts.append(context("NEW_VERIFICATION", "Another claim", topic=None))
    out = output_from_state(await graph.ainvoke({"message": "Another claim"}, config))
    assert out.claim == "Another claim" and len(out.state.turns) == 1


def test_collect_results_keeps_only_current_turn() -> None:
    def r(turn, cid):
        return ComponentResult(turn_id=turn, call_id=cid, component="knowledge_base", query="q")

    left = [r("old", "1"), r("new", "2")]
    assert [x.call_id for x in collect_results(left, [r("new", "3")])] == ["2", "3"]
    assert collect_results(left, []) == left
