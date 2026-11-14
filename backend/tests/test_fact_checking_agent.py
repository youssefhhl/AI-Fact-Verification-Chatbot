import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.fact_checking_agent import (
    FactCheckingAgent,
    calibrate_confidence,
    independence_key,
)
from app.config import Settings
from app.schemas.conversation import ContextDecision, ConversationState
from app.schemas.evidence import Evidence, Stance
from app.schemas.verification import Verdict
from app.services.llm_service import LLMMessage, LLMTimeoutError, ToolCall

pytestmark = pytest.mark.anyio

CLAIM = "Is vitamin D effective in preventing bone fractures?"
KB_1 = Evidence(source="SciFact: Vitamin D trial", url="https://api.semanticscholar.org/CorpusID:1",
                content="Vitamin D alone did not reduce fracture risk.", relevance=0.7)
KB_2 = Evidence(source="SciFact: Calcium and vitamin D", url="https://api.semanticscholar.org/CorpusID:2",
                content="Vitamin D with calcium reduced hip fractures in nursing-home residents.", relevance=0.6)
PAPER = Evidence(source="Europe PMC: Meta-analysis", url="https://doi.org/10.1/meta",
                 content="Meta-analysis of RCTs in older adults: no fracture reduction with vitamin D alone.")
WEB = Evidence(source="go.dev: Go", url="https://go.dev/blog/kubernetes",
               content="Kubernetes is written in Go.")
WIKI = [Evidence(source="FEVER (Wikipedia): Kubernetes", url="https://en.wikipedia.org/wiki/Kubernetes",
                 content=f"Kubernetes sentence {i}.") for i in range(3)]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def call(name: str, i: int = 0, **arguments) -> ToolCall:
    return ToolCall(id=f"{name}_{i}", name=name, arguments=arguments)


def plan(*calls: ToolCall) -> LLMMessage:
    return LLMMessage(tool_calls=list(calls))


FINISH = plan(call("finish_research", reason="enough"))


def synthesis(verdict="NOT_ENOUGH_EVIDENCE", confidence=0.5, assessments=(), answer="Answer.", gaps=()) -> str:
    return json.dumps({
        "assessments": [{"index": i, "relevant": rel, "stance": st} for i, st, rel in assessments],
        "verdict": verdict, "confidence": confidence, "explanation": f"Explanation for {verdict}.",
        "answer": answer, "evidence_gaps": list(gaps),
    })


def context(decision, claim, topic=0, requires_verdict=True, focus="") -> str:
    return "```json\n" + json.dumps({"decision": decision, "topic": topic, "claim": claim,
                                     "requires_verdict": requires_verdict, "research_focus": focus}) + "\n```"


class FakeLLM:
    """Scripted Groq: context resolution (generate), planning (chat with tools), synthesis (chat)."""

    def __init__(self, contexts=(), plans=(), syntheses=()) -> None:
        self.contexts, self.plans, self.syntheses = list(contexts), list(plans), list(syntheses)
        self.calls: list[str] = []
        self.plan_tools: list[set[str]] = []
        self.prompts: list[str] = []

    async def generate(self, prompt, system_prompt=None, temperature=0.0) -> str:
        self.calls.append("context")
        self.prompts.append(prompt)
        return _next(self.contexts)

    async def chat(self, messages, tools=None, temperature=0.0) -> LLMMessage:
        if tools:
            self.calls.append("plan")
            self.plan_tools.append({t["function"]["name"] for t in tools})
            self.prompts.append(messages[1]["content"])
            return _next(self.plans) if self.plans else FINISH
        self.calls.append("synthesis")
        self.prompts.append(messages[-1]["content"])
        return LLMMessage(content=_next(self.syntheses))


def _next(queue):
    item = queue.pop(0)
    if isinstance(item, Exception):
        raise item
    return item


def make_agent(llm: FakeLLM, kb=None, research=None, web=None, **settings):
    order: list[str] = []

    def component(name, result, method):
        mock = MagicMock()

        async def run(*args, **kwargs):
            order.append(name)
            value = result(*args) if callable(result) else result
            if isinstance(value, Exception):
                raise value
            return list(value)

        setattr(mock, method, AsyncMock(side_effect=run))
        return mock

    retrieval = component("kb", kb if kb is not None else [], "search_evidence")
    research_agent = component("research", research if research is not None else [], "research")
    web_agent = component("web", web if web is not None else [], "research")
    agent = FactCheckingAgent(llm, retrieval=retrieval, research_agent=research_agent, web_agent=web_agent,
                              settings=Settings(**settings))
    return agent, order, retrieval, research_agent, web_agent


# --- New claims and research routing -------------------------------------------------

async def test_new_claim_calls_rag_first_then_selected_research_agent() -> None:
    llm = FakeLLM(
        plans=[plan(call("run_research_agent", instructions="Find RCTs and meta-analyses on fractures")), FINISH],
        syntheses=[synthesis("FALSE", 0.9, [(0, "CONTRADICTS", True), (1, "SUPPORTS", True), (2, "CONTRADICTS", True)])],
    )
    agent, order, kb, research, web = make_agent(llm, kb=[KB_1, KB_2], research=[PAPER])

    out = await agent.handle_message(CLAIM)

    assert out.decision == ContextDecision.NEW_VERIFICATION
    assert order == ["kb", "research"]  # RAG first, then the agent the LLM chose
    assert llm.calls == ["plan", "plan", "synthesis"]  # no context-resolution call on turn 1
    web.research.assert_not_awaited()
    request = research.research.await_args.args[0]
    assert request.claim == CLAIM and "meta-analyses" in request.instructions
    assert "Vitamin D alone did not reduce" in llm.prompts[0]  # RAG results reached the planner

    v = out.verification
    assert v.verdict == Verdict.FALSE
    assert [e.stance for e in v.evidence] == [Stance.SUPPORTS, Stance.CONTRADICTS, Stance.CONTRADICTS]
    # Two independent contradicting papers (cap 0.8) minus 0.15 for the supporting one.
    assert v.confidence == 0.65
    assert [s.tool for s in out.research_steps] == ["knowledge_base", "research_agent"]


async def test_llm_can_choose_web_agent_only() -> None:
    llm = FakeLLM(plans=[plan(call("run_web_agent", instructions="Check the Kubernetes repository language"))],
                  syntheses=[synthesis("TRUE", 0.9, [(0, "SUPPORTS", True)])])
    agent, order, _, research, _ = make_agent(llm, web=[WEB])

    out = await agent.handle_message("Kubernetes is written in Go")

    assert order == ["kb", "web"]
    research.research.assert_not_awaited()
    assert out.verification.verdict == Verdict.TRUE and out.verification.confidence == 0.6  # 1 source


async def test_rag_sufficient_means_no_external_research() -> None:
    llm = FakeLLM(plans=[FINISH], syntheses=[synthesis("TRUE", 0.7, [(0, "SUPPORTS", True), (1, "SUPPORTS", True)])])
    agent, order, *_ = make_agent(llm, kb=[KB_1, KB_2])
    out = await agent.handle_message("claim")
    assert order == ["kb"]
    assert out.verification.confidence == 0.7  # 2 independent papers -> cap 0.8, LLM said 0.7


async def test_both_agents_called_when_llm_asks_for_both() -> None:
    llm = FakeLLM(plans=[plan(call("run_research_agent", instructions="scientific studies"),
                              call("run_web_agent", instructions="official health guidance"))],
                  syntheses=[synthesis("PARTIALLY_TRUE", 0.7, [(0, "SUPPORTS", True), (1, "CONTRADICTS", True)])])
    agent, order, *_ = make_agent(llm, research=[PAPER], web=[WEB])

    out = await agent.handle_message(CLAIM)

    assert sorted(order[1:]) == ["research", "web"] and order[0] == "kb"
    assert out.verification.verdict == Verdict.PARTIALLY_TRUE


async def test_external_research_budget_and_iteration_limit() -> None:
    llm = FakeLLM(plans=[
        plan(call("run_research_agent", 0, instructions="first request"),
             call("run_web_agent", 1, instructions="second request"),
             call("run_research_agent", 2, instructions="third request")),  # over budget
        plan(call("search_knowledge_base", query="vitamin D elderly")),
    ], syntheses=[synthesis()])
    agent, order, *_ = make_agent(llm, research=[PAPER], web=[WEB],
                                  fact_check_max_external_runs=2, fact_check_max_research_iterations=2)

    await agent.handle_message(CLAIM)

    assert order.count("research") + order.count("web") == 2
    assert order.count("kb") == 2
    # After the first turn, only knowledge-base search and finish remain available.
    assert llm.plan_tools[1] == {"search_knowledge_base", "finish_research"}
    assert llm.calls.count("plan") == 2  # iteration limit


async def test_duplicate_and_invalid_requests_are_refused() -> None:
    llm = FakeLLM(plans=[plan(call("search_knowledge_base", query=CLAIM),  # same as the automatic search
                              call("run_web_agent", 1, instructions=""),
                              call("delete_database"))],
                  syntheses=[synthesis()])
    agent, order, *_ = make_agent(llm, kb=[KB_1])
    await agent.handle_message(CLAIM)
    assert order == ["kb"]


# --- Context resolution and multi-turn state -----------------------------------------

async def first_turn(llm_contexts=(), plans=(), syntheses=(), **kwargs):
    llm = FakeLLM(contexts=list(llm_contexts), plans=[FINISH, *plans],
                  syntheses=[synthesis("FALSE", 0.8, [(0, "CONTRADICTS", True)]), *syntheses])
    agent, order, kb, research, web = make_agent(llm, **kwargs)
    out = await agent.handle_message(CLAIM)
    order.clear()
    return llm, agent, order, out, research


async def test_follow_up_reuses_context_without_research() -> None:
    llm, agent, order, first, _ = await first_turn(
        llm_contexts=[context("REUSE_CONTEXT", "Biological mechanism of vitamin D in bone health",
                              requires_verdict=False)],
        syntheses=[synthesis(answer="Vitamin D increases calcium absorption.", assessments=[(0, "NEUTRAL", True)])],
        kb=[KB_1],
    )

    out = await agent.handle_message("Can you explain its biological mechanism?", first.state)

    assert out.decision == ContextDecision.REUSE_CONTEXT
    assert order == []  # no research component called
    assert out.verification is None  # explanation, not a claim
    assert out.answer == "Vitamin D increases calcium absorption."
    assert "Vitamin D alone did not reduce" in llm.prompts[-1]  # previous evidence reused
    assert "Earlier topics" in llm.prompts[-2] and CLAIM in llm.prompts[-2]


async def test_follow_up_with_gap_triggers_targeted_research() -> None:
    elderly = "Vitamin D prevents bone fractures in elderly people"
    llm, agent, order, first, research = await first_turn(
        llm_contexts=[context("TARGETED_RESEARCH", elderly, focus="studies in people over 65")],
        plans=[plan(call("run_research_agent", instructions="RCTs in adults over 65"))],
        syntheses=[synthesis("FALSE", 0.9, [(0, "CONTRADICTS", True), (1, "CONTRADICTS", True)])],
        kb=[KB_1], research=[PAPER],
    )

    out = await agent.handle_message("What about elderly people?", first.state)

    assert out.decision == ContextDecision.TARGETED_RESEARCH and out.claim == elderly
    assert order == ["kb", "research"]
    assert research.research.await_args.args[0].claim == elderly
    assert "studies in people over 65" in llm.prompts[-2]  # focus given to the planner
    # Previous evidence (KB_1) and new evidence (PAPER) were both judged.
    assert {e.content for e in out.verification.evidence} == {KB_1.content, PAPER.content}
    assert len(out.state.topics) == 1 and len(out.state.topics[0].verifications) == 2


async def test_unrelated_message_starts_new_verification_and_topic() -> None:
    llm, agent, order, first, _ = await first_turn(
        llm_contexts=[context("NEW_VERIFICATION", "Kubernetes is written in Go", topic=None)],
        plans=[FINISH], syntheses=[synthesis("TRUE", 0.9, [(0, "SUPPORTS", True)])],
        kb=lambda q: [KB_1] if "vitamin" in q.lower() else [WEB],
    )

    out = await agent.handle_message("Is Kubernetes written in Go?", first.state)

    assert out.decision == ContextDecision.NEW_VERIFICATION
    assert order == ["kb"]
    assert len(out.state.topics) == 2 and out.state.active_topic == 1
    assert "Vitamin D" not in llm.prompts[-1]  # old topic's evidence not mixed in
    assert [e.content for e in out.verification.evidence] == [WEB.content]


async def test_reuse_without_stored_evidence_becomes_targeted_research() -> None:
    llm, agent, order, first, _ = await first_turn(
        llm_contexts=[context("REUSE_CONTEXT", "Vitamin D and fractures in elderly")],
        syntheses=[synthesis()], kb=[],
    )
    out = await agent.handle_message("And for the elderly?", first.state)
    assert out.decision == ContextDecision.TARGETED_RESEARCH and order == ["kb"]


async def test_unparseable_context_resolution_falls_back_to_new_claim() -> None:
    llm, agent, order, first, _ = await first_turn(llm_contexts=["no json here"], kb=[KB_1],
                                                    syntheses=[synthesis()])
    out = await agent.handle_message("Something else entirely", first.state)
    assert out.decision == ContextDecision.NEW_VERIFICATION and out.claim == "Something else entirely"


async def test_follow_up_to_unknown_topic_uses_active_topic() -> None:
    llm, agent, order, first, _ = await first_turn(
        llm_contexts=[context("REUSE_CONTEXT", "explain", topic=7, requires_verdict=False)],
        syntheses=[synthesis()], kb=[KB_1],
    )
    out = await agent.handle_message("Explain that", first.state)
    assert out.decision == ContextDecision.REUSE_CONTEXT and out.state.turns[-1].topic == 0


async def test_state_is_copied_and_recorded() -> None:
    _, agent, _, first, _ = await first_turn(kb=[KB_1])
    state = first.state
    assert state.turns[0].verdict == Verdict.FALSE and state.turns[0].decision == ContextDecision.NEW_VERIFICATION
    assert state.topics[0].evidence[0].stance is None  # stored without claim-specific stance
    assert first.verification.evidence[0].stance == Stance.CONTRADICTS
    snapshot = state.model_dump()
    agent.llm.contexts.append(context("REUSE_CONTEXT", "explain", requires_verdict=False))
    agent.llm.syntheses.append(synthesis())
    await agent.handle_message("Explain", state)
    assert state.model_dump() == snapshot  # input state not mutated
    assert ConversationState.model_validate_json(state.model_dump_json()) == state  # JSON round trip


# --- Evidence classification, contradictions and guards ------------------------------

async def test_irrelevant_evidence_is_dropped_and_unassessed_is_neutral() -> None:
    llm = FakeLLM(syntheses=[synthesis("TRUE", 0.9, [(0, "SUPPORTS", True), (1, "SUPPORTS", False)])])
    agent, *_ = make_agent(llm, kb=[KB_1, KB_2, PAPER])
    v = (await agent.handle_message("claim")).verification
    assert [e.content for e in v.evidence] == [KB_1.content, PAPER.content]
    assert v.evidence[1].stance == Stance.NEUTRAL


async def test_verdict_without_backing_evidence_is_downgraded() -> None:
    llm = FakeLLM(syntheses=[synthesis("TRUE", 0.95, [(0, "NEUTRAL", True)])])
    agent, *_ = make_agent(llm, kb=[KB_1])
    out = await agent.handle_message("claim")
    assert out.verification.verdict == Verdict.NOT_ENOUGH_EVIDENCE
    assert "leaned towards TRUE" in out.verification.explanation
    assert out.verification.confidence <= 0.6


async def test_same_source_counts_once_for_confidence() -> None:
    llm = FakeLLM(syntheses=[synthesis("TRUE", 0.99, [(i, "SUPPORTS", True) for i in range(3)])])
    agent, *_ = make_agent(llm, kb=WIKI)
    v = (await agent.handle_message("Kubernetes is written in Go")).verification
    assert v.verdict == Verdict.TRUE and v.confidence == 0.6  # 3 sentences, 1 Wikipedia page


def test_calibrate_confidence() -> None:
    def ev(url, stance):
        return Evidence(source="s", url=url, content="c", stance=stance)

    a, b, c = (ev(f"https://doi.org/10.1/{i}", Stance.SUPPORTS) for i in range(3))
    against = ev("https://who.int/x", Stance.CONTRADICTS)
    assert calibrate_confidence(Verdict.TRUE, 1.0, [a]) == 0.6
    assert calibrate_confidence(Verdict.TRUE, 1.0, [a, b]) == 0.8
    assert calibrate_confidence(Verdict.TRUE, 1.0, [a, b, c]) == 0.95
    assert calibrate_confidence(Verdict.TRUE, 1.0, [a, b, c, against]) == 0.8  # contradiction
    assert calibrate_confidence(Verdict.TRUE, 0.3, [a, b, c]) == 0.3  # never raises the LLM's value
    assert calibrate_confidence(Verdict.NOT_ENOUGH_EVIDENCE, 0.9, []) == 0.6
    assert calibrate_confidence(Verdict.NOT_ENOUGH_EVIDENCE, 0.9, [], had_failures=True) == 0.4


def test_independence_key() -> None:
    def key(url):
        return independence_key(Evidence(source="x: y", url=url, content="c"))

    assert key("https://en.wikipedia.org/wiki/A") == key("https://en.wikipedia.org/wiki/B")
    assert key("https://www.nih.gov/a") == key("https://nih.gov/b")
    assert key("https://doi.org/10.1/a") != key("https://doi.org/10.1/b")
    assert independence_key(Evidence(source="SciFact: t", content="c")) == "scifact"


# --- Failures and empty evidence -----------------------------------------------------

async def test_rag_failure_does_not_stop_external_research() -> None:
    llm = FakeLLM(plans=[plan(call("run_research_agent", instructions="studies please"))],
                  syntheses=[synthesis("FALSE", 0.9, [(0, "CONTRADICTS", True)])])
    agent, order, *_ = make_agent(llm, kb=ConnectionError("qdrant down"), research=[PAPER])

    out = await agent.handle_message(CLAIM)

    assert order == ["kb", "research"]
    assert out.verification.verdict == Verdict.FALSE
    assert out.research_steps[0].error.startswith("ConnectionError")
    assert any("knowledge base failed" in x for x in out.limitations)
    assert "knowledge base failed" in llm.prompts[0]  # planner was told


async def test_all_sources_failing_is_not_reported_as_no_evidence() -> None:
    llm = FakeLLM(plans=[plan(call("run_research_agent", instructions="studies please"),
                              call("run_web_agent", instructions="web sources please"))])
    agent, *_ = make_agent(llm, kb=RuntimeError("down"), research=RuntimeError("APIs down"),
                           web=RuntimeError("TAVILY_API_KEY is not set"))

    out = await agent.handle_message(CLAIM)

    v = out.verification
    assert v.verdict == Verdict.NOT_ENOUGH_EVIDENCE and v.confidence == 0.0
    assert "does not mean that no evidence exists" in v.explanation
    assert "synthesis" not in llm.calls  # nothing to analyse, no LLM call wasted
    assert len(out.limitations) == 3


async def test_no_evidence_found_without_failures() -> None:
    agent, *_ = make_agent(FakeLLM())
    v = (await agent.handle_message(CLAIM)).verification
    assert v.verdict == Verdict.NOT_ENOUGH_EVIDENCE and "found no relevant evidence" in v.explanation


async def test_failures_are_mentioned_in_insufficient_verdict() -> None:
    llm = FakeLLM(plans=[plan(call("run_web_agent", instructions="web sources please"))],
                  syntheses=[synthesis("NOT_ENOUGH_EVIDENCE", 0.9, [(0, "NEUTRAL", True)])])
    agent, *_ = make_agent(llm, kb=[KB_1], web=RuntimeError("timeout"))
    v = (await agent.handle_message(CLAIM)).verification
    assert "some research sources failed" in v.explanation and v.confidence == 0.4


async def test_planner_failure_still_produces_a_verdict() -> None:
    llm = FakeLLM(plans=[LLMTimeoutError("t")], syntheses=[synthesis("FALSE", 0.5, [(0, "CONTRADICTS", True)])])
    agent, *_ = make_agent(llm, kb=[KB_1])
    out = await agent.handle_message(CLAIM)
    assert out.verification.verdict == Verdict.FALSE
    assert any("planning was interrupted" in x for x in out.limitations)


async def test_synthesis_retries_once_on_bad_json() -> None:
    llm = FakeLLM(syntheses=["not json", synthesis("FALSE", 0.5, [(0, "CONTRADICTS", True)])])
    agent, *_ = make_agent(llm, kb=[KB_1])
    assert (await agent.handle_message(CLAIM)).verification.verdict == Verdict.FALSE
    assert llm.calls.count("synthesis") == 2


async def test_synthesis_failure_returns_safe_result() -> None:
    llm = FakeLLM(syntheses=["bad", "still bad"])
    agent, *_ = make_agent(llm, kb=[KB_1])
    v = (await agent.handle_message(CLAIM)).verification
    assert v.verdict == Verdict.NOT_ENOUGH_EVIDENCE and v.confidence == 0.0 and len(v.evidence) == 1


async def test_missing_components_are_reported() -> None:
    llm = FakeLLM(syntheses=[synthesis()])
    agent = FactCheckingAgent(llm, retrieval=None, research_agent=None, web_agent=None, settings=Settings())
    out = await agent.handle_message(CLAIM)
    assert len(out.limitations) == 3 and out.verification.verdict == Verdict.NOT_ENOUGH_EVIDENCE
    assert "plan" not in llm.calls  # no tool available, no planning call


async def test_empty_message_rejected() -> None:
    agent, *_ = make_agent(FakeLLM())
    with pytest.raises(ValueError):
        await agent.handle_message("   ")
