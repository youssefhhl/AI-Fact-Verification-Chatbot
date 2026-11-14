"""Deterministic stand-ins for Groq and the research components (no network)."""

import json
import re

from app.agents.fact_checking_agent import FactCheckingAgent
from app.config import Settings
from app.schemas.evidence import Evidence
from app.services.llm_service import LLMMessage, ToolCall

VITAMIN = Evidence(source="SciFact: Vitamin D trial", url="https://api.semanticscholar.org/CorpusID:1",
                   content="Vitamin D with calcium reduced hip fractures.", relevance=0.7)
PAPER = Evidence(source="Europe PMC: Meta-analysis", url="https://doi.org/10.1/meta",
                 content="Meta-analysis: vitamin D lowers fracture risk in older adults.")
GO = Evidence(source="go.dev: Kubernetes", url="https://go.dev/k8s", content="Kubernetes is written in Go.")
ELDERLY = "Vitamin D prevents bone fractures in elderly people"


class RuleLLM:
    """Context resolution, planning and synthesis decided by simple rules."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate(self, prompt, system_prompt=None, temperature=0.0) -> str:
        self.calls.append("context")
        message = prompt.rsplit("New message:", 1)[-1].strip()
        active = re.search(r"Active topic: \[(\d+)\]", prompt)
        topic = int(active.group(1)) if active else 0
        if "explain" in message.lower():
            data = {"decision": "REUSE_CONTEXT", "topic": topic, "claim": "How vitamin D works",
                    "requires_verdict": False}
        elif "elderly" in message.lower():
            data = {"decision": "TARGETED_RESEARCH", "topic": topic, "claim": ELDERLY,
                    "research_focus": "people over 65"}
        else:
            data = {"decision": "NEW_VERIFICATION", "topic": None, "claim": message}
        return json.dumps(data)

    async def chat(self, messages, tools=None, temperature=0.0) -> LLMMessage:
        if tools:
            self.calls.append("plan")
            names = {t["function"]["name"] for t in tools}
            researched = any(m["role"] == "tool" for m in messages)
            if not researched and "vitamin" in messages[1]["content"].lower() and "run_research_agent" in names:
                return LLMMessage(tool_calls=[ToolCall(id="r1", name="run_research_agent",
                                                       arguments={"instructions": "Find trials and reviews"})])
            return LLMMessage(tool_calls=[ToolCall(id="f", name="finish_research", arguments={})])
        self.calls.append("synthesis")
        prompt = messages[-1]["content"]
        claim = re.search(r"Claim: (.*)", prompt).group(1)
        count = len(re.findall(r"^\[\d+\]", prompt, re.MULTILINE))
        return LLMMessage(content=json.dumps({
            "assessments": [{"index": i, "relevant": True, "stance": "SUPPORTS"} for i in range(count)],
            "verdict": "TRUE", "confidence": 0.9, "explanation": f"Sources support: {claim}",
            "answer": f"Checked: {claim}", "evidence_gaps": [],
        }))


class FakeRetrieval:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search_evidence(self, claim, top_k=None, source=None):
        self.queries.append(claim)
        return [VITAMIN] if "vitamin" in claim.lower() else [GO]


class FakeResearcher:
    def __init__(self, evidence: list[Evidence]) -> None:
        self.evidence = evidence
        self.requests: list = []

    async def research(self, request):
        self.requests.append(request)
        return list(self.evidence)


class FakeStack:
    """One fake LLM + components; `factory` builds a FactCheckingAgent on them."""

    def __init__(self) -> None:
        self.llm = RuleLLM()
        self.retrieval = FakeRetrieval()
        self.research = FakeResearcher([PAPER])
        self.web = FakeResearcher([GO])

    def factory(self) -> FactCheckingAgent:
        return FactCheckingAgent(self.llm, retrieval=self.retrieval, research_agent=self.research,
                                 web_agent=self.web, settings=Settings())
