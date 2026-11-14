import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.evidence import Evidence, Stance
from app.schemas.verification import Verdict, VerificationResult


def make_evidence(**overrides) -> Evidence:
    data = {
        "source": "Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Earth",
        "content": "Earth orbits the Sun.",
        "stance": "SUPPORTS",
        "relevance": 0.9,
    }
    return Evidence(**(data | overrides))


def test_valid_evidence() -> None:
    ev = make_evidence()
    assert ev.stance is Stance.SUPPORTS
    assert ev.relevance == 0.9


def test_evidence_url_optional() -> None:
    assert make_evidence(url=None).url is None


@pytest.mark.parametrize("relevance", [-0.1, 1.1])
def test_invalid_relevance(relevance: float) -> None:
    with pytest.raises(ValidationError):
        make_evidence(relevance=relevance)


def test_invalid_stance() -> None:
    with pytest.raises(ValidationError):
        make_evidence(stance="MAYBE")


def test_valid_verification_result() -> None:
    result = VerificationResult(
        claim="Earth orbits the Sun",
        verdict="TRUE",
        explanation="Supported by sources.",
        confidence=0.95,
        evidence=[make_evidence()],
    )
    assert result.verdict is Verdict.TRUE
    assert isinstance(result.evidence[0], Evidence)


@pytest.mark.parametrize("confidence", [-0.01, 1.5])
def test_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValidationError):
        VerificationResult(claim="x", verdict="TRUE", explanation="", confidence=confidence)


def test_invalid_verdict() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(claim="x", verdict="MOSTLY_TRUE", explanation="", confidence=0.5)


def test_valid_chat_request_and_response() -> None:
    req = ChatRequest(message="  Is the Earth flat?  ")
    assert req.message == "Is the Earth flat?"
    assert req.conversation_id is None

    resp = ChatResponse(conversation_id="abc", answer="No.")
    assert resp.verification is None


@pytest.mark.parametrize("message", ["", "   "])
def test_empty_message_rejected(message: str) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message=message)
