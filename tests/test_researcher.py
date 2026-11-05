import json
from datetime import date

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agents import researcher
from agents.researcher import (
    Findings,
    FoundClaim,
    ResearchFailed,
    recording_fetch_page,
    render_task,
    research_agent,
    to_evidence_and_claims,
)
from graph.builder import build_graph
from graph.context import RunContext
from schemas import RevisionRequest, SubQuestion
from tools.fetch_page_tool import FetchError

SQ = SubQuestion(id="q1", text="How tall is the Eiffel Tower?")
URL = "https://example.com/eiffel"
PAGE = {"url": URL, "title": "Eiffel Tower", "published_at": date(2024, 1, 2),
        "text": "The Eiffel Tower is 330 metres tall."}


def _task(revision=None):
    return {"run_id": "r1", "round": 1 if revision else 0, "sub_question": SQ, "revision": revision}


class FakeRuntime:
    def __init__(self, context=None):
        self.context = context


class FakeToolModel(GenericFakeChatModel):
    """Plays back scripted AIMessages; bind_tools is a no-op so create_agent accepts it."""
    def bind_tools(self, tools, **kwargs):
        return self


def _ai(content="", tool_calls=(), input_tokens=100, output_tokens=20):
    return AIMessage(
        content=content,
        tool_calls=list(tool_calls),
        usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens},
        response_metadata={"model_name": "claude-sonnet-5-5"},
    )


def _fetch_call(url, call_id="t1"):
    return {"name": "fetch_page", "args": {"url": url}, "id": call_id, "type": "tool_call"}


def _answer(*claims):
    return _ai(json.dumps({"claims": [{"text": t, "source_urls": urls} for t, urls in claims]}))


@pytest.fixture
def script(monkeypatch):
    """Patch the model factory and the network; call with the AIMessages the model should emit."""
    monkeypatch.setattr(researcher, "fetch_page_content", lambda url: PAGE)
    holder = {}

    def set_script(*messages):
        def factory(model_id, **kwargs):
            holder["model_id"], holder["kwargs"] = model_id, kwargs
            return FakeToolModel(messages=iter(messages))
        monkeypatch.setattr(researcher, "chat_model", factory)
        return holder
    return set_script


# --- to_evidence_and_claims ---

def test_claims_link_to_evidence_built_from_fetched_pages():
    findings = Findings(claims=[FoundClaim(text="The Eiffel Tower is 330 m tall.", source_urls=[URL])])
    evidence, claims = to_evidence_and_claims(findings, {URL: PAGE}, "q1")

    [ev] = evidence.values()
    [claim] = claims.values()
    assert (ev.url, ev.title, ev.text, ev.published_at) == (URL, "Eiffel Tower", PAGE["text"], date(2024, 1, 2))
    assert claim.evidence_ids == [ev.id]
    assert claim.sub_question_id == "q1"


def test_unfetched_urls_are_dropped_and_unbacked_claims_skipped():
    findings = Findings(claims=[
        FoundClaim(text="Backed claim.", source_urls=["https://never-fetched.com", URL]),
        FoundClaim(text="Made-up claim.", source_urls=["https://never-fetched.com"]),
        FoundClaim(text="   ", source_urls=[URL]),
    ])
    evidence, claims = to_evidence_and_claims(findings, {URL: PAGE}, "q1")
    assert [c.text for c in claims.values()] == ["Backed claim."]
    assert len(evidence) == 1


def test_trailing_slash_differences_still_match():
    findings = Findings(claims=[FoundClaim(text="c", source_urls=[URL + "/"])])
    _, claims = to_evidence_and_claims(findings, {URL: PAGE}, "q1")
    assert len(claims) == 1


def test_ids_are_stable_content_hashes():
    findings = Findings(claims=[FoundClaim(text="Same  claim.", source_urls=[URL])])
    ev1, c1 = to_evidence_and_claims(findings, {URL: PAGE}, "q1")
    ev2, c2 = to_evidence_and_claims(findings, {URL: PAGE}, "q1")
    _, c3 = to_evidence_and_claims(findings, {URL: PAGE}, "q2")
    assert ev1.keys() == ev2.keys() and c1.keys() == c2.keys()
    assert c1.keys() != c3.keys()  # different sub-question, different claim ID
    assert [c.text for c in c1.values()] == ["Same claim."]


def test_evidence_text_is_capped(monkeypatch):
    monkeypatch.setattr(researcher, "MAX_CHARS", 10)
    findings = Findings(claims=[FoundClaim(text="c", source_urls=[URL])])
    evidence, _ = to_evidence_and_claims(findings, {URL: PAGE}, "q1")
    assert [e.text for e in evidence.values()] == [PAGE["text"][:10]]


# --- tools and prompt ---

def test_fetch_tool_records_pages_under_requested_and_final_url(monkeypatch):
    monkeypatch.setattr(researcher, "fetch_page_content", lambda url: PAGE)
    pages = {}
    out = recording_fetch_page(pages).invoke({"url": "http://short.link/x"})
    assert "<untrusted_document" in out
    assert pages.keys() == {"http://short.link/x", URL}


def test_fetch_tool_returns_errors_as_text(monkeypatch):
    def fail(url):
        raise FetchError("HTTP 404")
    monkeypatch.setattr(researcher, "fetch_page_content", fail)
    pages = {}
    assert recording_fetch_page(pages).invoke({"url": URL}) == "ERROR: HTTP 404"
    assert pages == {}


def test_render_task_wraps_question_and_adds_revision_feedback():
    assert render_task(_task()) == f"<sub_question>{SQ.text}</sub_question>"
    rr = RevisionRequest(round=1, sub_question_id="q1", claim_ids=["old"], missing="no source for height")
    assert "no source for height" in render_task(_task(rr))


def test_prompt_formats():
    text = researcher.load_prompt("researcher_v1").format(max_claims=3, max_tool_calls=5)
    assert "1 and 3 claims" in text and "budget of 5 tool calls" in text


# --- research_agent ---

def test_agent_fetches_then_returns_claims_with_usage(script):
    holder = script(
        _ai(tool_calls=[_fetch_call(URL)]),
        _answer(("The Eiffel Tower is 330 metres tall.", [URL])),
    )
    out = research_agent(_task(), FakeRuntime(RunContext(researcher_model="claude-opus-5-5")))

    assert holder["model_id"] == "claude-opus-5-5"
    [claim] = out["claims"].values()
    assert claim.text == "The Eiffel Tower is 330 metres tall."
    assert list(out["evidence"]) == claim.evidence_ids
    assert out["usage"].input_tokens == 200  # both model calls counted


def test_claims_citing_unfetched_pages_are_dropped(script):
    script(_answer(("Hallucinated.", ["https://example.com/never-read"])))
    out = research_agent(_task(), FakeRuntime())
    assert out["claims"] == {} and out["evidence"] == {}


def test_revision_tombstones_replaced_claims(script):
    script(_ai(tool_calls=[_fetch_call(URL)]), _answer(("New claim.", [URL])))
    rr = RevisionRequest(round=1, sub_question_id="q1", claim_ids=["old1", "old2"], missing="x")
    out = research_agent(_task(rr), FakeRuntime())
    assert out["claims"]["old1"] is None and out["claims"]["old2"] is None
    assert [c.text for c in out["claims"].values() if c] == ["New claim."]


def test_refound_claim_is_not_tombstoned(script):
    script(_ai(tool_calls=[_fetch_call(URL)]), _answer(("Same claim.", [URL])))
    first = research_agent(_task(), FakeRuntime())
    [cid] = first["claims"]

    script(_ai(tool_calls=[_fetch_call(URL)]), _answer(("Same claim.", [URL])))
    rr = RevisionRequest(round=1, sub_question_id="q1", claim_ids=[cid], missing="x")
    out = research_agent(_task(rr), FakeRuntime())
    assert out["claims"][cid] is not None


def test_no_structured_response_raises(script):
    # model keeps calling tools until the model-call backstop ends the loop
    ctx = RunContext(max_research_tool_calls=1)
    script(*[_ai(tool_calls=[_fetch_call(URL, f"t{i}")]) for i in range(10)])
    with pytest.raises(ResearchFailed):
        research_agent(_task(), FakeRuntime(ctx))


# --- in the graph ---

def test_researcher_drives_the_stub_pipeline(script):
    # every parallel branch gets a fresh copy of this script, so all answer the same text
    script(_ai(tool_calls=[_fetch_call(URL)]), _answer(("Claim A.", [URL])))
    out = build_graph({"research": research_agent}).invoke({"run_id": "r1", "topic": "Eiffel"},
                                                           context=RunContext())
    # stub plan has two sub-questions: same text, but distinct claim IDs, so both survive the merge
    assert out["report"].markdown.splitlines() == ["Claim A.", "Claim A."]
    assert len(set(out["report"].claim_ids)) == 2
    assert out["unverified_claim_ids"] == []
