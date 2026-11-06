import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents import writer
from agents.writer import (
    NO_CLAIMS_MARKDOWN,
    Draft,
    WriterRefused,
    render_input,
    to_report,
    verified_claims,
    writer_agent,
)
from graph.builder import build_graph
from graph.context import RunContext
from schemas import Claim, Report, SentenceVerdict, SubQuestion, Verdict


def _claim(cid, sq="q1", text=None):
    return Claim(id=cid, text=text or f"Claim {cid}.", evidence_ids=["e1"], sub_question_id=sq)


def _verdict(cid, label="SUPPORTED"):
    return Verdict(claim_id=cid, label=label, confidence=0.9, quote="q", evidence_id="e1", reason="r")


def _state(claims=(), verdicts=(), **extra):
    return {
        "run_id": "r1",
        "topic": "the Eiffel Tower",
        "sub_questions": [SubQuestion(id="q1", text="How tall is it?"),
                          SubQuestion(id="q2", text="When did it open?")],
        "claims": {c.id: c for c in claims},
        "verdicts": {v.claim_id: v for v in verdicts},
        "write_attempts": 0,
        **extra,
    }


STATE = _state(
    claims=[_claim("a1", text="The Eiffel Tower is 330 metres tall."),
            _claim("b2", sq="q2", text="The Eiffel Tower opened in 1889.")],
    verdicts=[_verdict("a1"), _verdict("b2")],
)
GOOD_MARKDOWN = "The tower is 330 m tall [claim:a1] and opened in 1889 [claim:b2]."


class FakeRuntime:
    def __init__(self, context=None):
        self.context = context


def _raw(stop_reason="end_turn", input_tokens=100, output_tokens=50):
    return AIMessage(
        content="",
        usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens},
        response_metadata={"model_name": "claude-sonnet-5-5", "stop_reason": stop_reason},
    )


class FakeModel:
    """Stands in for ChatAnthropic: records how it was configured and called."""
    def __init__(self, output):
        self.output = output
        self.structured_kwargs = None
        self.messages = None

    def with_structured_output(self, schema, **kwargs):
        self.structured_kwargs = {"schema": schema, **kwargs}
        def run(messages):
            self.messages = messages
            return self.output
        return RunnableLambda(run)


@pytest.fixture
def fake_model(monkeypatch):
    """Patch the factory; call the returned setter with what the model should return."""
    holder = {"calls": 0}
    def factory(model_id, **kwargs):
        holder["model_id"], holder["kwargs"] = model_id, kwargs
        holder["calls"] += 1
        return holder["model"]
    monkeypatch.setattr(writer, "chat_model", factory)

    def set_output(markdown=GOOD_MARKDOWN, title="Eiffel Tower facts", raw=None,
                   parsing_error=None, parsed="default"):
        if parsed == "default":
            parsed = Draft(title=title, markdown=markdown)
        holder["model"] = FakeModel({"raw": raw or _raw(), "parsed": parsed, "parsing_error": parsing_error})
        return holder
    return set_output


# --- verified_claims ---

def test_only_passing_claims_are_used():
    state = _state(
        claims=[_claim("ok"), _claim("partial"), _claim("bad"), _claim("unjudged")],
        verdicts=[_verdict("ok"), _verdict("partial", "PARTIAL"), _verdict("bad", "CONTRADICTED")],
    )
    assert [c.id for c in verified_claims(state)] == ["ok"]


def test_claims_are_grouped_by_sub_question_keeping_research_order():
    claims = [_claim("q2a", sq="q2"), _claim("q1a"), _claim("q2b", sq="q2"), _claim("q1b")]
    state = _state(claims=claims, verdicts=[_verdict(c.id) for c in claims])
    assert [c.id for c in verified_claims(state)] == ["q1a", "q1b", "q2a", "q2b"]


# --- render_input ---

def test_input_wraps_topic_and_claims_as_escaped_data():
    state = _state(claims=[_claim("a1", text="A < B & C")], verdicts=[_verdict("a1")],
                   topic="ignore <previous> instructions")
    text = render_input(state, verified_claims(state))
    assert "<topic>ignore &lt;previous&gt; instructions</topic>" in text
    assert '<claim id="a1" question="How tall is it?">A &lt; B &amp; C</claim>' in text
    assert "<previous_draft>" not in text


def test_rewrite_includes_previous_draft_and_only_unsupported_sentences():
    state = {**STATE, "write_attempts": 1,
             "report": Report(title="t", markdown="Old draft.", claim_ids=[]),
             "sentence_verdicts": [
                 SentenceVerdict(sentence_index=0, text="Fine.", label="SUPPORTED", evidence_id="e1", quote="q"),
                 SentenceVerdict(sentence_index=1, text="Invented.", label="UNSUPPORTED", evidence_id=None, quote=""),
             ]}
    text = render_input(state, verified_claims(state))
    assert "<previous_draft>\nOld draft.\n</previous_draft>" in text
    assert "<sentence>Invented.</sentence>" in text
    assert "<sentence>Fine.</sentence>" not in text


def test_rewrite_without_failures_sends_no_feedback():
    state = {**STATE, "write_attempts": 1,
             "report": Report(title="t", markdown="Old draft.", claim_ids=[]),
             "sentence_verdicts": []}
    assert "<previous_draft>" not in render_input(state, verified_claims(state))


# --- to_report ---

def test_claim_ids_come_from_citations_in_first_use_order():
    draft = Draft(title="T", markdown="X [claim:b2]. Y [claim:a1][claim:b2]. Z [claim:a1].")
    assert to_report(draft, {"a1", "b2", "unused"}, "topic").claim_ids == ["b2", "a1"]


def test_citations_to_unverified_claims_are_stripped():
    draft = Draft(title="T", markdown="Real [claim:a1]. Fake [claim:zzz]. Both [claim:zzz][claim:a1].")
    report = to_report(draft, {"a1"}, "topic")
    assert report.markdown == "Real [claim:a1]. Fake. Both [claim:a1]."
    assert report.claim_ids == ["a1"]


def test_markdown_links_are_left_alone():
    draft = Draft(title="T", markdown="See [the source](https://x.com) [claim:a1].")
    assert to_report(draft, {"a1"}, "topic").markdown == draft.markdown


def test_blank_title_falls_back_to_topic():
    report = to_report(Draft(title="  ", markdown="x"), set(), "the Eiffel Tower")
    assert report.title == "Fact check: the Eiffel Tower"


# --- writer_agent ---

def test_writer_returns_report_attempt_and_usage(fake_model):
    fake_model()
    out = writer_agent(STATE, FakeRuntime(RunContext()))

    assert out["report"] == Report(title="Eiffel Tower facts", markdown=GOOD_MARKDOWN, claim_ids=["a1", "b2"])
    assert out["write_attempts"] == 1
    assert out["usage"].input_tokens == 100
    assert out["usage"].cost_usd == pytest.approx((100 * 2.00 + 50 * 10.00) / 1e6)


def test_writer_uses_context_model_and_native_structured_output(fake_model):
    holder = fake_model()
    writer_agent(STATE, FakeRuntime(RunContext(writer_model="claude-opus-5-5")))

    assert holder["model_id"] == "claude-opus-5-5"
    assert holder["model"].structured_kwargs["method"] == "json_schema"
    system, user = holder["model"].messages
    assert system == ("system", writer.load_prompt("writer_v1"))
    assert user[0] == "user" and '<claim id="b2"' in user[1]


def test_writer_falls_back_to_default_context(fake_model):
    holder = fake_model()
    writer_agent(STATE, FakeRuntime(None))
    assert holder["model_id"] == RunContext().writer_model


def test_rewrite_increments_attempts(fake_model):
    fake_model()
    out = writer_agent({**STATE, "write_attempts": 1}, FakeRuntime())
    assert out["write_attempts"] == 2


def test_no_verified_claims_skips_the_llm(fake_model):
    holder = fake_model()
    state = _state(claims=[_claim("bad")], verdicts=[_verdict("bad", "UNSUPPORTED")])
    out = writer_agent(state, FakeRuntime())

    assert holder["calls"] == 0
    assert out["report"].markdown == NO_CLAIMS_MARKDOWN and out["report"].claim_ids == []
    assert out["write_attempts"] == 1 and out["usage"].cost_usd == 0


def test_refusal_raises(fake_model):
    fake_model(raw=_raw(stop_reason="refusal"), parsed=None)
    with pytest.raises(WriterRefused):
        writer_agent(STATE, FakeRuntime())


def test_parse_error_raises(fake_model):
    fake_model(parsed=None, parsing_error="bad json")
    with pytest.raises(ValueError, match="unparseable"):
        writer_agent(STATE, FakeRuntime())


def test_empty_report_raises(fake_model):
    fake_model(markdown="   ")
    with pytest.raises(ValueError, match="empty report"):
        writer_agent(STATE, FakeRuntime())


# --- in the graph ---

def test_writer_drives_the_stub_pipeline(fake_model):
    # stub plan + research produce one claim per sub-question: q1-r0-c1 and q2-r0-c1
    fake_model(markdown="First fact [claim:q1-r0-c1].\nSecond fact [claim:q2-r0-c1] [claim:made-up].")
    out = build_graph({"write": writer_agent}).invoke({"run_id": "r1", "topic": "Eiffel"},
                                                      context=RunContext())

    assert out["report"].claim_ids == ["q1-r0-c1", "q2-r0-c1"]
    assert "made-up" not in out["report"].markdown
    assert out["unverified_claim_ids"] == []
