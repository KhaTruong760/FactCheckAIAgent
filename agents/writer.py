"""write node: one structured-output LLM call that turns verified claims into a report.

No tools and no loop, like the planner. The model cites claims inline as
[claim:ID]; which claims the report uses is read back from those citations in
code, and citations to claims that aren't verified are stripped, so the report
can't point at anything the critic didn't pass.
"""
import re
from xml.sax.saxutils import escape, quoteattr

from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from graph.context import RunContext
from graph.state import PASSING, FactCheckState
from llm import chat_model, load_prompt, usage_from
from schemas import Claim, Report, Usage

CITATION = re.compile(r"(\s*)\[claim:([^\]\s]+)\]")
NO_CLAIMS_MARKDOWN = "None of the researched claims could be verified against their sources."


class Draft(BaseModel):
    """What the model fills in. Claim IDs are read from the citations in code, not listed by the LLM."""
    title: str = Field(description="Short, specific report title")
    markdown: str = Field(description="Report body in Markdown, citing claims inline as [claim:ID]")


class WriterRefused(RuntimeError):
    pass


def writer_agent(state: FactCheckState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context or RunContext()
    attempts = state["write_attempts"] + 1
    claims = verified_claims(state)
    if not claims:
        # nothing to write about; an LLM call could only invent content
        report = Report(title=default_title(state["topic"]), markdown=NO_CLAIMS_MARKDOWN, claim_ids=[])
        return {"report": report, "write_attempts": attempts, "usage": Usage()}

    model = chat_model(ctx.writer_model, effort="medium").with_structured_output(
        Draft,
        method="json_schema",  # native structured output, as in the planner
        include_raw=True,      # keep the raw AIMessage so we can read usage and stop_reason
    )
    out = model.invoke([
        ("system", load_prompt("writer_v1")),
        ("user", render_input(state, claims)),
    ])

    raw = out["raw"]
    if raw.response_metadata.get("stop_reason") == "refusal":
        raise WriterRefused(f"Model declined to write a report on {state['topic']!r}")
    if out["parsing_error"] or out["parsed"] is None:
        raise ValueError(f"Writer returned unparseable output: {out['parsing_error']}")

    report = to_report(out["parsed"], {c.id for c in claims}, state["topic"])
    if not report.markdown:
        raise ValueError(f"Writer returned an empty report for {state['topic']!r}")
    return {"report": report, "write_attempts": attempts, "usage": usage_from(raw)}


def verified_claims(state: FactCheckState) -> list[Claim]:
    """Claims with a passing verdict, grouped in sub-question order."""
    order = {sq.id: i for i, sq in enumerate(state["sub_questions"])}
    verdicts = state["verdicts"]
    passed = [
        c for cid, c in state["claims"].items()
        if c is not None and cid in verdicts and verdicts[cid].label in PASSING
    ]
    # stable sort keeps the researcher's "most important first" order within a sub-question
    return sorted(passed, key=lambda c: order.get(c.sub_question_id, len(order)))


def render_input(state: FactCheckState, claims: list[Claim]) -> str:
    texts = {sq.id: sq.text for sq in state["sub_questions"]}
    lines = [f"<topic>{escape(state['topic'])}</topic>", "", "<claims>"]
    for c in claims:
        question = quoteattr(texts.get(c.sub_question_id, ""))
        lines.append(f"<claim id={quoteattr(c.id)} question={question}>{escape(c.text)}</claim>")
    lines.append("</claims>")

    unsupported = [sv.text for sv in state.get("sentence_verdicts") or [] if sv.label not in PASSING]
    if state["write_attempts"] and state.get("report") and unsupported:
        lines += ["", f"<previous_draft>\n{escape(state['report'].markdown)}\n</previous_draft>",
                  "", "<unsupported_sentences>"]
        lines += [f"<sentence>{escape(s)}</sentence>" for s in unsupported]
        lines.append("</unsupported_sentences>")
    return "\n".join(lines)


def to_report(draft: Draft, allowed_ids: set[str], topic: str) -> Report:
    """Strip citations to unverified claims and list the cited claims in first-use order."""
    cited: list[str] = []

    def keep_or_strip(m: re.Match) -> str:
        cid = m.group(2)
        if cid not in allowed_ids:
            # drop the space before it too, unless another citation follows and needs it
            return m.group(1) if m.string.startswith("[claim:", m.end()) else ""
        if cid not in cited:
            cited.append(cid)
        return m.group(0)

    markdown = CITATION.sub(keep_or_strip, draft.markdown).strip()
    title = " ".join(draft.title.split()) or default_title(topic)
    return Report(title=title, markdown=markdown, claim_ids=cited)


def default_title(topic: str) -> str:
    return f"Fact check: {topic}"
