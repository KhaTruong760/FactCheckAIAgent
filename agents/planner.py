"""plan node: one structured-output LLM call that splits the topic into sub-questions.

No tools and no loop: the planner doesn't need to look anything up, so a single
call is cheaper, faster and easier to test than an agent.
"""
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from graph.context import RunContext
from graph.state import FactCheckState, initial_run_fields
from llm import chat_model, load_prompt, usage_from
from schemas import SubQuestion


class Plan(BaseModel):
    """What the model fills in. Just text: IDs are assigned in code, not by the LLM."""
    sub_questions: list[str] = Field(description="Self-contained factual questions, most important first")


class PlannerRefused(RuntimeError):
    pass


def planner_agent(state: FactCheckState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context or RunContext()
    model = chat_model(ctx.planner_model, effort="low").with_structured_output(
        Plan,
        method="json_schema",  # native structured output; the default (forced tool use) 400s on Sonnet 5.5
        include_raw=True,      # keep the raw AIMessage so we can read usage and stop_reason
    )
    system = load_prompt("planner_v1").format(max_sub_questions=ctx.max_sub_questions)
    out = model.invoke([
        ("system", system),
        ("user", f"<topic>{state['topic']}</topic>"),
    ])

    raw = out["raw"]
    if raw.response_metadata.get("stop_reason") == "refusal":
        raise PlannerRefused(f"Model declined to plan topic {state['topic']!r}")
    if out["parsing_error"] or out["parsed"] is None:
        raise ValueError(f"Planner returned unparseable output: {out['parsing_error']}")

    sub_questions = to_sub_questions(out["parsed"].sub_questions, ctx.max_sub_questions)
    if not sub_questions:
        # zero Sends would end the graph silently with no report
        raise ValueError(f"Planner returned no sub-questions for {state['topic']!r}")

    return {
        **initial_run_fields(),
        "sub_questions": sub_questions,
        "usage": usage_from(raw),
    }


def to_sub_questions(texts: list[str], limit: int) -> list[SubQuestion]:
    """Clean the model's list (blanks, duplicates, over-long) and assign stable IDs."""
    seen, kept = set(), []
    for text in texts:
        text = " ".join(text.split())
        if text and text.lower() not in seen:
            seen.add(text.lower())
            kept.append(text)
    return [SubQuestion(id=f"q{i}", text=t) for i, t in enumerate(kept[:limit], 1)]
