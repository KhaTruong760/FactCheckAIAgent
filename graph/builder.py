"""Graph wiring for the fact-check pipeline.

LLM nodes are stubs that return fixed data, so the topology (fan-out, revision
loop, write/audit retry) can be tested before any prompts exist. Replace them
one at a time with the real agents via build_graph(nodes={...}).
"""
from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from graph.context import RunContext
from graph.state import FactCheckInput, FactCheckOutput, FactCheckState, ResearchTask, initial_run_fields
from schemas import (
    Claim,
    Evidence,
    Report,
    RevisionRequest,
    SentenceVerdict,
    SubQuestion,
    Usage,
    Verdict,
)

MAX_ROUNDS = 2           # revision rounds after the first research pass
MAX_WRITE_ATTEMPTS = 2   # write -> audit cycles before we finalize anyway
PASSING = {"SUPPORTED"}  # labels that count as verified


# --- stub LLM nodes (to be replaced by agents/*) ---

def plan(state: FactCheckState) -> dict:
    topic = state["topic"]
    return {
        "sub_questions": [
            SubQuestion(id="q1", text=f"What are the key facts about {topic}?"),
            SubQuestion(id="q2", text=f"What do primary sources say about {topic}?"),
        ],
        **initial_run_fields(),
        "usage": Usage(input_tokens=1),
    }


def research(task: ResearchTask) -> dict:
    sq, rnd = task["sub_question"], task["round"]
    eid, cid = f"{sq.id}-r{rnd}-e1", f"{sq.id}-r{rnd}-c1"
    evidence = Evidence(
        id=eid,
        url=f"https://example.com/{eid}",
        title="stub source",
        published_at=None,
        retrieved_at=datetime.now(timezone.utc),
        text=f"Stub evidence for: {sq.text}",
        domain_score=0.5,
    )
    claims: dict[str, Claim | None] = {
        cid: Claim(id=cid, text=f"Stub claim for {sq.id}", evidence_ids=[eid], sub_question_id=sq.id)
    }
    if task["revision"]:
        # a revision replaces the claims it was asked to fix
        claims.update({old: None for old in task["revision"].claim_ids})
    return {"evidence": {eid: evidence}, "claims": claims, "usage": Usage(input_tokens=1)}


def critique(state: FactCheckState) -> dict:
    # only judge claims that don't have a verdict yet
    pending = [c for cid, c in state["claims"].items() if cid not in state["verdicts"]]
    verdicts = {
        c.id: Verdict(
            claim_id=c.id,
            label="SUPPORTED",
            confidence=1.0,
            quote="stub quote",
            evidence_id=c.evidence_ids[0] if c.evidence_ids else None,
            reason="stub",
        )
        for c in pending
    }
    return {"verdicts": verdicts, "usage": Usage(input_tokens=1)}


def write(state: FactCheckState) -> dict:
    ids = [cid for cid, v in state["verdicts"].items() if v.label in PASSING]
    report = Report(
        title=f"Fact check: {state['topic']}",
        markdown="\n".join(state["claims"][cid].text for cid in ids),
        claim_ids=ids,
    )
    return {"report": report, "write_attempts": state["write_attempts"] + 1, "usage": Usage(input_tokens=1)}


def audit(state: FactCheckState) -> dict:
    lines = state["report"].markdown.splitlines()
    sentence_verdicts = [
        SentenceVerdict(sentence_index=i, text=line, label="SUPPORTED", evidence_id=None, quote="stub")
        for i, line in enumerate(lines)
    ]
    return {"sentence_verdicts": sentence_verdicts, "usage": Usage(input_tokens=1)}


# --- deterministic nodes (no LLM) ---

def ground_check(state: FactCheckState) -> dict:
    """Turn failing verdicts into revision requests, one per sub-question."""
    failing = [v for v in state["verdicts"].values() if v.label not in PASSING]
    if not failing or state["round"] >= MAX_ROUNDS:
        return {"revision_requests": []}

    next_round = state["round"] + 1
    grouped: dict[str, list[Verdict]] = {}
    for v in failing:
        grouped.setdefault(state["claims"][v.claim_id].sub_question_id, []).append(v)

    requests = [
        RevisionRequest(
            round=next_round,
            sub_question_id=sq_id,
            claim_ids=[v.claim_id for v in vs],
            missing="; ".join(v.reason for v in vs),
        )
        for sq_id, vs in grouped.items()
    ]
    # drop the old verdicts so critique re-judges whatever research brings back
    return {
        "round": next_round,
        "revision_requests": requests,
        "verdicts": {v.claim_id: None for v in failing},
    }


def finalize(state: FactCheckState) -> dict:
    unverified = sorted(
        cid for cid in state["claims"]
        if cid not in state["verdicts"] or state["verdicts"][cid].label not in PASSING
    )
    return {"unverified_claim_ids": unverified}


# --- routers ---

def route_after_plan(state: FactCheckState) -> list[Send]:
    return [
        Send("research", ResearchTask(run_id=state["run_id"], round=0, sub_question=sq, revision=None))
        for sq in state["sub_questions"]
    ]


def route_after_ground_check(state: FactCheckState) -> str | list[Send]:
    if not state["revision_requests"]:
        return "write"
    by_id = {sq.id: sq for sq in state["sub_questions"]}
    return [
        Send("research", ResearchTask(
            run_id=state["run_id"],
            round=state["round"],
            sub_question=by_id[r.sub_question_id],
            revision=r,
        ))
        for r in state["revision_requests"]
    ]


def route_after_audit(state: FactCheckState) -> str:
    failed = any(sv.label not in PASSING for sv in state["sentence_verdicts"])
    if failed and state["write_attempts"] < MAX_WRITE_ATTEMPTS:
        return "write"
    return "finalize"


# --- assembly ---

DEFAULT_NODES = {
    "plan": plan,
    "research": research,
    "critique": critique,
    "ground_check": ground_check,
    "write": write,
    "audit": audit,
    "finalize": finalize,
}


def build_graph(nodes: dict | None = None):
    """Compile the pipeline. `nodes` overrides individual node functions by name."""
    unknown = set(nodes or {}) - set(DEFAULT_NODES)
    if unknown:
        raise ValueError(f"Unknown node override(s): {sorted(unknown)}")
    impl = {**DEFAULT_NODES, **(nodes or {})}

    g = StateGraph(
        FactCheckState,
        input_schema=FactCheckInput,
        output_schema=FactCheckOutput,
        context_schema=RunContext,
    )
    for name, fn in impl.items():
        g.add_node(name, fn)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", route_after_plan, ["research"])
    g.add_edge("research", "critique")
    g.add_edge("critique", "ground_check")
    g.add_conditional_edges("ground_check", route_after_ground_check, ["research", "write"])
    g.add_edge("write", "audit")
    g.add_conditional_edges("audit", route_after_audit, ["write", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()
