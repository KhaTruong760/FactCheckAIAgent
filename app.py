"""CLI entry point: `python app.py "topic"` prints a cited Markdown fact-check report.

Uses the real planner, researcher and writer; critique, audit and finalize are
still the stubs in graph/builder.py, so every claim currently counts as verified.
Progress and cost go to stderr, the report to stdout (or --out).
"""
import argparse
import os
import re
import sys
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from agents.planner import planner_agent
from agents.researcher import research_agent
from agents.writer import CITATION, writer_agent
from graph.builder import build_graph
from graph.context import RunContext
from llm import PRICES
from schemas import Claim, Evidence, Report, Usage

REAL_NODES = {"plan": planner_agent, "research": research_agent, "write": writer_agent}
REQUIRED_KEYS = ("ANTHROPIC_API_KEY", "TAVILY_API_KEY")


def run(topic: str, ctx: RunContext, run_id: str | None = None, nodes: dict | None = None,
        log=lambda msg: None) -> tuple[dict, dict[str, Usage]]:
    """Run the pipeline; return its full final state (not just FactCheckOutput) and usage per node."""
    run_id = run_id or uuid.uuid4().hex[:8]
    # the checkpointer is only here so we can read claims/evidence back for the sources list
    graph = build_graph(REAL_NODES if nodes is None else nodes, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": 50}
    by_node: dict[str, Usage] = {}
    for update in graph.stream({"run_id": run_id, "topic": topic}, config, context=ctx,
                               stream_mode="updates"):
        for node, delta in update.items():
            log(f"  done: {node}")
            # each update is that node's own return value, so its usage is the node's delta
            if delta and delta.get("usage"):
                by_node[node] = by_node.get(node, Usage()) + delta["usage"]
    return graph.get_state(config).values, by_node


def format_usage(by_node: dict[str, Usage]) -> str:
    """One line per node plus a total, e.g. for stderr after a run."""
    rows = [*by_node.items(), ("total", sum(by_node.values(), Usage()))]
    width = max(len(name) for name, _ in rows)
    return "\n".join(
        f"  {name:<{width}}  {u.input_tokens:>9,} in ({u.cache_read_tokens:>9,} cached)"
        f"  {u.output_tokens:>7,} out  ${u.cost_usd:.4f}"
        for name, u in rows
    )


def render_markdown(report: Report, claims: dict[str, Claim], evidence: dict[str, Evidence],
                    unverified_claim_ids: list[str] = ()) -> str:
    """Swap [claim:ID] citations for numbered sources [1] and append the source list.

    Sources are numbered in order of first citation, and a claim backed by several
    pages cites all of them.
    """
    numbers: dict[str, int] = {}  # evidence id -> source number

    def cite(m: re.Match) -> str:
        claim = claims.get(m.group(2))
        if claim is None:
            return ""
        refs = []
        for eid in claim.evidence_ids:
            if eid in evidence:
                n = numbers.setdefault(eid, len(numbers) + 1)
                if n not in refs:
                    refs.append(n)
        return m.group(1) + "".join(f"[{n}]" for n in refs)

    body = CITATION.sub(cite, report.markdown)
    body = re.sub(r"\[(\d+)\](?:\s*\[\1\])+", r"[\1]", body)  # adjacent claims from the same page
    lines = [f"# {report.title}", "", body]

    if numbers:
        lines += ["", "## Sources", ""]
        for eid, n in numbers.items():
            ev = evidence[eid]
            date = f" ({ev.published_at.isoformat()})" if ev.published_at else ""
            lines.append(f"{n}. [{ev.title or ev.url}]({ev.url}){date}")

    unverified = [claims[cid].text for cid in unverified_claim_ids if cid in claims]
    if unverified:
        lines += ["", "## Could not verify", ""]
        lines += [f"- {text}" for text in unverified]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = RunContext()
    p = argparse.ArgumentParser(description="Research a topic and print a cited fact-check report.")
    p.add_argument("topic", help="what to fact-check, e.g. \"the Eiffel Tower\"")
    p.add_argument("--out", type=Path, help="write the Markdown report here instead of stdout")
    p.add_argument("--run-id", help="thread id for this run (default: random)")
    p.add_argument("--max-sub-questions", type=int, default=defaults.max_sub_questions)
    p.add_argument("--max-tool-calls", type=int, default=defaults.max_research_tool_calls,
                   help="searches + fetches per sub-question")
    for role in ("planner", "researcher", "writer"):
        p.add_argument(f"--{role}-model", default=getattr(defaults, f"{role}_model"), choices=sorted(PRICES))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        print(f"Missing {', '.join(missing)}; set them in .env", file=sys.stderr)
        return 2

    ctx = RunContext(
        planner_model=args.planner_model,
        researcher_model=args.researcher_model,
        writer_model=args.writer_model,
        max_sub_questions=args.max_sub_questions,
        max_research_tool_calls=args.max_tool_calls,
    )
    log = lambda msg: print(msg, file=sys.stderr, flush=True)
    log(f"Fact-checking {args.topic!r}...")
    state, by_node = run(args.topic, ctx, run_id=args.run_id, log=log)

    markdown = render_markdown(state["report"], state["claims"], state["evidence"],
                               state.get("unverified_claim_ids", []))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown)
        log(f"Report written to {args.out}")
    else:
        print(markdown)

    log("Usage by node:\n" + format_usage(by_node))
    return 0


if __name__ == "__main__":
    sys.exit(main())
