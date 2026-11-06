from datetime import date, datetime, timezone

import pytest

import app
from app import format_usage, main, parse_args, render_markdown, run
from graph.context import RunContext
from schemas import Claim, Evidence, Report, Usage


def _ev(eid, title="Page", published_at=None):
    return Evidence(id=eid, url=f"https://example.com/{eid}", title=title, published_at=published_at,
                    retrieved_at=datetime.now(timezone.utc), text="t", domain_score=0.5)


def _claim(cid, *evidence_ids):
    return Claim(id=cid, text=f"Claim {cid}.", evidence_ids=list(evidence_ids), sub_question_id="q1")


EVIDENCE = {"e1": _ev("e1", "Official site", date(2024, 5, 1)), "e2": _ev("e2", "")}
CLAIMS = {"a": _claim("a", "e2"), "b": _claim("b", "e1", "e2"), "c": _claim("c", "e2")}


def _report(markdown):
    return Report(title="Eiffel Tower", markdown=markdown, claim_ids=[])


# --- render_markdown ---

def test_citations_become_numbered_sources_in_first_use_order():
    out = render_markdown(_report("X [claim:a]. Y [claim:b]."), CLAIMS, EVIDENCE)
    assert out.startswith("# Eiffel Tower\n\nX [1]. Y [2][1].\n")
    assert "1. [https://example.com/e2](https://example.com/e2)\n" in out  # blank title falls back to URL
    assert "2. [Official site](https://example.com/e1) (2024-05-01)\n" in out


def test_adjacent_claims_from_the_same_page_cite_it_once():
    out = render_markdown(_report("X [claim:a][claim:c]. Y [claim:a] [claim:c]."), CLAIMS, EVIDENCE)
    assert "X [1]. Y [1]." in out


def test_unknown_claims_are_dropped_and_no_sources_section_without_citations():
    out = render_markdown(_report("Nothing cited [claim:zzz]."), CLAIMS, EVIDENCE)
    assert "Nothing cited." in out
    assert "## Sources" not in out


def test_unverified_claims_get_their_own_section():
    out = render_markdown(_report("X [claim:a]."), CLAIMS, EVIDENCE, unverified_claim_ids=["c", "gone"])
    assert out.endswith("## Could not verify\n\n- Claim c.\n")


def test_rating_is_shown_under_the_title():
    report = Report(title="Eiffel Tower", markdown="X [claim:a].", claim_ids=[], rating="NOT_ENOUGH_EVIDENCE")
    out = render_markdown(report, CLAIMS, EVIDENCE)
    assert out.startswith("# Eiffel Tower\n\n**Verdict: Not enough evidence**\n\nX [1].")


def test_no_rating_line_in_topic_mode():
    assert "Verdict" not in render_markdown(_report("X [claim:a]."), CLAIMS, EVIDENCE)


# --- run / main on the stub graph (no network) ---

def test_run_returns_full_state_with_claims_and_evidence():
    state, _ = run("Eiffel", RunContext(), run_id="t1", nodes={})
    assert state["report"].claim_ids
    assert set(state["report"].claim_ids) <= set(state["claims"])
    assert state["evidence"]


def test_run_logs_each_node():
    seen = []
    run("Eiffel", RunContext(), nodes={}, log=seen.append)
    assert seen[0] == "  done: plan" and seen[-1] == "  done: finalize"
    assert seen.count("  done: research") == 2  # stub plan has two sub-questions


def test_run_sums_usage_per_node():
    # every stub LLM node reports Usage(input_tokens=1)
    _, by_node = run("Eiffel", RunContext(), nodes={})
    assert {n: u.input_tokens for n, u in by_node.items()} == {
        "plan": 1, "research": 2, "critique": 1, "write": 1, "audit": 1}


def test_format_usage_lists_nodes_then_total():
    text = format_usage({
        "plan": Usage(input_tokens=1_000, output_tokens=100, cost_usd=0.003),
        "research": Usage(input_tokens=90_000, output_tokens=4_000, cost_usd=0.2, cache_read_tokens=60_000),
    })
    plan, research, total = text.splitlines()
    assert plan.split()[0] == "plan" and "1,000 in" in plan
    assert "60,000 cached" in research
    assert total.split()[0] == "total" and "91,000 in" in total and "$0.2030" in total


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Run main() on the stub nodes with fake API keys."""
    monkeypatch.setattr(app, "REAL_NODES", {})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("TAVILY_API_KEY", "x")


def test_main_prints_report_and_cost(stub_pipeline, capsys):
    assert main(["Eiffel"]) == 0
    out, err = capsys.readouterr()
    assert out.startswith("# Fact check: Eiffel\n\nStub claim for q1")  # stub writer adds no citations
    assert "Usage by node:" in err and "total" in err


def test_main_writes_out_file(stub_pipeline, tmp_path, capsys):
    path = tmp_path / "runs" / "report.md"
    assert main(["Eiffel", "--out", str(path)]) == 0
    assert path.read_text().startswith("# Fact check: Eiffel")
    assert capsys.readouterr().out == ""


def test_main_fails_fast_without_api_keys(monkeypatch, capsys):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert main(["Eiffel"]) == 2
    assert "TAVILY_API_KEY" in capsys.readouterr().err


def test_claim_flag_runs_in_claim_mode(stub_pipeline, monkeypatch, capsys):
    seen = {}
    real_run = app.run
    def spy(*args, **kwargs):
        seen["mode"] = kwargs["mode"]
        return real_run(*args, **kwargs)
    monkeypatch.setattr(app, "run", spy)
    assert main(["The Eiffel Tower opened in 1899", "--claim"]) == 0
    assert seen["mode"] == "claim"
    assert parse_args(["x"]).claim is False


def test_args_map_onto_run_context():
    args = parse_args(["topic", "--writer-model", "claude-opus-5-5", "--max-sub-questions", "2"])
    assert (args.writer_model, args.max_sub_questions) == ("claude-opus-5-5", 2)
    assert args.planner_model == RunContext().planner_model
    with pytest.raises(SystemExit):
        parse_args(["topic", "--writer-model", "gpt-4"])
