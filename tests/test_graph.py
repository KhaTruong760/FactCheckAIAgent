import pytest

from graph import builder
from graph.builder import MAX_ROUNDS, MAX_WRITE_ATTEMPTS, build_graph
from schemas import SentenceVerdict, SubQuestion, Usage

INPUT = {"run_id": "r1", "topic": "the Eiffel Tower"}


def _recording(fn, calls):
    """Wrap a node so the test can see every input it was called with."""
    def wrapped(arg):
        calls.append(arg)
        return fn(arg)
    return wrapped


def _critique_with(label_for):
    """Run the stub critique, then relabel each verdict with label_for(claim_id)."""
    def critique(state):
        out = builder.critique(state)
        out["verdicts"] = {
            cid: v.model_copy(update={"label": label_for(cid)}) for cid, v in out["verdicts"].items()
        }
        return out
    return critique


# --- happy path ---

def test_happy_path_returns_only_output_fields():
    out = build_graph().invoke(INPUT)
    assert set(out) == {"report", "unverified_claim_ids", "usage"}
    assert out["unverified_claim_ids"] == []
    assert sorted(out["report"].claim_ids) == ["q1-r0-c1", "q2-r0-c1"]


def test_usage_sums_across_all_llm_nodes():
    # plan + 2x research + critique + write + audit, each stub reports 1 input token
    out = build_graph().invoke(INPUT)
    assert isinstance(out["usage"], Usage)
    assert out["usage"].input_tokens == 6


# --- fan-out ---

def test_one_research_branch_per_sub_question():
    def plan(state):
        out = builder.plan(state)
        out["sub_questions"] = [SubQuestion(id=f"q{i}", text=f"question {i}") for i in range(3)]
        return out

    calls = []
    out = build_graph({"plan": plan, "research": _recording(builder.research, calls)}).invoke(INPUT)

    assert sorted(t["sub_question"].id for t in calls) == ["q0", "q1", "q2"]
    assert len(out["report"].claim_ids) == 3


def test_research_receives_only_its_task():
    calls = []
    build_graph({"research": _recording(builder.research, calls)}).invoke(INPUT)
    for task in calls:
        assert set(task) == {"run_id", "round", "sub_question", "revision"}
        assert task["revision"] is None


# --- revision loop ---

def test_failing_claims_are_researched_again_and_replaced():
    # round-0 claims fail, anything researched later passes
    critique = _critique_with(lambda cid: "UNSUPPORTED" if "-r0-" in cid else "SUPPORTED")
    calls = []
    out = build_graph({"critique": critique, "research": _recording(builder.research, calls)}).invoke(INPUT)

    revisions = [t for t in calls if t["revision"] is not None]
    assert len(calls) == 4
    assert {t["sub_question"].id for t in revisions} == {"q1", "q2"}
    assert all(t["round"] == 1 for t in revisions)
    assert sorted(out["report"].claim_ids) == ["q1-r1-c1", "q2-r1-c1"]
    assert out["unverified_claim_ids"] == []


def test_revision_loop_stops_at_max_rounds():
    critique = _critique_with(lambda cid: "CONTRADICTED")
    calls = []
    out = build_graph({"critique": critique, "research": _recording(builder.research, calls)}).invoke(INPUT)

    assert len(calls) == 2 * (1 + MAX_ROUNDS)
    assert out["report"].claim_ids == []
    assert out["unverified_claim_ids"] == [f"q1-r{MAX_ROUNDS}-c1", f"q2-r{MAX_ROUNDS}-c1"]


# --- write / audit retry ---

def test_write_retries_are_capped():
    def audit(state):
        return {"sentence_verdicts": [
            SentenceVerdict(sentence_index=0, text="x", label="UNSUPPORTED", evidence_id=None, quote="")
        ]}

    calls = []
    build_graph({"audit": audit, "write": _recording(builder.write, calls)}).invoke(INPUT)
    assert len(calls) == MAX_WRITE_ATTEMPTS


# --- assembly ---

def test_unknown_node_override_is_rejected():
    with pytest.raises(ValueError, match="reserch"):
        build_graph({"reserch": builder.research})
