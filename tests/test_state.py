import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from graph.state import merge_by_id
from schemas import Usage


# --- merge_by_id ---

def test_merge_adds_new_keys():
    assert merge_by_id({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_merge_right_side_wins():
    assert merge_by_id({"a": 1}, {"a": 2}) == {"a": 2}


def test_merge_none_deletes_key():
    assert merge_by_id({"a": 1, "b": 2}, {"a": None}) == {"b": 2}


def test_merge_tombstone_for_missing_key_is_noop():
    assert merge_by_id({"a": 1}, {"zzz": None}) == {"a": 1}


def test_merge_is_idempotent():
    update = {"a": 1}
    assert merge_by_id(merge_by_id({}, update), update) == {"a": 1}


# --- Usage ---

def test_usage_add():
    total = Usage(input_tokens=10, output_tokens=5, cost_usd=0.01) + Usage(input_tokens=1, output_tokens=2, cost_usd=0.02)
    assert (total.input_tokens, total.output_tokens) == (11, 7)
    assert abs(total.cost_usd - 0.03) < 1e-9


# --- reducers inside a real graph ---

class _State(TypedDict):
    items: list[str]
    claims: Annotated[dict[str, str | None], merge_by_id]
    usage: Annotated[Usage, operator.add]


def test_parallel_branches_merge_without_conflict():
    """Parallel Send branches write the same keys in one step; reducers must combine them."""
    def fan_out(state):
        return [Send("worker", {"item": i}) for i in state["items"]]

    def worker(task):
        return {"claims": {task["item"]: task["item"].upper()}, "usage": Usage(input_tokens=1)}

    g = StateGraph(_State)
    g.add_node("worker", worker)
    g.add_conditional_edges(START, fan_out, ["worker"])
    g.add_edge("worker", END)

    out = g.compile().invoke({"items": ["a", "b", "c"]})
    assert out["claims"] == {"a": "A", "b": "B", "c": "C"}
    assert out["usage"].input_tokens == 3
