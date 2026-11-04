import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llm import chat_model, load_prompt, usage_from


def _ai(input_tokens, output_tokens, model="claude-sonnet-5-5", cache_read=0, cache_write=0):
    return AIMessage(
        content="",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cache_read, "cache_creation": cache_write},
        },
        response_metadata={"model_name": model},
    )


def test_usage_cost_for_one_message():
    u = usage_from(_ai(1_000_000, 100_000))
    assert (u.input_tokens, u.output_tokens) == (1_000_000, 100_000)
    assert u.cost_usd == pytest.approx(2.00 + 1.00)


def test_cached_tokens_are_billed_at_cache_rates():
    # 1M input of which 600k cache reads (0.1x) and 200k cache writes (1.25x)
    u = usage_from(_ai(1_000_000, 0, cache_read=600_000, cache_write=200_000))
    assert u.cost_usd == pytest.approx(200_000 * 2 / 1e6 + 600_000 * 0.2 / 1e6 + 200_000 * 2.5 / 1e6)


def test_usage_sums_ai_messages_in_a_transcript():
    transcript = [
        HumanMessage("hi"),
        _ai(100, 10),
        ToolMessage("result", tool_call_id="t1"),
        _ai(200, 20, model="claude-haiku-4-5-20251001"),
    ]
    u = usage_from(transcript)
    assert (u.input_tokens, u.output_tokens) == (300, 30)
    assert u.cost_usd == pytest.approx((100 * 2 + 10 * 10 + 200 * 1 + 20 * 5) / 1e6)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="No price"):
        chat_model("claude-made-up")


def test_effort_is_skipped_for_haiku():
    assert chat_model("claude-sonnet-5-5", effort="low").output_config == {"effort": "low"}
    assert chat_model("claude-haiku-4-5-20251001", effort="low").output_config is None


def test_planner_prompt_loads():
    assert "{max_sub_questions}" in load_prompt("planner_v1")
