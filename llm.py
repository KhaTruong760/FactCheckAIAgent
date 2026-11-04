"""The only module that knows about Anthropic: model factory, prompts, usage/cost.

Nodes ask for a model by ID and get back a LangChain chat model; if the provider
or pricing changes, this is the one file to edit.
"""
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage

from schemas import Usage

load_dotenv()

PROMPTS_DIR = Path(__file__).parent / "prompts"

# USD per million tokens: (input, output). Cache reads bill at 0.1x input,
# cache writes at 1.25x input.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}
# Haiku 4.5 predates the `effort` parameter and rejects it.
SUPPORTS_EFFORT = {"claude-opus-5-5", "claude-sonnet-5-5"}


def chat_model(model: str, effort: str | None = None, max_tokens: int = 16_000) -> ChatAnthropic:
    """Build a ChatAnthropic client. Fails fast on models we can't price."""
    if model not in PRICES:
        raise ValueError(f"No price for {model!r}; add it to llm.PRICES")
    kwargs = {}
    if effort and model in SUPPORTS_EFFORT:
        kwargs["output_config"] = {"effort": effort}
    # max_tokens covers thinking too, so leave generous headroom
    return ChatAnthropic(model=model, max_tokens=max_tokens, max_retries=3, **kwargs)


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text()


def usage_from(messages: BaseMessage | list[BaseMessage]) -> Usage:
    """Sum token usage and cost over one AIMessage or a whole agent transcript."""
    if isinstance(messages, BaseMessage):
        messages = [messages]
    total = Usage()
    for m in messages:
        if isinstance(m, AIMessage) and m.usage_metadata:
            total += _message_usage(m)
    return total


def _message_usage(m: AIMessage) -> Usage:
    meta = m.usage_metadata
    details = meta.get("input_token_details") or {}
    cache_read = details.get("cache_read") or 0
    cache_write = details.get("cache_creation") or 0
    # LangChain's input_tokens already includes cached tokens; split them back out
    fresh = meta["input_tokens"] - cache_read - cache_write

    model = m.response_metadata.get("model_name", "")
    in_price, out_price = PRICES.get(model, (0.0, 0.0))
    cost = (
        fresh * in_price
        + cache_read * in_price * 0.1
        + cache_write * in_price * 1.25
        + meta["output_tokens"] * out_price
    ) / 1_000_000
    return Usage(input_tokens=meta["input_tokens"], output_tokens=meta["output_tokens"], cost_usd=cost)
