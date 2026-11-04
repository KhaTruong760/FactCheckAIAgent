from dataclasses import dataclass


@dataclass(frozen=True)
class RunContext:
    """Static per-run config, passed as `invoke(..., context=RunContext())`.

    Unlike graph state it is not checkpointed and no node can change it, so it's
    the place for knobs an ablation swaps (models, limits) without code changes.
    """
    planner_model: str = "claude-sonnet-5-5"
    researcher_model: str = "claude-sonnet-5-5"
    critic_model: str = "claude-haiku-4-5-20251001"
    writer_model: str = "claude-sonnet-5-5"
    max_sub_questions: int = 5
