import operator
from typing import Annotated, TypedDict

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


def merge_by_id(left: dict, right: dict) -> dict:
    """Dict-merge reducer. A value of None deletes the key (tombstone)."""
    merged = {**left, **right}
    return {k: v for k, v in merged.items() if v is not None}


class FactCheckInput(TypedDict):
    run_id: str
    topic: str


class FactCheckState(FactCheckInput):
    # plan (single writer)
    sub_questions: list[SubQuestion]
    # research: parallel writers, so these need reducers
    evidence: Annotated[dict[str, Evidence], merge_by_id]
    claims: Annotated[dict[str, Claim | None], merge_by_id]
    # critique + ground_check
    verdicts: Annotated[dict[str, Verdict | None], merge_by_id]
    # revision loop: written by ground_check, only read by the router
    round: int
    revision_requests: list[RevisionRequest]
    # write / audit / finalize
    report: Report | None
    write_attempts: int
    sentence_verdicts: list[SentenceVerdict]
    unverified_claim_ids: list[str]
    # every LLM node returns its own delta
    usage: Annotated[Usage, operator.add]


class FactCheckOutput(TypedDict):
    report: Report
    unverified_claim_ids: list[str]
    usage: Usage


class ResearchTask(TypedDict):
    """Payload of each Send to `research`; a Send target only sees this, not the full state."""
    run_id: str
    round: int
    sub_question: SubQuestion
    revision: RevisionRequest | None
