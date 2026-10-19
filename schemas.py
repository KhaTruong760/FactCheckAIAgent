from datetime import date, datetime
from typing import Literal
from pydantic import BaseModel

class SubQuestion(BaseModel):
    id: str
    text: str

class Evidence(BaseModel):
    id: str
    url: str
    title: str
    published_at: date | None
    retrieved_at: datetime
    text: str
    domain_score: float

class Claim(BaseModel):
    id: str
    text: str
    evidence_ids: list[str]
    sub_question_id: str


class Verdict(BaseModel):
    claim_id: str
    label: Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"]
    confidence: float
    quote: str
    evidence_id: str | None
    reason: str

class RevisionRequest(BaseModel):
    round: int
    sub_question_id: str
    claim_ids: list[str]
    missing: str

class ResearchResult(BaseModel):
    evidence: list[Evidence]
    claims: list[Claim]

class Report(BaseModel):
    title: str
    markdown: str
    claim_ids: list[str]
