from datetime import date, datetime
from langchain_protocol import Literal
from pydantic import BaseModel

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
    sub_questions_id: str


class Verdict(BaseModel):
    claim_id: str
    label: Literal["SUPPORTED", "PARTIAL", "UNSUPPORTTED", "CONTRADICTED"]
    confidence: float
    quote: str
    evidence_id: str | None
    reason: str

class RevisionRequest(BaseModel):
    round: int
    sub_question_id: str
    claim_ids: list[str]
    missing: str



