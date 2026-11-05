"""research node: a create_agent tool loop that answers one sub-question.

The model only returns claim texts and the URLs that support them. Evidence is
built in code from the pages the fetch tool actually downloaded, so a claim can't
cite a page the agent never read, and evidence text is never written by the LLM.
"""
import hashlib
from datetime import datetime, timezone

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.tools import tool
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from graph.context import RunContext
from graph.state import ResearchTask
from llm import chat_model, load_prompt, usage_from
from schemas import Claim, Evidence
from tools.fetch_page_tool import MAX_CHARS, FetchError, _wrap_untrusted, fetch_page, fetch_page_content
from tools.tavily_tool import web_search

DEFAULT_DOMAIN_SCORE = 0.5  # neutral until tools/credibility.py exists


class FoundClaim(BaseModel):
    text: str = Field(description="One atomic, self-contained factual claim")
    source_urls: list[str] = Field(description="URLs of fetched pages that state this claim")


class Findings(BaseModel):
    """What the model fills in. IDs and evidence are assigned in code, not by the LLM."""
    claims: list[FoundClaim] = Field(description="Claims answering the sub-question, most important first")


class ResearchFailed(RuntimeError):
    pass


def research_agent(task: ResearchTask, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context or RunContext()
    pages: dict[str, dict] = {}  # url -> page, filled by the fetch tool as the agent reads
    agent = create_agent(
        model=chat_model(ctx.researcher_model, effort="medium"),
        tools=[web_search, recording_fetch_page(pages)],
        system_prompt=load_prompt("researcher_v1").format(
            max_claims=ctx.max_claims_per_question,
            max_tool_calls=ctx.max_research_tool_calls,
        ),
        response_format=ProviderStrategy(Findings),  # native structured output, as in the planner
        middleware=[
            # over budget, tool calls get an error back so the model answers with what it has
            ToolCallLimitMiddleware(run_limit=ctx.max_research_tool_calls),
            # hard backstop against a model that keeps calling tools anyway
            ModelCallLimitMiddleware(run_limit=ctx.max_research_tool_calls + 2),
        ],
    )
    out = agent.invoke({"messages": [("user", render_task(task))]})

    sq = task["sub_question"]
    findings: Findings | None = out.get("structured_response")
    if findings is None:
        raise ResearchFailed(f"Researcher returned no findings for {sq.id} ({sq.text!r})")

    evidence, claims = to_evidence_and_claims(findings, pages, sq.id)
    if task["revision"]:
        # replace the claims we were asked to fix (unless re-found with the same ID)
        claims.update({old: None for old in task["revision"].claim_ids if old not in claims})
    return {"evidence": evidence, "claims": claims, "usage": usage_from(out["messages"])}


def recording_fetch_page(pages: dict[str, dict]):
    """fetch_page, but every page it downloads is also stored in `pages`."""
    @tool("fetch_page", description=fetch_page.description)
    def fetch(url: str) -> str:
        try:
            page = fetch_page_content(url)
        except FetchError as e:
            return f"ERROR: {e}"
        # key by both: the model may cite the requested URL or the final one after redirects
        pages[url] = pages[page["url"]] = page
        return _wrap_untrusted(page)
    return fetch


def render_task(task: ResearchTask) -> str:
    msg = f"<sub_question>{task['sub_question'].text}</sub_question>"
    if task["revision"]:
        msg += (
            "\n\nA fact-checker could not verify your earlier claims for this sub-question. "
            f"What was missing: {task['revision'].missing}\n"
            "Look for better sources that directly state the facts."
        )
    return msg


def to_evidence_and_claims(
    findings: Findings, pages: dict[str, dict], sub_question_id: str
) -> tuple[dict[str, Evidence], dict[str, Claim]]:
    """Keep claims citing at least one fetched page; build Evidence for the pages they cite."""
    retrieved_at = datetime.now(timezone.utc)
    evidence: dict[str, Evidence] = {}
    claims: dict[str, Claim] = {}
    for found in findings.claims:
        text = " ".join(found.text.split())
        cited = [_lookup(pages, url) for url in found.source_urls]
        cited = [p for p in cited if p is not None]
        if not text or not cited:
            continue  # nothing we downloaded backs it, so the critic couldn't check it
        evidence_ids = []
        for page in cited:
            ev = _to_evidence(page, retrieved_at)
            evidence.setdefault(ev.id, ev)
            if ev.id not in evidence_ids:
                evidence_ids.append(ev.id)
        cid = _hash(sub_question_id, text.lower())
        claims[cid] = Claim(id=cid, text=text, evidence_ids=evidence_ids, sub_question_id=sub_question_id)
    return evidence, claims


def _lookup(pages: dict[str, dict], url: str) -> dict | None:
    url = url.strip()
    return pages.get(url) or pages.get(url.rstrip("/")) or pages.get(url.rstrip("/") + "/")


def _to_evidence(page: dict, retrieved_at: datetime) -> Evidence:
    text = page["text"][:MAX_CHARS]  # what the agent read; the critic checks quotes against this
    return Evidence(
        id=_hash(page["url"], text),
        url=page["url"],
        title=page["title"],
        published_at=page["published_at"],
        retrieved_at=retrieved_at,
        text=text,
        domain_score=DEFAULT_DOMAIN_SCORE,
    )


def _hash(*parts: str) -> str:
    # content hashes: parallel researchers can't collide, and re-finding the same thing merges
    return hashlib.sha1("\x00".join(parts).encode()).hexdigest()[:8]
