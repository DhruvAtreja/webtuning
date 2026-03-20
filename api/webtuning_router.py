"""WebTuning router — crawl websites and run expert-assisted browser agents.

Endpoints:
- POST /webtuning/crawl          Start a web crawl and expert model training run.
- GET  /webtuning/crawl/{run_id} Poll crawl status; updates registry on completion.
- POST /webtuning/run-agent      Run the browser agent (with optional navigation expert).
"""

import json
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from api.dependencies import AuthResult, FlexibleAuth
from schemas.auto_agent import AutoAgentRunStatus
from schemas.webtuning import (
    WebTuningCrawlRequest,
    WebTuningCrawlResponse,
    WebTuningRunAgentRequest,
    WebTuningRunAgentResponse,
)
from services.auto_agent.auto_agent_runner import start_run_background
from services.auto_agent.run_store import create_run, get_run
from services.browser_agent.agent import run_browser_agent
from services.web_tuner.task_builder import build_crawl_task
from shared.rate_limiting import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webtuning", tags=["webtuning"])

REGISTRY_PATH = Path("data/models/registry.json")


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def _read_registry() -> dict:
    try:
        return json.loads(REGISTRY_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n")


def _maybe_update_registry(deliverables: dict, run_id: str) -> None:
    """If deliverables contain domain + job_id, write them to the registry."""
    domain = deliverables.get("domain")
    job_id = (deliverables.get("final_model") or {}).get("job_id")
    if not domain or not job_id:
        return
    registry = _read_registry()
    registry[domain] = {
        "job_id": job_id,
        "model_name": (deliverables.get("final_model") or {}).get("model_name", ""),
        "run_id": run_id,
    }
    _write_registry(registry)
    logger.info("Registered navigation expert for %s (job_id=%s)", domain, job_id)


def _registry_lookup(url: str) -> Optional[str]:
    """Return the job_id for the given URL's domain, or None."""
    domain = urlparse(url).netloc
    return _read_registry().get(domain, {}).get("job_id")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/crawl", response_model=WebTuningCrawlResponse)
@limiter.limit("10/minute")
async def start_crawl(
    request: Request,
    body: WebTuningCrawlRequest,
    background_tasks: BackgroundTasks,
    auth: AuthResult = Depends(FlexibleAuth()),
) -> WebTuningCrawlResponse:
    """Start a web crawl and expert model fine-tuning run.

    Creates an Auto Agent run that crawls the target URL, generates decoder
    training data, and fine-tunes a Qwen3-8B navigation expert. Returns
    immediately with a run_id; poll /webtuning/crawl/{run_id} for results.

    Args:
        request: FastAPI request (required by rate limiter).
        body: Crawl request containing the target URL.
        background_tasks: FastAPI background task manager.
        auth: Authenticated user.

    Returns:
        run_id and status_url to poll for completion.
    """
    try:
        user_id = str(auth.user_id)
        task = build_crawl_task(body.url)
        run_id = await create_run(user_id=user_id, message=task)

        background_tasks.add_task(
            start_run_background,
            run_id=run_id,
            user_id=user_id,
            message=task,
            history=None,
        )

        return WebTuningCrawlResponse(
            run_id=run_id,
            status_url=f"/webtuning/crawl/{run_id}",
        )
    except Exception as e:
        logger.error("Failed to start crawl: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start crawl: {e}")


@router.get("/crawl/{run_id}", response_model=AutoAgentRunStatus)
async def get_crawl_status(
    run_id: str,
    auth: AuthResult = Depends(FlexibleAuth()),
) -> AutoAgentRunStatus:
    """Poll a crawl run for status and results.

    When the run completes and deliverables.json contains a domain + job_id,
    the registry is updated automatically so subsequent /run-agent calls can
    find the expert without providing job_id explicitly.

    Args:
        run_id: UUID of the crawl run to check.
        auth: Authenticated user (ownership enforced).

    Returns:
        Current run status and deliverables when complete.

    Raises:
        HTTPException: 404 if run not found or not owned by user.
    """
    run = await get_run(run_id=run_id, user_id=str(auth.user_id))
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run["status"] == "complete" and run.get("deliverables_json"):
        _maybe_update_registry(run["deliverables_json"], run_id)

    return AutoAgentRunStatus(
        run_id=str(run["id"]),
        status=run["status"],
        deliverables_json=run.get("deliverables_json"),
        curation_report=run.get("curation_report"),
        final_report=run.get("final_report"),
        error_message=run.get("error_message"),
        tool_calls_made=run.get("tool_calls_made", 0),
        created_at=str(run["created_at"]),
        completed_at=str(run["completed_at"]) if run.get("completed_at") else None,
    )


@router.post("/run-agent", response_model=WebTuningRunAgentResponse)
async def run_agent_endpoint(
    request: Request,
    body: WebTuningRunAgentRequest,
    auth: AuthResult = Depends(FlexibleAuth()),
) -> WebTuningRunAgentResponse:
    """Run the browser agent to complete a task on a website.

    If job_id is not provided, the domain registry is checked automatically.
    If no expert is found, the agent runs in exploration-only mode.

    Args:
        request: FastAPI request object.
        body: Agent run request (url, task, optional job_id).
        auth: Authenticated user.

    Returns:
        Agent result, steps taken, trajectory, and success flag.
    """
    job_id = body.job_id or _registry_lookup(body.url)
    try:
        result = run_browser_agent(url=body.url, task=body.task, job_id=job_id)
        return WebTuningRunAgentResponse(**result)
    except Exception as e:
        logger.error("Browser agent run failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Browser agent failed: {e}")
