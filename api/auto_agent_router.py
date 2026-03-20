"""Auto Agent router — user-triggered autonomous fine-tuning.

Provides two endpoints:
- POST /auto-agent/run: Start a new agent run (fire-and-forget).
- GET  /auto-agent/run/{run_id}: Poll run status and retrieve results when complete.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from api.dependencies import AuthResult, FlexibleAuth
from schemas.auto_agent import (
    AutoAgentRunRequest,
    AutoAgentRunResponse,
    AutoAgentRunStatus,
)
from services.auto_agent.auto_agent_runner import start_run_background
from services.auto_agent.run_store import (
    create_run,
    get_run,
)
from shared.rate_limiting import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auto-agent", tags=["auto-agent"])


@router.post("/run", response_model=AutoAgentRunResponse)
@limiter.limit("20/minute")
async def start_run(
    request: Request,
    body: AutoAgentRunRequest,
    background_tasks: BackgroundTasks,
    auth: AuthResult = Depends(FlexibleAuth()),
) -> AutoAgentRunResponse:
    """Start a new Auto Agent run.

    Creates a run record immediately and returns a results URL.
    The agent executes asynchronously in the background (or Modal sandbox).

    Args:
        request: FastAPI request (required by rate limiter).
        body: Run request with task description and optional history.
        background_tasks: FastAPI background task manager.
        auth: Authenticated user.

    Returns:
        run_id and results_url to poll for completion.

    Raises:
        HTTPException: If the run record cannot be created.
    """
    try:
        user_id = str(auth.user_id)
        run_id = await create_run(user_id=user_id, message=body.message)

        history = None
        if body.history:
            history = [{"role": h.role, "content": h.content} for h in body.history]

        background_tasks.add_task(
            start_run_background,
            run_id=run_id,
            user_id=user_id,
            message=body.message,
            history=history,
        )

        return AutoAgentRunResponse(
            run_id=run_id,
            results_url=f"/auto-agent/results/{run_id}",
        )
    except Exception as e:
        logger.error("Failed to start auto agent run: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start auto agent run: {e}",
        )


@router.get("/run/{run_id}", response_model=AutoAgentRunStatus)
async def get_run_status(
    run_id: str,
    auth: AuthResult = Depends(FlexibleAuth()),
) -> AutoAgentRunStatus:
    """Get the status and results of an Auto Agent run.

    Poll this endpoint to check if the run is still running or has completed.

    Args:
        run_id: UUID of the run to check.
        auth: Authenticated user (ownership is enforced).

    Returns:
        Current run status and results if complete.

    Raises:
        HTTPException: 404 if run not found or not owned by user.
    """
    run = await get_run(run_id=run_id, user_id=str(auth.user_id))
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

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
