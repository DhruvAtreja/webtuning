"""Shared background execution logic for the Auto Agent.

Used by both the router (POST /auto-agent/run) and the clarification agent's
trigger_auto_agent tool so both paths go through the same execution flow.
"""

import logging
from typing import Optional

from services.auto_agent.run_store import (
    parse_workspace_results,
    update_run_complete,
    update_run_failed,
)

logger = logging.getLogger(__name__)


async def start_run_background(
    run_id: str,
    user_id: str,
    message: str,
    history: Optional[list[dict[str, str]]] = None,
) -> None:
    """Run the Auto Agent in the background and persist results.

    Intended to be called as an asyncio task or FastAPI BackgroundTask.

    Args:
        run_id: The DB record UUID to update on completion.
        user_id: Authenticated user ID.
        message: The user's task description.
        history: Optional conversation history.
    """
    try:
        from services.auto_agent import get_auto_agent_service  # noqa: PLC0415

        service = get_auto_agent_service()
        result = await service.chat(
            user_id=user_id,
            message=message,
            history=history,
        )

        workspace_files = result.get("workspace_files", {})
        deliverables, curation_report, final_report = parse_workspace_results(workspace_files)

        await update_run_complete(
            run_id=run_id,
            deliverables_json=deliverables,
            curation_report=curation_report,
            final_report=final_report,
            tool_calls_made=result.get("tool_calls", 0),
        )
    except Exception as e:
        logger.error("Auto agent background run %s failed: %s", run_id, e, exc_info=True)
        await update_run_failed(run_id, str(e))
