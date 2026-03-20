"""Supabase CRUD operations for auto_agent_runs table.

Provides simple helpers to create, update, and retrieve Auto Agent
run records. Uses the service role client to bypass RLS for
server-side updates (background tasks completing after request lifecycle).
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from utils.supabase.supabase_client import get_service_role_client

logger = logging.getLogger(__name__)

TABLE = "auto_agent_runs"


async def create_run(user_id: str, message: str) -> str:
    """Create a new auto agent run record with status 'running'.

    Args:
        user_id: Authenticated user ID.
        message: The user's task description.

    Returns:
        The new run's UUID string.

    Raises:
        Exception: If the Supabase insert fails.
    """
    client = get_service_role_client()
    result = (
        client.table(TABLE)
        .insert({"user_id": user_id, "message": message, "status": "running"})
        .execute()
    )
    run_id: str = result.data[0]["id"]
    logger.info("Created auto agent run %s for user %s", run_id, user_id)
    return run_id


async def update_run_complete(
    run_id: str,
    deliverables_json: Optional[dict[str, Any]],
    curation_report: Optional[str],
    final_report: Optional[str],
    tool_calls_made: int,
) -> None:
    """Mark a run as complete and store its output files.

    Args:
        run_id: Run UUID.
        deliverables_json: Parsed deliverables.json content.
        curation_report: Contents of data-curation.md.
        final_report: Contents of final_report.md.
        tool_calls_made: Number of tool calls the agent executed.
    """
    client = get_service_role_client()
    client.table(TABLE).update({
        "status": "complete",
        "deliverables_json": deliverables_json,
        "curation_report": curation_report,
        "final_report": final_report,
        "tool_calls_made": tool_calls_made,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", run_id).execute()
    logger.info("Auto agent run %s completed with %d tool calls", run_id, tool_calls_made)


async def update_run_failed(run_id: str, error_message: str) -> None:
    """Mark a run as failed with an error message.

    Args:
        run_id: Run UUID.
        error_message: Human-readable error description.
    """
    client = get_service_role_client()
    client.table(TABLE).update({
        "status": "failed",
        "error_message": error_message,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", run_id).execute()
    logger.error("Auto agent run %s failed: %s", run_id, error_message)


async def get_run(run_id: str, user_id: str) -> Optional[dict[str, Any]]:
    """Fetch a run record, ensuring it belongs to the given user.

    Args:
        run_id: Run UUID.
        user_id: Authenticated user ID (ownership check).

    Returns:
        Run dict, or None if not found or not owned by user.
    """
    client = get_service_role_client()
    result = (
        client.table(TABLE)
        .select("*")
        .eq("id", run_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    return result.data if result.data else None


def _extract_deliverables(workspace_files: dict[str, str]) -> Optional[dict[str, Any]]:
    """Extract deliverables.json from workspace files.

    Args:
        workspace_files: Dict mapping file path to file contents.

    Returns:
        Parsed deliverables dict, or None if not found.
    """
    for path, content in workspace_files.items():
        if "deliverables.json" in path:
            try:
                return json.loads(content)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Could not parse deliverables.json from %s", path)
    return None


def _extract_report(workspace_files: dict[str, str], filename: str) -> Optional[str]:
    """Extract a named markdown report from workspace files.

    Args:
        workspace_files: Dict mapping file path to file contents.
        filename: Report filename to look for (e.g. 'final_report.md').

    Returns:
        File contents string, or None if not found.
    """
    for path, content in workspace_files.items():
        if filename in path:
            return content
    return None


def parse_workspace_results(
    workspace_files: dict[str, str],
) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[str]]:
    """Parse deliverables.json, data-curation.md, and final_report.md from workspace files.

    Args:
        workspace_files: Dict mapping file path to file contents.

    Returns:
        Tuple of (deliverables_json, curation_report, final_report).
    """
    deliverables = _extract_deliverables(workspace_files)
    curation_report = _extract_report(workspace_files, "data-curation.md")
    final_report = _extract_report(workspace_files, "final_report.md")
    return deliverables, curation_report, final_report
