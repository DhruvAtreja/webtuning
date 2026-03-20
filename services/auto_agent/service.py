"""Auto Agent service layer.

Orchestrates agent invocations and run persistence.
Routes to Modal sandbox or local execution based on configuration.
Follows Pioneer's singleton service pattern.
"""

import logging
import os
from typing import Any, Optional

from services.auto_agent.agent import run_auto_agent

logger = logging.getLogger(__name__)

# When true, the agent runs inside a Modal container with felix helpers
USE_MODAL = os.getenv("AUTO_AGENT_USE_MODAL", "false").lower() == "true"


class AutoAgentService:
    """Service for the Auto Agent conversational agent.

    Routes to Modal sandbox (production) or local execution (dev)
    based on the AUTO_AGENT_USE_MODAL environment variable.
    """

    async def chat(
        self,
        user_id: str,
        message: str,
        history: Optional[list[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """Process a user message through the auto agent.

        Args:
            user_id: Authenticated user ID.
            message: User's task description for the agent.
            history: Optional conversation history as [{role, content}].

        Returns:
            Dict with answer, tool_calls, workspace_files.
        """
        message_history = None
        if history:
            message_history = [(h["role"], h["content"]) for h in history]

        if USE_MODAL:
            result = await self._run_in_modal(
                question=message,
                user_id=user_id,
                message_history=message_history,
            )
        else:
            result = await run_auto_agent(
                question=message,
                user_id=user_id,
                message_history=message_history,
            )

        return {
            "answer": result["answer"],
            "tool_calls": result.get("tool_calls", 0),
            "workspace_files": result.get("workspace_files", {}),
        }

    async def _run_in_modal(
        self,
        question: str,
        user_id: str,
        message_history: list[tuple[str, str]] | None,
    ) -> dict[str, Any]:
        """Run the agent inside a Modal sandbox.

        Calls the synchronous run_agent_in_modal_sync inside asyncio.to_thread
        so that all blocking Modal I/O runs in a thread pool and the asyncio
        event loop stays unblocked.

        Args:
            question: User's task description.
            user_id: Authenticated user ID.
            message_history: Optional conversation history.

        Returns:
            Dict with answer, tool_calls, workspace_files.
        """
        import asyncio  # noqa: PLC0415
        from services.auto_agent.modal_sandbox import run_agent_in_modal_sync  # noqa: PLC0415 — lazy import for Modal

        logger.info("Running auto agent in Modal sandbox")
        return await asyncio.to_thread(
            run_agent_in_modal_sync,
            question,
            user_id,
            message_history,
        )


# Singleton
_service: Optional[AutoAgentService] = None


def get_auto_agent_service() -> AutoAgentService:
    """Get or create the AutoAgentService singleton.

    Returns:
        The global AutoAgentService instance.
    """
    global _service
    if _service is None:
        _service = AutoAgentService()
        logger.info("AutoAgentService singleton created")
    return _service
