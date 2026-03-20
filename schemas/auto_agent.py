"""Auto Agent API schemas.

Pydantic v2 models for the Auto Agent endpoints that allow users to trigger
autonomous fine-tuning runs from a task description.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

from schemas.adaptive_finetuning import MessageHistoryItem


class AutoAgentClarifyRequest(BaseModel):
    """Request body for a clarification agent turn.

    Either `message` (new turn) or `resume_value` (MCQ selection resume) must be set.

    Args:
        message: The user's text input for a new turn.
        resume_value: The option the user selected from an MCQ (resumes interrupted graph).
        conversation_id: Conversation ID (required for resume; generated on first call).
        history: Optional conversation history [{role, content}].
    """

    message: Optional[str] = Field(
        None,
        max_length=10000,
        description="User text input (for new turns)",
    )
    resume_value: Optional[str] = Field(
        None,
        max_length=10000,
        description="MCQ option selected by user (resumes interrupted graph)",
    )
    conversation_id: Optional[str] = Field(
        None, description="Existing conversation ID"
    )
    history: Optional[list[dict[str, str]]] = Field(
        None, description="Conversation history [{role, content}]"
    )


class AutoAgentClarifyResponse(BaseModel):
    """Response from the clarification agent.

    Args:
        answer: The agent's text response (markdown).
        conversation_id: Conversation ID for follow-up messages.
        interrupted: True when the graph paused on present_options (waiting for MCQ selection).
        options: MCQ options when interrupted=True.
        question: The question text when interrupted=True.
        run_id: Set when the agent triggered the Auto Agent sandbox.
        results_url: Frontend URL to view results (set with run_id).
    """

    answer: str = Field(..., description="Agent's markdown response")
    conversation_id: str = Field(..., description="Conversation ID for follow-ups")
    interrupted: bool = Field(False, description="True when waiting for MCQ selection")
    options: Optional[list[str]] = Field(
        None, description="MCQ options when interrupted"
    )
    question: Optional[str] = Field(
        None, description="Question text when interrupted"
    )
    run_id: Optional[str] = Field(
        None, description="Run UUID if agent triggered the Auto Agent"
    )
    results_url: Optional[str] = Field(
        None, description="Results page URL if agent triggered the Auto Agent"
    )


class AutoAgentRunRequest(BaseModel):
    """Request body to start a new Auto Agent run.

    Args:
        message: The user's task description for the agent.
        history: Optional conversation history for multi-turn context.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Task description for the Auto Agent",
    )
    history: Optional[list[MessageHistoryItem]] = Field(
        None, description="Conversation history for multi-turn context"
    )


class AutoAgentRunResponse(BaseModel):
    """Response after creating a new Auto Agent run.

    Args:
        run_id: UUID of the created run.
        results_url: Frontend URL to check results when the run completes.
    """

    run_id: str = Field(..., description="UUID of the created run")
    results_url: str = Field(..., description="URL to view results when run completes")


class AutoAgentRunStatus(BaseModel):
    """Current status and results of an Auto Agent run.

    Args:
        run_id: UUID of the run.
        status: Current status (running | complete | failed).
        deliverables_json: Parsed deliverables.json when complete.
        curation_report: Contents of data-curation.md when complete.
        final_report: Contents of final_report.md when complete.
        error_message: Error description when failed.
        tool_calls_made: Number of tool calls the agent executed.
        created_at: ISO timestamp when the run was created.
        completed_at: ISO timestamp when the run completed, or None.
    """

    run_id: str = Field(..., description="Run UUID")
    status: str = Field(..., description="running | complete | failed")
    deliverables_json: Optional[dict[str, Any]] = Field(
        None, description="Parsed deliverables.json (available when complete)"
    )
    curation_report: Optional[str] = Field(
        None, description="data-curation.md contents (available when complete)"
    )
    final_report: Optional[str] = Field(
        None, description="final_report.md contents (available when complete)"
    )
    error_message: Optional[str] = Field(
        None, description="Error description (available when failed)"
    )
    tool_calls_made: int = Field(0, description="Number of tool calls executed")
    created_at: str = Field(..., description="ISO timestamp of run creation")
    completed_at: Optional[str] = Field(None, description="ISO timestamp of completion")
