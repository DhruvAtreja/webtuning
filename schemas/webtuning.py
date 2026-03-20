"""WebTuning API schemas.

Pydantic v2 models for the WebTuning endpoints that kick off website crawls,
expose crawl status, and run the browser agent with an optional navigation expert.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class WebTuningCrawlRequest(BaseModel):
    """Request body to start a website crawl and expert model training.

    Args:
        url: Website URL to crawl and build an expert model for.
    """

    url: str = Field(..., description="Website URL to crawl and build expert model for")


class WebTuningCrawlResponse(BaseModel):
    """Response after starting a crawl run.

    Args:
        run_id: UUID of the created Auto Agent run.
        status_url: URL to poll for crawl status and registry updates.
    """

    run_id: str = Field(..., description="UUID of the created Auto Agent run")
    status_url: str = Field(..., description="URL to poll for crawl status")


class WebTuningRunAgentRequest(BaseModel):
    """Request body to run the browser agent on a website.

    Args:
        url: Website URL to navigate.
        task: High-level task description (e.g. "Find laptops under $500").
        job_id: Pioneer job_id of the navigation expert. If None, the registry
                is checked by domain; if still None the agent runs without an expert.
    """

    url: str = Field(..., description="Website URL to navigate")
    task: str = Field(
        ..., description="High-level task, e.g. 'Find laptops under $500'"
    )
    job_id: Optional[str] = Field(
        None, description="Pioneer job_id; if None uses registry lookup"
    )


class WebTuningRunAgentResponse(BaseModel):
    """Response from a completed browser agent run.

    Args:
        result: Final text result produced by the agent.
        steps_taken: Number of tool calls executed.
        trajectory: List of tool call records (tool, input, output).
        success: True if the agent reached end_turn before max steps.
    """

    result: str = Field(..., description="Final result from the browser agent")
    steps_taken: int = Field(..., description="Number of tool calls executed")
    trajectory: list[dict[str, Any]] = Field(
        ..., description="Tool call records (tool, input, output)"
    )
    success: bool = Field(
        ..., description="True if agent completed task before max turns"
    )
