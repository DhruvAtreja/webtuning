"""Anthropic SDK browser agent for WebTuning.

Runs an agentic loop that uses bash (Playwright) and ask_website_expert tools
to complete navigation tasks on websites.

LangSmith tracing is enabled automatically when LANGSMITH_API_KEY is set.
Each call to run_browser_agent appears as a top-level trace; individual LLM
turns are traced via wrap_anthropic().
"""

import logging
import os
import subprocess
from typing import Optional
from urllib.parse import urlparse

import anthropic

from services.browser_agent.training_recorder import record_successful_trajectory
from services.browser_agent.website_expert import SENTINEL, ask_website_expert

logger = logging.getLogger(__name__)

MAX_TURNS = 20

SYSTEM_PROMPT = """You are a web automation agent.

ALWAYS start by calling ask_website_expert with your navigation question.

**CRITICAL — how to use the expert answer:**
- If ask_website_expert returns numbered steps or a URL → FOLLOW THEM DIRECTLY with bash/Playwright. Do NOT re-explore or re-verify with extra Playwright steps. Trust the expert and execute.
- If ask_website_expert returns "I don't have information about this." → the expert doesn't know. Use bash/Playwright to explore and figure it out yourself.

Use bash to run Playwright (Python sync API). On task completion, take a final screenshot."""

_TOOLS = [
    {
        "name": "bash",
        "description": "Run shell commands. Playwright (sync API) is available.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "ask_website_expert",
        "description": (
            "Ask the fine-tuned navigation expert what it knows about the target website. "
            f"Returns numbered steps or '{SENTINEL}'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
]


def _make_client() -> anthropic.Anthropic:
    """Create an Anthropic client, wrapped for LangSmith tracing if available.

    Returns:
        anthropic.Anthropic instance (possibly LangSmith-wrapped).
    """
    client = anthropic.Anthropic()
    if not os.getenv("LANGSMITH_API_KEY"):
        return client
    try:
        from langsmith.wrappers import wrap_anthropic  # type: ignore[import]
        return wrap_anthropic(client)
    except ImportError:
        logger.debug("langsmith not installed; browser agent will run without tracing")
        return client


def _traceable(fn):
    """Wrap fn with LangSmith @traceable if available, else return as-is."""
    if not os.getenv("LANGSMITH_API_KEY"):
        return fn
    try:
        from langsmith import traceable  # type: ignore[import]
        return traceable(name="browser_agent_run", run_type="chain")(fn)
    except ImportError:
        return fn


def _enable_langsmith_env() -> None:
    """Set LangSmith env vars when the API key is present."""
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "auto-agent")


def _run_browser_agent_impl(
    url: str,
    task: str,
    job_id: Optional[str] = None,
) -> dict:
    """Core browser agent loop (called by the traced public function).

    Args:
        url: Target website URL.
        task: High-level task description.
        job_id: Pioneer job_id for the navigation expert. If None, the
                ask_website_expert tool returns SENTINEL for every call.

    Returns:
        Dict with keys: result (str), steps_taken (int), trajectory (list), success (bool).
    """
    domain = urlparse(url).netloc
    client = _make_client()
    messages = [{"role": "user", "content": f"URL: {url}\nTask: {task}"}]
    steps = 0
    trajectory: list[dict] = []

    for _ in range(MAX_TURNS):
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=_TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            result_text = next(
                (b.text for b in resp.content if hasattr(b, "text")), "Done"
            )
            record_successful_trajectory(domain, task, trajectory, job_id or "")
            return {
                "result": result_text,
                "steps_taken": steps,
                "trajectory": trajectory,
                "success": True,
            }

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            steps += 1

            if block.name == "bash":
                proc = subprocess.run(
                    block.input["command"],
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                tool_output = (proc.stdout + proc.stderr)[:5000]

            elif block.name == "ask_website_expert" and job_id:
                tool_output = ask_website_expert(
                    domain, block.input["question"], job_id
                )
            else:
                tool_output = SENTINEL

            trajectory.append(
                {
                    "tool": block.name,
                    "input": block.input,
                    "output": tool_output[:200],
                }
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_output,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return {
        "result": "Max turns reached",
        "steps_taken": steps,
        "trajectory": trajectory,
        "success": False,
    }


# Public entry point — apply @traceable at import time so the decorator
# sees the LANGSMITH_API_KEY env var that was set before the module loaded.
def run_browser_agent(
    url: str,
    task: str,
    job_id: Optional[str] = None,
) -> dict:
    """Run the browser agent to complete a navigation task.

    LangSmith tracing is enabled automatically when LANGSMITH_API_KEY is set.
    Each call creates a top-level 'browser_agent_run' trace; individual LLM
    turns are captured via the wrapped Anthropic client.

    Args:
        url: Target website URL.
        task: High-level task description.
        job_id: Pioneer job_id for the navigation expert. If None, the
                ask_website_expert tool returns SENTINEL for every call.

    Returns:
        Dict with keys: result (str), steps_taken (int), trajectory (list), success (bool).
    """
    _enable_langsmith_env()
    traced_impl = _traceable(_run_browser_agent_impl)
    return traced_impl(url=url, task=task, job_id=job_id)
