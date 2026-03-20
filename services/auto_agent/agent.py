"""LangGraph agent for the Auto Agent.

Implements an autonomous agent that understands a user's task description,
researches the domain via web search, builds curated synthetic datasets,
trains models, and iterates until the model meets the user's requirements.
Uses 4 tools: web_search, bash, read_file, and edit_file.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import subprocess
import uuid
from typing import Annotated, Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from services.auto_agent.context_manager import ContextManager
from services.auto_agent.prompts import (
    build_system_message,
    build_system_prompt,
)

logger = logging.getLogger(__name__)

# Model configuration
MAIN_MODEL = os.getenv("AUTO_AGENT_MODEL", "claude-sonnet-4-6")
RECURSION_LIMIT = 500

# Bash command timeout (seconds)
BASH_TIMEOUT = 3600


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class State(TypedDict):
    """LangGraph agent state — accumulates messages via the add_messages reducer."""

    messages: Annotated[list[AnyMessage], add_messages]


# ---------------------------------------------------------------------------
# Context compaction helper
# ---------------------------------------------------------------------------


def _run_compaction_sync(
    context_manager: ContextManager,
    messages: list[AnyMessage],
) -> tuple[list[AnyMessage], int, int]:
    """Run async context compaction from a synchronous LangGraph node.

    Args:
        context_manager: The ContextManager instance.
        messages: Current message list.

    Returns:
        Tuple of (compacted messages, original tokens, new tokens).
    """
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                context_manager.truncate_context(messages),
            )
            return future.result(timeout=60)
    except RuntimeError:
        return asyncio.run(context_manager.truncate_context(messages))


# ---------------------------------------------------------------------------
# Assistant node
# ---------------------------------------------------------------------------


class Assistant:
    """LangGraph node that invokes the LLM with context compaction.

    Before each LLM call, checks if the conversation exceeds the token
    limit and compacts the middle messages into a summary if needed.
    The graph state retains the full history; only the LLM sees the
    compacted view.

    Args:
        runnable: The LLM chain (prompt | model.bind_tools).
        context_manager: Manages token counting and summarization.
    """

    def __init__(self, runnable: Runnable, context_manager: ContextManager) -> None:
        self.runnable = runnable
        self.context_manager = context_manager

    def __call__(self, state: State, config: RunnableConfig) -> dict[str, Any]:
        """Invoke the LLM, compacting context if over token limit.

        Args:
            state: Current graph state with messages.
            config: LangGraph runtime config.

        Returns:
            Dict with updated messages.
        """
        # Compact messages for LLM if over context limit
        invoke_state = state
        if self.context_manager.needs_truncation(state["messages"]):
            compacted, orig_tokens, new_tokens = _run_compaction_sync(
                self.context_manager, state["messages"]
            )
            logger.info(
                "Context compacted for LLM: %d → %d tokens",
                orig_tokens,
                new_tokens,
            )
            invoke_state = {**state, "messages": compacted}

        max_retries = 3
        for attempt in range(max_retries):
            result = self.runnable.invoke(invoke_state)
            if result.tool_calls or _has_text_content(result):
                _log_assistant_response(result)
                return {"messages": result}
            if attempt < max_retries - 1:
                invoke_state = {
                    **invoke_state,
                    "messages": invoke_state["messages"]
                    + [("user", "Please respond with a real output.")],
                }
        return {"messages": result}


def _has_text_content(result: Any) -> bool:
    """Check if an LLM result has actual text content.

    Args:
        result: The LLM response message.

    Returns:
        True if the result contains text content.
    """
    content = result.content
    if isinstance(content, str) and content.strip():
        return True
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                return True
    return False


# ---------------------------------------------------------------------------
# Agent logging
# ---------------------------------------------------------------------------

# Step counter for readable log prefixes
_step_counter = 0


def _next_step() -> int:
    """Increment and return the global step counter."""
    global _step_counter
    _step_counter += 1
    return _step_counter


def _truncate_for_log(text: str, max_len: int = 500) -> str:
    """Truncate text for log output.

    Args:
        text: Text to truncate.
        max_len: Maximum characters.

    Returns:
        Truncated string.
    """
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... ({len(text)} chars total)"


def _log_assistant_response(result: Any) -> None:
    """Log the assistant's LLM response: text content and tool calls.

    Args:
        result: The LLM response message.
    """
    step = _next_step()

    # Log text content
    text = extract_text_content(result.content)
    if text.strip():
        logger.info(
            "[Step %d] 🤖 ASSISTANT:\n%s",
            step,
            _truncate_for_log(text, 1000),
        )

    # Log each tool call
    tool_calls = getattr(result, "tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            name = tc.get("name", "unknown")
            args = tc.get("args", {})
            args_str = json.dumps(args, default=str)
            logger.info(
                "[Step %d] 🔧 TOOL CALL: %s(%s)",
                step,
                name,
                _truncate_for_log(args_str, 300),
            )


def _log_tool_result(tool_message: ToolMessage) -> None:
    """Log a tool execution result.

    Args:
        tool_message: The ToolMessage with tool output.
    """
    step = _next_step()
    tool_name = getattr(tool_message, "name", "unknown")
    content = str(tool_message.content)
    logger.info(
        "[Step %d] 📋 TOOL RESULT [%s]:\n%s",
        step,
        tool_name,
        _truncate_for_log(content, 800),
    )


class LoggingToolNode:
    """ToolNode wrapper that logs every tool call and result.

    Wraps the standard LangGraph ToolNode to add logging of each
    tool execution's input arguments and output results.

    Args:
        tool_node: The underlying ToolNode (with fallback).
    """

    def __init__(self, tool_node: Any) -> None:
        self.tool_node = tool_node

    def __call__(self, state: dict, config: Optional[RunnableConfig] = None) -> dict:
        """Execute tools and log results.

        Args:
            state: Graph state with messages.
            config: LangGraph runtime config.

        Returns:
            Dict with tool result messages.
        """
        result = self.tool_node.invoke(state, config)

        # Log each tool result message
        for msg in result.get("messages", []):
            if isinstance(msg, ToolMessage):
                _log_tool_result(msg)

        return result


# ---------------------------------------------------------------------------
# Tool error fallback
# ---------------------------------------------------------------------------


def _handle_tool_error(state: dict[str, Any]) -> dict[str, Any]:
    """Return friendly error messages for failed tool calls.

    Args:
        state: Graph state including the error and last message.

    Returns:
        Dict with ToolMessage error responses.
    """
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {error!r}\nPlease fix and try again.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def _create_tool_node_with_fallback(tools_list: list[Any]) -> Any:
    """Create a ToolNode with error fallback, wrapped with logging.

    Args:
        tools_list: List of LangChain tools.

    Returns:
        LoggingToolNode wrapping ToolNode with fallback error handler.
    """
    base_node = ToolNode(tools_list).with_fallbacks(
        [RunnableLambda(_handle_tool_error)], exception_key="error"
    )
    return LoggingToolNode(base_node)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def web_search(query: str) -> str:
    """Search the web using Anthropic's built-in web search capability.

    Use for: researching domain knowledge, finding labeled data examples,
    understanding entity types, best practices for classification tasks,
    finding similar datasets, checking what labels a task should use, etc.

    Args:
        query: Natural language search query.

    Returns:
        Search results as synthesized text.
    """
    import json  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = json.dumps({
        "model": "claude-opus-4-6",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": query}],
        "tools": [{"type": "web_search_20260209", "name": "web_search"}],
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text_parts = [
            block["text"]
            for block in data.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(text_parts) if text_parts else "(no results)"
    except Exception as e:
        return f"Web search error: {e}"


@tool
def bash(command: str) -> str:
    """Execute a bash command in the sandbox. Unrestricted access. 1-hour timeout.

    Use for: running Python scripts, calling felix helper functions,
    file operations, data processing, installing packages, etc.

    The sandbox has functions.py pre-loaded with felix helpers.
    Example: python3 -c "from functions import *; print(list_datasets())"

    For long operations (inference loops over hundreds of examples), add
    progress prints so you can see it's working. If a command times out,
    you'll get the partial output captured up to that point.

    Args:
        command: The bash command to execute.

    Returns:
        Command output (stdout + stderr) or error message.
    """
    try:
        proc = subprocess.Popen(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=BASH_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            output = stdout or ""
            if stderr:
                output += f"\n(stderr: {stderr.strip()})"
            output += f"\n\n... COMMAND TIMED OUT after {BASH_TIMEOUT}s. Above is the partial output captured before timeout."
            if len(output) > 10000:
                output = output[:10000] + "\n... (truncated)"
            return output

        output = stdout
        if stderr:
            output += f"\n(stderr: {stderr.strip()})"
        if not output:
            return "(no output)"
        if len(output) > 10000:
            return output[:10000] + (
                f"\n\n... OUTPUT TRUNCATED at 10000 chars (total: {len(output)}). "
                "Write results to a file and read specific sections, or make more targeted queries."
            )
        return output
    except Exception as e:
        return f"Error executing command: {e}"


@tool
def read_file(file_path: str) -> str:
    """Read a file from the sandbox filesystem.

    Args:
        file_path: Path to the file to read.

    Returns:
        File contents or error message.
    """
    try:
        with open(file_path) as f:
            content = f.read()
        if len(content) > 100_000:
            return content[:100_000] + "\n... (truncated at 100K chars)"
        return content
    except FileNotFoundError:
        return f"Error: file not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing old_string with new_string.

    If old_string is empty, creates the file with new_string as content.

    Args:
        file_path: Path to the file to edit.
        old_string: Text to find and replace. Empty string to create new file.
        new_string: Replacement text.

    Returns:
        Success message or error.
    """
    try:
        if not old_string:
            # Create new file
            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
            with open(file_path, "w") as f:
                f.write(new_string)
            return f"Created file: {file_path}"

        with open(file_path) as f:
            content = f.read()

        if old_string not in content:
            return f"Error: old_string not found in {file_path}"

        count = content.count(old_string)
        new_content = content.replace(old_string, new_string, 1)

        with open(file_path, "w") as f:
            f.write(new_content)

        return f"Replaced 1 of {count} occurrence(s) in {file_path}"
    except FileNotFoundError:
        return f"Error: file not found: {file_path}"
    except Exception as e:
        return f"Error editing file: {e}"


# ---------------------------------------------------------------------------
# Delegate task subagent
# ---------------------------------------------------------------------------

SUBAGENT_MODEL = os.getenv("AUTO_AGENT_SUBAGENT_MODEL", "claude-sonnet-4-6")
SUBAGENT_RECURSION_LIMIT = 100


@tool
def delegate_task(
    task: str,
    context: Optional[str] = None,
) -> str:
    """Delegate an independent task to a subagent with full tool access.

    The subagent is a fresh instance of this agent with its own conversation,
    context window, and tool access (web_search, bash, read_file, edit_file).
    Use it to parallelize independent work — e.g. researching one domain area
    while building a dataset for another, or training a model while writing
    evaluation scripts.

    The subagent has a 100-turn limit and shares the same sandbox filesystem.
    Files written by the subagent are visible to you and vice versa.

    Args:
        task: Clear, self-contained instruction for the subagent. Include all
            context it needs (task description, file paths, etc.).
        context: Optional background context about the broader goal.

    Returns:
        The subagent's final text response, or error message.
    """
    try:
        message = f"Context: {context}\n\nTask: {task}" if context else task

        def _run_subagent() -> dict[str, Any]:
            """Build and run a fresh subagent graph."""
            return asyncio.run(
                _run_delegate_subagent(question=message)
            )

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run_subagent)
                result = future.result(timeout=1800)  # 30 min max
        except RuntimeError:
            result = asyncio.run(
                _run_delegate_subagent(question=message)
            )

        return json.dumps({
            "task": task[:200],
            "answer": result["answer"],
            "tool_calls": result["tool_calls"],
        }, indent=2, default=str)

    except Exception as e:
        logger.error("Delegate subagent failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Subagent failed: {e}"}, indent=2)


async def _run_delegate_subagent(question: str) -> dict[str, Any]:
    """Build and run a fresh subagent graph for a delegated task.

    Creates an independent graph with the same tools (minus delegate_task
    to prevent recursion) and a separate conversation thread.

    Args:
        question: The task instruction.

    Returns:
        Dict with: answer (str), tool_calls (int), conversation_id (str).
    """
    prompt_text = build_system_prompt()
    system_msg = build_system_message(prompt_text)

    # Subagent gets all tools EXCEPT delegate_task (no recursion)
    subagent_tools = [web_search, bash, read_file, edit_file]

    context_manager = ContextManager()
    graph = _build_graph(
        agent_tools=subagent_tools,
        checkpointer=MemorySaver(),
        model_name=SUBAGENT_MODEL,
        system_message=system_msg,
        context_manager=context_manager,
    )

    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": SUBAGENT_RECURSION_LIMIT,
    }

    result = graph.invoke({"messages": [("user", question)]}, config)

    answer = extract_text_content(result["messages"][-1].content)
    tool_calls = sum(
        1 for m in result["messages"]
        if hasattr(m, "tool_calls") and m.tool_calls
    )

    return {
        "answer": answer,
        "tool_calls": tool_calls,
        "conversation_id": thread_id,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

ALL_TOOLS = [web_search, bash, read_file, edit_file, delegate_task]


def _build_graph(
    agent_tools: list[Any],
    checkpointer: MemorySaver,
    model_name: str,
    system_message: Any,
    context_manager: ContextManager,
) -> Any:
    """Build a compiled LangGraph agent with context compaction.

    Args:
        agent_tools: Tools available to the agent.
        checkpointer: Memory checkpointer for conversation state.
        model_name: Anthropic model ID.
        system_message: Pre-built SystemMessage (with caching).
        context_manager: Handles context window summarization.

    Returns:
        Compiled LangGraph StateGraph.
    """
    llm = ChatAnthropic(
        model=model_name,
        max_tokens=128000,
        thinking={
            "type": "enabled",
            "budget_tokens": 32768,
        },
        model_kwargs={
            "extra_headers": {
                "anthropic-beta": "context-1m-2025-08-07",
            },
        },
    )

    prompt = ChatPromptTemplate.from_messages(
        [system_message, ("placeholder", "{messages}")]
    )

    agent_runnable = prompt | llm.bind_tools(agent_tools, parallel_tool_calls=True)

    builder = StateGraph(State)
    builder.add_node("assistant", Assistant(agent_runnable, context_manager))
    builder.add_node("tools", _create_tool_node_with_fallback(agent_tools))
    builder.set_entry_point("assistant")
    builder.add_conditional_edges(
        "assistant", tools_condition, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "assistant")

    return builder.compile(checkpointer=checkpointer)


def extract_text_content(content: Any) -> str:
    """Extract text from LLM response content.

    Handles extended thinking format where content is a list of blocks.

    Args:
        content: The content field from an LLM response message.

    Returns:
        Extracted text string.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts) if text_parts else str(content)

    return str(content)


# ---------------------------------------------------------------------------
# Agent initialization and entry point
# ---------------------------------------------------------------------------

# Lazy singleton
_main_graph: Any = None


def _initialize_graph() -> None:
    """Initialize the agent graph.

    Builds the system prompt and creates the LangGraph StateGraph.
    """
    global _main_graph

    prompt_text = build_system_prompt()
    system_msg = build_system_message(prompt_text)

    context_manager = ContextManager()

    _main_graph = _build_graph(
        agent_tools=ALL_TOOLS,
        checkpointer=MemorySaver(),
        model_name=MAIN_MODEL,
        system_message=system_msg,
        context_manager=context_manager,
    )

    logger.info("Auto Agent graph initialized")


async def run_auto_agent(
    question: str,
    user_id: str,
    message_history: Optional[list] = None,
) -> dict:
    """Run the auto agent on a user question.

    Initializes the graph if needed and invokes the agent.

    Args:
        question: The user's task description or instruction.
        user_id: Authenticated user ID (used by felix helpers for API calls).
        message_history: Optional [(role, content)] conversation history.

    Returns:
        Dict with: answer (str), tool_calls (int), conversation_id (str),
                   workspace_files (dict[str, str]).
    """
    global _main_graph

    # Reset step counter for clean log output
    global _step_counter
    _step_counter = 0

    # Enable LangSmith tracing if API key is available
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "auto-agent")

    try:
        # Initialize graph if not yet built
        if _main_graph is None:
            _initialize_graph()

        thread_id = str(uuid.uuid4())
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": RECURSION_LIMIT,
        }

        messages: list[tuple[str, str]] = []
        if message_history:
            messages.extend(message_history)

        messages.append(("user", question))
        logger.info(
            "[Step 0] 👤 USER:\n%s",
            _truncate_for_log(question, 1000),
        )

        result = _main_graph.invoke({"messages": messages}, config)

        answer = extract_text_content(result["messages"][-1].content)
        tool_calls = sum(
            1
            for m in result["messages"]
            if hasattr(m, "tool_calls") and m.tool_calls
        )

        return {
            "answer": answer,
            "tool_calls": tool_calls,
            "conversation_id": thread_id,
        }

    except Exception as e:
        logger.error("Auto agent failed: %s", e, exc_info=True)
        return {
            "answer": f"I encountered an error: {e}",
            "tool_calls": 0,
            "conversation_id": str(uuid.uuid4()),
        }
