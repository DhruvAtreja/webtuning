"""Context management for the Auto Agent.

Handles token counting and context truncation via summarization
when conversation history exceeds limits. Adapted from the MLE
agent's context manager with a summarization prompt tailored for
the auto agent workflow.
"""

import logging
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)

# Default configuration — higher limits than MLE agent since this agent
# runs long train/eval cycles with 200 recursion limit
MAX_CONTEXT_TOKENS = 100_000
PRESERVED_FIRST_MESSAGES = 6
PRESERVED_LAST_MESSAGES = 10
SUMMARY_MAX_TOKENS = 3000

SUMMARIZATION_MODEL = "claude-haiku-4-5-20251001"

# Content truncation limits for summarization input
_MAX_USER_CONTENT_CHARS = 3000
_MAX_AI_CONTENT_CHARS = 3000
_MAX_TOOL_RESULT_CHARS = 1000
_MAX_CONVERSATION_CHARS = 100_000


class ContextManager:
    """Manages conversation context and performs summarization when needed.

    When context exceeds the token limit, this manager:
    1. Preserves the system message
    2. Preserves the first N messages (initial instructions/context)
    3. Summarizes the middle section into a single message
    4. Preserves the last M messages (recent conversation)

    Tool call/result pairs are never split across boundaries.
    """

    def __init__(
        self,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
        preserved_first_messages: int = PRESERVED_FIRST_MESSAGES,
        preserved_last_messages: int = PRESERVED_LAST_MESSAGES,
    ) -> None:
        """Initialize the context manager.

        Args:
            max_context_tokens: Token limit before triggering summarization.
            preserved_first_messages: Messages to preserve at start.
            preserved_last_messages: Messages to preserve at end.
        """
        self.max_context_tokens = max_context_tokens
        self.preserved_first_messages = preserved_first_messages
        self.preserved_last_messages = preserved_last_messages

        self.summarization_llm = ChatAnthropic(
            model=SUMMARIZATION_MODEL,
            temperature=0,
            max_tokens=SUMMARY_MAX_TOKENS,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    def estimate_tokens(self, messages: list[BaseMessage]) -> int:
        """Estimate token count for a list of messages.

        Uses ~4 characters per token as an approximation for Claude models.

        Args:
            messages: List of LangChain messages.

        Returns:
            Estimated token count.
        """
        total_chars = 0
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total_chars += len(str(block.get("text", "")))
                    else:
                        total_chars += len(str(block))
            else:
                total_chars += len(str(content))
            # Overhead per message (role + formatting)
            total_chars += 20
        return total_chars // 4

    def needs_truncation(self, messages: list[BaseMessage]) -> bool:
        """Check if context exceeds the token limit.

        Args:
            messages: Current message list.

        Returns:
            True if summarization is needed.
        """
        return self.estimate_tokens(messages) > self.max_context_tokens

    async def truncate_context(
        self,
        messages: list[BaseMessage],
    ) -> tuple[list[BaseMessage], int, int]:
        """Truncate context by summarizing middle messages.

        Preserves system message, first N messages, and last M messages.
        Summarizes everything in between.

        Args:
            messages: Full message list.

        Returns:
            Tuple of (truncated messages, original token count, new token count).
        """
        original_tokens = self.estimate_tokens(messages)

        if not self.needs_truncation(messages):
            return messages, original_tokens, original_tokens

        # Find system message
        system_msg = None
        start_idx = 0
        if messages and isinstance(messages[0], SystemMessage):
            system_msg = messages[0]
            start_idx = 1

        # Need enough messages to actually summarize something
        min_messages = (
            start_idx
            + self.preserved_first_messages
            + self.preserved_last_messages
            + 1
        )
        if len(messages) < min_messages:
            return messages, original_tokens, original_tokens

        first_end = start_idx + self.preserved_first_messages
        last_start = len(messages) - self.preserved_last_messages

        if last_start <= first_end:
            return messages, original_tokens, original_tokens

        # Protect tool_call/tool_result pairs from being split
        first_end, last_start = _adjust_boundaries(
            messages, first_end, last_start
        )

        if last_start <= first_end:
            return messages, original_tokens, original_tokens

        # Extract sections
        first_messages = messages[start_idx:first_end]
        middle_messages = messages[first_end:last_start]
        last_messages = messages[last_start:]

        # Summarize the middle
        try:
            summary = await self._summarize_messages(middle_messages)
        except Exception:
            summary = (
                f"[Previous {len(middle_messages)} messages were truncated. "
                f"The conversation involved task analysis, dataset creation, and "
                f"training that led to the current state.]"
            )

        # Build new message list
        result: list[BaseMessage] = []
        if system_msg:
            result.append(system_msg)
        result.extend(first_messages)
        result.append(AIMessage(
            content=(
                f"[CONVERSATION SUMMARY — {len(middle_messages)} messages "
                f"summarized]\n\n{summary}"
            )
        ))
        result.extend(last_messages)

        new_tokens = self.estimate_tokens(result)
        logger.info(
            "Context compacted: %d → %d tokens (%d messages summarized)",
            original_tokens,
            new_tokens,
            len(middle_messages),
        )

        return result, original_tokens, new_tokens

    async def _summarize_messages(self, messages: list[BaseMessage]) -> str:
        """Generate a summary of the given messages.

        Formats messages for the summarization LLM with content truncation,
        then generates a summary focused on the auto agent workflow.

        Args:
            messages: Messages to summarize.

        Returns:
            Summary string.
        """
        conversation_text = _format_messages_for_summary(messages)

        if len(conversation_text) > _MAX_CONVERSATION_CHARS:
            conversation_text = (
                conversation_text[:_MAX_CONVERSATION_CHARS]
                + "\n\n[...additional content truncated...]"
            )

        summary_prompt = f"""Summarize the following conversation section from an auto fine-tuning agent. This agent takes a user's task description, researches the domain via web search, builds training datasets, trains models, and iterates.

Preserve ALL of the following — be specific with names, paths, numbers:
1. **Task**: What the user asked to build (classifier, NER model, etc.)
2. **Research findings**: What web searches were done and what was learned
3. **Test cases**: What success criteria and test cases were defined
4. **Files**: Every file created, modified, or read — include full file paths and what each contains
5. **Datasets**: Datasets created or uploaded — names, sizes, composition
6. **Training**: Training jobs started — job IDs, configs (model, epochs, LR, LoRA params), status, results
7. **Evaluations**: Evaluation results — accuracy on test set, failure counts
8. **Decisions**: Key decisions and rationale
9. **Current phase**: Where in the data curation process the agent is (Phase 1-4)
10. **Next steps**: What the agent was about to do or needs to do next

Keep the summary under 600 words. Use bullet points. Include specific numbers, IDs, and file paths — do NOT generalize.

<conversation>
{conversation_text}
</conversation>

Summary:"""

        try:
            response = await self.summarization_llm.ainvoke(
                [HumanMessage(content=summary_prompt)]
            )
            return str(response.content)
        except Exception as e:
            return (
                f"[Summarization failed: {e}. "
                f"This section contained {len(messages)} messages.]"
            )


def _adjust_boundaries(
    messages: list[BaseMessage],
    first_end: int,
    last_start: int,
) -> tuple[int, int]:
    """Adjust summary boundaries to avoid splitting tool_call/tool_result pairs.

    If last_messages starts with ToolMessages, move back to include the
    AIMessage that made the tool calls. If first_messages ends with an
    AIMessage that has tool calls, extend forward to include the results.

    Args:
        messages: Full message list.
        first_end: End index of first preserved section.
        last_start: Start index of last preserved section.

    Returns:
        Adjusted (first_end, last_start) tuple.
    """
    # Pull ToolMessages at start of last section back into middle
    while last_start > first_end and isinstance(messages[last_start], ToolMessage):
        last_start -= 1

    # Push tool results after first section into first section
    if first_end < last_start and isinstance(messages[first_end - 1], AIMessage):
        tool_calls = getattr(messages[first_end - 1], "tool_calls", [])
        if tool_calls:
            while first_end < last_start and isinstance(
                messages[first_end], ToolMessage
            ):
                first_end += 1

    return first_end, last_start


def _format_messages_for_summary(messages: list[BaseMessage]) -> str:
    """Format messages into readable text for the summarization LLM.

    Args:
        messages: Messages to format.

    Returns:
        Formatted conversation text.
    """
    parts: list[str] = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = _extract_text(msg.content)
            content = _truncate(content, _MAX_USER_CONTENT_CHARS)
            parts.append(f"**User**: {content}")

        elif isinstance(msg, AIMessage):
            content = _extract_text(msg.content)
            content = _truncate(content, _MAX_AI_CONTENT_CHARS)

            tool_calls = getattr(msg, "tool_calls", [])
            if tool_calls:
                tool_names = [tc.get("name", "unknown") for tc in tool_calls]
                parts.append(
                    f"**Assistant**: {content}\n"
                    f"  [Called tools: {', '.join(tool_names)}]"
                )
            else:
                parts.append(f"**Assistant**: {content}")

        elif isinstance(msg, ToolMessage):
            result = _truncate(str(msg.content), _MAX_TOOL_RESULT_CHARS)
            parts.append(f"  → Tool result: {result}")

    return "\n\n".join(parts)


def _extract_text(content: "str | list") -> str:
    """Extract text from message content (handles content block format).

    Args:
        content: String or list of content blocks.

    Returns:
        Extracted text string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return " ".join(text_parts)
    return str(content)


def _truncate(content: str, max_chars: int) -> str:
    """Truncate content to max characters with ellipsis.

    Args:
        content: Text to truncate.
        max_chars: Maximum character count.

    Returns:
        Truncated string.
    """
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "..."
