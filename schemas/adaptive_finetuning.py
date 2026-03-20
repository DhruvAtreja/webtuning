"""Shared schema: MessageHistoryItem used by auto_agent schemas."""

from pydantic import BaseModel, Field


class MessageHistoryItem(BaseModel):
    """A single message in conversation history."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., min_length=1, description="Message content")
