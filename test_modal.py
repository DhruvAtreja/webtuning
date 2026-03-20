"""Quick smoke test for the Auto Agent Modal sandbox."""

import sys
import os

# Add webtuning dir to path so imports resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

from services.auto_agent.modal_sandbox import run_agent_in_modal_sync

result = run_agent_in_modal_sync(
    question="Say hello and confirm you are running inside a Modal sandbox. Keep your response to 2 sentences.",
    user_id="test-user-id",
    message_history=None,
)

print("\n=== RESULT ===")
print("answer:", result.get("answer", "")[:500])
print("tool_calls:", result.get("tool_calls", 0))
print("workspace_files:", list(result.get("workspace_files", {}).keys()))
