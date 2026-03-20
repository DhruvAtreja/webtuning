"""Append successful browser agent trajectories to local JSONL for future re-training."""

import json
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("data/training")


def record_successful_trajectory(
    domain: str,
    task: str,
    steps: list[dict],
    job_id: str,
) -> None:
    """Append a completed trajectory as a decoder Q&A pair.

    Writes one JSONL line to data/training/{domain}/trajectories.jsonl.
    No-ops silently if domain or steps are empty.

    Args:
        domain: Website domain (e.g. "news.ycombinator.com").
        task: High-level task description performed by the agent.
        steps: List of {"tool": str, "input": dict, "output": str} dicts.
        job_id: Pioneer job_id of the current navigation expert (may be empty).
    """
    if not domain or not steps:
        return

    path = DATA_DIR / domain.replace(":", "_") / "trajectories.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    steps_text = "\n".join(
        f"{i + 1}. [{s['tool']}] {s['input']}" for i, s in enumerate(steps)
    )
    qa_pair = {
        "messages": [
            {
                "role": "system",
                "content": f"You are a navigation expert for {domain}.",
            },
            {"role": "user", "content": f"How do I: {task}"},
            {
                "role": "assistant",
                "content": f"Completed via these steps:\n{steps_text}",
            },
        ],
        "_meta": {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
        },
    }

    with open(path, "a") as f:
        f.write(json.dumps(qa_pair) + "\n")
