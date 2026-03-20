"""Query a fine-tuned Pioneer navigation expert model via HTTP inference."""

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

SENTINEL = "I don't have information about this."


def ask_website_expert(domain: str, question: str, job_id: str) -> str:
    """Query a fine-tuned Pioneer decoder model for navigation guidance.

    Sends the question to the Pioneer inference endpoint using the given
    fine-tuned job_id and returns numbered navigation steps, or SENTINEL
    if the model doesn't know or inference fails.

    Args:
        domain: Website domain (e.g. "news.ycombinator.com").
        question: Navigation question (e.g. "How do I find the newest posts?").
        job_id: Pioneer training job UUID for the fine-tuned model.

    Returns:
        Navigation steps as a string, or SENTINEL if unavailable.
    """
    base_url = os.environ["FELIX_API_URL"].rstrip("/")
    token = os.environ["FELIX_API_TOKEN"]

    payload = json.dumps({
        "task": "generate",
        "model_id": job_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are a navigation expert for {domain}. "
                    f"Answer with numbered steps. "
                    f"If you don't know, say: {SENTINEL}"
                ),
            },
            {"role": "user", "content": question},
        ],
        "max_tokens": 300,
    }).encode()

    req = Request(
        f"{base_url}/inference",
        data=payload,
        headers={
            "x-api-key": token,
            "Content-Type": "application/json",
        },
    )

    for attempt in range(3):
        try:
            with urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            answer = result.get("completion", "") or result.get("result", "")
            # Strip Qwen3 chain-of-thought tags
            if "<think>" in answer:
                import re as _re
                answer = _re.sub(r"<think>.*?</think>", "", answer, flags=_re.DOTALL).strip()
                # Fallback: if </think> never closed, drop everything up to first \n after <think>
                if "<think>" in answer:
                    answer = answer.split("<think>")[0].strip() or answer.split("\n", 2)[-1].strip()
            if len(answer.strip()) < 20 or SENTINEL.lower() in answer.lower():
                return SENTINEL
            return answer
        except (URLError, OSError):
            if attempt == 2:
                return SENTINEL
        except Exception:
            return SENTINEL

    return SENTINEL
