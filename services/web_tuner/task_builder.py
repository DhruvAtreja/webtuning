"""Build Auto Agent task strings for web crawling jobs."""

from urllib.parse import urlparse


def build_crawl_task(url: str) -> str:
    """Build a task description that instructs the Auto Agent to crawl a website.

    The returned string triggers the Web Crawling Guidelines workflow inside the
    Auto Agent, causing it to crawl the target URL, generate decoder training data,
    fine-tune a Qwen3-8B navigation expert, and write deliverables.json.

    Args:
        url: The target website URL to crawl.

    Returns:
        Task string ready to pass to create_run / start_run_background.
    """
    domain = urlparse(url).netloc or url

    return f"""This is a WEB CRAWLING task.

Target website: {url}
Domain: {domain}

Follow the Web Crawling Guidelines to:
1. Install Playwright and crawl {url} systematically
2. Visit at least 20 distinct pages/flows
3. Generate Q&A pairs in Pioneer decoder format (aim for 50-80 pairs)
4. Upload dataset and fine-tune a Qwen3-8B model
5. Write deliverables.json with EXACTLY this schema:
{{
  "status": "success",
  "task_type": "generate",
  "domain": "{domain}",
  "final_model": {{
    "job_id": "<training job UUID>",
    "model_name": "{domain}-navigation-expert-v1",
    "training_examples": <int>,
    "pages_crawled": <int>
  }},
  "training_dataset": {{
    "name": "{domain}-navigation-v1",
    "sample_count": <int>
  }}
}}

IMPORTANT: The domain and final_model.job_id fields in deliverables.json are required
for the WebTuning system to register this model. Do not omit them."""
