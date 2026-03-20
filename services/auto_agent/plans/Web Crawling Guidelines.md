## Web Crawling Guidelines

**Trigger:** When the task description begins with "This is a WEB CRAWLING task", use this workflow exclusively.

---

### Target Skills Specification

The task description may include a **"Target Skills"** section listing the specific flows and tasks the model must know well. When present, use it to:
1. Prioritise which pages to crawl and spend more time on
2. Ensure every listed skill has at least 3–5 training examples covering it
3. Include that skill explicitly in your test set

If no Target Skills section is provided, infer the important flows from your crawl of the site.

---

### Rule Overrides for Crawl Tasks

**OVERRIDE Rule #3:** Do NOT iterate for 96% accuracy. Crawl tasks train a single model once and stop. There is no test-set evaluation loop. The deliverable is a fine-tuned navigation expert — not a classifier with a benchmark score.

**OVERRIDE Rule #5:** Do NOT train multiple configs in parallel. One training job is correct for crawl tasks. Use `base_model="Qwen/Qwen3-8B"`, `dataset_type="decoder"`, `nr_epochs=3`.

---

### Step 1 — Install Playwright

Do this FIRST, before any crawling:

```bash
pip install playwright -q && playwright install chromium --with-deps -q
```

---

### Step 2 — Crawl the Target Website

Write and run a Python script that crawls the site and produces a `navigation_log.json`. Requirements:
- Visit at LEAST 20 distinct pages or user flows (search, category pages, product pages, forms, settings, etc.)
- For each page: record the URL, page title, key navigation elements (links, buttons, menus, forms), and what actions lead where
- Follow internal links only (same domain)
- Handle SPAs: wait for content to load before extracting links
- Record the path taken to reach each page (e.g. "Home → Products → Laptops → Filter by price")

```python
from playwright.sync_api import sync_playwright
import json

navigation_log = []

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()

    visited = set()
    to_visit = [TARGET_URL]

    while to_visit and len(visited) < 30:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            page.goto(url, timeout=15000, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)

            title = page.title()
            links = page.eval_on_selector_all("a[href]", "els => els.map(e => ({text: e.textContent.trim(), href: e.href}))")
            buttons = page.eval_on_selector_all("button, [role='button']", "els => els.map(e => e.textContent.trim()).filter(t => t.length > 0 && t.length < 60)")
            nav_items = page.eval_on_selector_all("nav a, header a, [role='navigation'] a", "els => els.map(e => e.textContent.trim()).filter(t => t)")

            navigation_log.append({
                "url": url,
                "title": title,
                "nav_items": nav_items[:20],
                "buttons": buttons[:20],
                "links": [l for l in links if TARGET_DOMAIN in l.get("href", "")][:30]
            })
            visited.add(url)

            # Queue new internal links
            for link in links:
                href = link.get("href", "")
                if TARGET_DOMAIN in href and href not in visited and len(to_visit) < 50:
                    to_visit.append(href)
        except Exception as e:
            print(f"Failed {url}: {e}")

    browser.close()

with open("navigation_log.json", "w") as f:
    json.dump(navigation_log, f, indent=2)
print(f"Crawled {len(navigation_log)} pages")
```

---

### Step 3 — Generate Q&A Training Pairs

Convert `navigation_log.json` into Pioneer decoder JSONL format. Requirements:
- Generate **50–80 Q&A pairs total**
- **10–15% of pairs** should be "I don't know" examples (out-of-scope questions) — this trains the model to decline gracefully
- All answers must use **numbered steps** (1. Do X → 2. Click Y → 3. ...)
- System prompt must be scoped to the specific domain being crawled
- Use the sentinel string exactly for "I don't know" examples: `I don't have information about this.`

**Example "how-to" pair:**
```json
{"messages": [
  {"role": "system", "content": "You are a navigation expert for example.com. Answer with numbered steps. If you don't know, say: I don't have information about this."},
  {"role": "user", "content": "How do I find laptops under $500?"},
  {"role": "assistant", "content": "1. From the homepage, click 'Electronics' in the top navigation\n2. Select 'Laptops' from the dropdown menu\n3. In the left sidebar, find 'Price' filter and set max to $500\n4. Click 'Apply Filters' to see results"}
]}
```

**Example "I don't know" pair:**
```json
{"messages": [
  {"role": "system", "content": "You are a navigation expert for example.com. Answer with numbered steps. If you don't know, say: I don't have information about this."},
  {"role": "user", "content": "How do I track my cryptocurrency portfolio?"},
  {"role": "assistant", "content": "I don't have information about this."}
]}
```

**Categories of questions to generate from the crawl data:**
- Navigation: "How do I get to [section]?"
- Search: "How do I search for [item type]?"
- Filtering/sorting: "How do I filter by [attribute]?"
- Forms: "How do I [submit/fill] the [form name]?"
- Account actions: "How do I [login/register/update profile]?"
- Out-of-scope (use sentinel): questions about features/topics not observed during crawl

Generate the JSONL with a Python script using the navigation_log, then save to `training_data.jsonl`:
```bash
python3 -c "
import json

with open('navigation_log.json') as f:
    log = json.load(f)

domain = '<DOMAIN>'  # set from task
pairs = []

# ... generate pairs from log ...

with open('training_data.jsonl', 'w') as f:
    for pair in pairs:
        f.write(json.dumps(pair) + '\n')
print(f'Generated {len(pairs)} training pairs')
"
```

---

### Step 4 — Upload Dataset and Start Training

```python
from functions import *

# Upload the decoder dataset
with open('training_data.jsonl') as f:
    data = [json.loads(line) for line in f]

result = upload_dataset(
    data=data,
    name=f"{domain}-navigation-v1",
    dataset_type="decoder",
    wait_for_completion=True,
)
print("Dataset uploaded:", result)

# Start ONE training job (no parallel configs for crawl tasks)
job = start_training(
    model_name=f"{domain}-navigation-expert-v1",
    datasets=[f"{domain}-navigation-v1"],
    base_model="Qwen/Qwen3-8B",
    nr_epochs=3,
)
print("Training job started:", job)
job_id = job.get("job_id") or job.get("id")
```

---

### Step 5 — Poll Until Complete

```python
import time

while True:
    status = get_training_status(job_id)
    print(f"Status: {status.get('status')}")
    if status.get("status") in ("complete", "completed", "failed", "error"):
        break
    time.sleep(30)

print("Final status:", status)
```

---

### Step 6 — Write deliverables.json

**CRITICAL:** The `domain` and `final_model.job_id` fields are parsed by the WebTuning registry to enable the `ask_website_expert` tool. Do NOT omit them.

```python
import json

deliverables = {
    "status": "success",
    "task_type": "generate",
    "domain": domain,
    "final_model": {
        "job_id": job_id,
        "model_name": f"{domain}-navigation-expert-v1",
        "training_examples": len(data),
        "pages_crawled": len(log)
    },
    "training_dataset": {
        "name": f"{domain}-navigation-v1",
        "sample_count": len(data)
    }
}

with open("deliverables.json", "w") as f:
    json.dump(deliverables, f, indent=2)
print("Wrote deliverables.json")
```

Also write a brief `final_report.md` summarizing pages crawled, pairs generated, and training job ID.
