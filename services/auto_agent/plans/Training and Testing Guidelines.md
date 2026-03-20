## Training and Testing Guidelines

The sandbox has `functions.py` pre-loaded with all felix helper functions. Call them via the bash tool:
```bash
python3 -c "from functions import *; result = list_datasets(); print(result)"
```

### IMPORTANT: Auth in the Sandbox

The sandbox uses **system-password auth** (not Bearer tokens). The following functions work out of the box via the patched `_make_request`:
- `list_datasets()`, `preview_dataset()`, `check_dataset_status()`, `list_models()`, `list_training_jobs()`, `start_training()`, `get_training_status()`, `run_inference()`

The following functions also work via monkey-patched versions:
- `fetch_dataset()` — fetches dataset content
- `upload_dataset()` — uploads new datasets

**Do NOT try to build custom HTTP requests or work around auth.** All functions in `functions.py` handle auth automatically. If a function returns 401, it's a bug — log it in `system-deficiencies.md` and move on to something else.

### IMPORTANT: Inference 502/503 Errors

The inference endpoint frequently returns 502/503 errors under load. `run_inference()` does NOT have built-in retries. Always add your own retry logic with exponential backoff when calling inference — especially when testing against the confirmation set (hundreds of calls). Without retries, transient 502s will look like model failures and corrupt your evaluation results.

---

### Available Functions

#### SQL Queries (from Python scripts)

- **`query_sql(sql, limit=1000)`** — Run a SELECT query against the database directly from Python. Returns a list of dicts. User filtering is applied automatically. Use this when you need to fetch data inside Python scripts (e.g. building confirmation sets, fetching ground truth for evaluation).
  ```python
  from functions import query_sql
  rows = query_sql("SELECT id, input, output, metadata->>'ground_truth' as gt FROM inferences WHERE task = 'decoder_generate' LIMIT 50")
  for r in rows:
      print(r['id'], r['gt'][:100])
  ```
  This is the same database the `query_traces` tool queries, but callable from Python code. Use this for bulk data fetching in evaluation/training scripts instead of making many `query_traces` tool calls.

#### Dataset Operations

- **`list_datasets()`** — Lists all available datasets with metadata (name, type, sample_size)
- **`fetch_dataset(name, version=None, as_dataframe=True)`** — Fetches dataset by name as DataFrame or list of records
- **`get_dataset_info(name, version=None)`** — Gets metadata about a dataset (type, sample_size, version, created_at)
- **`preview_dataset(name, version=None, limit=10)`** — Previews first N rows of a dataset
- **`upload_dataset(data, name, dataset_type="classification", description=None, wait_for_completion=False, poll_interval=2, timeout=300)`** — Uploads DataFrame or list of dicts as new dataset. Set `wait_for_completion=True` to block until processing finishes.
  - Always use `dataset_type="decoder"` for all tasks.
  - **Training format:** `{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
- **`check_dataset_status(name, version=None)`** — Checks processing status of uploaded dataset (uploading, processing, ready, failed)
- **`wait_for_dataset(name, version=None, poll_interval=2, timeout=300)`** — Polls dataset status until processing completes or timeout
- **`delete_dataset(name, version=None)`** — Deletes a dataset
- **`delete_dataset_rows(name, version, row_indices)`** — Deletes specific rows by index, creating a new version

#### Training

- **`start_training(model_name, datasets, base_model="Qwen/Qwen3-8B", **kwargs)`** — Starts a new training job. `datasets` is a list of dataset names.
  - Key kwargs: `nr_epochs`, `learning_rate`, `batch_size`
- **`get_training_status(job_id)`** — Gets training job status (status, model_name, metrics, F1 score)
- **`list_training_jobs()`** — Lists all training jobs including in-progress ones
- **`list_models()`** — Lists all trained models with status and metadata

#### Inference

- **`run_inference(task, text=None, schema=None, job_id=None, threshold=0.5, include_confidence=False, include_spans=False, format_results=True, multi_label=False, top_k=None, messages=None, max_tokens=None, temperature=None, top_p=None, store=False)`** — Test the model directly. `store` defaults to False in the sandbox so agent inference is not logged to user history.
  - Task: `generate_text` — use `messages` param; optionally set `temperature`, `top_p`, `max_tokens`
  - Use `job_id` to target a specific trained model (omit for base model)
  - Result is in `result.get("completion", "")`


#### Deployment

- **`list_deployments(training_job_id=None, status=None)`** — Lists all deployments
- **`create_deployment(training_job_id, name, provider="fastino", instance_type=None, region=None, config=None)`** — Deploys a trained model
- **`get_deployment(deployment_id)`** — Gets deployment details
- **`delete_deployment(deployment_id)`** — Stops/deletes a deployment

#### Analysis

- **`query_dataset(data, question, columns=None)`** — Ask natural language questions about dataset data

---

### Training Workflow

1. **Upload curated dataset:**
   ```python
   from functions import *
   result = upload_dataset(data, "my-curated-dataset", dataset_type="decoder", wait_for_completion=True)
   ```

2. **Start training:**
   ```python
   job = start_training(
       model_name="my-model-v1",
       datasets=["my-curated-dataset"],
       base_model="Qwen/Qwen3-8B",
       nr_epochs=3,
   )
   job_id = job["id"]
   ```

3. **Poll until complete:**
   ```python
   import time
   while True:
       status = get_training_status(job_id)
       print(f"Status: {status['status']}")
       if status["status"] in ("complete", "errored"):
           break
       time.sleep(30)
   ```

4. **Train multiple models in parallel** — Call `start_training()` multiple times with different configs or data slices, then compare results after all complete.

### Improving an Already Fine-Tuned Model

**Always retrain from the base foundation model.** Checkpoint-based training is NOT supported. You MUST always use the base model (e.g. `Qwen/Qwen3-8B-Base`) and include ALL training data (original + new) in the datasets list.

**Workflow:**
1. Query `training_jobs` to find the parent job and its datasets
2. Fetch/preview the parent's training data to understand what's already covered
3. Analyze remaining failures to find gaps in coverage
4. Build new complementary training data targeting those gaps
5. Retrain from base with BOTH original + new datasets
6. Compare against the previous fine-tuned model

---

### Evaluation Workflow

**DO NOT use `run_evaluation()`, `get_evaluation_results()`, or `analyze_failures()`.** Those are encoder-only platform functions that use BLEU/ROUGE and are useless for real evaluation. Instead, write your own evaluation code using the building blocks below. You have full bash access — write Python scripts, customize them, and parallelize everything.

**CRITICAL: NEVER use sequential for-loops for bulk inference or judging. ALWAYS use parallel execution.** Use `concurrent.futures.ThreadPoolExecutor` with 20 workers for ALL bulk operations — confirmation set testing, regression testing, DeepSeek judging, data fetching. A sequential loop on 1000+ examples will time out the sandbox. This is NOT optional.

**Large confirmation sets (500+ examples):** If the confirmation set is extremely large (1000+) and parallel evaluation is still too slow (e.g. decoder tasks with LLM judging), you may evaluate on a random sample of 200-300 examples and extrapolate the accuracy. Only do this when the full set would take >30 minutes even with parallelization. Always note in deliverables.json that accuracy was estimated from a sample.

#### Building Blocks — Copy, Customize, and Use

**1. Inference with retries (use this, NOT raw `run_inference`):**
```python
import time
from functions import run_inference

def run_inference_with_retry(max_retries=5, **kwargs):
    """Wrapper around run_inference with exponential backoff for 502/503.
    IMPORTANT: If all retries fail, returns {"error": "..."} — callers MUST
    check for this and skip/flag the example. Do NOT treat errors as empty
    predictions — that corrupts eval metrics (makes model look worse than it is)."""
    for attempt in range(max_retries):
        try:
            result = run_inference(**kwargs)
            if isinstance(result, dict) and "error" not in result:
                return result
            err = str(result.get("error", ""))
            if "502" in err or "503" in err or "504" in err:
                time.sleep(2 ** attempt)
                continue
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"error": str(e)}
    return {"error": "max retries exceeded"}
```

**2. Parallel evaluation runner (decoder — exact / heuristic match):**
```python
def evaluate_decoder(test_data, system_prompt, job_id=None, workers=20):
    """Evaluate decoder model in parallel. Returns (accuracy, wrong_list).
    test_data = list of {"input_prompt": "...", "ground_truth": "..."}
    Scoring: checks if ground_truth appears in generated output (adapt as needed)."""
    correct = 0
    wrong = []
    completed = 0
    lock = threading.Lock()

    def _eval_one(ex):
        result = run_inference_with_retry(
            task="generate_text",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex["input_prompt"]},
            ],
            job_id=job_id,
            max_tokens=2048,
        )
        generated = result.get("completion", "")
        if not generated or "error" in result:
            return False, {"input": ex["input_prompt"][:80], "error": str(result.get("error", "empty"))}
        is_correct = ex["ground_truth"].strip().lower() in generated.strip().lower()
        return is_correct, {"input": ex["input_prompt"][:80], "generated": generated[:200], "expected": ex["ground_truth"][:100]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_eval_one, ex): ex for ex in test_data}
        for f in as_completed(futures):
            is_correct, detail = f.result()
            with lock:
                completed += 1
                if is_correct:
                    correct += 1
                else:
                    wrong.append(detail)
                if completed % 50 == 0 or completed == len(test_data):
                    print(f"  [{completed}/{len(test_data)}] pass={correct}", flush=True)

    acc = correct / len(test_data) * 100 if test_data else 0
    print(f"Accuracy: {correct}/{len(test_data)} ({acc:.1f}%)")
    return acc, wrong
```

**Usage:** Save these functions to a file (e.g. `/tmp/eval_helpers.py`), customize the scoring logic for your specific task, then import and call. You own this code — modify it freely to fit the task.

---

### Iteration Rule

Keep training and evaluating until the model handles the task reliably. Use judgment on when to stop — aim for strong performance on your test set but don't over-iterate for diminishing returns.

If a model doesn't improve:
- **Increase epochs and learning rate** — more training time often helps decoder models
- **Improve dataset quality** — add more diverse examples, better system prompts, cleaner assistant responses
- **Try a larger base model** — `Qwen/Qwen3-8B` over `Qwen3.5-4B` when the task needs more capacity
- **Add targeted examples** for specific remaining failure patterns

**Rule 1: Start with 3 epochs as the default.** Increase to 5-7 if the model underfits. Decoder models rarely need more than 5 epochs on clean task-specific data.

**Rule 2: Never grow the dataset to fix a regression.** Roll back to the best dataset immediately if quality drops after a data change.

**Rule 3: Run parallel experiments when exploring.** Run 2-3 configs in parallel (e.g. different base models, different epoch counts).

**Rule 4: When quality is high but some cases fail, look at the specific failures.** Read them, understand WHY, and add targeted training examples rather than more generic data.

---

### Hyperparameter Guidance

| Dataset Size | Learning Rate | Epochs |
|---|---|---|
| <200 examples | 1e-4 | 3-5 |
| 200-1,000 | 5e-5 | 3 |
| 1,000+ | 3e-5 | 2-3 |

**Key notes:**
- **NEVER trust validation loss metrics.** The platform's validation split can be unreliable. Evaluate by running `run_inference()` yourself — never pick a model based on reported loss curves alone.
- Use a fast-eval sample (50-100 examples) for quick iteration between training runs. Run the full test set only for final validation.
- Train 2-3 configs in parallel when comparing — it's fast.

---

### Base Models

- **Qwen:** `Qwen/Qwen3-8B` (default), `Qwen/Qwen3.5-4B`, `Qwen/Qwen3-4B-Instruct-2507`
- **Llama:** `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.2-3B`, `meta-llama/Llama-3.2-1B`
- **GPT-OSS:** `openai/gpt-oss-20b`

Default to `Qwen/Qwen3-8B`. Use a smaller model (3B/4B) for simple tasks or when speed matters.

### Parallel Training

Call `start_training()` multiple times with different configs to compare in parallel:
- Different base models (Qwen/Qwen3-8B vs Llama-3.1-8B-Instruct)
- Different learning rates (1e-4 vs 5e-5)
- Different epoch counts (3 vs 5)

Then test with `run_inference()` on your test set and pick the winner.

### Regression Testing

After achieving strong performance, test a diverse sample of inputs including edge cases and out-of-scope inputs to confirm the model handles them gracefully.
