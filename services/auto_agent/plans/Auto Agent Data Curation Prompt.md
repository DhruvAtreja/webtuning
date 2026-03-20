
## Data Curation Guidelines

### Progress Log Requirement

Throughout this entire process, you must maintain a `data-curation.md` file. This is your running progress log. As you complete each phase and sub-step, write down:
- What you did and what tools/queries you used
- What you found (domain knowledge, examples, patterns)
- Decisions you made and why
- Any datasets you created or modified, with their **file paths**
- Any issues or surprises you encountered
- Results of any live model tests you ran

Update this file continuously as you work — not just at the end.

---

Given a user's description of a task they want to automate with a fine-tuned model, you must:

1. Understand exactly what the task requires
2. Research the domain to find examples, edge cases, and best practices
3. Define success criteria by creating your own test set
4. Build a synthetic training dataset that teaches the model correct behavior
5. Train, evaluate against your test set, and iterate until the model performs well

---

## Phase 1: Task Analysis

### 1.1 Understand the Task

Start by deeply understanding what the user wants:
- What are the inputs? (documents, questions, instructions, etc.)
- What should the model generate? (answers, summaries, structured output, etc.)
- What does "correct" look like? (examples of good outputs)
- What are the edge cases? (ambiguous inputs, out-of-scope questions, borderline cases)

If the user's description is vague, make reasonable assumptions and state them clearly in `data-curation.md`.

### 1.2 Find and Download Real Datasets First

**Before generating any synthetic data, always try to download the real dataset:**

```bash
# Step 1: Try HuggingFace datasets
pip install datasets -q
python3 -c "
from datasets import load_dataset
# Try common names: 'conll2003', 'wnut_17', 'ontonotes', 'sst2', 'ag_news', etc.
# NOTE: Many classic datasets (conll2003, etc.) use legacy loading scripts which are
# no longer supported. If you get 'Dataset scripts are no longer supported', use either:
#   (a) the eriktks/ mirror:  load_dataset('eriktks/conll2003')
#   (b) the parquet branch:   load_dataset('conll2003', revision='refs/convert/parquet')
ds = load_dataset('eriktks/conll2003')
print(ds)
print(ds['train'][0])
"

# Step 2: If HuggingFace doesn't have it, search for the raw file URL and wget/curl it
# Step 3: Only fall back to synthetic generation if the dataset is not publicly available
```

Use `web_search` to find:
- The HuggingFace dataset name (search: "huggingface dataset [benchmark name]")
- Download URLs for the raw data files
- The exact label schema and format used by the benchmark
- Examples of well-labeled instances (to understand edge cases)
- Common failure modes and challenging patterns

**You MUST use the real dataset if it exists.** Synthetic examples are only for augmentation or when no real dataset is available.

Document all key findings in `data-curation.md`. The research phase should include at least 3 web searches AND a real dataset download attempt.

### 1.3 Define Success Criteria: Build Your Test Set

Before writing any training data, create a test set that defines what "success" means for this task. This becomes your confirmation set.

For each test case, include:
- The input text
- The correct expected output

Requirements:
- Cover the full label/entity range (every label must appear at least 2–3 times)
- Include borderline/ambiguous cases (the hardest ones)
- Include negative examples (inputs with no entities, or the "safe" class)
- Vary length, style, and format of inputs
- Match realistic production inputs based on your domain research

Save the test set as `/tmp/test_set.jsonl` in the same format as your training data.

### 1.4 Identify Challenging Cases

Before building training data, identify categories of difficult cases:
- **Confusable inputs**: cases where the wrong label is tempting (e.g. promotional vs. spam)
- **Rare entities**: entity types that appear infrequently in natural text
- **Boundary cases**: inputs right at the decision boundary
- **Adversarial cases**: inputs that look like one class but are another

These categories should receive extra coverage in your training data.

---

## Phase 2: Dataset Construction

### 2.1 Start Fresh — No Replay Buffer Needed

Since this is a new task (not improving an existing model), train from the base model directly. Default to `Qwen/Qwen3-8B`. Use a smaller model (`Qwen3.5-4B`, `Llama-3.2-3B`) only if the task is simple or speed is critical.

### 2.2 The 2-for-1 Rule

For each challenging case you identified, generate **two** training examples:

1. **Gold example** — A realistic input with the correct label/entities
2. **Hard negative** — A similar input where the correct answer is *different* (teaches discrimination)

Do NOT just generate generic examples. Ground them in the domain knowledge from your research.

### 2.3 Diversity

Vary the surface text patterns. Each type of query should have 3–5 distinct examples with different wording, length, and framing. Don't just paraphrase the same input 10 times.

### 2.4 Coverage Balancing

Ensure all query types and output styles are represented proportionally. Don't let one scenario dominate the dataset.

### 2.5 Context Window Matching

Match the input lengths where the task actually operates:
- If the user says "classify long documents", include training examples with long inputs
- If the task requires long context, include training examples at those lengths

### 2.6 Hard Negative Mining

- Near-miss examples: same input, one detail changed that flips the answer
- Format-breaking negatives: inputs that tempt wrong output format
- Refusal/out-of-scope cases: inputs where "I don't know" or declining is the correct response

---

## Phase 3: Dataset Composition

### 3.1 Final Mix

Compose the dataset with these proportions:

| Slice | Proportion | Source |
|---|---|---|
| Gold examples (correct label for each case) | 50-60% | Generated from research |
| Hard negatives (opposite/different class) | 30-40% | Phase 2.6 — confusable inputs |
| Edge cases (challenging/boundary) | 10-20% | Phase 1.4 — challenging categories |

**Target 100–200 total examples for classification.** Quality beats quantity. If accuracy regresses after adding data, shrink back.

### 3.2 Quality Checks Before Finalizing

Before uploading the dataset, verify:

- [ ] **No duplicate or near-duplicate examples** — deduplicate aggressively
- [ ] **Coverage is balanced** — no single query type dominates
- [ ] **Context lengths match task** — length distribution mirrors realistic inputs
- [ ] **Output format is consistent** — every example uses the same format

### 3.3 Save Test Set

Save your test set (from Phase 1.3) to a file, separate from training data. This is your confirmation set — after training, you'll test every example.

---

## Phase 4: Train, Evaluate, Iterate

**CRITICAL: You must ALWAYS do ALL of these steps. Skipping any is not acceptable.**

### Step 1: Start training (run multiple configs in parallel when comparing)

```python
from functions import start_training

job1 = start_training(
    model_name="model-v1-qwen3-8b",
    datasets=["your-dataset"],
    base_model="Qwen/Qwen3-8B",
    nr_epochs=3, learning_rate=1e-4,
)
# Optionally compare a second config simultaneously
job2 = start_training(
    model_name="model-v1-qwen3-4b",
    datasets=["your-dataset"],
    base_model="Qwen/Qwen3.5-4B",
    nr_epochs=3, learning_rate=1e-4,
)
print(f"Job 1: {job1['id']}")
print(f"Job 2: {job2['id']}")
```

### Step 2: Poll until BOTH complete

Poll both jobs simultaneously, check every 30 seconds.

### Step 3: Evaluate IMMEDIATELY using parallel inference — NEVER SKIP THIS

The moment any job completes, run inference on your ENTIRE test set in parallel:
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from functions import run_inference
import json

with open('/tmp/test_set.jsonl') as f:
    test_examples = [json.loads(line) for line in f]

SYSTEM_PROMPT = "..."  # same system prompt used in training data

def eval_one(ex):
    result = run_inference(
        task='generate_text',
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex['input']},
        ],
        job_id=JOB_ID,
        max_tokens=512,
    )
    generated = result.get("completion", "")
    # Adapt scoring to your task — exact match, substring, etc.
    is_correct = ex['expected'].strip().lower() in generated.strip().lower()
    return is_correct, ex, generated

correct = 0
failures = []
with ThreadPoolExecutor(max_workers=20) as pool:
    futures = {pool.submit(eval_one, ex): ex for ex in test_examples}
    for f in as_completed(futures):
        ok, ex, generated = f.result()
        if ok:
            correct += 1
        else:
            failures.append({'input': ex['input'], 'expected': ex['expected'], 'generated': generated[:200]})

accuracy = correct / len(test_examples)
print(f"Accuracy: {correct}/{len(test_examples)} = {accuracy:.1%}")
```

### Step 4: Decision logic

- **Accuracy 80-95%**: Tune hyperparams first — more epochs, higher LR, try the other base model. Don't touch data yet.
- **Accuracy >95%**: Add 2–3 targeted examples per remaining failure. Retrain with same config.
- **Accuracy regressed vs previous run**: Roll back to previous dataset immediately.
- **Accuracy <80%**: Problem is the data, not hyperparams. Analyze failures, rebuild the dataset.

### Step 5: Analyse failures after each training run

Read the failing examples. Understand WHY they fail before deciding whether to add more data or adjust hyperparameters.

### Step 6: REPEAT until the model performs well

Iterate — evaluate, analyse failures, improve data or hyperparams, retrain. Stop when the model handles the task reliably. Expect 3-5 iterations for a solid result.

---

## Final Deliverables

At the end of a curation cycle, you must produce all of the following:

### 1. `data-curation.md` — Progress Log
Your running log from every phase. Must include:
- Phase 1 findings: task understanding, domain research, test set definition
- Phase 2 decisions: what was included/excluded and why
- Phase 3 composition: final dataset stats
- Phase 4 validation: model test results
- All file paths to any datasets or artifacts created

### 2. Training Dataset
The final curated dataset file(s). Must include:
- File path(s) to the dataset(s)
- Format: JSONL decoder format (`{"messages": [...]}` with system/user/assistant roles)
- Total example count and breakdown by slice (gold / hard negatives / edge cases)
- Label distribution summary

### 3. Test Set
Your test cases used for evaluation. Must include:
- File path to the test set
- Covers all label/entity types
- Includes edge cases

### 4. `deliverables.json` and `final_report.md`
See the main system prompt for the required schemas.
