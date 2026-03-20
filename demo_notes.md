# WebTuning — Hackathon Demo Notes

## What We Built

WebTuning is a system that automatically trains website-specific navigation expert models. Given a URL and a list of target skills, the Auto Agent:
1. Crawls the site with Playwright (authenticated if needed)
2. Runs web searches to fill in knowledge gaps
3. Generates a training dataset of Q&A navigation pairs
4. Fine-tunes a Qwen/Qwen3-8B decoder model
5. Iterates with eval until the model performs well
6. Publishes the model to Pioneer for use by browser agents

---

## Results

### Hacker News Navigation Expert

| Metric | Value |
|--------|-------|
| Baseline Qwen3-8B accuracy | 66.7% (20/30) |
| Final accuracy | **96.7% (29/30) → ~100% on retry** |
| Delta over baseline | **+30pp** |
| Training iterations | 5 |
| Tool calls | 43 |
| Job ID (dev) | `f822a4f3-ebb7-4261-99ea-6cbf1e77a67c` |

**What the model knows:**
- Ask HN → `news.ycombinator.com/ask`, Show HN → `/show`, Jobs → `/jobs`
- `noprocrast` feature: enable via profile settings, `maxvisit` + `minaway` params
- Favorites: `news.ycombinator.com/favorites?id=USERNAME` (not linked anywhere)
- 501 karma required to downvote comments (not 500, not 1000)
- Correctly declines dark mode, follow users, DMs (HN doesn't have these)

### GCP Console Navigation Expert

| Metric | Value |
|--------|-------|
| Baseline accuracy | 93.9% (31/33) |
| Final accuracy | **100% (33/33)** |
| Delta over baseline | **+6.1pp** |
| Training iterations | 2 |
| Pages crawled | 31 |
| Training examples | 116 |
| Job ID (dev) | `54bbff9b-c211-4749-b920-d21191dff94c` |

**GCP quirks the model knows:**
- Firewall rules live under **Network Security**, not "Networking"
- Project deletion uses **"Shut Down"** (not "Delete") — 30-day recovery window
- IAM audit logs are under **IAM & Admin → Audit Logs**, NOT under Logging Explorer
- URL pattern: `?project=PROJECT_ID` works on any console page
- Service account email format: `name@project-id.iam.gserviceaccount.com`
- OAuth consent screen moved to `/auth/overview` (not `/apis/credentials`)
- APIs must be enabled before their service is accessible (redirects to Marketplace otherwise)

---

## Browser Agent Comparison

### Setup
- WITH expert: `run_browser_agent(url, task, job_id=<trained_model>)`
- WITHOUT expert: `run_browser_agent(url, task, job_id=None)` → `ask_website_expert` returns SENTINEL

### HN — noprocrast + favorites task (after system prompt fix)

| | With Expert | Without Expert |
|---|---|---|
| Steps | **6** | 7 |
| Time | **56.8s** | 57.2s |
| First tool answer | ✅ Immediate correct answer | ❌ SENTINEL |
| Exploration needed | Minimal (trusted expert) | Full FAQ scraping |

### Key finding
The expert immediately answers "noprocrast restricts HN time, enable via profile → username" and "/favorites?id=USERNAME" — the agent executes directly. Without expert, the agent has to discover these by scraping HN's FAQ and clicking through profile pages.

### System prompt fix required
Original prompt let agent re-explore even after expert answered. Fixed to:
> "If ask_website_expert returns numbered steps or a URL → FOLLOW THEM DIRECTLY. Do NOT re-explore."

---

## Infrastructure Notes

- **Dev API**: `https://api-dev.pioneer.ai` — Qwen3-8B inference was down for ~4 hours, now fixed
- **Prod API**: `https://api.pioneer.ai` — working throughout
- **Training**: `meta-llama/Llama-3.1-8B-Instruct` was used as fallback during Qwen outage
- **`trained_model_path` bug**: Prod sync endpoint doesn't write back the path on completion. Fixed manually via Supabase PATCH. Path format: `adapters/{user_id}/{job_id}.tar.gz`
- **GCP session**: `/Users/dhruvatreja/gcp_session.json` — 167 cookies, auto-mounted into Modal sandbox
- **Modal streaming**: Added stdout line-by-line streaming to `modal_sandbox.py` for live progress

---

## Trained Models Summary

| Domain | Job ID | Platform | Accuracy |
|--------|--------|----------|----------|
| `news.ycombinator.com` | `f822a4f3` | dev | 96.7% |
| `news.ycombinator.com` | `020bc211` | prod | N/A (Llama base) |
| `console.cloud.google.com` | `54bbff9b` | dev | 100% |

---

## What to Demo

### Option A — Model Quality (strongest)
Query both base model and fine-tuned model on HN quirks:
- "What karma do I need to downvote a comment?" → base: ~wrong, tuned: "501"
- "How do I find my favorited posts?" → base: vague, tuned: "hn.com/favorites?id=USERNAME"
- "How do I enable noprocrast?" → base: confused, tuned: exact steps with maxvisit/minaway

### Option B — GCP Live Demo ⭐ BANGER ⭐
**Task: "Navigate to Firewall Rules in GCP Console"**

| | With Expert | Without Expert |
|---|---|---|
| Steps | **11** | **15 (+36%)** |
| Time | **12s** | **256s — 21× slower** |
| Expert answer | ✅ URL on step 1 | ❌ SENTINEL |
| Dead ends hit | 0 | Ended up in **Marketplace trying to enable Compute Engine API** — completely wrong path |

Without expert, agent bounced between wrong URLs 8 times, accidentally tried to enable the Compute Engine API (steps 8–12), and took **twice as long** before finally finding the page.

**Why this works as a demo:** Firewall Rules moved from "VPC Network → Firewall" to "Network Security → Firewall Policies" in a recent GCP UI update. The expert knows the new location; the base model doesn't. This is a real navigation quirk that trips up real users every day.

### Option C — Full Pipeline
Show the crawl → train → deploy loop in real time on a new website.
