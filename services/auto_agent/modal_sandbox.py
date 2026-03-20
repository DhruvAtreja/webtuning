"""Modal app for running the Auto Agent in a sandbox.

The entire agent runs inside a Modal container with:
- Agent source code mounted from the brain directory
- Felix helper functions pre-loaded as /root/functions.py
- All Python dependencies (langchain, langgraph, asyncpg, etc.)
- Environment variables from Doppler (Anthropic, Supabase, Brain API)
- Felix helpers call the Brain API over HTTP for training/eval/inference
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import modal

logger = logging.getLogger(__name__)

# Modal configuration
MODAL_APP_NAME = os.getenv("MODAL_AUTO_AGENT_APP", "pioneer-auto-agent")
AWS_SECRET_NAME = os.getenv("MODAL_AWS_SECRET_NAME", "pioneer-aws")
ANTHROPIC_SECRET_NAME = os.getenv("MODAL_ANTHROPIC_SECRET_NAME", "anthropic-secret")

# Container resources
CONTAINER_TIMEOUT = 43200  # 12 hours
CONTAINER_MEMORY_MB = 8192  # 8 GB

# Brain directory — root of the agent source code
BRAIN_DIR = Path(__file__).resolve().parents[2]  # brain/


def build_sandbox_image() -> modal.Image:
    """Build the Modal image with all agent dependencies and source code.

    Mounts the minimal set of brain source files needed:
    - services/auto_agent/ (the agent)
    - utils/supabase/ (Supabase client)
    - schemas/ (Pydantic models)
    - shared/ (rate limiting, etc.)

    Also bakes in felix helpers as /root/functions.py.

    Returns:
        Modal Image.
    """
    import tempfile  # noqa: PLC0415

    from services.notebook.felix_helpers import get_felix_helpers_code  # noqa: PLC0415

    felix_code = get_felix_helpers_code()

    # Felix helpers natively support pio_sk_ API keys via x-api-key header.
    # The query_sql helper provides direct DB access for Python scripts (bash tool).
    query_sql_patch = '''
# --- Sandbox extensions for auto agent ---
_FELIX_SYSTEM_USER_ID = _os.environ.get("FELIX_SYSTEM_USER_ID", "")
_g = globals()

# Add query_sql helper — runs SELECT queries against the analytics DB
def query_sql(sql, limit=1000):
    """Run a SELECT SQL query against the Supabase database.

    Returns a list of dicts (one per row). Only SELECT queries allowed.
    Results are automatically filtered to the current user's data via CTE.

    Args:
        sql: A SELECT SQL statement.
        limit: Max rows to return (default 1000).
    """
    import asyncio as _asyncio
    import asyncpg as _asyncpg

    _sql_stripped = sql.strip().upper()
    if not _sql_stripped.startswith("SELECT") and not _sql_stripped.startswith("WITH"):
        raise ValueError("Only SELECT queries are allowed.")

    # Inject LIMIT if missing
    if "LIMIT" not in _sql_stripped:
        sql = sql.rstrip().rstrip(";") + f" LIMIT {limit}"

    # Build pooler connection URL
    _sb_url = _os.environ.get("SUPABASE_URL", "")
    _sb_pwd = _os.environ.get("SUPABASE_DB_PASSWORD", "")
    _sb_region = _os.environ.get("SUPABASE_POOLER_REGION", "aws-1-us-west-1")
    _sb_ref = _sb_url.replace("https://", "").replace(".supabase.co", "")
    _pooler_host = f"{_sb_region}.pooler.supabase.com"
    _pooler_url = f"postgresql://postgres.{_sb_ref}:{_sb_pwd}@{_pooler_host}:6543/postgres"

    # Inject user_id CTE filter
    _uid = _FELIX_SYSTEM_USER_ID
    if _uid:
        _cte = (
            f"WITH _user_projects AS (SELECT id FROM projects WHERE user_id = \'{_uid}\'), "
            f"_user_inferences AS (SELECT * FROM inferences WHERE user_id = \'{_uid}\'), "
            f"_user_training_jobs AS (SELECT * FROM training_jobs WHERE user_id = \'{_uid}\'), "
            f"_user_datasets AS (SELECT * FROM datasets WHERE user_id = \'{_uid}\')"
        )
        if _sql_stripped.startswith("WITH"):
            sql = _cte + ", " + sql[4:]
        else:
            sql = _cte + " " + sql

    async def _run():
        import uuid as _uuid_mod
        from decimal import Decimal as _Decimal
        conn = await _asyncpg.connect(_pooler_url, timeout=15, statement_cache_size=0)
        try:
            rows = await conn.fetch(sql)
            result = []
            for r in rows:
                row_dict = {}
                for k in r.keys():
                    v = r[k]
                    if v is None:
                        row_dict[k] = None
                    elif hasattr(v, \'isoformat\'):
                        row_dict[k] = v.isoformat()
                    elif isinstance(v, _uuid_mod.UUID):
                        row_dict[k] = str(v)
                    elif isinstance(v, _Decimal):
                        row_dict[k] = float(v)
                    else:
                        row_dict[k] = v
                result.append(row_dict)
            return result
        finally:
            await conn.close()

    return _asyncio.run(_run())

_g[\'query_sql\'] = query_sql
# --- End sandbox extensions ---
'''
    felix_code = felix_code + "\n" + query_sql_patch

    # Write felix helpers to a temp file so we can mount it
    felix_tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="felix_helpers_"
    )
    felix_tmp.write(felix_code)
    felix_tmp.close()

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "curl", "jq")
        .uv_pip_install(
            # Agent core
            "langchain-anthropic",
            "langchain-core",
            "langgraph",
            "anthropic",
            # Database
            "asyncpg",
            "supabase",
            # Data science (for felix helpers in bash)
            "numpy",
            "pandas",
            "pyarrow",
            "requests",
            "httpx",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "pydantic>=2.0",
            "python-dotenv",
            "fastapi",
            "slowapi",
        )
        # Create package structure (before add_local_dir)
        .run_commands(
            "mkdir -p /root/brain/services /root/brain/utils /root/brain/schemas /root/brain/shared",
            "touch /root/brain/__init__.py /root/brain/services/__init__.py /root/brain/utils/__init__.py",
        )
        # Mount agent source code + felix helpers
        .add_local_file(felix_tmp.name, remote_path="/root/functions.py")
        .add_local_dir(
            str(BRAIN_DIR / "services" / "auto_agent"),
            remote_path="/root/brain/services/auto_agent",
            ignore=["__pycache__", "*.pyc"],
        )
        .add_local_dir(
            str(BRAIN_DIR / "utils" / "supabase"),
            remote_path="/root/brain/utils/supabase",
            ignore=["__pycache__", "*.pyc"],
        )
        .add_local_dir(
            str(BRAIN_DIR / "schemas"),
            remote_path="/root/brain/schemas",
            ignore=["__pycache__", "*.pyc"],
        )
        .add_local_dir(
            str(BRAIN_DIR / "shared"),
            remote_path="/root/brain/shared",
            ignore=["__pycache__", "*.pyc"],
        )
    )

    # Mount GCP session cookies if present (enables authenticated GCP Console crawling)
    gcp_session = Path(__file__).resolve().parents[2].parent / "gcp_session.json"
    if not gcp_session.exists():
        gcp_session = Path.home() / "gcp_session.json"
    if gcp_session.exists():
        image = image.add_local_file(str(gcp_session), remote_path="/root/gcp_session.json")
        logger.info("Mounted GCP session cookies into sandbox image")

    return image


# Lazy image singleton (built once, cached by Modal)
_image: Optional[modal.Image] = None


def get_sandbox_image() -> modal.Image:
    """Get or build the Modal image.

    Returns:
        Cached Modal Image.
    """
    global _image
    if _image is None:
        _image = build_sandbox_image()
    return _image


def get_sandbox_secrets() -> list:
    """Collect Modal secrets for the container.

    Returns:
        List of Modal Secret references.
    """
    secrets = []

    try:
        secrets.append(modal.Secret.from_name(AWS_SECRET_NAME))
    except Exception:
        logger.warning("AWS secret '%s' not found", AWS_SECRET_NAME)

    return secrets


def build_sandbox_env(user_id: str) -> dict:
    """Build environment variables to pass into the container.

    Args:
        user_id: Authenticated user ID for felix helpers auth.

    Returns:
        Dict of env var name to value.
    """
    env: dict[str, str] = {}

    passthrough_vars = [
        "ANTHROPIC_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_DB_PASSWORD",
        "SUPABASE_POOLER_REGION",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_PROJECT",
        "HF_TOKEN",
    ]

    for var in passthrough_vars:
        val = os.getenv(var, "")
        if val:
            env[var] = val

    # Modal sandboxes run on Azure — use CloudFront domain which is globally accessible
    env["FELIX_API_URL"] = os.getenv("SANDBOX_FELIX_API_URL", "https://api.pioneer.ai")
    env.pop("BRAIN_API_URL", None)

    # Felix helpers auth: set user ID and API key
    env["FELIX_SYSTEM_USER_ID"] = user_id
    api_key = os.getenv("SANDBOX_API_KEY", "")
    if api_key:
        env["FELIX_API_TOKEN"] = api_key

    # Force LangSmith tracing on if key is present
    if env.get("LANGSMITH_API_KEY"):
        env.setdefault("LANGCHAIN_TRACING_V2", "true")
        env.setdefault("LANGSMITH_PROJECT", "auto-agent")

    # Drop DATABASE_URL so sql helpers fall back to pooler URL
    env.pop("DATABASE_URL", None)

    if "SUPABASE_POOLER_REGION" not in env:
        env["SUPABASE_POOLER_REGION"] = "aws-1-us-west-1"

    return env


def _extract_workspace_files(sandbox: "modal.Sandbox") -> dict:
    """Extract key files from the sandbox before termination.

    Reads data-curation.md, deliverables.json, final_report.md, and other
    .md/.json files from /workspace/ and /root/ so they aren't lost when
    the sandbox terminates.

    Args:
        sandbox: The Modal sandbox to extract files from.

    Returns:
        Dict mapping file path to file contents.
    """
    files = {}
    try:
        proc = sandbox.exec(
            "bash", "-c",
            "{ find /workspace /root /tmp -maxdepth 3 -type f 2>/dev/null; "
            "find / -maxdepth 1 -type f 2>/dev/null; } | "
            "grep -E '\\.(md|jsonl|json|txt)$' | "
            "grep -v '/brain/' | grep -v 'functions.py'"
        )
        file_list = proc.stdout.read().strip()
        proc.wait()

        if not file_list:
            return files

        for fpath in file_list.split("\n"):
            fpath = fpath.strip()
            if not fpath:
                continue
            try:
                cat_proc = sandbox.exec("cat", fpath)
                content = cat_proc.stdout.read()
                cat_proc.wait()
                if len(content) <= 500_000:
                    files[fpath] = content
                else:
                    files[fpath] = content[:500_000] + f"\n... (truncated, {len(content)} bytes total)"
            except Exception as e:
                files[fpath] = f"(error reading: {e})"

        logger.info("Extracted %d workspace files from sandbox", len(files))
    except Exception as e:
        logger.warning("Failed to extract workspace files: %s", e)

    return files


def _run_sandbox_blocking(
    app_name: str,
    image: "modal.Image",
    secrets: list,
    env: dict,
    agent_script: str,
) -> dict:
    """Execute the agent script in a Modal sandbox (blocking).

    All Modal calls are synchronous here. This function is intended to be
    called via asyncio.to_thread() so it doesn't block the event loop.

    Args:
        app_name: Modal app name string (App.lookup is done inside the thread).
        image: Modal image.
        secrets: Modal secrets list.
        env: Environment variables dict.
        agent_script: Python script to run inside the sandbox.

    Returns:
        Dict with answer, tool_calls, conversation_id, workspace_files.
    """
    app = modal.App.lookup(app_name, create_if_missing=True)
    sandbox = modal.Sandbox.create(
        app=app,
        image=image,
        timeout=CONTAINER_TIMEOUT,
        idle_timeout=CONTAINER_TIMEOUT,
        memory=CONTAINER_MEMORY_MB,
        secrets=secrets,
    )

    try:
        proc = sandbox.exec("python3", "-c", agent_script, env=env)

        # Stream stdout line-by-line so logs are visible in real time
        stdout_lines = []
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                stdout_lines.append(line)
        except Exception:
            pass
        stdout = "".join(stdout_lines)

        try:
            stderr = proc.stderr.read()
        except Exception:
            stderr = ""
        try:
            proc.wait()
        except Exception as wait_err:
            logger.warning("proc.wait() failed: %s", wait_err)

        exit_code = None
        try:
            exit_code = proc.returncode
        except Exception:
            pass

        if exit_code and exit_code != 0:
            logger.error("Modal agent failed (exit %d): %s", exit_code, stderr)
            return {
                "answer": f"Agent execution failed in sandbox: {stderr[:1000]}",
                "tool_calls": 0,
                "conversation_id": "",
                "workspace_files": {},
            }

        # Parse result from the marker line
        if "===AGENT_RESULT===" in stdout:
            result_line = stdout.split("===AGENT_RESULT===\n")[-1].strip()
            result = json.loads(result_line)
        else:
            output_lines = stdout.strip().split("\n")
            result = json.loads(output_lines[-1])

        # Extract workspace files before terminating
        try:
            result["workspace_files"] = _extract_workspace_files(sandbox)
        except Exception as e:
            logger.warning("Failed to extract workspace files: %s", e)
            result["workspace_files"] = {}

        return result

    except Exception as e:
        logger.error("Modal sandbox execution failed: %s", e, exc_info=True)
        return {
            "answer": f"Sandbox execution error: {e}",
            "tool_calls": 0,
            "conversation_id": "",
            "workspace_files": {},
        }
    finally:
        try:
            sandbox.terminate()
        except Exception:
            pass


def run_agent_in_modal_sync(
    question: str,
    user_id: str,
    message_history: Optional[list] = None,
) -> dict:
    """Spawn the auto agent inside a Modal container (synchronous).

    Intended to be called via asyncio.to_thread() from the service layer.
    All Modal operations are blocking; do NOT call from the asyncio event loop directly.

    Args:
        question: The user's task description.
        user_id: Authenticated user ID.
        message_history: Optional conversation history.

    Returns:
        Dict with answer, tool_calls, conversation_id, workspace_files.
    """
    image = get_sandbox_image()
    secrets = get_sandbox_secrets()
    env = build_sandbox_env(user_id)

    request_payload = json.dumps({
        "question": question,
        "user_id": user_id,
        "message_history": message_history,
    })

    agent_script = """
import json, sys, os, asyncio, logging

# Override env vars before anything imports functions.py
for _k, _v in json.loads(ENV_OVERRIDES_PLACEHOLDER).items():
    os.environ[_k] = _v

sys.path.insert(0, "/root/brain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

request = json.loads(REQUEST_PAYLOAD_PLACEHOLDER)

from services.auto_agent.agent import run_auto_agent

result = asyncio.run(run_auto_agent(
    question=request["question"],
    user_id=request["user_id"],
    message_history=request.get("message_history"),
))

# Print result as last line of stdout for parsing
print("===AGENT_RESULT===")
print(json.dumps(result, default=str))
""".replace("REQUEST_PAYLOAD_PLACEHOLDER", repr(request_payload)).replace(
        "ENV_OVERRIDES_PLACEHOLDER", repr(json.dumps(env))
    )

    # Run ALL blocking Modal calls (App.lookup + Sandbox.create + exec + read) in a thread
    # so we never block the asyncio event loop.
    return _run_sandbox_blocking(MODAL_APP_NAME, image, secrets, env, agent_script)
