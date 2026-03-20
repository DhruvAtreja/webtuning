"""
Felix Helper Functions for Notebook.

These functions are injected into the Modal sandbox kernel at startup,
providing convenient access to the Felix API from within notebook cells.

Usage in notebook:
    # List available datasets
    datasets = list_datasets()

    # Fetch a dataset as a DataFrame
    df = fetch_dataset("my-dataset")

    # Upload a DataFrame as a new dataset (async processing)
    result = upload_dataset(df, "new-dataset", "classification", wait_for_completion=True)

    # Or upload and check status manually
    result = upload_dataset(df, "new-dataset", "classification")
    status = check_dataset_status(result['dataset_name'], result['version_number'])
    final = wait_for_dataset(result['dataset_name'], result['version_number'])

    # Start a training job
    job = start_training("my-model", ["dataset-1", "dataset-2"])

    # Check training status
    status = get_training_status(job["id"])

    # Run evaluation
    eval_result = run_evaluation(model_id="model-123", dataset_name="test-data")

    # Generate synthetic data
    data = generate_data("classification", "Product reviews", num_examples=100, dataset_name="product-reviews")
"""

# This is the Python code that will be injected into the Modal sandbox kernel.
# It's stored as a string so it can be executed in the kernel context.

FELIX_HELPERS_CODE = '''
# =============================================================================
# Felix API Helper Functions
# =============================================================================
# These functions are automatically available in your notebook.
# They provide easy access to Felix datasets, training, and evaluation APIs.

import os as _os
import json as _json
import urllib.request as _urllib_request
import urllib.parse as _urllib_parse
import urllib.error as _urllib_error
import time as _time
import sys as _sys
from typing import Optional, List, Dict, Any, Union

# Configuration from environment
_FELIX_API_URL = _os.environ.get("FELIX_API_URL", "http://localhost:5001")
if _FELIX_API_URL and not _FELIX_API_URL.startswith(("http://", "https://")):
    _FELIX_API_URL = f"https://{_FELIX_API_URL}"
_FELIX_API_URL = _FELIX_API_URL.rstrip("/")
_FELIX_API_TOKEN = _os.environ.get("FELIX_API_TOKEN", "")
_TRANSIENT_GATEWAY_STATUS_CODES = {502, 503, 504}
_MAX_TRANSIENT_GATEWAY_RETRIES = 2
_TRANSIENT_GATEWAY_RETRY_BASE_DELAY_SECONDS = 0.75


class _FelixHelperAPIError(Exception):
    """Raised when Felix API returns an HTTP error response."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API Error ({status_code}): {detail}")


def _make_request(
    method: str,
    endpoint: str,
    data: Optional[Dict] = None,
    params: Optional[Dict] = None,
    timeout: int = 60,
    max_retries: int = _MAX_TRANSIENT_GATEWAY_RETRIES,
) -> Dict[str, Any]:
    """Make an HTTP request to the Felix API."""
    url = f"{_FELIX_API_URL}{endpoint}"

    if params:
        query_string = _urllib_parse.urlencode(params)
        url = f"{url}?{query_string}"

    # Send appropriate auth header based on token type
    # API keys start with 'pio_sk_', JWTs are used as Bearer tokens
    headers = {
        "Content-Type": "application/json",
    }
    if _FELIX_API_TOKEN.startswith("pio_sk_"):
        headers["x-api-key"] = _FELIX_API_TOKEN
    else:
        headers["Authorization"] = f"Bearer {_FELIX_API_TOKEN}"

    body = None
    if data is not None:
        body = _json.dumps(data).encode("utf-8")

    attempt = 0
    while True:
        request = _urllib_request.Request(url, data=body, headers=headers, method=method)

        try:
            response = _urllib_request.urlopen(request, timeout=timeout)
            response_data = response.read().decode("utf-8")
            if response_data:
                return _json.loads(response_data)
            return {}
        except _urllib_error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            if (
                e.code in _TRANSIENT_GATEWAY_STATUS_CODES
                and attempt < max_retries
            ):
                attempt += 1
                delay_seconds = _TRANSIENT_GATEWAY_RETRY_BASE_DELAY_SECONDS * attempt
                _time.sleep(delay_seconds)
                continue

            try:
                error_data = _json.loads(error_body)
                detail = str(error_data.get("detail", error_body))
            except _json.JSONDecodeError:
                detail = error_body
            raise _FelixHelperAPIError(e.code, detail)
        except _urllib_error.URLError as e:
            if attempt < max_retries:
                attempt += 1
                delay_seconds = _TRANSIENT_GATEWAY_RETRY_BASE_DELAY_SECONDS * attempt
                _time.sleep(delay_seconds)
                continue
            raise Exception(f"Connection Error: {e.reason}")


def _poll_generation_job(
    job_id: str,
    timeout: int = 600,
    poll_interval: int = 5,
) -> List[Dict[str, Any]]:
    """Poll a generation job until completion and return data.

    Args:
        job_id: The job/dataset ID to poll.
        timeout: Maximum time in seconds to wait.
        poll_interval: Seconds between poll requests.

    Returns:
        List of generated data records.

    Raises:
        Exception: If job fails or times out.
    """
    import time as _time

    elapsed = 0
    while elapsed < timeout:
        result = _make_request("GET", f"/generate/jobs/{job_id}", timeout=30)
        status = result.get("status", "")

        if status == "ready":
            return result.get("data") or []
        elif status == "failed":
            error_msg = result.get("error", "Generation job failed")
            raise Exception(f"Generation failed: {error_msg}")

        _time.sleep(poll_interval)
        elapsed += poll_interval

    raise Exception(f"Generation job {job_id} timed out after {timeout}s")


# =============================================================================
# Dataset Operations
# =============================================================================

def list_datasets() -> List[Dict[str, Any]]:
    """
    List all available datasets.

    Returns:
        List of dataset objects with keys: name, dataset_name, dataset_type, sample_size, etc.

    Example:
        >>> datasets = list_datasets()
        >>> for ds in datasets:
        ...     print(f"{ds['name']}: {ds['sample_size']} samples")
    """
    result = _make_request("GET", "/felix/datasets")
    datasets = result.get("datasets") or []
    # Add 'name' alias for convenience (API returns 'dataset_name')
    for ds in datasets:
        if "dataset_name" in ds and "name" not in ds:
            ds["name"] = ds["dataset_name"]
    return datasets


def fetch_dataset(name: str, version: Optional[str] = None, as_dataframe: bool = True):
    """
    Fetch a dataset by name.

    Args:
        name: Dataset name.
        version: Optional version number (defaults to latest).
        as_dataframe: If True, return as pandas DataFrame. If False, return raw list.

    Returns:
        pandas DataFrame or list of records.

    Example:
        >>> df = fetch_dataset("my-dataset")
        >>> print(df.head())
    """
    version_str = version if version else "latest"
    endpoint = f"/felix/datasets/{_urllib_parse.quote(name)}/{version_str}/download"
    params = {"format": "jsonl"}  # Valid formats: jsonl, csv, parquet

    # Build the URL with query params
    url = f"{_FELIX_API_URL}{endpoint}"
    query_string = _urllib_parse.urlencode(params)
    url = f"{url}?{query_string}"

    # Set up auth headers
    headers = {"Content-Type": "application/json"}
    if _FELIX_API_TOKEN.startswith("pio_sk_"):
        headers["x-api-key"] = _FELIX_API_TOKEN
    else:
        headers["Authorization"] = f"Bearer {_FELIX_API_TOKEN}"

    request = _urllib_request.Request(url, headers=headers, method="GET")

    try:
        response = _urllib_request.urlopen(request, timeout=120)
        response_data = response.read().decode("utf-8")

        # Parse JSONL format (one JSON object per line)
        data = []
        for line in response_data.strip().split(chr(10)):  # chr(10) = newline
            line = line.strip()
            if line:  # Skip empty lines
                data.append(_json.loads(line))

    except _urllib_error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            error_data = _json.loads(error_body)
            raise Exception(f"API Error ({e.code}): {error_data.get('detail', error_body)}")
        except _json.JSONDecodeError:
            raise Exception(f"API Error ({e.code}): {error_body}")
    except _urllib_error.URLError as e:
        raise Exception(f"Connection Error: {e.reason}")

    if as_dataframe:
        try:
            import pandas as pd
            return pd.DataFrame(data)
        except ImportError:
            print("Warning: pandas not available, returning list instead")
            return data
    return data


def get_dataset_info(name: str, version: Optional[str] = None) -> Dict[str, Any]:
    """
    Get metadata about a dataset.

    Args:
        name: Dataset name.
        version: Optional version number (defaults to latest).

    Returns:
        Dataset info dict with keys: dataset_name, dataset_type, sample_size, size,
        version_number, created_at, etc.

    Example:
        >>> info = get_dataset_info("my-dataset")
        >>> print(f"Type: {info['dataset_type']}, Size: {info['sample_size']}")
    """
    version_str = version if version else "latest"
    endpoint = f"/felix/datasets/{_urllib_parse.quote(name)}/{version_str}"
    return _make_request("GET", endpoint)


def preview_dataset(name: str, version: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """
    Preview the first few rows of a dataset.

    Args:
        name: Dataset name.
        version: Optional version number (defaults to latest).
        limit: Number of rows to preview (default 10).

    Returns:
        List of row dictionaries.

    Example:
        >>> rows = preview_dataset("my-dataset", limit=5)
        >>> for row in rows:
        ...     print(row)
    """
    version_str = version if version else "latest"
    endpoint = f"/felix/datasets/{_urllib_parse.quote(name)}/{version_str}/preview"
    params = {"limit": limit}
    result = _make_request("GET", endpoint, params=params)
    return result.get("rows") or []


def upload_dataset(
    data,  # DataFrame or list of dicts
    name: str,
    dataset_type: str = "classification",
    description: Optional[str] = None,
    wait_for_completion: bool = True,
    poll_interval: int = 2,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Upload data as a new dataset (asynchronous processing).

    **IMPORTANT:** Dataset uploads are processed asynchronously in the background.
    This function returns immediately with status="uploading". The dataset will not
    be ready for use until processing completes.

    **To check processing status:**
    - Set wait_for_completion=True to automatically poll until ready (recommended)
    - Or manually call check_dataset_status(name, version) to check status
    - Or use wait_for_dataset(name, version) to poll until ready

    **Processing status values:**
    - "uploading" - Initial upload in progress
    - "processing" - Converting and validating data
    - "ready" - Dataset is ready to use
    - "failed" - Processing failed (check error message)

    Args:
        data: pandas DataFrame or list of dictionaries.
        name: Name for the new dataset.
        dataset_type: Type of dataset ("classification", "ner", "custom").
        description: Optional description.
        wait_for_completion: If True, polls until dataset is ready (default: False).
        poll_interval: Seconds between status checks when waiting (default: 2).
        timeout: Max seconds to wait for completion (default: 300).

    Returns:
        Upload result with keys: id, dataset_name, version_number, status, etc.
        If wait_for_completion=True, returns final status (ready or failed).

    Raises:
        Exception: If upload fails or timeout occurs while waiting.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"text": ["hello", "world"], "label": ["greeting", "noun"]})

        >>> # Option 1: Wait for completion (recommended for interactive use)
        >>> result = upload_dataset(df, "my-dataset", wait_for_completion=True)
        >>> print(f"Dataset ready: {result['dataset_name']} v{result['version_number']}")

        >>> # Option 2: Return immediately and check status manually
        >>> result = upload_dataset(df, "my-dataset")
        >>> print(f"Upload initiated: {result['dataset_name']} v{result['version_number']}")
        >>> # ... do other work ...
        >>> status = check_dataset_status(result['dataset_name'], result['version_number'])
        >>> print(f"Current status: {status['status']}")

        >>> # Option 3: Poll for completion later
        >>> result = upload_dataset(df, "my-dataset")
        >>> final = wait_for_dataset(result['dataset_name'], result['version_number'])
        >>> print(f"Processing complete: {final['status']}")
    """
    import io as _io
    import uuid as _uuid

    # Convert DataFrame to list of dicts if needed
    if hasattr(data, "to_dict"):
        records = data.to_dict(orient="records")
    else:
        records = list(data)

    # Convert records to JSONL format (use chr(10) for newline to avoid escaping issues)
    newline = chr(10)
    jsonl_content = newline.join(_json.dumps(record) for record in records)
    file_bytes = jsonl_content.encode("utf-8")

    # Build multipart form data
    boundary = f"----WebKitFormBoundary{_uuid.uuid4().hex[:16]}"

    body_parts = []

    # Add file field
    body_parts.append(f"--{boundary}")
    body_parts.append(f'Content-Disposition: form-data; name="file"; filename="{name}.jsonl"')
    body_parts.append("Content-Type: application/jsonl")
    body_parts.append("")
    body_parts.append(jsonl_content)

    # Add dataset_name field
    body_parts.append(f"--{boundary}")
    body_parts.append('Content-Disposition: form-data; name="dataset_name"')
    body_parts.append("")
    body_parts.append(name)

    # Add dataset_type field
    body_parts.append(f"--{boundary}")
    body_parts.append('Content-Disposition: form-data; name="dataset_type"')
    body_parts.append("")
    body_parts.append(dataset_type)

    # Add format field
    body_parts.append(f"--{boundary}")
    body_parts.append('Content-Disposition: form-data; name="format"')
    body_parts.append("")
    body_parts.append("jsonl")

    # End boundary
    body_parts.append(f"--{boundary}--")
    body_parts.append("")

    # Use chr(13)+chr(10) for CRLF to avoid escaping issues in triple-quoted string
    crlf = chr(13) + chr(10)
    body = crlf.join(body_parts).encode("utf-8")

    url = f"{_FELIX_API_URL}/felix/datasets/upload"
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    if _FELIX_API_TOKEN.startswith("pio_sk_"):
        headers["x-api-key"] = _FELIX_API_TOKEN
    else:
        headers["Authorization"] = f"Bearer {_FELIX_API_TOKEN}"

    request = _urllib_request.Request(url, data=body, headers=headers, method="POST")

    try:
        response = _urllib_request.urlopen(request, timeout=120)
        response_data = response.read().decode("utf-8")
        if response_data:
            result = _json.loads(response_data)

            # Pretty print status for user
            status = result.get('status', 'unknown')
            dataset_name = result.get('dataset_name', name)
            version = result.get('version_number', '?')

            print(f"✓ Dataset upload initiated: {dataset_name} v{version}")
            print(f"  Status: {status}")

            # If requested, wait for completion
            if wait_for_completion:
                print(f"  Waiting for processing to complete...")
                result = wait_for_dataset(
                    dataset_name,
                    str(version),
                    poll_interval=poll_interval,
                    timeout=timeout
                )
                print(f"✓ Dataset ready: {dataset_name} v{version}")
            else:
                print(f"  Use check_dataset_status('{dataset_name}', '{version}') to check progress")

            return result
        return {}
    except _urllib_error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            error_data = _json.loads(error_body)
            raise Exception(f"API Error ({e.code}): {error_data.get('detail', error_body)}")
        except _json.JSONDecodeError:
            raise Exception(f"API Error ({e.code}): {error_body}")
    except _urllib_error.URLError as e:
        raise Exception(f"Connection Error: {e.reason}")


def check_dataset_status(name: str, version: Optional[str] = None) -> Dict[str, Any]:
    """
    Check the processing status of a dataset.

    Use this to check if an uploaded dataset has finished processing and is ready to use.

    Args:
        name: Dataset name.
        version: Version number (defaults to latest).

    Returns:
        Dataset info dict including 'status' field:
        - "uploading" - Initial upload in progress
        - "processing" - Converting and validating data
        - "ready" - Dataset is ready to use
        - "failed" - Processing failed

    Example:
        >>> result = upload_dataset(df, "my-dataset")
        >>> status_info = check_dataset_status(result['dataset_name'], result['version_number'])
        >>> print(f"Status: {status_info['status']}")
    """
    return get_dataset_info(name, version)


def wait_for_dataset(
    name: str,
    version: Optional[str] = None,
    poll_interval: int = 2,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Poll dataset status until processing completes or timeout occurs.

    Blocks until the dataset status becomes "ready" or "failed", or until timeout.

    Args:
        name: Dataset name.
        version: Version number (defaults to latest).
        poll_interval: Seconds to wait between status checks (default: 2).
        timeout: Maximum seconds to wait (default: 300).

    Returns:
        Final dataset info dict with status "ready" or "failed".

    Raises:
        Exception: If timeout occurs or processing fails.

    Example:
        >>> result = upload_dataset(df, "my-dataset")
        >>> final = wait_for_dataset(result['dataset_name'], result['version_number'])
        >>> if final['status'] == 'ready':
        ...     print("Dataset is ready!")
    """
    import time as _time

    start_time = _time.time()
    last_status = None

    while True:
        elapsed = _time.time() - start_time
        if elapsed > timeout:
            raise Exception(f"Timeout waiting for dataset '{name}' to be ready after {timeout}s")

        try:
            info = get_dataset_info(name, version)
            status = info.get('status', 'unknown')

            # Print status updates (only when status changes)
            if status != last_status:
                if status == 'uploading':
                    print(f"  Status: Uploading...")
                elif status == 'processing':
                    print(f"  Status: Processing and validating data...")
                elif status == 'ready':
                    print(f"  Status: Ready!")
                elif status == 'failed':
                    error_msg = info.get('error_message', 'Unknown error')
                    print(f"  Status: Failed - {error_msg}")
                last_status = status

            # Terminal states
            if status == 'ready':
                return info
            elif status == 'failed':
                error_msg = info.get('error_message', 'Processing failed')
                raise Exception(f"Dataset processing failed: {error_msg}")

            # Wait before next poll
            _time.sleep(poll_interval)

        except _FelixHelperAPIError as e:
            if e.status_code in (401, 403):
                raise Exception(
                    "Session expired while waiting for dataset processing. "
                    "Reconnect notebook session and retry."
                )
            _time.sleep(poll_interval)
        except Exception as e:
            # If it's our own exception, re-raise
            if "processing failed" in str(e) or "Timeout waiting" in str(e):
                raise
            # For other errors (network, etc), wait and retry
            _time.sleep(poll_interval)


def delete_dataset(name: str, version: Optional[str] = None) -> Dict[str, Any]:
    """
    Delete a dataset.

    Args:
        name: Dataset name.
        version: Optional specific version to delete (deletes all versions if not specified).

    Returns:
        Deletion result.

    Example:
        >>> delete_dataset("old-dataset")
    """
    if version:
        endpoint = f"/felix/datasets/{_urllib_parse.quote(name)}/{version}"
    else:
        endpoint = f"/felix/datasets/{_urllib_parse.quote(name)}"
    return _make_request("DELETE", endpoint)


def delete_dataset_rows(
    name: str,
    version: str,
    row_indices: List[int],
) -> Dict[str, Any]:
    """
    Delete specific rows from a dataset by index, creating a new version.

    Args:
        name: Dataset name.
        version: Version number (use "latest" for most recent).
        row_indices: List of zero-based row indices to remove.

    Returns:
        Dict with success, rows_deleted, rows_remaining, new_version.

    Example:
        >>> delete_dataset_rows("my-dataset", "1", [0, 2, 4])
    """
    endpoint = f"/felix/datasets/{_urllib_parse.quote(name)}/{version}/rows"
    return _make_request("DELETE", endpoint, data={"row_indices": row_indices})


def dismiss_dataset_outlier(
    dataset_name: str,
    fingerprint: str,
) -> Dict[str, Any]:
    """
    Dismiss an outlier so it no longer appears in analysis results.

    Appends the content fingerprint to the dataset's dismissed outliers list.
    On subsequent analysis runs the outlier will be filtered out.

    Args:
        dataset_name: Name of the dataset containing the outlier.
        fingerprint: Content fingerprint of the outlier sample to dismiss.

    Returns:
        Dict with success status.

    Example:
        >>> dismiss_dataset_outlier("my-dataset", "abc123fingerprint")
    """
    endpoint = "/felix/dataset/outliers/dismiss"
    return _make_request(
        "POST", endpoint, data={"dataset_name": dataset_name, "fingerprint": fingerprint}
    )


# =============================================================================
# Training Operations
# =============================================================================

def list_models() -> List[Dict[str, Any]]:
    """
    List all trained models.

    Returns:
        List of model objects with keys: id, model_name, status, etc.

    Example:
        >>> models = list_models()
        >>> for m in models:
        ...     print(f"{m['model_name']}: {m['status']}")
    """
    result = _make_request("GET", "/felix/trained-models")
    models = result.get("jobs") or result.get("models") or []
    # Add 'job_id' alias for convenience (API returns 'id')
    for m in models:
        if "id" in m and "job_id" not in m:
            m["job_id"] = m["id"]
    return models


def list_training_jobs() -> List[Dict[str, Any]]:
    """
    List all training jobs (including in-progress).

    Returns:
        List of training job objects with keys: id, model_name, status, etc.

    Example:
        >>> jobs = list_training_jobs()
        >>> running = [j for j in jobs if j['status'] == 'running']
        >>> print(f"{len(running)} jobs currently running")
    """
    result = _make_request("GET", "/felix/training-jobs")
    jobs = result.get("jobs") or []
    # Add 'job_id' alias for convenience (API returns 'id')
    for j in jobs:
        if "id" in j and "job_id" not in j:
            j["job_id"] = j["id"]
    return jobs


def get_training_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status of a training job.

    Args:
        job_id: Training job ID.

    Returns:
        Training job status with keys: id, status, model_name, metrics, etc.

    Example:
        >>> status = get_training_status("job-123")
        >>> print(f"Status: {status['status']}")
        >>> if status.get('metrics'):
        ...     print(f"F1: {status['metrics'].get('f1_score')}")
    """
    endpoint = f"/felix/training-jobs/{_urllib_parse.quote(job_id)}"
    result = _make_request("GET", endpoint)
    # Add 'job_id' alias for convenience (API returns 'id')
    if "id" in result and "job_id" not in result:
        result["job_id"] = result["id"]
    return result


def start_training(
    model_name: str,
    datasets: List[str],
    base_model: str = "fastino/gliner2-base-v1",
    **kwargs,
) -> Dict[str, Any]:
    """
    Start a new training job.

    Args:
        model_name: Name for the trained model.
        datasets: List of dataset names to train on.
        base_model: Base model to fine-tune (default: fastino/gliner2-base-v1 for encoder).
            For decoder datasets, defaults to Qwen/Qwen3-8B. Supported decoder models:
            Qwen/Qwen3-8B, meta-llama/Llama-3.1-8B-Instruct.
        **kwargs: Additional training parameters.

    Returns:
        Training job info with keys: id, status, etc.

    Example:
        >>> # Encoder (GLiNER) training
        >>> job = start_training(
        ...     model_name="my-classifier",
        ...     datasets=["training-data-v1"],
        ...     base_model="fastino/gliner2-base-v1"
        ... )
        >>> print(f"Started training job: {job['id']}")
        >>>
        >>> # Decoder training — auto-routes when all datasets are decoder type
        >>> job = start_training("my-chatbot", ["decoder-training-data"])
        >>>
        >>> # Decoder training with a specific model
        >>> job = start_training("my-chatbot", ["decoder-data"],
        ...     base_model="meta-llama/Llama-3.1-8B-Instruct")
    """
    payload = {
        "model_name": model_name,
        "datasets": [{"name": ds} for ds in datasets],
        "base_model": base_model,
        **kwargs,
    }
    result = _make_request("POST", "/felix/training-jobs", data=payload, timeout=120)
    # Add 'job_id' alias for convenience (API returns 'id')
    if "id" in result and "job_id" not in result:
        result["job_id"] = result["id"]
    return result


# =============================================================================
# Evaluation Operations
# =============================================================================

def list_evaluations() -> List[Dict[str, Any]]:
    """
    List all evaluation runs.

    Returns:
        List of evaluation objects with keys: id, model_id, dataset_name, status, f1_score, etc.

    Example:
        >>> evals = list_evaluations()
        >>> completed = [e for e in evals if e['status'] == 'complete']
        >>> for e in completed:
        ...     print(f"{e['model_id']} on {e['dataset_name']}: F1={e.get('f1_score', 'N/A')}")
    """
    result = _make_request("GET", "/felix/evaluations")
    return result.get("evaluations") or []


def run_evaluation(
    model_id: str,
    dataset_name: str,
    provider: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run an evaluation of a model on a dataset.

    Args:
        model_id: ID of the model to evaluate (training job ID, 'gliner2' for base, or LLM model like 'gpt-4o').
        dataset_name: Name of the dataset to evaluate on.
        provider: Inference provider (auto-detected from model_id if not specified).
        max_examples: Maximum number of examples to evaluate (None for all).

    Returns:
        Evaluation job info with keys: id, status, provider, f1_score, etc.

    Example:
        >>> eval_job = run_evaluation(
        ...     model_id="model-123",
        ...     dataset_name="test-data",
        ...     max_examples=100
        ... )
        >>> print(f"Started evaluation: {eval_job['id']}")
    """
    payload = {
        "base_model": model_id,  # API expects base_model (model_id is deprecated)
        "dataset_name": dataset_name,
    }
    if provider is not None:
        payload["provider"] = provider
    if max_examples is not None:
        payload["max_examples"] = max_examples

    result = _make_request("POST", "/felix/evaluations", data=payload, timeout=120)

    # API returns EvaluationCreateResponse with evaluations list
    evaluations = result.get("evaluations") or []

    # Find the matching evaluation for this model/dataset combination
    for eval_item in evaluations:
        if eval_item.get("model_id") == model_id and eval_item.get("dataset_name") == dataset_name:
            return eval_item

    # Return first evaluation if only one exists (common case)
    if evaluations:
        return evaluations[0]

    # If no evaluations found, return the raw response for debugging
    return result


def get_evaluation_results(evaluation_id: str) -> Dict[str, Any]:
    """
    Get results of an evaluation.

    Args:
        evaluation_id: Evaluation ID.

    Returns:
        Evaluation results with keys: id, model_id, dataset_name, status,
        f1_score, precision_score, recall_score, accuracy, bleu_score, rouge_l_score,
        sample_count, error_message, predictions (per-example results for error analysis).

    Example (Encoder):
        >>> results = get_evaluation_results("eval-123")
        >>> if results['status'] == 'complete':
        ...     print(f"F1: {results['f1_score']:.4f}")

    Example (Decoder):
        >>> results = get_evaluation_results("decoder-eval-123")
        >>> if results['status'] == 'complete':
        ...     print(f"BLEU: {results['bleu_score']:.4f}")
        ...     print(f"ROUGE-L: {results['rouge_l_score']:.4f}")
    """
    endpoint = f"/felix/evaluations/{_urllib_parse.quote(evaluation_id)}"
    return _make_request("GET", endpoint)


def analyze_failures(evaluation_id: str, limit: int = 10):
    """
    Analyze evaluation failures to identify patterns.

    Fetches evaluation results and returns only the examples that failed,
    helping identify what the model struggles with.

    Args:
        evaluation_id: Evaluation ID to analyze.
        limit: Maximum number of failures to return (default: 10).

    Returns:
        pandas DataFrame or list of failed examples.
        For single-label: columns are 'text', 'true_label', 'predicted_label', 'correct'
        For multi-label: columns are 'text', 'true_labels', 'predicted_labels', 'correct'
        For decoder: columns are 'text', 'expected_response', 'generated_response', 'bleu', 'rouge_l', 'correct'

    Example (Single-label):
        >>> failures = analyze_failures(eval_result['id'])
        >>> for i, f in failures.iterrows():
        ...     print(f"Expected: {f['true_label']}, Got: {f['predicted_label']}")

    Example (Decoder):
        >>> failures = analyze_failures(eval_result['id'])
        >>> for i, f in failures.iterrows():
        ...     print(f"BLEU={f['bleu']:.3f}: {f['generated_response'][:80]}")
    """
    results = get_evaluation_results(evaluation_id)

    if results.get('status') != 'complete':
        raise Exception(f"Evaluation not complete yet. Status: {results.get('status')}")

    predictions = results.get('predictions') or []
    if not predictions:
        raise Exception("No per-example predictions found. This evaluation may have been run before per-example tracking was enabled.")

    # Filter for failures only
    failures = [p for p in predictions if not p.get('correct', True)]

    # Limit results
    failures = failures[:limit]

    # Try to return as DataFrame if pandas is available
    try:
        import pandas as pd
        return pd.DataFrame(failures)
    except ImportError:
        return failures


def get_failure_stats(evaluation_id: str) -> Dict[str, Any]:
    """
    Get summary statistics about evaluation failures.

    Works for classification (single and multi-label), NER, and decoder evaluations.

    Args:
        evaluation_id: Evaluation ID to analyze.

    Returns:
        Dict with failure counts, most common errors, etc.
        For single-label classification: includes confusion matrix (true -> predicted label pairs).
        For multi-label classification: includes missing/extra label analysis.
        For NER: includes overall stats only (confusion matrix not applicable).
        For decoder: includes bleu_score, rouge_l_score, avg_bleu, avg_rouge_l,
        and worst_examples (10 lowest-BLEU examples with text/expected/generated).

    Example (Single-label Classification):
        >>> stats = get_failure_stats('eval-123')
        >>> print(f"Total failures: {stats['total_failures']}")
        >>> print(f"Accuracy: {stats['accuracy']:.1%}")
        >>> print(f"\\nMost confused labels:")
        >>> for error in stats['common_errors'][:5]:
        ...     print(f"  {error['true']} -> {error['predicted']}: {error['count']} times")

    Example (Decoder):
        >>> stats = get_failure_stats('decoder-eval-123')
        >>> print(f"BLEU: {stats['bleu_score']:.4f}")
        >>> print(f"ROUGE-L: {stats['rouge_l_score']:.4f}")
        >>> for ex in stats['worst_examples'][:3]:
        ...     print(f"  BLEU={ex['bleu']:.3f}: {ex['generated'][:80]}...")
    """
    results = get_evaluation_results(evaluation_id)

    if results.get('status') != 'complete':
        raise Exception(f"Evaluation not complete yet. Status: {results.get('status')}")

    predictions = results.get('predictions') or []
    if not predictions:
        raise Exception("No per-example predictions found.")

    total = len(predictions)
    failures = [p for p in predictions if not p.get('correct', True)]
    successes = [p for p in predictions if p.get('correct', False)]

    # Detect task type from prediction structure
    task_type = 'unknown'
    if predictions:
        first_pred = predictions[0]
        if 'expected_response' in first_pred and 'generated_response' in first_pred:
            task_type = 'decoder'
        elif 'true_label' in first_pred and 'predicted_label' in first_pred:
            task_type = 'classification'
        elif 'true_labels' in first_pred and 'predicted_labels' in first_pred:
            task_type = 'multilabel_classification'
        elif 'true_entities' in first_pred and 'predicted_entities' in first_pred:
            task_type = 'ner'

    # Count confusion pairs (for single-label classification only)
    confusion_counts = {}
    missing_label_counts = {}
    extra_label_counts = {}

    if task_type == 'classification':
        for f in failures:
            true_label = f.get('true_label')
            pred_label = f.get('predicted_label')
            if true_label and pred_label:
                key = (true_label, pred_label)
                confusion_counts[key] = confusion_counts.get(key, 0) + 1
    elif task_type == 'multilabel_classification':
        for f in failures:
            true_labels = set(f.get('true_labels') or [])
            pred_labels = set(f.get('predicted_labels') or [])
            for label in true_labels - pred_labels:
                missing_label_counts[label] = missing_label_counts.get(label, 0) + 1
            for label in pred_labels - true_labels:
                extra_label_counts[label] = extra_label_counts.get(label, 0) + 1

    # Sort by frequency
    common_errors = [
        {"true": true_l, "predicted": pred_l, "count": count}
        for (true_l, pred_l), count in sorted(confusion_counts.items(), key=lambda x: -x[1])
    ]

    missing_labels = sorted(missing_label_counts.items(), key=lambda x: -x[1])
    extra_labels = sorted(extra_label_counts.items(), key=lambda x: -x[1])

    result: Dict[str, Any] = {
        "task_type": task_type,
        "total_examples": total,
        "total_failures": len(failures),
        "total_successes": len(successes),
        "accuracy": len(successes) / total if total > 0 else 0,
    }

    if task_type == 'decoder':
        result["bleu_score"] = results.get('bleu_score')
        result["rouge_l_score"] = results.get('rouge_l_score')
        bleu_scores = [p['bleu'] for p in predictions if p.get('bleu') is not None]
        rouge_scores = [p['rouge_l'] for p in predictions if p.get('rouge_l') is not None]
        result["avg_bleu"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else None
        result["avg_rouge_l"] = sum(rouge_scores) / len(rouge_scores) if rouge_scores else None
        worst = sorted(predictions, key=lambda p: p.get('bleu', 0))[:10]
        result["worst_examples"] = [
            {
                "text": p.get('text', '')[:200],
                "expected": p.get('expected_response', '')[:200],
                "generated": p.get('generated_response', '')[:200],
                "bleu": p.get('bleu'),
                "rouge_l": p.get('rouge_l'),
            }
            for p in worst
        ]
    else:
        result["f1_score"] = results.get('f1_score')
        result["precision"] = results.get('precision_score')
        result["recall"] = results.get('recall_score')
        result["common_errors"] = common_errors

    if task_type == 'multilabel_classification':
        result["missing_labels"] = missing_labels
        result["extra_labels"] = extra_labels
        result["subset_accuracy"] = results.get('subset_accuracy')
        result["hamming_loss"] = results.get('hamming_loss')

    return result


def delete_evaluation(evaluation_id: str) -> Dict[str, Any]:
    """
    Delete an evaluation.

    Args:
        evaluation_id: The evaluation ID to delete.

    Returns:
        Deletion result.

    Example:
        >>> delete_evaluation("eval-123")
    """
    endpoint = f"/felix/evaluations/{_urllib_parse.quote(evaluation_id)}"
    return _make_request("DELETE", endpoint)


# =============================================================================
# Prompt Improvement
# =============================================================================

def improve_prompt(
    prompt: str,
    data_type: Optional[str] = None,
) -> Dict[str, str]:
    """
    Improve a dataset generation prompt with best practices and expanded coverage.

    Takes a short or vague prompt and returns a comprehensive, detailed version
    that covers niche cases and edge scenarios for higher-quality data generation.

    Call this before generate_data() to produce better training data.

    Args:
        prompt: The original prompt to improve (e.g. "smart home detection").
        data_type: Optional dataset type: "classification", "entity_extraction",
                   or "json_extraction".

    Returns:
        Dict with keys:
        - ``improved_prompt``: The expanded, detailed prompt text.
        - ``summary``: Short description of what was improved.

    Example:
        >>> result = improve_prompt("smart home detection", data_type="entity_extraction")
        >>> print(result["summary"])
        'Expanded coverage for smart home device categories and edge cases'
        >>> better_prompt = result["improved_prompt"]
        >>> data = generate_data("ner", better_prompt, labels=["device", "room", "action"])
    """
    payload: Dict[str, Any] = {"prompt": prompt}
    if data_type:
        payload["data_type"] = data_type

    result = _make_request("POST", "/generate/improve-prompt", data=payload, timeout=60)
    return {
        "improved_prompt": result.get("improved_prompt", ""),
        "summary": result.get("summary", ""),
    }


# =============================================================================
# Generation Operations
# =============================================================================

def generate_data(
    task_type: str,
    domain_description: str,
    num_examples: int = 10,
    labels: Optional[List[str]] = None,
    save_to_cloud: Optional[bool] = None,
    dataset_name: str = "",
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic data using AI.

    If dataset_name is provided, the dataset is automatically saved to cloud
    storage (save_to_cloud is set to True). You can then retrieve it with
    fetch_dataset(dataset_name).

    Args:
        task_type: Type of data to generate ("classification", "ner", "custom",
            "decoder", "records", "fields").
        domain_description: Description of what kind of data to generate.
        num_examples: Number of examples to generate.
        labels: Optional list of labels (for classification/NER).
        save_to_cloud: If True, save the generated dataset to Felix cloud storage.
            Automatically set to True when dataset_name is provided.
        dataset_name: Name for the saved dataset. When provided, the dataset is
            automatically saved to cloud and can be fetched with fetch_dataset().
        **kwargs: Additional generation parameters.
            For decoder: instruction (str) - optional system instruction for chat messages.
            For records: fields (list[dict]) - field definitions.
            For fields: input_fields / output_fields (list[dict]).

    Returns:
        List of generated data records. For NER datasets, each record has:
        - 'text': the input text
        - 'entities': list of [text_span, label] pairs, e.g. [["New York", "Location"], ["March", "Date"]]

    Example:
        >>> # Generate and save classification data
        >>> data = generate_data(
        ...     "classification",
        ...     "Customer support tickets about software bugs",
        ...     num_examples=50,
        ...     labels=["bug", "feature_request", "question"],
        ...     dataset_name="support-tickets"
        ... )
        >>>
        >>> # Generate NER data
        >>> ner_data = generate_data(
        ...     "ner",
        ...     "Tech news articles",
        ...     num_examples=20,
        ...     labels=["COMPANY", "PRODUCT", "PERSON"],
        ...     dataset_name="tech-ner-data"
        ... )
        >>> # Each record: {"text": "...", "entities": [["Apple", "COMPANY"], ["iPhone", "PRODUCT"]]}
        >>>
        >>> # Generate decoder (instruction-tuning) data
        >>> decoder_data = generate_data(
        ...     "decoder",
        ...     "Customer support conversations about SaaS products",
        ...     num_examples=100,
        ...     instruction="You are a helpful support agent.",
        ...     dataset_name="support-chatbot"
        ... )
    """
    # Auto-enable save_to_cloud when dataset_name is provided
    if save_to_cloud is None:
        save_to_cloud = dataset_name is not None

    if not dataset_name:
        raise ValueError("dataset_name is required for generate_data()")

    payload: Dict[str, Any] = {
        "task_type": task_type,
        "dataset_name": dataset_name,
        "num_examples": num_examples,
        "use_meta_felix": False,
    }

    if task_type in ("classification", "ner", "decoder", "records", "fields"):
        payload["domain_description"] = domain_description
    else:
        payload["prompt"] = domain_description

    if labels:
        payload["labels"] = labels

    if save_to_cloud:
        payload["save_dataset"] = True

    payload.update(kwargs)

    create_result = _make_request(
        "POST", "/generate", data=payload,
        params={"is_seed": "false"}, timeout=30,
    )
    job_id = create_result.get("job_id")
    if not job_id:
        raise Exception("No job_id returned from /generate")

    return _poll_generation_job(job_id)


def label_existing_data(
    dataset_type: str,
    inputs: List[str],
    labels: List[str],
    domain_description: Optional[str] = None,
    multi_label: bool = False,
    temperature: float = 0.3,
    save_to_cloud: Optional[bool] = None,
    dataset_name: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Label existing text data with NER entities or classification labels.

    Takes a list of raw text inputs and annotates them using AI, applying the
    specified entity types (NER) or class labels (classification).

    Args:
        dataset_type: Type of labeling task ("ner" or "classification").
        inputs: List of text strings to label.
        labels: List of labels to apply (entity types for NER, class names for classification).
        domain_description: Optional description of the domain for better labeling accuracy.
        multi_label: Allow multiple labels per sample (classification only, default False).
        temperature: Generation temperature (0.0-2.0, default 0.3 for consistent labeling).
        save_to_cloud: If True, save the labeled dataset to Felix cloud storage.
        dataset_name: Name for the saved dataset (required if save_to_cloud=True).
        **kwargs: Additional labeling parameters.

    Returns:
        List of labeled data records.

    Raises:
        ValueError: If dataset_type is not "ner" or "classification", or if
            save_to_cloud=True but dataset_name is not provided.

    Example:
        >>> # Label text with NER entities
        >>> labeled = label_existing_data(
        ...     "ner",
        ...     inputs=["Apple CEO Tim Cook announced new products in Cupertino."],
        ...     labels=["PERSON", "ORG", "LOC"],
        ...     domain_description="Tech news articles",
        ... )
        >>>
        >>> # Label text with classification labels and save
        >>> labeled = label_existing_data(
        ...     "classification",
        ...     inputs=["Great product, love it!", "Broke after one day."],
        ...     labels=["positive", "negative", "neutral"],
        ...     save_to_cloud=True,
        ...     dataset_name="my-labeled-reviews",
        ... )
    """
    valid_types = ("ner", "classification")
    if dataset_type not in valid_types:
        raise ValueError(
            f"dataset_type must be one of {valid_types}, got '{dataset_type}'"
        )

    endpoint = f"/generate/{_urllib_parse.quote(dataset_type)}/label-existing"

    payload: Dict[str, Any] = {
        "inputs": inputs,
        "labels": labels,
        "temperature": temperature,
        "use_meta_felix": False,  # Notebook helpers use basic Felix
    }

    if domain_description:
        payload["domain_description"] = domain_description

    if dataset_type == "classification" and multi_label:
        payload["multi_label"] = multi_label

    # Auto-enable save_to_cloud when dataset_name is provided
    if save_to_cloud is None:
        save_to_cloud = dataset_name is not None

    if save_to_cloud:
        payload["save_dataset"] = True
        if dataset_name:
            payload["dataset_name"] = dataset_name
        else:
            raise ValueError("dataset_name is required when save_to_cloud=True")

    payload.update(kwargs)

    result = _make_request("POST", endpoint, data=payload, timeout=300)
    return result.get("data") or []


# =============================================================================
# Inference Operations
# =============================================================================

def run_inference(
    task: str,
    text: Union[str, List[str]] = None,
    schema: Union[List[str], Dict] = None,
    job_id: Optional[str] = None,
    threshold: float = 0.5,
    include_confidence: bool = False,
    include_spans: bool = False,
    format_results: bool = True,
    multi_label: bool = False,
    top_k: Optional[int] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    store: bool = False,
) -> Dict[str, Any]:
    """
    Run inference using a trained model. Supports encoder (GLiNER) and decoder (Qwen3/Llama) models.

    Args:
        task: Task type:
            - 'extract_entities': Extract named entities (encoder, schema: list of entity types)
            - 'classify_text': Classify text (encoder, schema: dict with classification labels)
            - 'extract_json': Extract structured JSON (encoder, schema: dict with field specs)
            - 'schema': Combined extraction (encoder)
            - 'generate_text': Generate text with a decoder model (requires messages).
              Omit job_id to use the base Qwen3-8B model.
        text: Text to process (for encoder tasks). Single string or list for batch.
        schema: Schema for extraction (for encoder tasks, format depends on task type).
        job_id: Training job ID or base model identifier. For 'generate_text': omit to
            use the base Qwen3-8B model. For encoder tasks, omit for base GLiNER-2.
        threshold: Confidence threshold (0.0-1.0, default: 0.5). Encoder only.
        include_confidence: Include confidence scores in results. Encoder only.
        include_spans: Include character-level span offsets in entity results. Encoder only.
        format_results: Format results for readability (default: True). Encoder only.
        multi_label: Return all labels above threshold. Only for 'classify_text'.
        top_k: Return top k labels by confidence. Only for 'classify_text'.
        messages: Chat messages for decoder inference. List of dicts with 'role' and 'content'.
                 Roles: 'system', 'user', 'assistant'. Last message must be 'user'.
        max_tokens: Maximum tokens to generate. Only for 'generate_text'.
        temperature: Sampling temperature (0.0-2.0). Only for 'generate_text'.
        top_p: Top-p sampling parameter (0.0-1.0). Only for 'generate_text'.
        store: Whether to store this inference in history (default: False). Pass True to
            persist the inference for adaptive fine-tuning feedback loops.

    Returns:
        For encoder tasks: Dict with 'result', 'token_usage', 'model_used'.
        For generate_text: Dict with 'completion', 'model_id', 'latency_ms'.

    Example:
        >>> # Extract entities using base model
        >>> result = run_inference(
        ...     task="extract_entities",
        ...     text="Apple CEO Tim Cook announced the new iPhone.",
        ...     schema=["company", "person", "product"]
        ... )
        >>>
        >>> # Classify text using a trained model
        >>> result = run_inference(
        ...     task="classify_text",
        ...     text="This product is amazing!",
        ...     schema={"categories": ["positive", "negative", "neutral"]},
        ...     job_id="your-training-job-id"
        ... )
        >>>
        >>> # Generate text with a fine-tuned decoder model
        >>> result = run_inference(
        ...     task="generate_text",
        ...     messages=[
        ...         {"role": "system", "content": "You are a Python expert."},
        ...         {"role": "user", "content": "Explain decorators."}
        ...     ],
        ...     job_id="decoder-model-job-id",
        ...     max_tokens=512
        ... )
        >>> print(result['completion'])
        >>>
        >>> # Generate text with the base Qwen3-8B model (no fine-tuning needed)
        >>> result = run_inference(
        ...     task="generate_text",
        ...     messages=[{"role": "user", "content": "What is machine learning?"}],
        ...     max_tokens=256
        ... )
        >>> print(result['completion'])
    """
    # Decoder (text generation) inference
    if task == "generate_text":
        if not messages:
            raise ValueError("messages required for generate_text task.")
        effective_model_id = job_id or "base:Qwen/Qwen3-8B"
        payload = {
            "model_id": effective_model_id,
            "task": "generate",
            "messages": messages,
            "store": store,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        return _make_request("POST", "/inference", data=payload, timeout=120)

    # Encoder inference
    # For classify_text with multi_label or top_k, add flags to schema
    if task == "classify_text" and (multi_label or top_k):
        if isinstance(schema, dict):
            schema = {**schema}
        else:
            schema = {"categories": schema}
        if multi_label:
            schema["multi_label"] = True
        if top_k:
            schema["top_k"] = top_k

    payload = {
        "model_id": job_id or "base",
        "task": task,
        "text": text,
        "schema": schema,
        "threshold": threshold,
        "include_confidence": include_confidence,
        "include_spans": include_spans,
        "format_results": format_results,
        "store": store,
    }

    return _make_request("POST", "/inference", data=payload, timeout=120)


# =============================================================================
# Analysis Operations
# =============================================================================

def query_dataset(
    data: List[Dict],
    question: str,
    columns: Optional[List[str]] = None,
) -> str:
    """
    Ask a natural language question about dataset data.

    Args:
        data: Sample of data rows (list of dicts or DataFrame).
        question: Natural language question about the data.
        columns: Optional list of column names (auto-detected if not provided).

    Returns:
        Answer to the question.

    Example:
        >>> df = fetch_dataset("customer-feedback")
        >>> answer = query_dataset(
        ...     df.head(100).to_dict('records'),
        ...     "What are the most common complaint topics?"
        ... )
        >>> print(answer)
    """
    # Convert DataFrame to list if needed
    if hasattr(data, "to_dict"):
        records = data.to_dict(orient="records")
    else:
        records = list(data)

    # Auto-detect columns if not provided
    if columns is None and records:
        columns = list(records[0].keys())

    payload = {
        "data": records[:100],  # Limit to 100 rows to avoid token limits
        "question": question,
        "columns": columns or [],
    }

    result = _make_request("POST", "/felix/dataset/query", data=payload, timeout=120)
    return result.get("answer", "")


# =============================================================================
# Deployment Operations
# =============================================================================

def list_deployments(
    training_job_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List all deployments for the current user.

    Args:
        training_job_id: Optional filter by training job ID.
        status: Optional filter by status ('pending', 'deploying', 'active', 'failed', 'stopped').

    Returns:
        List of deployment objects with keys: id, name, provider, status, endpoint_url, etc.

    Example:
        >>> deployments = list_deployments()
        >>> for d in deployments:
        ...     print(f"{d['name']}: {d['status']}")
        >>>
        >>> # Filter by status
        >>> active = list_deployments(status='active')
    """
    params = {}
    if training_job_id:
        params["training_job_id"] = training_job_id
    if status:
        params["status"] = status

    result = _make_request("GET", "/felix/deployments", params=params)
    return result.get("deployments") or []


def get_deployment(deployment_id: str) -> Dict[str, Any]:
    """
    Get details of a specific deployment.

    Args:
        deployment_id: The deployment UUID.

    Returns:
        Deployment details dict with keys: id, name, provider, status, endpoint_url,
        training_job_id, model_name, instance_type, region, etc.

    Example:
        >>> dep = get_deployment("deployment-123")
        >>> print(f"Status: {dep['status']}, Endpoint: {dep.get('endpoint_url')}")
    """
    endpoint = f"/felix/deployments/{_urllib_parse.quote(deployment_id)}"
    return _make_request("GET", endpoint)


def create_deployment(
    training_job_id: str,
    name: str,
    provider: str = "fastino",
    instance_type: Optional[str] = None,
    region: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a new deployment for a trained model.

    Deploy your trained model to Fastino Compute or HuggingFace Hub
    for inference.

    Args:
        training_job_id: ID of the completed training job to deploy.
        name: Name for the deployment.
        provider: Deployment provider - 'fastino' or 'huggingface' (default: fastino).
        instance_type: Instance type for deployment (optional, uses default if not specified).
        region: Deployment region (optional, uses default if not specified).
        config: Provider-specific configuration (optional).
                For HuggingFace: {"repo_name": "my-model", "hf_token": "hf_xxx", "private": True}

    Returns:
        Deployment response with keys: success, deployment (dict), message.

    Example:
        >>> # Deploy to Fastino Compute
        >>> result = create_deployment(
        ...     training_job_id="job-123",
        ...     name="my-model-prod"
        ... )
        >>> print(f"Deployed! Endpoint: {result['deployment']['endpoint_url']}")
        >>>
        >>> # Deploy to HuggingFace Hub
        >>> result = create_deployment(
        ...     training_job_id="job-123",
        ...     name="my-hf-model",
        ...     provider="huggingface",
        ...     config={"repo_name": "my-org/my-model", "hf_token": "hf_xxx"}
        ... )
    """
    payload = {
        "training_job_id": training_job_id,
        "name": name,
        "provider": provider,
    }
    if instance_type:
        payload["instance_type"] = instance_type
    if region:
        payload["region"] = region
    if config:
        payload["config"] = config

    return _make_request("POST", "/felix/deployments", data=payload, timeout=120)


def delete_deployment(deployment_id: str) -> Dict[str, Any]:
    """
    Delete/stop a deployment.

    Args:
        deployment_id: The deployment UUID to delete.

    Returns:
        Success response dict.

    Example:
        >>> delete_deployment("deployment-123")
    """
    endpoint = f"/felix/deployments/{_urllib_parse.quote(deployment_id)}"
    return _make_request("DELETE", endpoint)


def get_deployment_options() -> Dict[str, Any]:
    """
    Get available deployment options (instance types and regions).

    Returns:
        Dict with 'instance_types' and 'regions' lists.

    Example:
        >>> options = get_deployment_options()
        >>> print("Available instance types:")
        >>> for it in options['instance_types']:
        ...     print(f"  {it['id']}: {it['name']}")
    """
    return _make_request("GET", "/felix/deployments/options")


def combine_datasets(
    source_datasets: List[Dict[str, Any]],
    output_name: str,
) -> Dict[str, Any]:
    """Combine multiple datasets into a single new dataset, removing duplicates.

    All source datasets must have the same dataset type (ner, classification,
    custom, or decoder). If the output name already exists, a new version is
    created automatically.

    Args:
        source_datasets: List of datasets to combine. Each must have
            'dataset_name' and optionally 'version' (defaults to latest).
            Example: [{"dataset_name": "batch-1"}, {"dataset_name": "batch-2"}]
        output_name: Name for the combined output dataset.

    Returns:
        Dict with dataset_name, version, sample_size, and source_counts.

    Example:
        >>> result = combine_datasets(
        ...     source_datasets=[
        ...         {"dataset_name": "training-v1"},
        ...         {"dataset_name": "new-targeted-examples"},
        ...     ],
        ...     output_name="training-v2"
        ... )
        >>> print(f"Combined {result['sample_size']} rows into {result['dataset_name']}")
    """
    if len(source_datasets) < 2:
        raise ValueError(
            f"At least 2 source datasets are required to combine. Got {len(source_datasets)}."
        )

    sources = []
    for ds in source_datasets:
        source: Dict[str, Any] = {"dataset_name": ds.get("dataset_name") or ds.get("name", "")}
        if not source["dataset_name"]:
            raise ValueError("Each source dataset must have a 'dataset_name' field.")
        if ds.get("version"):
            source["version"] = ds["version"]
        sources.append(source)

    # Derive dataset_type from the first source before merging.
    # /felix/dataset/augment only accepts "ner" or "classification"; skip dedup
    # for other types (decoder, custom, records, fields).
    _DEDUP_SUPPORTED_TYPES = {"ner", "classification"}
    first_info = _make_request("GET", f"/felix/datasets/{_urllib_parse.quote(sources[0]['dataset_name'])}/latest")
    dataset_type = first_info.get("dataset_type", "custom")

    result = _make_request(
        "POST",
        "/felix/datasets/merge",
        data={"sources": sources, "output_name": output_name},
    )

    dataset_name = result.get("dataset_name", output_name)
    version = result.get("version", "1")
    result["duplicates_removed"] = 0

    # Dedup is only supported for ner/classification
    if dataset_type in _DEDUP_SUPPORTED_TYPES:
        dedup_result = _make_request(
            "POST",
            "/felix/dataset/augment",
            data={
                "task_type": dataset_type,
                "dataset_name": dataset_name,
                "dataset_version": str(version),
                "operations": [{"type": "remove_duplicates", "enabled": True}],
                "new_dataset_name": dataset_name,
            },
        )
        duplicates_removed = (
            dedup_result.get("modifications", {}).get("duplicates_removed", 0)
        )
        merged_rows = result.get("sample_size", 0)
        if merged_rows > 0 and 0 <= duplicates_removed <= merged_rows:
            result["sample_size"] = merged_rows - duplicates_removed
        result["duplicates_removed"] = duplicates_removed

    return result


# =============================================================================
# Utility Functions
# =============================================================================

def felix_help():
    """Print help information about available Felix helper functions."""
    help_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        Felix Notebook Helper Functions                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DATASET OPERATIONS                                                          ║
║  ─────────────────                                                           ║
║  list_datasets()                        - List all available datasets        ║
║  fetch_dataset(name)                    - Fetch dataset as DataFrame         ║
║  get_dataset_info(name)                 - Get dataset metadata               ║
║  preview_dataset(name, limit=10)        - Preview first N rows               ║
║  upload_dataset(data, name, dataset_type) - Upload DataFrame as new dataset  ║
║  delete_dataset(name)                   - Delete a dataset                   ║
║  combine_datasets(source_datasets, output_name) - Merge datasets, dedup     ║
║                                                                              ║
║  TRAINING OPERATIONS                                                         ║
║  ───────────────────                                                         ║
║  list_models()                          - List all trained models            ║
║  list_training_jobs()                   - List all training jobs             ║
║  get_training_status(job_id)            - Get training job status            ║
║  start_training(model_name, datasets)   - Start a new training job           ║
║                                                                              ║
║  INFERENCE OPERATIONS                                                        ║
║  ────────────────────                                                        ║
║  run_inference(task, text, schema, job_id, ...) - Run model inference        ║
║    Encoder tasks: extract_entities, classify_text, extract_json, schema      ║
║    Decoder tasks: generate_text (pass messages=, max_tokens=)                ║
║    job_id: trained model ID, omit for base model                             ║
║    multi_label=True: all labels above threshold                              ║
║    top_k=N: top N labels by confidence                                       ║
║                                                                              ║
║  EVALUATION OPERATIONS                                                       ║
║  ─────────────────────                                                       ║
║  list_evaluations()                     - List all evaluations               ║
║  run_evaluation(model_id, dataset_name) - Run model evaluation               ║
║  get_evaluation_results(id)             - Get evaluation results             ║
║  analyze_failures(eval_id)              - Analyze failed predictions         ║
║  get_failure_stats(eval_id)             - Get failure statistics             ║
║  delete_evaluation(eval_id)             - Delete an evaluation               ║
║                                                                              ║
║  GENERATION OPERATIONS                                                       ║
║  ─────────────────────                                                       ║
║  generate_data(task_type, domain_description, ...) - Generate synthetic data ║
║    task_type: classification, ner, decoder, custom, records, fields          ║
║    labels: required for classification/ner                                   ║
║    instruction: system prompt for decoder                                    ║
║    dataset_name: auto-saves to cloud when provided                           ║
║  combine_datasets(source_datasets, output_name) - Merge & dedup datasets    ║
║                                                                              ║
║  ANALYSIS OPERATIONS                                                         ║
║  ───────────────────                                                         ║
║  query_dataset(data, question)          - Ask questions about data           ║
║                                                                              ║
║  DEPLOYMENT OPERATIONS                                                       ║
║  ─────────────────────                                                       ║
║  list_deployments()                     - List all deployments               ║
║  get_deployment(id)                     - Get deployment details             ║
║  create_deployment(training_job_id, name) - Deploy a trained model           ║
║  delete_deployment(id)                  - Delete/stop a deployment           ║
║  get_deployment_options()               - Get available instance types       ║
║                                                                              ║
║  For detailed help on any function, use: help(function_name)                 ║
║  Example: help(generate_data)                                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(help_text)


'''


def get_felix_helpers_code() -> str:
    """Return the felix helpers Python code for injection into the kernel."""
    return FELIX_HELPERS_CODE
