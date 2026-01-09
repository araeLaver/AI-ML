"""GitHub webhook handler for automated PR reviews with error handling and retry logic."""
import asyncio
import hashlib
import hmac
import logging
import os
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])

# In-memory job tracking (use Redis/DB in production)
_review_jobs: dict[str, dict] = {}


class WebhookPayload(BaseModel):
    """GitHub webhook payload model."""
    action: str
    number: int | None = None
    pull_request: dict | None = None
    repository: dict | None = None


class ReviewJobStatus(BaseModel):
    """Review job status model."""
    job_id: str
    status: str  # pending, processing, completed, failed
    repo: str
    pr_number: int
    started_at: str | None = None
    completed_at: str | None = None
    files_reviewed: int = 0
    files_total: int = 0
    errors: list[str] = []
    result_url: str | None = None


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature.

    Args:
        payload: Raw request body
        signature: X-Hub-Signature-256 header
        secret: Webhook secret

    Returns:
        True if signature is valid
    """
    if not signature:
        return False

    expected = "sha256=" + hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


async def process_pr_review_with_retry(
    repo_name: str,
    pr_number: int,
    job_id: str,
    max_retries: int = 3,
    timeout: float = 300.0
) -> None:
    """Background task to process PR review with retry logic.

    Args:
        repo_name: Repository name (owner/repo)
        pr_number: Pull request number
        job_id: Unique job identifier
        max_retries: Maximum retry attempts
        timeout: Overall timeout in seconds
    """
    from langchain_openai import ChatOpenAI
    from tools.github_tools import GitHubTools
    from workflows.review_workflow import run_review

    # Update job status
    _review_jobs[job_id] = {
        "status": "processing",
        "repo": repo_name,
        "pr_number": pr_number,
        "started_at": datetime.utcnow().isoformat(),
        "files_reviewed": 0,
        "files_total": 0,
        "errors": []
    }

    try:
        # Initialize with configurable model
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        llm = ChatOpenAI(model=model_name, temperature=0, request_timeout=60)

        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_TOKEN environment variable not set")

        github = GitHubTools(token=github_token)

        # Fetch PR data with retry
        pr_data = None
        for attempt in range(max_retries):
            try:
                pr_data = github.get_pr_data(repo_name, pr_number)
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to fetch PR data: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

        if not pr_data:
            raise ValueError("Failed to fetch PR data")

        # Filter reviewable files
        reviewable_files = [
            f for f in pr_data.files
            if f.content and f.status != "removed" and _is_reviewable_file(f.filename)
        ]

        _review_jobs[job_id]["files_total"] = len(reviewable_files)
        logger.info(f"Starting review of {len(reviewable_files)} files for PR #{pr_number}")

        # Review each file with timeout and error handling
        all_reviews = []
        failed_files = []

        for idx, file in enumerate(reviewable_files):
            try:
                # Per-file timeout
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda f=file: run_review(llm, f.content, {
                            "file_path": f.filename,
                            "language": _detect_language(f.filename)
                        })
                    ),
                    timeout=timeout / max(len(reviewable_files), 1)
                )

                all_reviews.append({
                    "file": file.filename,
                    "report": result.get("final_report", ""),
                    "security": result.get("security_analysis", {}),
                    "performance": result.get("performance_analysis", {}),
                    "style": result.get("style_analysis", {})
                })

                _review_jobs[job_id]["files_reviewed"] = idx + 1
                logger.info(f"Reviewed {idx + 1}/{len(reviewable_files)}: {file.filename}")

            except asyncio.TimeoutError:
                logger.error(f"Timeout reviewing {file.filename}")
                failed_files.append({"file": file.filename, "error": "Timeout"})
                _review_jobs[job_id]["errors"].append(f"Timeout: {file.filename}")

            except Exception as e:
                logger.error(f"Failed to review {file.filename}: {e}")
                failed_files.append({"file": file.filename, "error": str(e)})
                _review_jobs[job_id]["errors"].append(f"{file.filename}: {e}")

        # Generate and post combined report
        if all_reviews or failed_files:
            combined_report = format_combined_report(pr_data, all_reviews, failed_files)

            # Post review with retry
            for attempt in range(max_retries):
                try:
                    github.post_review_comment(repo_name, pr_number, combined_report)
                    logger.info(f"Posted review comment for PR #{pr_number}")
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed to post comment: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        _review_jobs[job_id]["errors"].append(f"Failed to post comment: {e}")

        # Mark as completed
        _review_jobs[job_id]["status"] = "completed" if not failed_files else "completed_with_errors"
        _review_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        logger.error(f"PR review failed for {repo_name}#{pr_number}: {e}")
        _review_jobs[job_id]["status"] = "failed"
        _review_jobs[job_id]["errors"].append(str(e))
        _review_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()


def _is_reviewable_file(filename: str) -> bool:
    """Check if file should be reviewed."""
    reviewable_extensions = {
        ".py", ".js", ".ts", ".tsx", ".jsx",
        ".java", ".go", ".rs", ".cpp", ".c",
        ".rb", ".php", ".swift", ".kt"
    }
    return any(filename.endswith(ext) for ext in reviewable_extensions)


def _detect_language(filename: str) -> str:
    """Detect programming language from filename."""
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin"
    }
    for ext, lang in ext_map.items():
        if filename.endswith(ext):
            return lang
    return "unknown"


def format_combined_report(pr_data: Any, reviews: list[dict], failed_files: list[dict] | None = None) -> str:
    """Format all file reviews into a single PR comment."""
    # Count issues across all files
    total_critical = 0
    total_high = 0
    total_medium = 0

    for review in reviews:
        security = review.get("security", {})
        for finding in security.get("findings", []):
            severity = finding.get("severity", "").upper()
            if severity == "CRITICAL":
                total_critical += 1
            elif severity == "HIGH":
                total_high += 1
            elif severity == "MEDIUM":
                total_medium += 1

    # Determine overall verdict
    if total_critical > 0:
        verdict_emoji = ""
        verdict_text = "REQUEST_CHANGES"
    elif total_high > 2:
        verdict_emoji = ""
        verdict_text = "REQUEST_CHANGES"
    elif total_high > 0 or total_medium > 3:
        verdict_emoji = ""
        verdict_text = "COMMENT"
    else:
        verdict_emoji = ""
        verdict_text = "APPROVE"

    report = f"""## AI Code Review {verdict_emoji}

**PR:** #{pr_data.number} - {pr_data.title}
**Files reviewed:** {len(reviews)}{f" ({len(failed_files)} failed)" if failed_files else ""}
**Verdict:** {verdict_text}

### Issue Summary
| Severity | Count |
|----------|-------|
| Critical | {total_critical} |
| High | {total_high} |
| Medium | {total_medium} |

---

"""

    for review in reviews:
        report += f"### `{review['file']}`\n\n"
        report += review.get("report", "No report generated")
        report += "\n\n---\n\n"

    if failed_files:
        report += "### Failed Files\n\n"
        for f in failed_files:
            report += f"- `{f['file']}`: {f['error']}\n"
        report += "\n---\n\n"

    report += "\n*Generated by [Code Review Agent](https://github.com/your-repo/code-review-agent)*"
    return report


@router.post("/github")
async def handle_github_webhook(
    request: Request,
    background_tasks: BackgroundTasks
) -> dict:
    """Handle incoming GitHub webhooks.

    Triggers automated PR review on PR open/synchronize events.
    """
    # Verify signature
    secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")
    signature = request.headers.get("X-Hub-Signature-256", "")
    body = await request.body()

    if secret and not verify_signature(body, signature, secret):
        logger.warning("Invalid webhook signature received")
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse payload
    try:
        payload = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse webhook payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event = request.headers.get("X-GitHub-Event", "")
    logger.info(f"Received GitHub event: {event}")

    # Handle PR events
    if event == "pull_request":
        action = payload.get("action")
        if action in ("opened", "synchronize", "reopened"):
            pr = payload.get("pull_request", {})
            repo = payload.get("repository", {})

            repo_name = repo.get("full_name")
            pr_number = pr.get("number")

            if repo_name and pr_number:
                # Generate unique job ID
                job_id = f"{repo_name.replace('/', '_')}_{pr_number}_{datetime.utcnow().timestamp()}"

                # Initialize job status
                _review_jobs[job_id] = {
                    "status": "pending",
                    "repo": repo_name,
                    "pr_number": pr_number,
                    "started_at": None,
                    "errors": []
                }

                background_tasks.add_task(
                    process_pr_review_with_retry,
                    repo_name,
                    pr_number,
                    job_id
                )

                logger.info(f"Queued review for {repo_name}#{pr_number}, job_id: {job_id}")

                return {
                    "status": "processing",
                    "message": f"Review queued for PR #{pr_number}",
                    "job_id": job_id
                }

    return {"status": "ignored", "event": event}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict:
    """Get the status of a review job."""
    if job_id not in _review_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _review_jobs[job_id]
    return {
        "job_id": job_id,
        **job
    }


@router.get("/jobs")
async def list_jobs(limit: int = 10) -> dict:
    """List recent review jobs."""
    jobs = list(_review_jobs.items())[-limit:]
    return {
        "jobs": [{"job_id": k, **v} for k, v in jobs],
        "total": len(_review_jobs)
    }
