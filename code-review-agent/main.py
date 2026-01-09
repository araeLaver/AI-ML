"""Code Review Agent - Main entry point with extended API."""
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api import webhook_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Request/Response Models
class ReviewRequest(BaseModel):
    """Code review request model."""
    code: str = Field(..., min_length=1, description="Source code to review")
    language: str = Field(default="auto", description="Programming language")
    file_path: str | None = Field(default=None, description="Optional file path for context")
    check_types: list[str] = Field(
        default=["security", "performance", "style"],
        description="Types of checks to run"
    )


class ReviewResponse(BaseModel):
    """Code review response model."""
    review_id: str
    status: str
    security: dict | None = None
    performance: dict | None = None
    style: dict | None = None
    synthesis: dict | None = None
    final_report: str | None = None
    errors: list[str] | None = None
    duration_ms: int | None = None


class QuickCheckRequest(BaseModel):
    """Quick code check request (no LLM)."""
    code: str = Field(..., min_length=1)
    language: str = Field(default="python")


# In-memory storage for async reviews
_async_reviews: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Code Review Agent starting...")
    logger.info(f"Environment: {os.getenv('APP_ENV', 'production')}")
    yield
    logger.info("Code Review Agent shutting down...")


app = FastAPI(
    title="Code Review Agent",
    description="AI-powered code review using multi-agent system",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - configurable origins
allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(webhook_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "name": "Code Review Agent",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "review": "POST /api/review",
            "quick_check": "POST /api/quick-check",
            "webhook": "POST /api/webhook/github",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "api": "ok",
            "llm": _check_llm_availability()
        }
    }


def _check_llm_availability() -> str:
    """Check if LLM is configured."""
    if os.getenv("OPENAI_API_KEY"):
        return "openai_configured"
    elif os.getenv("OLLAMA_URL"):
        return "ollama_configured"
    return "not_configured"


@app.post("/api/review", response_model=ReviewResponse)
async def review_code(request: ReviewRequest) -> ReviewResponse:
    """Run AI-powered code review.

    Analyzes code for security vulnerabilities, performance issues,
    and style/quality problems using specialized AI agents.
    """
    import time
    from langchain_openai import ChatOpenAI
    from langchain_community.chat_models import ChatOllama
    from agents import ReviewOrchestrator

    start_time = time.time()
    review_id = f"review_{datetime.utcnow().timestamp()}"

    try:
        # Initialize LLM
        if os.getenv("OPENAI_API_KEY"):
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            llm = ChatOpenAI(model=model, temperature=0, request_timeout=60)
        elif os.getenv("OLLAMA_URL"):
            model = os.getenv("OLLAMA_MODEL", "llama3.2")
            llm = ChatOllama(model=model, base_url=os.getenv("OLLAMA_URL"))
        else:
            raise HTTPException(
                status_code=500,
                detail="No LLM configured. Set OPENAI_API_KEY or OLLAMA_URL."
            )

        # Run review
        orchestrator = ReviewOrchestrator(llm, timeout=60.0)
        context = {
            "language": request.language,
            "file_path": request.file_path
        }

        result = orchestrator.review(request.code, context)

        duration_ms = int((time.time() - start_time) * 1000)

        return ReviewResponse(
            review_id=review_id,
            status="completed",
            security=result.get("security"),
            performance=result.get("performance"),
            style=result.get("style"),
            synthesis=result.get("synthesis"),
            final_report=_generate_markdown_report(result),
            errors=result.get("errors"),
            duration_ms=duration_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Review failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/review/async")
async def review_code_async(
    request: ReviewRequest,
    background_tasks: BackgroundTasks
) -> dict:
    """Start an asynchronous code review.

    Returns immediately with a review_id that can be used to check status.
    """
    review_id = f"review_{datetime.utcnow().timestamp()}"

    _async_reviews[review_id] = {
        "status": "pending",
        "started_at": datetime.utcnow().isoformat()
    }

    background_tasks.add_task(
        _run_async_review,
        review_id,
        request.code,
        request.language,
        request.file_path
    )

    return {
        "review_id": review_id,
        "status": "pending",
        "message": "Review started. Check status at /api/review/{review_id}"
    }


async def _run_async_review(
    review_id: str,
    code: str,
    language: str,
    file_path: str | None
) -> None:
    """Background task for async review."""
    import time
    from langchain_openai import ChatOpenAI
    from agents import ReviewOrchestrator

    start_time = time.time()

    try:
        _async_reviews[review_id]["status"] = "processing"

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not configured")

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=model, temperature=0, request_timeout=60)

        orchestrator = ReviewOrchestrator(llm, timeout=60.0)
        result = orchestrator.review(code, {"language": language, "file_path": file_path})

        _async_reviews[review_id].update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "duration_ms": int((time.time() - start_time) * 1000),
            "result": result,
            "final_report": _generate_markdown_report(result)
        })

    except Exception as e:
        logger.error(f"Async review failed: {e}")
        _async_reviews[review_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error": str(e)
        })


@app.get("/api/review/{review_id}")
async def get_review_status(review_id: str) -> dict:
    """Get the status of an async review."""
    if review_id not in _async_reviews:
        raise HTTPException(status_code=404, detail="Review not found")

    return {
        "review_id": review_id,
        **_async_reviews[review_id]
    }


@app.post("/api/quick-check")
async def quick_check(request: QuickCheckRequest) -> dict:
    """Quick code check without LLM (static analysis only).

    Fast, local-only analysis for basic issues.
    """
    from tools.code_analyzer import CodeAnalyzer

    analyzer = CodeAnalyzer()

    # Get metrics
    metrics = analyzer.extract_metrics(request.code, request.language)

    # Find potential issues
    issues = analyzer.find_potential_issues(request.code, request.language)

    return {
        "language": request.language,
        "metrics": {
            "lines_total": metrics.lines_total,
            "lines_code": metrics.lines_code,
            "lines_comment": metrics.lines_comment,
            "functions": metrics.functions,
            "classes": metrics.classes
        },
        "issues": issues,
        "issue_count": len(issues)
    }


@app.get("/api/supported-languages")
async def get_supported_languages() -> dict:
    """Get list of supported programming languages."""
    return {
        "languages": [
            {"code": "python", "name": "Python", "extensions": [".py"]},
            {"code": "javascript", "name": "JavaScript", "extensions": [".js", ".jsx"]},
            {"code": "typescript", "name": "TypeScript", "extensions": [".ts", ".tsx"]},
            {"code": "java", "name": "Java", "extensions": [".java"]},
            {"code": "go", "name": "Go", "extensions": [".go"]},
            {"code": "rust", "name": "Rust", "extensions": [".rs"]},
            {"code": "cpp", "name": "C++", "extensions": [".cpp", ".cc", ".h", ".hpp"]},
            {"code": "c", "name": "C", "extensions": [".c", ".h"]},
            {"code": "ruby", "name": "Ruby", "extensions": [".rb"]},
            {"code": "php", "name": "PHP", "extensions": [".php"]},
            {"code": "swift", "name": "Swift", "extensions": [".swift"]},
            {"code": "kotlin", "name": "Kotlin", "extensions": [".kt"]}
        ]
    }


def _generate_markdown_report(result: dict[str, Any]) -> str:
    """Generate a markdown report from review results."""
    synthesis = result.get("synthesis", {})

    # Get verdict and score
    verdict = synthesis.get("verdict", "COMMENT")
    score = synthesis.get("health_score", 5)
    summary = synthesis.get("summary", "")

    # Count issues
    issue_counts = synthesis.get("issue_counts", {})
    critical = issue_counts.get("critical", 0)
    high = issue_counts.get("high", 0)
    medium = issue_counts.get("medium", 0)
    total = issue_counts.get("total", 0)

    # Verdict emoji
    if verdict == "APPROVE":
        verdict_emoji = ""
    elif verdict == "REQUEST_CHANGES":
        verdict_emoji = ""
    else:
        verdict_emoji = ""

    report = f"""# Code Review Report {verdict_emoji}

## Summary
{summary}

**Verdict:** {verdict}
**Health Score:** {score}/10

## Issue Summary
| Severity | Count |
|----------|-------|
| Critical | {critical} |
| High | {high} |
| Medium | {medium} |
| **Total** | **{total}** |

"""

    # Critical issues
    critical_issues = synthesis.get("critical_issues", [])
    if critical_issues:
        report += "## Critical Issues\n"
        for issue in critical_issues:
            report += f"- {issue}\n"
        report += "\n"

    # Important improvements
    improvements = synthesis.get("important_improvements", [])
    if improvements:
        report += "## Important Improvements\n"
        for imp in improvements:
            report += f"- {imp}\n"
        report += "\n"

    # Minor suggestions
    suggestions = synthesis.get("minor_suggestions", [])
    if suggestions:
        report += "## Minor Suggestions\n"
        for sug in suggestions:
            report += f"- {sug}\n"
        report += "\n"

    report += "\n---\n*Generated by Code Review Agent*"

    return report


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("APP_ENV") == "development"

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
