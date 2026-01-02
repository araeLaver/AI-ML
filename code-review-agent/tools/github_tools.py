"""GitHub integration tools for fetching PR data and posting comments."""
import os
from dataclasses import dataclass
from github import Github, PullRequest
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


@dataclass
class PRFile:
    """Represents a file changed in a PR."""
    filename: str
    status: str  # added, modified, removed
    additions: int
    deletions: int
    patch: str | None
    content: str | None = None


@dataclass
class PRData:
    """Represents PR data for review."""
    number: int
    title: str
    body: str
    base_branch: str
    head_branch: str
    files: list[PRFile]
    author: str


class GitHubTools:
    """Tools for interacting with GitHub PRs."""

    def __init__(self, token: str | None = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub token required")
        self.client = Github(self.token)

    def get_pr_data(self, repo_name: str, pr_number: int) -> PRData:
        """Fetch PR data including changed files.

        Args:
            repo_name: Repository name (owner/repo)
            pr_number: Pull request number

        Returns:
            PRData with all PR information
        """
        repo = self.client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)

        files = []
        for f in pr.get_files():
            content = None
            if f.status != "removed":
                try:
                    content = repo.get_contents(f.filename, ref=pr.head.sha).decoded_content.decode()
                except Exception:
                    content = None

            files.append(PRFile(
                filename=f.filename,
                status=f.status,
                additions=f.additions,
                deletions=f.deletions,
                patch=f.patch,
                content=content
            ))

        return PRData(
            number=pr.number,
            title=pr.title,
            body=pr.body or "",
            base_branch=pr.base.ref,
            head_branch=pr.head.ref,
            files=files,
            author=pr.user.login
        )

    def post_review_comment(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        event: str = "COMMENT"
    ) -> None:
        """Post a review comment on a PR.

        Args:
            repo_name: Repository name (owner/repo)
            pr_number: Pull request number
            body: Comment body
            event: APPROVE, REQUEST_CHANGES, or COMMENT
        """
        repo = self.client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        pr.create_review(body=body, event=event)

    def post_line_comment(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        path: str,
        line: int
    ) -> None:
        """Post a comment on a specific line.

        Args:
            repo_name: Repository name (owner/repo)
            pr_number: Pull request number
            body: Comment body
            path: File path
            line: Line number
        """
        repo = self.client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        commit = repo.get_commit(pr.head.sha)
        pr.create_review_comment(body=body, commit=commit, path=path, line=line)


# LangChain Tool Wrappers
class GetPRInput(BaseModel):
    repo_name: str = Field(description="Repository name (owner/repo)")
    pr_number: int = Field(description="Pull request number")


class GetPRTool(BaseTool):
    """LangChain tool for fetching PR data."""
    name: str = "get_pull_request"
    description: str = "Fetch pull request data including changed files and patches"
    args_schema: type[BaseModel] = GetPRInput

    github_tools: GitHubTools = None

    def __init__(self, github_tools: GitHubTools):
        super().__init__()
        self.github_tools = github_tools

    def _run(self, repo_name: str, pr_number: int) -> str:
        pr_data = self.github_tools.get_pr_data(repo_name, pr_number)
        return f"PR #{pr_data.number}: {pr_data.title}\nFiles: {len(pr_data.files)}"
