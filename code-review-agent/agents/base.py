"""Base agent class for code review agents."""
from abc import ABC, abstractmethod
from typing import Any
from langchain_core.language_models import BaseChatModel


class BaseReviewAgent(ABC):
    """Abstract base class for all review agents."""

    def __init__(self, llm: BaseChatModel, name: str):
        self.llm = llm
        self.name = name

    @abstractmethod
    def analyze(self, code: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Analyze code and return findings.

        Args:
            code: Source code to analyze
            context: Optional context (file path, language, etc.)

        Returns:
            Dictionary containing analysis results
        """
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    def _create_analysis_prompt(self, code: str, context: dict[str, Any] | None = None) -> str:
        """Create analysis prompt with code and context."""
        file_info = ""
        if context:
            file_info = f"\nFile: {context.get('file_path', 'unknown')}"
            file_info += f"\nLanguage: {context.get('language', 'unknown')}"

        return f"""Analyze the following code:{file_info}

```
{code}
```

Provide your analysis in a structured format."""
