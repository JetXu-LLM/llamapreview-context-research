from abc import ABC, abstractmethod
from typing import Dict, Any

class ContextStrategy(ABC):
    """Abstract base class for all context retrieval strategies."""

    @abstractmethod
    def execute(self, pr_details: str, pr_content: Dict[str, Any], repo_full_name: str) -> str:
        """
        Execute the strategy to retrieve context.
        
        Args:
            pr_details: Markdown description of the PR.
            pr_content: JSON object of the PR data.
            repo_full_name: "owner/repo" string.
            
        Returns:
            A string containing the assembled context (code snippets, summaries, etc.).
        """
        pass
