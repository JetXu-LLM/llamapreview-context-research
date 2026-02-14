import logging
import os
from typing import Any, Dict, Optional

import requests

from core.github_client import GithubClient
from core.llm_client import DeepSeekClient
from strategies.base_strategy import ContextStrategy

logger = logging.getLogger("LlamaPReview")


class CodeMeshConfigurationError(RuntimeError):
    """Raised when Code Mesh adapter is not properly configured."""


class CodeMeshExecutionError(RuntimeError):
    """Raised when the proprietary Code Mesh endpoint call fails."""


class CodeMeshStrategy(ContextStrategy):
    """
    Closed-core adapter for deterministic context retrieval.

    This repository intentionally does not include the core graph RAG engine.
    Instead, this strategy calls a private Code Mesh service endpoint.

    Required env vars:
      - CODE_MESH_ENDPOINT: HTTPS endpoint for context generation

    Optional env vars:
      - CODE_MESH_API_KEY: bearer token for endpoint auth
      - CODE_MESH_TIMEOUT_SEC: request timeout seconds (default: 60)
    """

    ENDPOINT_ENV = "CODE_MESH_ENDPOINT"
    API_KEY_ENV = "CODE_MESH_API_KEY"
    TIMEOUT_ENV = "CODE_MESH_TIMEOUT_SEC"

    def __init__(
        self,
        llm_client: DeepSeekClient,
        github_client: GithubClient,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_sec: Optional[int] = None,
    ):
        super().__init__(llm_client=llm_client, github_client=github_client)
        self.endpoint = (endpoint or os.getenv(self.ENDPOINT_ENV, "")).strip()
        self.api_key = (api_key or os.getenv(self.API_KEY_ENV, "")).strip()

        if timeout_sec is not None:
            self.timeout_sec = timeout_sec
        else:
            raw_timeout = os.getenv(self.TIMEOUT_ENV, "60").strip()
            self.timeout_sec = int(raw_timeout) if raw_timeout.isdigit() else 60

    @property
    def name(self) -> str:
        return "Code Mesh (Closed-Core Adapter)"

    def _validate_configuration(self) -> None:
        if not self.endpoint:
            raise CodeMeshConfigurationError(
                "Code Mesh strategy is unavailable: set CODE_MESH_ENDPOINT to your private engine endpoint. "
                "This repository ships only the adapter, not the core graph engine."
            )

    def _build_payload(self, pr_details: str, pr_content: Dict[str, Any], repo_full_name: str) -> Dict[str, Any]:
        """Build the stable adapter contract sent to the proprietary backend."""
        return {
            "repo_full_name": repo_full_name,
            "pr_details": pr_details,
            "pr_content": pr_content,
            "options": {
                "mode": "review_context",
                "include_trace": False,
            },
        }

    def execute(self, pr_details: str, pr_content: Dict[str, Any], repo_full_name: str) -> str:
        self._validate_configuration()

        payload = self._build_payload(pr_details, pr_content, repo_full_name)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.info("🚀 Starting Code Mesh retrieval via closed-core adapter...")

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout_sec,
            )
        except requests.Timeout as exc:
            raise CodeMeshExecutionError(
                f"Code Mesh endpoint timed out after {self.timeout_sec}s."
            ) from exc
        except requests.RequestException as exc:
            raise CodeMeshExecutionError(
                f"Failed to call Code Mesh endpoint: {exc}"
            ) from exc

        if response.status_code >= 400:
            message = response.text[:500].strip()
            raise CodeMeshExecutionError(
                f"Code Mesh endpoint returned HTTP {response.status_code}: {message}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise CodeMeshExecutionError("Code Mesh endpoint returned non-JSON response.") from exc

        context = data.get("context")
        if not context and isinstance(data.get("data"), dict):
            context = data["data"].get("context")

        if not isinstance(context, str) or not context.strip():
            raise CodeMeshExecutionError(
                "Code Mesh response missing 'context' string field. "
                "Expected {'context': '<text>'} or {'data': {'context': '<text>'}}."
            )

        logger.info(f"✅ Code Mesh retrieval complete. Context size: {len(context)} chars.")
        return context
