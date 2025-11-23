import logging
from typing import Dict, Any
from collections import Counter

from strategies.base_strategy import ContextStrategy
from core.llm_client import DeepSeekClient
from core.github_client import GithubClient
from core.code_analysis import extract_diff_entities, format_diff_entities_block
from core.pr_processor import get_modified_files_from_pr

from .query_generator import (
    query_generator_prompt,
    build_queries_response,
    language_noise_filter
)
from .search_engine import generate_search_context_for_review

logger = logging.getLogger("LlamaPReview")

class SearchRAGStrategy(ContextStrategy):
    """
    Solution 2: Search-based Context Retrieval (The "Needle in a Haystack" Approach).
    
    Mechanism:
    1. Analyze PR Diff to extract heuristic entities (added symbols, params).
    2. Use LLM to generate targeted GitHub Search queries based on the PR theme.
    3. Refine queries (deduplication, expansion, language filtering).
    4. Execute searches via GitHub API.
    5. Extract and assemble relevant code snippets into a context block.
    
    Pros: Fast, covers the whole repo.
    Cons: Lower recall on implicit dependencies, no semantic understanding.
    """

    def __init__(self, llm_client: DeepSeekClient, github_client: GithubClient):
        self.llm = llm_client
        self.github = github_client

    def execute(self, pr_details: str, pr_content: Dict[str, Any], repo_full_name: str) -> str:
        """
        Execute the Search RAG pipeline.
        
        Args:
            pr_details: Markdown description of the PR.
            pr_content: JSON object of the PR data (files, diffs, etc.).
            repo_full_name: "owner/repo" string.
            
        Returns:
            A string containing the assembled context (code snippets, summaries).
        """
        logger.info("🚀 Starting Search RAG Pipeline...")

        # 1. Heuristic Analysis: Extract entities from Diff
        # This helps ground the LLM with specific symbols found in the code changes.
        logger.info("   Analyzing Diff for entities...")
        entities = extract_diff_entities(pr_content)
        diff_entities_block = format_diff_entities_block(entities)
        logger.debug(f"   Extracted entities summary: {diff_entities_block.replace(chr(10), '; ')}")

        # 2. Detect Primary Language
        # Needed for language-specific query filtering and prompt hints.
        primary_language = self._infer_primary_language(pr_content)
        logger.info(f"   Inferred primary language: {primary_language}")

        # 3. Generate Queries (LLM)
        # Ask the LLM to brainstorm search queries based on the PR details.
        logger.info("   Generating search queries via LLM...")
        try:
            response_json_str = self.llm.query_with_template(
                template=query_generator_prompt,
                parameters={
                    "pr_details": pr_details,
                    "diff_entities_block": diff_entities_block,
                    "primary_language": primary_language
                },
                model="deepseek-chat", # Use V3 for fast query generation
                response_type="json_object",
                temperature=0.2
            )
        except Exception as e:
            logger.error(f"   LLM Query Generation failed: {e}")
            return "Error: Failed to generate search queries."

        if not response_json_str:
            logger.warning("   LLM returned empty response.")
            return "Error: LLM returned no queries."

        # 4. Query Refinement & Assembly
        # Clean up LLM output, add "call variants" (e.g. "func("), and filter noise.
        logger.info("   Refining and assembling queries...")
        queries_response = build_queries_response(
            response=response_json_str,
            entities=entities,
            max_queries=5
        )
        
        # Filter out cross-language noise (e.g., "extends" in Python)
        queries_response = language_noise_filter(
            queries_json=queries_response,
            primary_language=primary_language,
            pr_content=pr_content
        )

        # 5. Execute Search & Context Assembly
        # Run the queries against GitHub API and assemble the results.
        logger.info("   Executing GitHub searches and assembling context...")
        
        # Get list of modified files to potentially exclude them from results 
        # (we usually want external context, not what's already in the PR)
        pr_modified_files = get_modified_files_from_pr(pr_content)
        
        search_context = generate_search_context_for_review(
            queries_response=queries_response,
            github_client=self.github,
            repo_full_name=repo_full_name,
            pr_modified_files=pr_modified_files,
            primary_language=primary_language
        )

        if not search_context:
            logger.warning("   No context found after search.")
            return "No relevant context found via search."

        logger.info(f"✅ Search RAG Complete. Context size: {len(search_context)} chars.")
        return search_context

    def _infer_primary_language(self, pr_content: Dict[str, Any]) -> str:
        """
        Infer the primary language of the PR based on file extensions in file_changes.
        Falls back to "Unknown" if no clear winner.
        """
        extensions = []
        file_changes = pr_content.get('file_changes', [])
        
        if not file_changes:
            return "Unknown"

        for fc in file_changes:
            path = fc.get('file_path', '')
            if '.' in path:
                ext = path.split('.')[-1].lower()
                extensions.append(ext)
        
        if not extensions:
            return "Unknown"

        # Map extensions to Languages
        ext_map = {
            'py': 'Python',
            'js': 'JavaScript', 'jsx': 'JavaScript', 'mjs': 'JavaScript',
            'ts': 'TypeScript', 'tsx': 'TypeScript',
            'java': 'Java',
            'go': 'Go',
            'rs': 'Rust',
            'rb': 'Ruby',
            'php': 'PHP',
            'cs': 'C#',
            'cpp': 'C++', 'c': 'C', 'h': 'C++', 'hpp': 'C++',
            'swift': 'Swift',
            'kt': 'Kotlin'
        }

        langs = [ext_map.get(ext) for ext in extensions if ext in ext_map]
        
        if not langs:
            return "Unknown"
            
        # Return the most common language
        most_common = Counter(langs).most_common(1)
        return most_common[0][0]
