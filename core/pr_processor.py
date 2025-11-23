import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger("LlamaPReview")

def trim_pr_data(pr_data: dict, max_size: int = 20000) -> dict:
    """
    Trim PR data by first cleaning SVG paths, then removing specific fields if size exceeds the limit,
    and finally cleaning URLs and SHA values if still necessary.
    
    Args:
        pr_data (dict): Pull request data dictionary
        max_size (int): Maximum allowed size 300000
    
    Returns:
        dict: Trimmed PR data dictionary
    """

    def clean_svg_paths(text: str) -> str:
        """
        Replace SVG path data with "..." while preserving the path tag structure.
        
        Args:
            text (str): Input text containing SVG path data
            
        Returns:
            str: Text with simplified SVG path data
        """
        if not isinstance(text, str):
            return text
            
        pattern = r'(<path[^>]*?d=")[^"]*("(?:[^>]*?>|[^>]*\n[^>]*>))'
        
        def replace_path(match):
            start, end = match.groups()
            return start + "..." + end
        
        return re.sub(pattern, replace_path, text, flags=re.DOTALL)

    def clean_dict_svg_paths(data: Any) -> Any:
        """
        Recursively clean SVG paths from all string values in a dictionary or list.
        
        Args:
            data: Input data (can be dict, list, or primitive type)
            
        Returns:
            Data structure with cleaned SVG paths
        """
        if isinstance(data, dict):
            return {k: clean_dict_svg_paths(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [clean_dict_svg_paths(item) for item in data]
        elif isinstance(data, str):
            return clean_svg_paths(data)
        return data

    def clean_text(text: str, key: str = "") -> str:
        """
        Remove URLs and SHA values from text while preserving descriptive text.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text with URLs and SHAs removed
        """
        if not isinstance(text, str):
            return text
        
        # Remove only the URL part from markdown links, preserving the description, only when key is not "diff"
        if key != "diff":
            text = re.sub(r'\(https?://[^)]+\)', '', text)
            text = re.sub(r'"https?://[^)]+"', '"..."', text)
        
        # Remove SHA values
        text = re.sub(r'sha(?:\d+)?-[A-Za-z0-9+/=]+', '', text)
        
        return text

    def clean_dict_urls_shas(data: Any, key: str="") -> Any:
        """
        Recursively clean URLs and SHA values from all string values in a dictionary or list.
        
        Args:
            data: Input data (can be dict, list, or primitive type)
            
        Returns:
            Cleaned data structure
        """
        if isinstance(data, dict):
            return {k: clean_dict_urls_shas(v, k) for k, v in data.items()}
        elif isinstance(data, list):
            return [clean_dict_urls_shas(item) for item in data]
        elif isinstance(data, str):
            return clean_text(data, key)
        return data

    # Create a copy to avoid modifying the original data
    trimmed_data = pr_data.copy()
    current_size = len(str(trimmed_data))
    logger.info(f"Initial data size: {current_size} bytes")
    
    # Always clean SVG paths first, regardless of size
    trimmed_data = clean_dict_svg_paths(trimmed_data)
    
    current_size = len(str(trimmed_data))
    logger.info(f"Size after cleaning SVG paths: {current_size} bytes")
    
    if current_size <= max_size:
        return trimmed_data
        
    # Fields to potentially remove, in order of priority
    fields_to_remove = ['ci_cd_results', 'commits', 'related_issues']
    
    # Step 1: Remove fields until size is under limit
    for field in fields_to_remove:
        if field not in trimmed_data:
            logger.warning(f"Field '{field}' not found in PR data")
            continue
            
        # Remove the field
        trimmed_data[field] = []
        new_size = len(str(trimmed_data))
        
        logger.info(f"Removed {field}, new size: {new_size} bytes")
        
        if new_size <= max_size:
            return trimmed_data
    
    # Step 2: If still too large, clean URLs and SHA values
    if new_size > max_size:
        logger.info("Still exceeding size limit. Cleaning URLs and SHA values...")
        trimmed_data = clean_dict_urls_shas(trimmed_data)
        final_size = len(str(trimmed_data))
        logger.info(f"Size after cleaning URLs and SHAs: {final_size} bytes")
    
    # Step 3: If still too large, clean all interactions which author is not llamapreview[bot] but contains [bot]
    if final_size > max_size:
        logger.info("Still exceeding size limit. Cleaning interactions...")
        trimmed_data['interactions'] = [
            item for item in trimmed_data['interactions']
            if not ("author" in item 
                    and item['author'] != 'llamapreview[bot]' 
                    and '[bot]' in item['author'])
        ]
        final_size = len(str(trimmed_data))
        if final_size > max_size:
            logger.warning("Data size still exceeds maximum limit after all cleaning steps")
    
    return trimmed_data

def json_to_markdown(pr_data: Dict[str, Any]) -> str:
    def format_value(value: Any, indent: int = 0) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, dict):
            return '\n'.join(f"{'  ' * indent}- **{k}**: {format_value(v, indent + 1)}" for k, v in value.items())
        elif isinstance(value, list):
            return '\n'.join(f"{'  ' * indent}- {format_value(item, indent + 1)}" for item in value)
        else:
            return str(value).replace('\\n', '\n').replace('\\r', '')

    def format_file_change(file_change: Dict[str, Any]) -> str:
        md = f"### {file_change.get('file_path', 'Unknown file')}\n"
        md += f"**Change Type:** {file_change.get('change_type', 'Unknown')}\n"
        md += f"**Language:** {file_change.get('language', 'Unknown')}\n"
        md += f"**Additions:** {file_change.get('additions', 0)}\n"
        md += f"**Deletions:** {file_change.get('deletions', 0)}\n"
        md += f"**Changes:** {file_change.get('changes', 0)}\n"
        md += f"**Change Categories:** {', '.join(file_change.get('change_categories', []))}\n"
        
        related_modules = file_change.get('related_modules', [])
        if related_modules:
            md += f"**Related Modules:** {', '.join(related_modules)}\n"
        
        diff = file_change.get('diff', '')
        if diff:
            md += "\n```diff\n"
            md += diff.replace('\\n', '\n').replace('\\r', '')
            md += "\n```\n\n"
        return md
    
    def format_issue(issue: Dict[str, Any]) -> str:
        # Create markdown header with issue number
        md = f"### Issue #{issue['issue_number']}\n"
        
        # Get issue content with empty string as default
        issue_content = issue.get('issue_content', '')
        
        if issue_content is not None:
            # Check if content starts with the repo description header
            if issue_content.startswith('This is a Github Issue related to repo'):
                # Find the position of first double newline which marks the end of description
                start_pos = issue_content.find('\n\n')
                if start_pos != -1:
                    # Remove the description part and keep the actual content
                    issue_content = issue_content[start_pos + 2:]
            
            # Replace escaped newlines with actual newlines
            md += issue_content.replace('\\n', '\n').replace('\\r', '')
        else:
            # Handle case where no content is available
            md += "No issue content available"
        
        # Add final newlines
        md += "\n\n"
        return md

    def format_ci_cd_results(ci_cd_results: Dict[str, Any]) -> str:
        md = "## CI/CD Results\n\n"
        md += f"**Overall State:** {ci_cd_results['state']}\n\n"
        
        if ci_cd_results['statuses']:
            md += "### Statuses\n\n"
            for status in ci_cd_results['statuses']:
                md += f"- **{status['context']}**\n"
                md += f"  - State: {status['state']}\n"
                md += f"  - Description: {status['description']}\n"
                md += f"  - URL: {status['target_url']}\n"
                md += f"  - Created: {status['created_at']}\n"
                md += f"  - Updated: {status['updated_at']}\n\n"
        
        if ci_cd_results['check_runs']:
            md += "### Check Runs\n\n"
            for check_run in ci_cd_results['check_runs']:
                md += f"- **{check_run['name']}**\n"
                md += f"  - Status: {check_run['status']}\n"
                md += f"  - Conclusion: {check_run['conclusion']}\n"
                md += f"  - Started: {check_run['started_at']}\n"
                md += f"  - Completed: {check_run['completed_at']}\n"
                md += f"  - Details URL: {check_run['details_url']}\n\n"
        
        return md

    def format_interactions(interactions: List[Dict[str, Any]]) -> str:
        md = "## Interactions\n\n"
        for interaction in interactions:
            md += f"### {interaction['type'].replace('_', ' ').title()} by {interaction['author']}\n"
            md += f"**Author Association:** {interaction['author_association']}\n"
            md += f"**Created At:** {interaction['created_at']}\n"
            if 'state' in interaction:
                md += f"**State:** {interaction['state']}\n"
            if 'path' in interaction:
                md += f"**File:** {interaction['path']}\n"
            md += f"\n{interaction['content']}\n\n"
            if 'diff_hunk' in interaction:
                md += "```diff\n"
                md += interaction['diff_hunk']
                md += "\n```\n\n"
        return md
    
    def format_commits(commits: List[Dict[str, Any]]) -> str:
        """
        Format commit information into markdown.
        """
        if not commits:
            return ""
        
        md = "## Commits\n"
        for commit in commits:
            message_lines = commit.get('message', '').split('\n')
            title = message_lines[0]
            
            md += f"### Commit [{commit.get('sha', '')[:7]}]\n"
            md += f"**Author:** {commit.get('author', 'Unknown')}\n"
            md += f"**Date:** {commit.get('date', 'Unknown')}\n"
            
            stats = commit.get('stats', {})
            md += "**Changes:**\n"
            md += f"- Additions: {stats.get('additions', 0)}\n"
            md += f"- Deletions: {stats.get('deletions', 0)}\n"
            md += f"- Total: {stats.get('total', 0)}\n"
            
            md += "**Message:**\n```\n"
            md += commit.get('message', '').strip()
            md += "\n```\n"
            
            files = commit.get('files', [])
            if files:
                md += "**Modified files:**\n"
                for file in files:
                    if isinstance(file, str):
                        md += f"- {file}\n"
                    elif isinstance(file, dict):
                        md += f"- {file.get('filename', 'Unknown file')}\n"
            
            md += "\n---\n\n"
        
        return md

    markdown = f"# Pull Request #{pr_data['pr_metadata']['number']}: {pr_data['pr_metadata']['title']}\n\n"

    markdown += "## Metadata\n"
    markdown += format_value(pr_data['pr_metadata']) + "\n\n"

    markdown += "## Description\n"
    description = pr_data.get('pr_metadata', {}).get('description')
    if description:
        markdown += description.replace('\\n', '\n').replace('\\r', '')
    else:
        markdown += "No description provided"
    markdown += "\n\n"

    if pr_data.get('related_issues'):
        markdown += "## Related Issues\n"
        for issue in pr_data['related_issues']:
            markdown += format_issue(issue)
    
    if pr_data.get('commits'):
        markdown += format_commits(pr_data['commits'])

    markdown += "## File Changes\n"
    for file_change in pr_data['file_changes']:
        markdown += format_file_change(file_change)

    if pr_data.get('dependency_changes'):
        markdown += "## Dependency Changes\n"
        for dep_change in pr_data['dependency_changes']:
            markdown += f"### {dep_change.get('file_path', 'Unknown file')}\n"
            markdown += "```\n"
            content = dep_change.get('content', '')
            if content is not None:
                markdown += content.replace('\\n', '\n').replace('\\r', '')
            else:
                markdown += "No content available"
            markdown += "\n```\n\n"

    if pr_data.get('config_changes'):
        markdown += "## Configuration Changes\n"
        for config_change in pr_data['config_changes']:
            markdown += f"### {config_change.get('file_path', 'Unknown file')}\n"
            markdown += "```\n"
            content = config_change.get('content', '')
            if content is not None:
                markdown += content.replace('\\n', '\n').replace('\\r', '')
            else:
                markdown += "No content available"
            markdown += "\n```\n\n"

    if pr_data.get('ci_cd_results'):
        try:
            markdown += format_ci_cd_results(pr_data['ci_cd_results'])
        except Exception as e:
            logger.error(f"Error formatting CI/CD results: {str(e)}")

    if pr_data.get('interactions'):
        try:
            markdown += format_interactions(pr_data['interactions'])
        except Exception as e:
            logger.error(f"Error formatting interactions: {str(e)}")

    return markdown

def get_modified_files_from_pr(pr_content: Dict[str, Any]) -> Set[str]:
    """
    Extracts the set of modified file paths from the PR content dictionary.

    This function safely navigates the pr_content structure, which is expected
    to contain a 'file_changes' list. It iterates through this list and collects
    all 'file_path' values into a set, ensuring uniqueness and providing an
    efficient data structure for lookups.

    Args:
        pr_content: The dictionary containing PR data, including file changes.

    Returns:
        A set of strings, where each string is a file path modified in the PR.
        Returns an empty set if 'file_changes' is missing or malformed.
    """
    # Use a set comprehension for a concise and efficient implementation.
    # The .get() method safely handles cases where 'file_changes' might be missing.
    file_changes = pr_content.get('file_changes', [])

    # Ensure file_changes is a list before proceeding, returning an empty set if not.
    if not isinstance(file_changes, list):
        return set()

    return {
        change['file_path']
        for change in file_changes
        if isinstance(change, dict) and 'file_path' in change
    }

def get_pr_details(repo, pr_number) -> tuple[str, str]:
    pr_content = repo.get_pr_content(number=pr_number, force_update=True)
    keys = list(pr_content['pr_metadata'].keys())
    number_index = keys.index('number')
    keys.insert(number_index + 1, 'repo_description')
    pr_content['pr_metadata'] = {
        key: (pr_content['pr_metadata'][key] if key != 'repo_description' else 
                str(repo))
        for key in keys
    }
    pr_content['interactions'] = [
        item for item in pr_content['interactions'] if not '[bot]' in item['author']
    ]
    pr_content = trim_pr_data(pr_content)
    pr_details = json_to_markdown(pr_content)
    return pr_content, pr_details