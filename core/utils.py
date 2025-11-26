import logging
from typing import Tuple

logger = logging.getLogger("LlamaPReview")

def parse_github_url(url: str) -> Tuple[str, int]:
    """
    Parses a GitHub PR URL into repository full name and PR number.
    
    Args:
        url (str): The full URL (e.g., https://github.com/owner/repo/pull/123)
        
    Returns:
        Tuple[str, int]: ('owner/repo', 123)
        
    Raises:
        ValueError: If the URL format is invalid.
    """
    try:
        # Handle trailing slashes
        clean_url = url.rstrip('/')
        parts = clean_url.split('/')
        
        if 'pull' not in parts:
            raise ValueError("URL must be a Pull Request URL containing '/pull/'")
        
        pull_idx = parts.index('pull')
        
        # Ensure we have enough parts before 'pull'
        if pull_idx < 4: # https://github.com/owner/repo/pull...
            raise ValueError("URL path is too short")
            
        owner = parts[pull_idx - 2]
        repo = parts[pull_idx - 1]
        number_str = parts[pull_idx + 1]
        
        if not number_str.isdigit():
            raise ValueError(f"PR number must be an integer, got '{number_str}'")
            
        return f"{owner}/{repo}", int(number_str)
        
    except Exception as e:
        logger.error(f"Failed to parse GitHub URL '{url}': {str(e)}")
        raise ValueError(f"Invalid GitHub PR URL: {url}. Expected format: https://github.com/owner/repo/pull/123")