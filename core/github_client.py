import os
import logging
import requests
from typing import Optional, Dict, Any, List

try:
    from llama_github import GithubRAG
except ImportError:
    # Fallback or error handling if library is missing
    GithubRAG = None

logger = logging.getLogger("LlamaPReview")

def get_github_token() -> str:
    """
    Retrieves the GitHub Token from environment variables.
    
    For this research repository, we use a simple Personal Access Token (PAT)
    instead of the complex GitHub App JWT authentication used in the production
    LlamaPReview service.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.warning("GITHUB_TOKEN not found in environment variables.")

    return token or ""

class GithubClient:
    """
    A unified facade for GitHub API interactions.
    
    This class abstracts the underlying API calls, whether they are implemented
    via raw requests, PyGithub, or the LlamaPReview proprietary SDK.
    """

    def __init__(self, token: Optional[str] = None):
        self.token = token or get_github_token()
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
        if GithubRAG:
            self.github_rag=GithubRAG(
                github_access_token=self.token,
                simple_mode=True
            )
            self.api_handler = self.github_rag.github_api_handler
        else:
            logger.error("llama_github library not found!")
            self.github_rag = None
            self.api_handler = None

    def _get_llama_repo(self, repo_full_name: str):
        """
        Helper to get the repository object from llama-github's pool.
        """
        if not self.github_rag:
            raise RuntimeError("GithubRAG not initialized")
            
        return self.github_rag.RepositoryPool.get_repository(
            full_name=repo_full_name, 
            github_instance=self.token
        )

    def get_pr_content(self, repo_full_name: str, pr_number: int) -> Dict[str, Any]:
        """
        Fetches Pull Request metadata, diffs, and comments using llama-github.
        """
        repo = self._get_llama_repo(repo_full_name)
        return repo.get_pr_content(number=pr_number)

    def get_file_content(self, repo_full_name: str, file_path: str, sha: Optional[str] = None) -> Optional[str]:
        """
        Fetches raw file content.
        Used by Agentic Strategy.
        """
        try:
            repo = self._get_llama_repo(repo_full_name)
            return repo.get_file_content(file_path, sha=sha)
        except Exception as e:
            logger.error(f"Failed to get file content for {file_path}: {e}")
            return None

    def search_code(self, query: str, repo_full_name: str) -> List[Dict[str, Any]]:
        """
        Executes code search.
        Used by Search RAG Strategy.
        """
        if not self.api_handler:
            return []
            
        try:
            results = self.api_handler.search_code(
                query=query,
                repo_full_name=repo_full_name
            )
            return results or []
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []

    def get_repo_structure_for_llm(
        self,
        repo_full_name: str, 
        sha: str = '',
        target_path: str = None,
        max_depth: int = 3,
        show_files: bool = True,
        include_summary: bool = True,
        include_file_list: bool = False,
        file_extensions: list = None,
        exclude_hidden: bool = True
    ) -> dict:
        """
        Get repository structure optimized for LLM consumption and RAG file selection.
        
        This method provides a comprehensive yet token-efficient view of a repository,
        allowing LLMs to understand structure and intelligently select files to read.
        
        Args:
            repo_full_name (str): Repository in format 'owner/repo'
            sha (str): sha name (default: '')
            target_path (str): Specific directory path to explore (None for root)
                            Example: 'src/features/people'
            max_depth (int): Maximum depth from target_path (default: 3)
            show_files (bool): Whether to show files in tree (default: True)
            include_summary (bool): Include statistical summary (default: True)
            include_file_list (bool): Include flat list of files for easy selection (default: True)
            file_extensions (list): Filter files by extensions (e.g., ['.ts', '.tsx', '.py'])
                                None means all files
            exclude_hidden (bool): Exclude hidden files/directories (starting with .) (default: True)
        
        Returns:
            dict: {
                'tree': str,              # Tree visualization
                'summary': dict,          # Statistics and insights (if include_summary=True)
                'files': list,            # Flat list of selectable files (if include_file_list=True)
                'metadata': dict          # Context information
            }
        """
        import requests
        from collections import defaultdict
        
        owner, repo_name = repo_full_name.split('/')
        headers = {'Authorization': f'token {self.token}'}
        
        # Fetch repository tree
        if sha:
            tree_url = f'https://api.github.com/repos/{owner}/{repo_name}/git/trees/{sha}?recursive=1'
            logger.debug(f"fetch repo {sha} files with full paths")
        else:
            tree_url = f'https://api.github.com/repos/{owner}/{repo_name}/git/trees?recursive=1'
        response = requests.get(tree_url, headers=headers)
        
        if response.status_code != 200:
            return {
                'error': f"Failed to fetch repository: {response.status_code}",
                'details': response.json() if response.content else None
            }
        
        all_items = response.json().get('tree', [])
        
        # Normalize target_path
        if target_path:
            target_path = target_path.strip('/')
            prefix = target_path + '/'
        else:
            target_path = ''
            prefix = ''
        
        def is_hidden_path(path: str) -> bool:
            """
            Check if a path contains hidden files or directories.
            A path is considered hidden if any part of it starts with a dot.
            
            Examples:
                .github/workflows/ci.yml -> True
                src/.hidden/file.py -> True
                src/components/Button.tsx -> False
                .gitignore -> True
            """
            parts = path.split('/')
            return any(part.startswith('.') for part in parts)
        
        # Filter items based on target_path, file_extensions, and hidden status
        filtered_items = []
        for item in all_items:
            path = item['path']
            
            # Check if item is under target_path
            if target_path and not path.startswith(prefix):
                continue
            
            # Get relative path from target
            rel_path = path[len(prefix):] if prefix else path
            
            # Skip if empty (target_path itself)
            if not rel_path:
                continue
            
            # Filter hidden files/directories
            if exclude_hidden and is_hidden_path(rel_path):
                continue
            
            # Check depth constraint
            depth = rel_path.count('/') + 1
            if max_depth and depth > max_depth:
                continue
            
            # Check file extension filter
            if file_extensions and item['type'] == 'blob':
                if not any(rel_path.endswith(ext) for ext in file_extensions):
                    continue
            
            filtered_items.append({
                'path': rel_path,
                'full_path': path,
                'type': item['type'],
                'size': item.get('size', 0)
            })
        
        # Build tree structure
        tree_dict = {}
        file_list = []
        stats = {
            'total_files': 0,
            'total_dirs': 0,
            'total_size': 0,
            'file_types': defaultdict(int),
            'largest_files': [],
            'dir_file_counts': defaultdict(int)
        }
        
        for item in filtered_items:
            parts = item['path'].split('/')
            current = tree_dict
            
            # Build nested structure
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {'__type__': 'dir', '__children__': {}}
                    stats['total_dirs'] += 1
                current = current[part]['__children__']
            
            # Add final item
            final_name = parts[-1]
            if item['type'] == 'blob':
                if show_files:
                    current[final_name] = {
                        '__type__': 'file',
                        '__size__': item['size']
                    }
                
                # Collect statistics
                stats['total_files'] += 1
                stats['total_size'] += item['size']
                
                # File extension stats
                ext = '.' + final_name.split('.')[-1] if '.' in final_name else '[no extension]'
                stats['file_types'][ext] += 1
                
                # Track largest files
                stats['largest_files'].append({
                    'path': item['path'],
                    'size': item['size']
                })
                
                # Directory file count
                dir_path = '/'.join(parts[:-1]) if len(parts) > 1 else '[root]'
                stats['dir_file_counts'][dir_path] += 1
                
                # Add to selectable file list
                if include_file_list:
                    file_list.append({
                        'path': item['path'],
                        'full_path': item['full_path'],
                        'size': item['size'],
                        'extension': ext,
                        'directory': dir_path
                    })
            else:
                if final_name not in current:
                    current[final_name] = {'__type__': 'dir', '__children__': {}}
                    stats['total_dirs'] += 1
        
        # Generate tree visualization
        def dict_to_tree_text(node, prefix='', is_last=True, is_root=True):
            lines = []
            items = sorted(node.items(), key=lambda x: (x[1].get('__type__') != 'dir', x[0]))
            
            visible_items = [(k, v) for k, v in items if not k.startswith('__')]
            
            for i, (name, value) in enumerate(visible_items):
                is_last_item = (i == len(visible_items) - 1)
                
                if is_root and i == 0:
                    connector = ''
                    lines.append(f"{name}/" if value.get('__type__') == 'dir' else name)
                else:
                    connector = '└── ' if is_last_item else '├── '
                    
                    # Format name with type indicator
                    if value.get('__type__') == 'dir':
                        display = f"{name}/"
                    else:
                        size = value.get('__size__', 0)
                        size_str = f" ({_format_size(size)})" if size > 0 else ""
                        display = f"{name}{size_str}"
                    
                    lines.append(f"{prefix}{connector}{display}")
                
                # Recurse for directories
                if value.get('__type__') == 'dir' and '__children__' in value:
                    extension = '    ' if is_last_item else '│   '
                    new_prefix = prefix + extension if not is_root or i > 0 else ''
                    lines.extend(dict_to_tree_text(
                        value['__children__'], 
                        new_prefix, 
                        is_last_item,
                        False
                    ))
            
            return lines
        
        def _format_size(size_bytes):
            """Format file size for human readability."""
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.1f}{unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.1f}TB"
        
        # Generate tree text
        if tree_dict:
            tree_lines = dict_to_tree_text(tree_dict)
            tree_text = '\n'.join(tree_lines)
        else:
            tree_text = f"[Empty or no items match criteria in: {target_path or 'root'}]"
        
        # Build summary
        summary = None
        if include_summary:
            # Sort and limit
            stats['largest_files'].sort(key=lambda x: x['size'], reverse=True)
            stats['largest_files'] = stats['largest_files'][:10]
            
            top_extensions = sorted(
                stats['file_types'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            busiest_dirs = sorted(
                stats['dir_file_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            summary = {
                'total_files': stats['total_files'],
                'total_directories': stats['total_dirs'],
                'total_size': _format_size(stats['total_size']),
                'total_size_bytes': stats['total_size'],
                'file_type_distribution': [
                    {'extension': ext, 'count': count} 
                    for ext, count in top_extensions
                ],
                'largest_files': [
                    {
                        'path': f['path'],
                        'size': _format_size(f['size']),
                        'size_bytes': f['size']
                    }
                    for f in stats['largest_files']
                ],
                'busiest_directories': [
                    {'path': dir_path, 'file_count': count}
                    for dir_path, count in busiest_dirs
                ]
            }
        
        # Build metadata
        metadata = {
            'repository': repo_full_name,
            'sha': sha,
            'target_path': target_path or '[root]',
            'max_depth': max_depth,
            'filters': {
                'show_files': show_files,
                'file_extensions': file_extensions,
                'exclude_hidden': exclude_hidden
            },
            'items_displayed': len(filtered_items),
            'total_files': stats['total_files'],
            'total_dirs': stats['total_dirs']
        }
        
        # Construct result
        result = {
            'tree': tree_text,
            'metadata': metadata
        }
        
        if include_summary:
            result['summary'] = summary
        
        if include_file_list:
            result['files'] = sorted(file_list, key=lambda x: x['path'])
        
        return result
