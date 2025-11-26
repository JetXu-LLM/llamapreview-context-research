"""
Reusable code context extraction utilities.
A class that encapsulates language-agnostic code block extraction and classification methods.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Set, Optional

logger = logging.getLogger("LlamaPReview")

class CodeContextExtractor:
    """
    A utility class for extracting and classifying code blocks from source files.
    All methods are exact copies from the original code, organized as a reusable toolkit.
    """
    
    def __init__(self):
        """
        Initialize the CodeContextExtractor.
        Attempts to use DiffGenerator patterns, falls back to comprehensive patterns if unavailable.
        """
        # Try to import DiffGenerator patterns, use comprehensive fallback if not available
        try:
            from llama_github.utils import DiffGenerator
            self._FUNC_CONTEXT_PATTERNS = DiffGenerator._FUNC_CONTEXT_PATTERNS
            logger.info("[CodeContextExtractor] Using DiffGenerator patterns")
        except (ImportError, AttributeError):
            logger.warning("[CodeContextExtractor] DiffGenerator not available, using comprehensive fallback patterns")
            self._FUNC_CONTEXT_PATTERNS = [
                # Python
                re.compile(r'^\s*def\s+\w+'),
                re.compile(r'^\s*async\s+def\s+\w+'),
                re.compile(r'^\s*class\s+\w+'),
                # JavaScript/TypeScript
                re.compile(r'^\s*function\s+\w+'),
                re.compile(r'^\s*async\s+function\s+\w+'),
                re.compile(r'^\s*const\s+\w+\s*=\s*\(.*\)\s*=>'),
                re.compile(r'^\s*const\s+\w+\s*=\s*async\s*\(.*\)\s*=>'),
                re.compile(r'^\s*const\s+\w+\s*=\s*function'),
                re.compile(r'^\s*let\s+\w+\s*=\s*\(.*\)\s*=>'),
                re.compile(r'^\s*var\s+\w+\s*=\s*function'),
                re.compile(r'^\s*export\s+function\s+\w+'),
                re.compile(r'^\s*export\s+const\s+\w+\s*='),
                re.compile(r'^\s*export\s+default\s+function'),
                re.compile(r'^\s*export\s+async\s+function'),
                # TypeScript specific
                re.compile(r'^\s*interface\s+\w+'),
                re.compile(r'^\s*type\s+\w+\s*='),
                re.compile(r'^\s*enum\s+\w+'),
                re.compile(r'^\s*export\s+interface\s+\w+'),
                re.compile(r'^\s*export\s+type\s+\w+'),
                re.compile(r'^\s*export\s+enum\s+\w+'),
                # Java
                re.compile(r'^\s*(public|private|protected)\s+(static\s+)?[\w<>\[\]]+\s+\w+\s*\('),
                re.compile(r'^\s*(public|private|protected)\s+class\s+\w+'),
                re.compile(r'^\s*(public|private|protected)\s+interface\s+\w+'),
                re.compile(r'^\s*(public|private|protected)\s+enum\s+\w+'),
                # C++
                re.compile(r'^\s*[\w:]+\s+[\w:]+\s*\([^)]*\)\s*(const)?\s*{'),
                re.compile(r'^\s*class\s+\w+'),
                re.compile(r'^\s*struct\s+\w+'),
                re.compile(r'^\s*namespace\s+\w+'),
                # C#
                re.compile(r'^\s*(public|private|protected|internal)\s+(static\s+)?(async\s+)?[\w<>\[\]]+\s+\w+\s*\('),
                # Go
                re.compile(r'^\s*func\s+\w+'),
                re.compile(r'^\s*func\s+\(\w+\s+\*?\w+\)\s+\w+'),
                re.compile(r'^\s*type\s+\w+\s+struct'),
                re.compile(r'^\s*type\s+\w+\s+interface'),
                # Rust
                re.compile(r'^\s*fn\s+\w+'),
                re.compile(r'^\s*pub\s+fn\s+\w+'),
                re.compile(r'^\s*async\s+fn\s+\w+'),
                re.compile(r'^\s*pub\s+async\s+fn\s+\w+'),
                re.compile(r'^\s*struct\s+\w+'),
                re.compile(r'^\s*pub\s+struct\s+\w+'),
                re.compile(r'^\s*enum\s+\w+'),
                re.compile(r'^\s*pub\s+enum\s+\w+'),
                re.compile(r'^\s*trait\s+\w+'),
                re.compile(r'^\s*impl\s+'),
                # Ruby
                re.compile(r'^\s*def\s+\w+'),
                re.compile(r'^\s*class\s+\w+'),
                re.compile(r'^\s*module\s+\w+'),
                # PHP
                re.compile(r'^\s*(public|private|protected)\s+function\s+\w+'),
                re.compile(r'^\s*function\s+\w+'),
                re.compile(r'^\s*class\s+\w+'),
                # Swift
                re.compile(r'^\s*func\s+\w+'),
                re.compile(r'^\s*(public|private|internal|fileprivate)\s+func\s+\w+'),
                re.compile(r'^\s*class\s+\w+'),
                re.compile(r'^\s*struct\s+\w+'),
                re.compile(r'^\s*enum\s+\w+'),
                # Kotlin
                re.compile(r'^\s*fun\s+\w+'),
                re.compile(r'^\s*(public|private|protected|internal)\s+fun\s+\w+'),
                re.compile(r'^\s*class\s+\w+'),
                re.compile(r'^\s*interface\s+\w+'),
                re.compile(r'^\s*object\s+\w+'),
            ]

    def _is_definition_start_line(self, line: str) -> bool:
        """
        Checks if a line is a likely start of a definition using the robust patterns
        from the DiffGenerator class. This is the primary mechanism for identifying
        function/class/method starts across different languages.
        """
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            return False
        
        for pattern in self._FUNC_CONTEXT_PATTERNS:
            if pattern.search(line):
                return True
        return False
    
    def extract_enclosing_block(
        self,
        content: str,
        line_index: int,
        symbol: str,
        max_block_lines: int = 200
    ) -> Tuple[Optional[str], int, int]:
        """
        Extracts an enclosing definition block (function, class, etc.) using a language-agnostic approach.

        This function is optimized to first find the start of a definition using a robust set of regex
        patterns (shared with DiffGenerator) and then expand that block using either brace-balancing
        or indentation-based rules.

        Args:
            content (str): The full content of the file.
            line_index (int): The 0-based index of the line where a symbol was found.
            symbol (str): The symbol being searched for (used as a hint).
            max_block_lines (int): A safeguard to prevent runaway block extraction.

        Returns:
            A tuple of (block_text, start_line_1_based, end_line_1_based) or (None, -1, -1) if no
            enclosing block can be determined.
        """
        lines = content.splitlines()
        n = len(lines)
        if line_index < 0 or line_index >= n:
            return None, -1, -1

        # --- Step 1: Find the start of the enclosing block ---
        def_idx = -1
        for i in range(line_index, -1, -1):
            if self._is_definition_start_line(lines[i]):
                def_idx = i
                break

        if def_idx == -1:
            return None, -1, -1

        # --- Step 2: Expand upwards from the definition to include decorators ---
        start_of_block = def_idx
        for i in range(def_idx - 1, -1, -1):
            line = lines[i].strip()
            if not line:
                break
            
            if line.startswith('@') or (line.startswith("/*") and "*/" in line):
                start_of_block = i
            else:
                break
        
        def_idx = start_of_block

        # --- Step 3: Expand downwards to find the end of the block ---
        def_line_text = lines[def_idx]
        brace_mode = ("{" in def_line_text) or any("{" in lines[k] for k in range(def_idx, min(n, def_idx + 5)))

        if brace_mode:
            start = def_idx
            brace_depth = 0
            opened = False
            end = start
            for i in range(start, n):
                if not (lines[i].strip().startswith("//") or lines[i].strip().startswith("#")):
                    for char in lines[i]:
                        if char == "{":
                            brace_depth += 1
                            opened = True
                        elif char == "}":
                            brace_depth = max(0, brace_depth - 1)
                
                if opened and brace_depth == 0:
                    end = i
                    break
                end = i
                if end - start >= max_block_lines:
                    break
            block = "\n".join(lines[start:end + 1])
            return block, start + 1, end + 1
        else:
            start = def_idx

            content_start_idx = -1
            for i in range(start, n):
                line = lines[i].strip()
                if line and not line.startswith('@'):
                    content_start_idx = i
                    break
            
            if content_start_idx == -1: content_start_idx = start

            base_indent_line = lines[content_start_idx]
            base_indent = len(base_indent_line) - len(base_indent_line.lstrip())

            signature_end_idx = -1
            for i in range(content_start_idx, n):
                if lines[i].strip().endswith(':'):
                    signature_end_idx = i
                    break
            if signature_end_idx == -1: signature_end_idx = content_start_idx

            end = signature_end_idx
            for i in range(signature_end_idx + 1, n):
                current_line = lines[i]
                if not current_line.strip():
                    end = i
                    continue
                
                indent = len(current_line) - len(current_line.lstrip())
                
                if indent <= base_indent:
                    break
                end = i
                if end - start >= max_block_lines:
                    break
            
            block = "\n".join(lines[start:end + 1])
            return block, start + 1, end + 1
    
    def build_line_window(
        self,
        content: str,
        line_index: int,
        window: int = 3
    ) -> Tuple[str, int, int]:
        """
        Build a simple line window around a target line.
        This is used as a fallback when full block extraction fails.
        
        Args:
            content: The full file content.
            line_index: The 0-based index of the target line.
            window: Number of lines to include above and below the target line.
        
        Returns:
            A tuple of (window_text, start_line_1_based, end_line_1_based).
        """
        lines = content.splitlines()
        start = max(0, line_index - window)
        end = min(len(lines) - 1, line_index + window)
        return "\n".join(lines[start:end + 1]), start + 1, end + 1
    
    def pick_representative_line(
        self,
        candidate_lines: List[int],
        lines: List[str],
        symbol: str
    ) -> int:
        """
        Pick the most representative line from a list of candidate lines.
        Prefers lines with function calls (containing '(') and the symbol.
        
        Args:
            candidate_lines: List of line indices to choose from.
            lines: The full list of file lines.
            symbol: The symbol to look for.
        
        Returns:
            The index of the most representative line.
        """
        for idx in candidate_lines:
            if "(" in lines[idx] and (not symbol or symbol in lines[idx]):
                return idx
        return candidate_lines[0]
    
    def classify_snippet_kind(
        self,
        symbol: str,
        code: str,
        path: str
    ) -> str:
        """
        Classify a code snippet as 'definition', 'usage', or 'import'.
        
        Args:
            symbol: The symbol being searched for.
            code: The code snippet text.
            path: The file path (used for context).
        
        Returns:
            One of: 'definition', 'usage', 'import'.
        """
        lowered_lines = [l.strip().lower() for l in code.splitlines() if l.strip()]
        if not lowered_lines:
            return "usage"
        sym_re = re.compile(rf"\b{re.escape(symbol)}\b") if symbol else None
        def_prefixes = (
            "class ", "interface ", "struct ", "enum ", "trait ",
            "object ", "record ", "contract ", "module ", "def ",
            "fn ", "function ", "type "
        )
        for l in lowered_lines[:8]:
            if sym_re and sym_re.search(l):
                if l.endswith("{") or l.endswith(":") or any(l.startswith(p) for p in def_prefixes):
                    return "definition"
        import_like = 0
        for l in lowered_lines:
            if re.match(r"^(from\s+\S+\s+import\s+|import\s+\S+|using\s+\S+|export\s+|pub\s+use\s+)", l):
                import_like += 1
        if import_like >= max(2, 0.6 * len(lowered_lines)):
            return "import"
        return "usage"
    
    def is_barrel_file(self, path: str) -> bool:
        """
        Check if a file is a "barrel" file (index.js, __init__.py, etc.)
        These files typically just re-export other modules and should have lower priority.
        
        Args:
            path: The file path to check.
        
        Returns:
            True if the file is a barrel file, False otherwise.
        """
        name = path.split("/")[-1].lower()
        return bool(re.match(r"(?:__init__\.py|index\.(?:js|ts|jsx|tsx)|mod\.rs|barrel\.\w+|all\.\w+)$", name))

    def assemble_snippets_into_context(
        self,
        query_results: List[Dict[str, Any]],
        max_snippets_per_query: int = 5,
        max_total_chars: int = 10000,
        include_summary: bool = True
    ) -> str:
        """
        Assemble multiple code snippets from different queries into a final context string.
        
        This method implements a sophisticated assembly strategy:
        1. Enriches snippets with kind classification and scoring
        2. Sorts snippets by priority (definition > usage > import)
        3. Ensures diversity (picks different kinds first)
        4. Round-robin allocation across queries for fairness
        5. Respects character budget limits
        
        Args:
            query_results: List of query result dictionaries, each containing:
                - "query_item": Dict with "query", "intent", "reason", "_normalized"
                - "snippets": List of snippet dicts with "path", "code", "start", "end", 
                            "api_index", "fallback", and optionally "kind", "forced_kind"
                - "discard_reasons": List of reasons why snippets were discarded (optional)
            max_snippets_per_query: Maximum number of snippets to include per query
            max_total_chars: Maximum total characters in the final context
            include_summary: Whether to include a summary section at the beginning
        
        Returns:
            A formatted string containing:
            - Summary section (if include_summary=True) with statistics per query
            - Code snippets organized by query, with headers and metadata
            
        Example query_results structure:
        [
            {
                "query_item": {
                    "query": "MyClass",
                    "intent": "internal_definition",
                    "reason": "Find class definition",
                    "_normalized": "MyClass"
                },
                "snippets": [
                    {
                        "path": "src/main.py",
                        "code": "class MyClass:\n    ...",
                        "start": 10,
                        "end": 20,
                        "api_index": 0,
                        "fallback": False
                    }
                ],
                "discard_reasons": []
            }
        ]
        
        (Extracted and adapted from original generate_search_context_for_review)
        """
        final_context_blocks: Dict[int, List[str]] = {}
        total_chars = 0

        def _add_block(q_idx: int, text: str, header: Optional[str] = None) -> bool:
            """Helper to add a block while respecting character budget."""
            nonlocal total_chars
            addition = text
            if header:
                addition = header + addition
            if total_chars + len(addition) > max_total_chars:
                return False
            total_chars += len(addition)
            if q_idx not in final_context_blocks:
                final_context_blocks[q_idx] = []
            if header and header not in final_context_blocks[q_idx]:
                final_context_blocks[q_idx].append(header)
            final_context_blocks[q_idx].append(text)
            return True

        # Step 1: Enrich and sort snippets for each query
        for result in query_results:
            enriched: List[Dict[str, Any]] = []
            for snip in result.get("snippets", []):
                # Use forced_kind if available, otherwise classify
                if "forced_kind" in snip:
                    kind = snip["forced_kind"]
                elif "kind" in snip:
                    kind = snip["kind"]
                else:
                    kind = self.classify_snippet_kind(
                        snip.get("symbol", ""),
                        snip.get("code", ""),
                        snip.get("path", "")
                    )
                
                # Calculate priority score
                kind_score = {"definition": 0, "usage": 1, "import": 2}.get(kind, 3)
                
                # Penalize barrel files
                if kind == "import" and self.is_barrel_file(snip.get("path", "")):
                    kind_score += 1
                
                enriched.append({**snip, "kind": kind, "kind_score": kind_score})
            
            # Sort by priority: kind_score, fallback status, second_pass, api_index, start line
            enriched.sort(
                key=lambda x: (
                    x.get("kind_score", 999),
                    1 if x.get("fallback") else 0,
                    1 if x.get("second_pass_source") else 0,
                    x.get("api_index", 999),
                    x.get("start", 0)
                )
            )
            
            # Step 2: Pick diverse snippets (ensure different kinds are represented)
            picked_snippets: List[Dict[str, Any]] = []
            seen_kinds: Set[str] = set()
            
            # First pass: pick one of each kind
            for snip in enriched:
                if len(picked_snippets) >= max_snippets_per_query:
                    break
                if snip["kind"] not in seen_kinds:
                    picked_snippets.append(snip)
                    seen_kinds.add(snip["kind"])
            
            # Second pass: fill remaining slots
            if len(picked_snippets) < max_snippets_per_query:
                for snip in enriched:
                    if len(picked_snippets) >= max_snippets_per_query:
                        break
                    snip_key = f"{snip.get('path')}::{snip.get('start')}-{snip.get('end')}"
                    is_already_picked = any(
                        f"{p.get('path')}::{p.get('start')}-{p.get('end')}" == snip_key 
                        for p in picked_snippets
                    )
                    if not is_already_picked:
                        picked_snippets.append(snip)
            
            result["_picked_snippets"] = picked_snippets

        # Step 3: Round-robin allocation across queries
        for rank in range(max_snippets_per_query):
            for q_idx, result in enumerate(query_results):
                snips = result.get("_picked_snippets", [])
                if rank >= len(snips):
                    continue
                
                snip = snips[rank]
                qi = result.get("query_item", {})
                header_needed = (q_idx not in final_context_blocks)

                query_header = ""
                if header_needed:
                    query_header = (
                        f"--- Context for Query {q_idx + 1} ---\n"
                        f"Intent: {qi.get('intent', '?')}\n"
                        f"Reason: {qi.get('reason', 'N/A')}\n"
                        f"Query: `{qi.get('query', 'N/A')}` "
                        f"(normalized=`{qi.get('_normalized', qi.get('query', 'N/A'))}`)\n\n"
                    )

                snippet_header = (
                    f"File: `{snip.get('path', 'unknown')}` "
                    f"(lines {snip.get('start', '?')}-{snip.get('end', '?')})"
                    f"{' [fallback]' if snip.get('fallback') else ''}"
                    f" [{snip.get('kind', '?')}]\n"
                )
                code_block = f"```text\n{snip.get('code', '')}\n```\n"
                full_block = snippet_header + code_block

                if not _add_block(q_idx, full_block, header=query_header if header_needed else None):
                    logger.warning(
                        f"[CodeContextExtractor] Char budget reached before adding "
                        f"query {q_idx+1} rank {rank}."
                    )
                    continue

        # Step 4: Generate summary
        summary_lines = []
        if include_summary:
            summary_lines.append("**Code Context Assembly Summary:**")
            for idx, result in enumerate(query_results):
                qi = result.get("query_item", {})
                q_raw = qi.get("query", "N/A")
                intent = qi.get("intent", "?")
                snippets = result.get("_picked_snippets", [])
                fallback_count = sum(1 for s in snippets if s.get("fallback"))
                
                included_count = 0
                if idx in final_context_blocks:
                    included_count = sum(
                        1 for b in final_context_blocks[idx] 
                        if b.startswith("File:")
                    )
                
                kind_dist: Dict[str, int] = {}
                for s in snippets:
                    k = s.get("kind", "?")
                    kind_dist[k] = kind_dist.get(k, 0) + 1

                if not snippets:
                    discard_reasons = result.get("discard_reasons", [])
                    summary_lines.append(
                        f"- Query {idx+1} (`{q_raw}`) intent={intent}: "
                        f"No snippets (reasons={discard_reasons[:3]})"
                    )
                else:
                    if included_count:
                        fallback_ratio = 0.0 if not snippets else round(fallback_count / len(snippets), 2)
                        summary = (
                            f"- Query {idx+1} (`{q_raw}`) intent={intent}: "
                            f"Included {included_count}/{len(snippets)} snippets "
                            f"(fallback={fallback_count}, ratio={fallback_ratio}) "
                            f"kind_dist={kind_dist}"
                        )
                    else:
                        summary = (
                            f"- Query {idx+1} (`{q_raw}`) intent={intent}: "
                            f"Snippets found={len(snippets)} but excluded by budget"
                        )
                    summary_lines.append(summary)

        # Step 5: Assemble final output
        summary_section = "\n".join(summary_lines) + "\n\n" if summary_lines else ""
        ordered_blocks = []
        for q_idx in sorted(final_context_blocks.keys()):
            ordered_blocks.append("".join(final_context_blocks[q_idx]))
        
        final_output = summary_section + "".join(ordered_blocks)
        logger.info(
            f"[CodeContextExtractor] Assembled context: {len(final_output)} chars, "
            f"{len(query_results)} queries, {sum(len(r.get('_picked_snippets', [])) for r in query_results)} total snippets"
        )
        
        return final_output
    
RESERVED_KEYWORDS = {
    # Python
    'def', 'class', 'for', 'if', 'else', 'elif', 'import', 'from', 'as', 'return',
    # JS/TS
    'const', 'let', 'var', 'function', 'class', 'interface', 'type', 'enum', 'export', 'import',
    # Java/C#
    'public', 'private', 'protected', 'static', 'void', 'class', 'interface', 'new',
    # Go
    'func', 'package', 'import', 'type', 'struct',
    # Rust
    'fn', 'struct', 'enum', 'pub', 'use', 'mod',
    # General
    'extends', 'implements'
}

def extract_diff_entities(pr_content: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Language-agnostic heuristic extraction:
    - Added function-like identifiers
    - Added classes / interfaces / structs / types
    - Removed identifiers (potential cleanup target)
    - Parameter / property keys that appear newly (pattern: name:)
    This is intentionally approximate; noise is acceptable.

    Returns sets of:
      added_symbols, removed_symbols, added_params, added_files_removed_symbols
    """
    added_symbols = set()
    removed_symbols = set()
    added_params = set()

    func_like_pattern = re.compile(
        r'\b(?:def|func|function|fn|class|interface|struct|enum|type|public|private|protected|export|async|const)\s+([A-Za-z_][A-Za-z0-9_]*)'
    )
    # Generic parameter / property key match (key: or key= or key?:)
    param_pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\s*(?=[:=]\s)')

    for fc in pr_content.get("file_changes", []):
        diff = fc.get("diff") or ""
        for line in diff.splitlines():
            # A helper to check and add symbols
            def process_and_add_symbol(symbol_str: str, target_set: set):
                if symbol_str and symbol_str.lower() not in RESERVED_KEYWORDS:
                    target_set.add(symbol_str)

            if line.startswith('+'):
                # Added symbol definitions
                for m in func_like_pattern.finditer(line):
                    process_and_add_symbol(m.group(1), added_symbols)
                # Added param-like keys
                for m in param_pattern.finditer(line):
                    added_params.add(m.group(1)) # Params are less likely to be keywords, can skip filtering
            elif line.startswith('-'):
                for m in func_like_pattern.finditer(line):
                    process_and_add_symbol(m.group(1), removed_symbols)

    # Basic de-noising: drop overly generic tokens
    generic = {"data", "config", "util", "base", "core", "main", "test"}
    added_symbols = {s for s in added_symbols if s.lower() not in generic and len(s) > 2}
    removed_symbols = {s for s in removed_symbols if s.lower() not in generic and len(s) > 2}
    added_params = {p for p in added_params if p.lower() not in generic and len(p) > 2}

    return {
        "added_symbols": added_symbols,
        "removed_symbols": removed_symbols,
        "added_params": added_params
    }

def format_diff_entities_block(entities: Dict[str, Set[str]]) -> str:
    """
    Build a lightweight textual hint block for the prompt.
    """
    parts = []
    if entities["added_symbols"]:
        parts.append("Added symbols: " + ", ".join(sorted(entities["added_symbols"])))
    if entities["removed_symbols"]:
        parts.append("Removed symbols: " + ", ".join(sorted(entities["removed_symbols"])))
    if entities["added_params"]:
        parts.append("New parameter/property keys: " + ", ".join(sorted(entities["added_params"])))
    if not parts:
        return "None detected."
    return "\n".join(parts)