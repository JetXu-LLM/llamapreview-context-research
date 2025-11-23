import json
import logging
import re
from typing import List, Dict, Any, Set, Optional

from core.code_analysis import CodeContextExtractor
from core.github_client import GithubClient
from .query_generator import normalize_github_search_query

logger = logging.getLogger("LlamaPReview")

def _run_second_pass_search(
    query_term: str,
    lang: str,
    existing_paths: Set[str],
    github_client: GithubClient,
    repo_full_name: str
) -> List[Dict[str, Any]]:
    """
    Executes a second pass search for high-fallback queries using language-specific keywords.
    """
    is_pascal_case = bool(re.match(r'^[A-Z][a-zA-Z0-9]*$', query_term))
    lang_lower = (lang or "").lower()
    keyword = ""
    if lang_lower == "python":
        keyword = "class" if is_pascal_case else "def"
    elif lang_lower in ("javascript", "typescript"):
        keyword = "class" if is_pascal_case else "function"
    elif lang_lower in ("java", "c++", "c#"):
        keyword = "class"
    elif lang_lower == "go":
        keyword = "type" if is_pascal_case else "func"
    elif lang_lower == "rust":
        keyword = "struct" if is_pascal_case else "fn"

    candidate_queries: List[str] = []
    if keyword:
        candidate_queries.append(f"{keyword} {query_term}")
    if is_pascal_case:
        candidate_queries.append(f"function {query_term}")
        candidate_queries.append(f"const {query_term}")
        candidate_queries.append(f"export function {query_term}")
    if not query_term.endswith('('):
        candidate_queries.append(f"{query_term}(")

    aggregated: List[Dict[str, Any]] = []
    seen_paths_local: Set[str] = set()
    seen_queries: Set[str] = set()

    for cq in candidate_queries:
        q_clean = cq.strip()
        if not q_clean or q_clean in seen_queries:
            continue
        seen_queries.add(q_clean)
        try:
            new_results = github_client.search_code(
                query=q_clean,
                repo_full_name=repo_full_name
            ) or []
        except Exception as e:
            logger.error(f"[ContextGen] second-pass search error for '{q_clean}': {e}")
            continue
        for r in new_results:
            p = r.get("path")
            if not p or p in existing_paths or p in seen_paths_local:
                continue
            seen_paths_local.add(p)
            aggregated.append(r)
    return aggregated

def generate_search_context_for_review(
    queries_response: str,
    github_client: GithubClient,
    repo_full_name: str,
    pr_modified_files: set,
    primary_language: str = "Unknown",
    max_snippets_per_query: int = 5,
    max_total_chars: int = 10000
) -> str:
    """
    Build aggregated contextual code snippets for multiple LLM-generated queries.
    Enhanced with:
      - Intent-aware modified-file fallback retry
      - Cross-query snippet reuse (query-scoped keys)
      - JSX / PascalCase fallback expansion
      - Second-pass broader search (function/const/export function + call form)
      - High fallback suggestions in summary
    """
    
    # Initialize the extractor to handle code parsing
    extractor = CodeContextExtractor()

    try:
        queries_data = json.loads(queries_response)
        queries = queries_data.get("queries", [])
    except json.JSONDecodeError:
        logger.error("Invalid queries_response JSON.")
        return "No relevant context (invalid queries JSON)."

    if not queries:
        return "No relevant context (empty queries)."

    # Normalize queries
    seen_norm = set()
    normalized_queries = []
    for q in queries:
        raw_q = q.get("query")
        if not raw_q:
            continue
        norm = normalize_github_search_query(raw_q)
        if not norm:
            logger.info(f"[ContextGen] Skip empty norm query: {raw_q!r}")
            continue
        if norm in seen_norm:
            logger.info(f"[ContextGen] Duplicate normalized query skipped: {norm!r}")
            continue
        seen_norm.add(norm)
        q["_normalized"] = norm
        normalized_queries.append(q)

    if not normalized_queries:
        return "No relevant context (all queries filtered)."

    # Intent definitions
    INTENTS_ALLOW_MODIFIED = {
        "peer_dependency",
        "internal_definition",
        "parameter_adoption",
        "adoption_migration"
    }
    INTENTS_PREFER_EXTERNAL = {
        "external_usage",
        "interface_implementations",
        "removal_cleanup"
    }

    def allow_modified_files(intent: str, legacy_migration: bool) -> bool:
        if intent in INTENTS_ALLOW_MODIFIED:
            return True
        if intent in INTENTS_PREFER_EXTERNAL:
            return False
        return legacy_migration

    def skip_definitions_for_intent(intent: str, legacy_migration: bool) -> bool:
        return intent in ("removal_cleanup",) or legacy_migration

    all_query_results: List[Dict[str, Any]] = []
    processed_snippet_keys: Set[str] = set()

    definition_keyword_pattern = re.compile(
        r'\b('
        r'class|interface|struct|enum|trait|def|async\s+def|function|fn|module|type|record|contract'
        r')\b'
        r'|^\s*export\s+(?:const|function|class|interface|type)\b'
        r'|^\s*(?:interface|type)\s+\w+',
        re.IGNORECASE
    )

    # --------------------------------------------------------------------------
    # Main Search Loop
    # --------------------------------------------------------------------------
    for qi_index, query_item in enumerate(normalized_queries):
        raw_q = query_item.get("query")
        norm_q = query_item.get("_normalized", raw_q)
        reason_text = query_item.get("reason", "") or ""
        intent = (query_item.get("intent") or "").strip().lower()
        legacy_migration = bool(re.search(
            r'\b(remove|removed|removal|rename|renamed|migrate|migration|replace|replacing)\b',
            reason_text,
            re.IGNORECASE
        ))

        logger.info(
            f"[ContextGen] Query {qi_index+1}/{len(normalized_queries)} "
            f"intent={intent or '?'} raw='{raw_q}' norm='{norm_q}'"
        )

        try:
            search_results = github_client.search_code(
                query=norm_q,
                repo_full_name=repo_full_name
            ) or []
        except Exception as e:
            logger.error(f"[ContextGen] search_code error for '{norm_q}': {e}", exc_info=True)
            search_results = []

        if not search_results:
            all_query_results.append({
                "query_item": query_item,
                "snippets": [],
                "discard_reasons": ["no_search_results"]
            })
            continue

        clean_symbol = norm_q.rstrip('(').strip()
        if not clean_symbol:
            all_query_results.append({
                "query_item": query_item,
                "snippets": [],
                "discard_reasons": ["empty_core_token"]
            })
            continue

        can_use_modified = allow_modified_files(intent, legacy_migration)
        drop_definitions = skip_definitions_for_intent(intent, legacy_migration)
        discard_reasons_for_query: List[str] = []
        retry_relaxed = False

        collected_snippets_for_query: Dict[str, Dict[str, Any]] = {}

        # Retry loop for relaxed policy
        while True:
            unique_snippets_for_query: Dict[str, Dict[str, Any]] = {}
            used_paths: Set[str] = set()
            discard_reasons_for_query_iteration: List[str] = []
            excluded_modified_count = 0

            for result in search_results:
                content = result.get("content") or ""
                path = result.get("path") or "unknown_file"
                api_index = result.get("index", 999)

                if path in pr_modified_files and not can_use_modified:
                    excluded_modified_count += 1
                    reason = f"file_excluded_modified:{path}"
                    discard_reasons_for_query_iteration.append(reason)
                    logger.debug(f"[ContextGen][Discard] {norm_q} -> {reason}")
                    continue

                if path in used_paths:
                    continue

                lines = content.splitlines()
                candidate_lines = [ln for ln, text in enumerate(lines) if norm_q in text]
                if not candidate_lines:
                    continue

                picked_snippet = None
                was_discarded_as_duplicate = False

                # 1. Try to extract Definition Block
                if not drop_definitions:
                    for line_idx in candidate_lines:
                        line_text = lines[line_idx]
                        if (
                            definition_keyword_pattern.search(line_text.lower())
                            or line_text.endswith("{")
                            or line_text.endswith(":")
                        ):
                            block, s_line, e_line = extractor.extract_enclosing_block(
                                content, line_idx, clean_symbol
                            )
                            if block:
                                snippet_key = f"{norm_q}::{path}::{s_line}:{e_line}"
                                if snippet_key not in processed_snippet_keys:
                                    picked_snippet = {
                                        "path": path, "code": block, "start": s_line, "end": e_line,
                                        "api_index": api_index, "fallback": False, "forced_kind": "definition"
                                    }
                                    processed_snippet_keys.add(snippet_key)
                                else:
                                    reason = f"duplicate_block:{snippet_key}"
                                    logger.debug(f"[ContextGen][Discard] {norm_q} -> {reason}")
                                    discard_reasons_for_query_iteration.append(reason)
                                    was_discarded_as_duplicate = True
                                break

                # 2. Try to extract Usage Block (if not definition)
                if picked_snippet is None and not was_discarded_as_duplicate:
                    for line_idx in candidate_lines:
                        if re.match(r'^\s*(from\s+\S+\s+import|import\s+\S+)', lines[line_idx].strip()):
                            continue
                        block, s_line, e_line = extractor.extract_enclosing_block(
                            content, line_idx, clean_symbol
                        )
                        if block:
                            snippet_key = f"{norm_q}::{path}::{s_line}:{e_line}"
                            if snippet_key not in processed_snippet_keys:
                                picked_snippet = {
                                    "path": path, "code": block, "start": s_line, "end": e_line,
                                    "api_index": api_index, "fallback": False
                                }
                                processed_snippet_keys.add(snippet_key)
                            else:
                                reason = f"duplicate_block:{snippet_key}"
                                logger.debug(f"[ContextGen][Discard] {norm_q} -> {reason}")
                                discard_reasons_for_query_iteration.append(reason)
                                was_discarded_as_duplicate = True
                            break

                # 3. Fallback: Line Window
                if picked_snippet is None and not was_discarded_as_duplicate:
                    chosen = extractor.pick_representative_line(candidate_lines, lines, norm_q)
                    window_code, w_start, w_end = extractor.build_line_window(content, chosen, window=3)

                    # Heuristic expansion for JSX/Components
                    expanded_any = False
                    line_text_chosen = lines[chosen].strip()
                    if (norm_q.startswith("<") or line_text_chosen.startswith("<")) and "<" in line_text_chosen:
                        # Simple heuristic to expand JSX component usage
                        up = chosen
                        down = chosen
                        while up > 0:
                            t = lines[up].strip()
                            if not t: break
                            if re.match(r'^(export\s+)?(function|const|let|var)\s+[A-Za-z0-9_]+', t): break
                            up -= 1
                        
                        # Simple brace counting for end
                        brace_depth = 0
                        for i in range(chosen, min(len(lines), chosen + 80)):
                            brace_depth += lines[i].count('{')
                            brace_depth -= lines[i].count('}')
                            down = i
                            if brace_depth <= 0 and i > chosen and lines[i].strip().endswith('}'):
                                break
                        
                        expanded_block = "\n".join(lines[up:down+1])
                        if expanded_block.count('\n') >= window_code.count('\n'):
                            window_code = expanded_block
                            w_start = up + 1
                            w_end = down + 1
                            expanded_any = True

                    # Heuristic expansion for Class/Function definitions missed by regex
                    if not expanded_any and re.match(r'^[A-Z][A-Za-z0-9]+$', clean_symbol):
                        up = chosen
                        down = chosen
                        while up > 0 and lines[up].strip():
                            t = lines[up].strip()
                            if re.match(r'^(class|def |fn |function |export function|const )', t): break
                            up -= 1
                        for i in range(chosen + 1, min(len(lines), chosen + 60)):
                            t = lines[i].strip()
                            if re.match(r'^(class |def |fn |function )', t): break
                            down = i
                        expanded_block = "\n".join(lines[up:down+1])
                        if expanded_block.count('\n') > window_code.count('\n'):
                            window_code = expanded_block
                            w_start = up + 1
                            w_end = down + 1

                    fb_key = f"{norm_q}::{path}::{w_start}:{w_end}"
                    if fb_key not in processed_snippet_keys:
                        picked_snippet = {
                            "path": path, "code": window_code, "start": w_start, "end": w_end,
                            "api_index": api_index, "fallback": True
                        }
                        processed_snippet_keys.add(fb_key)
                    else:
                        reason = f"duplicate_fallback:{fb_key}"
                        logger.debug(f"[ContextGen][Discard] {norm_q} -> {reason}")
                        discard_reasons_for_query_iteration.append(reason)

                if picked_snippet:
                    unique_snippets_for_query[path] = picked_snippet
                    used_paths.add(path)

            for k, v in unique_snippets_for_query.items():
                collected_snippets_for_query[k] = v
            discard_reasons_for_query.extend(discard_reasons_for_query_iteration)

            if unique_snippets_for_query or retry_relaxed or excluded_modified_count == 0:
                break

            logger.info(
                f"[ContextGen] Query '{norm_q}' found no external snippets and "
                f"discarded {excluded_modified_count} modified files. "
                f"Retrying with relaxed policy (allow_modified=True)."
            )

            can_use_modified = True
            retry_relaxed = True
            discard_reasons_for_query.append("retry_relaxed_allow_modified")

        # Enrich snippets
        enriched: List[Dict[str, Any]] = []
        for snip in collected_snippets_for_query.values():
            if "forced_kind" in snip:
                kind = snip["forced_kind"]
            else:
                kind = extractor.classify_snippet_kind(clean_symbol, snip["code"], snip["path"])
            kind_score = {"definition": 0, "usage": 1, "import": 2}.get(kind, 3)
            if kind == "import" and extractor.is_barrel_file(snip["path"]):
                kind_score += 1
            enriched.append({**snip, "kind": kind, "kind_score": kind_score})

        # Check for high fallback ratio -> Second Pass
        fallback_total = sum(1 for e in enriched if e.get("fallback"))
        if len(enriched) >= 2 and (fallback_total / len(enriched)) > 0.5:
            discard_reasons_for_query.append(f"high_fallback_ratio:{fallback_total}/{len(enriched)}")
            existing_snippet_paths = {snip["path"] for snip in enriched}
            second_pass_results = _run_second_pass_search(
                query_term=clean_symbol,
                lang=primary_language,
                existing_paths=existing_snippet_paths,
                github_client=github_client,
                repo_full_name=repo_full_name
            )
            if second_pass_results:
                discard_reasons_for_query.append(f"second_pass_found:{len(second_pass_results)}")
                for result in second_pass_results:
                    content = result.get("content") or ""
                    path = result.get("path") or "unknown_file"
                    api_index = result.get("index", 999)
                    lines = content.splitlines()
                    candidate_lines = [ln for ln, text in enumerate(lines) if clean_symbol in text]
                    if not candidate_lines:
                        continue
                    block, s_line, e_line = extractor.extract_enclosing_block(
                        content, candidate_lines[0], clean_symbol
                    )
                    if block:
                        snippet_key = f"{norm_q}::{path}::{s_line}:{e_line}"
                        if snippet_key not in processed_snippet_keys:
                            new_snippet = {
                                "path": path,
                                "code": block,
                                "start": s_line,
                                "end": e_line,
                                "api_index": api_index,
                                "fallback": False,
                                "forced_kind": "definition",
                                "second_pass_source": True
                            }
                            processed_snippet_keys.add(snippet_key)
                            kind = "definition"
                            kind_score = 0
                            enriched.append({**new_snippet, "kind": kind, "kind_score": kind_score})
                            logger.debug(f"[ContextGen] Second-pass added snippet from {path}")

        # Add to results
        all_query_results.append({
            "query_item": query_item,
            "snippets": enriched, # We pass all enriched snippets, assembly logic will sort/limit
            "discard_reasons": discard_reasons_for_query
        })

    # --------------------------------------------------------------------------
    # Final Assembly
    # --------------------------------------------------------------------------
    # Use the reusable assembler from CodeContextExtractor to format the output
    final_output = extractor.assemble_snippets_into_context(
        query_results=all_query_results,
        max_snippets_per_query=max_snippets_per_query,
        max_total_chars=max_total_chars,
        include_summary=True
    )
    
    return final_output
