import json
import logging
import re
from typing import List, Dict, Set, Optional, Any

from core.code_analysis import RESERVED_KEYWORDS

logger = logging.getLogger("LlamaPReview")

# ==============================================================================
# Prompt Templates
# ==============================================================================

query_generator_prompt = """
You are a language-agnostic, expert code dependency analyst. Your primary mission is to analyze a Pull Request for any software project and generate a concise set of high-impact search queries. These queries will be used to gather essential context for a comprehensive code review.

## CORE ANALYSIS STRATEGY:

Your thinking process must follow these three steps:

**Step 1: Identify the PR's Core Theme**  
First, look beyond individual lines of code and understand the PR's primary purpose. Is it a bug fix, a new feature, a performance optimization, or a large-scale refactoring? This theme provides the context for all subsequent analysis.

**Step 2: Identify Key Structural Entities**  
Based on the theme, identify the most important structural entities involved in the change. Focus on the architectural components:  
- **Classes / Types / Prototypes** that were created, modified, or inherited from.  
- **Interfaces, Abstract/Base Types** that define a contract.  
- **Public Functions or Methods** whose signatures or core logic have changed.  
- **Key Data Structures or Configuration Objects**.  
- **Added or Removed Identifiers** that may cause ripple effects.

**Step 3: Generate Targeted Queries to Uncover Ripple Effects**  
For each key entity, generate queries to understand its impact on UNCHANGED code. The goal is to find call sites, implementations, and usages that might be affected by the PR but are not part of the PR's diff themselves.

---

### Query Patterns Cookbook (Regex-Free, Language-Agnostic)

* **To find INSTANTIATIONS or DIRECT CALLS (Usage):**  
  * Goal: Find where a class is instantiated or a function is called elsewhere (check argument / parameter propagation).  
  * Pattern: Use the identifier itself, optionally with an opening parenthesis `(` to bias toward call sites (e.g. `processData(`, `NewClient(`).  
  * For constructor-like patterns in some languages you may still just use the type name plus `(`.

* **To find SUBCLASSES / IMPLEMENTATIONS (Inheritance / Polymorphism):**  
  * Goal: Detect classes implementing or extending a changed contract.  
  * Pattern: Use phrases like `implements InterfaceName`, `extends BaseType`, `: BaseType` (colon form can surface inheritance in languages like C++, C#, Swift, Kotlin).  
  * Example: `implements IDataProcessor`, `extends AbstractComponent`, `: BaseRepository`.

* **To find the DEFINITION of a newly introduced entity (when necessary):**  
  * Goal: Provide context if the entity is entirely new and understanding its structure is essential.  
  * Pattern: Combine a language keyword with the identifier (e.g. `class NewExternalClient`, `def build_cache`, `func CalculateMetrics`, `struct MetricsCollector`). Use this sparingly—prefer usages over definitions when you already have the diff.

* **To find GENERAL REFERENCES (Configuration / Wiring / DI / Registration):**  
  * Goal: Capture non-call-site usages (config files, registries, factory bindings) when the name is distinctive.  
  * Pattern: The bare unique identifier or identifier plus a relevant keyword (e.g. `AuthTokenProvider`, `AuthTokenProvider register`).

* **To inspect adoption of NEW PARAMETERS / PROPERTIES:**  
  * Goal: If a new parameter/prop/field was added, combine the callee/owner identifier with that parameter key to locate partial/omitted updates (e.g. `RequestBuilder timeout`).  
  * Pattern: `<Identifier> <ParamKey>` (tokens are ANDed).

* **To inspect interactions BETWEEN MODIFIED FILES (Peer Dependency):**
  * Goal: If a modified entity (e.g., Component A) is used by another modified entity (e.g., Page B) within the same PR, this interaction is critical context.
  * Pattern: Use the identifier itself (e.g., `<ComponentA`, `functionB(`).
  * **Use this intent specifically for usages that cross file boundaries but are still *within* the PR's diff.**

---

## PLATFORM CONSTRAINTS (MANDATORY)

1. Produce at most **5** queries; return fewer if only a small number are truly high-impact. Returning 0 is allowed if nothing is actionable.  
2. Each query must be a plain substring expression (tokens separated by spaces = AND semantics).  
3. Forbidden: regular expressions, enclosing slashes `/.../`, alternation (`|`), grouping parentheses used as regex, escaped word boundaries (`\\b`), quantifiers (`* + ? {{}}`), lookarounds, or any regex metacharacter usage.  
4. Do NOT rely on path filters (`path:`), file filters, or language qualifiers here—the execution layer will scope repositories itself.  
5. No quoting unless the quote character is literally part of an identifier (rare).  
6. Prefer usage-oriented forms (e.g. `FunctionName(`, `implements InterfaceX`) over pure definitions unless the definition context is missing elsewhere.  
7. Avoid overly generic tokens (model, config, data, util, common, base, main, manager, helper) unless combined with a distinctive identifier.
8. DO NOT emit queries that are ONLY generic language types or containers (e.g. "List", "String", "int", "float", "Array", "Map") unless paired with a project-specific identifier. If all candidates are generic → return an empty list.
9. If all candidate identifiers are from standard libraries / ubiquitous language features, return an empty list.
10. Avoid spending more than half of the query budget on symbols that are newly introduced in this PR with no prior usages; prefer existing modified symbols, removed symbols, or adoption of newly added parameters in existing callers.
11. Avoid emitting both a symbol and its trivial parenthesis variant (e.g., "Config" and "Config("). Keep the simpler form.
12. When suggesting queries for removed or renamed symbols, prefer usage contexts and do not exclude files just because they are modified; only skip pure definition lines.
13. If after applying the steps you only identify highly overlapping or trivially low-impact symbols, return fewer (even 0–2) queries rather than forcing filler queries.

---

## STANDARD LIBRARY / GENERIC EXCLUSION

Exclude queries that target purely standard library or framework primitives (e.g. Python: `os`, `json`; Java: `List`, `Map`; JavaScript: `Promise`; Go: `fmt`; C++: `std::vector`). If a modified line only changes such usage, it is not a valid reason to search.

---

## DYNAMIC CONTEXT SIZING & OUTPUT FORMAT

Default `context_lines` MUST be **10**.  
Only increase (max 20) if the default 10 would omit the *entire* definition of a symbol (variable, constant, small helper function) that is directly referenced or modified in the changed lines, thereby breaking essential comprehension.  
Reject all other justifications (readability, style, “more clarity”) as insufficient.

Rules Recap:  
- Start from 10.  
- Increase only under the Critical Exception Rule above.  
- Justify clearly in `context_lines_reason`.  
- If no exception applies, keep 10 and explicitly state so.

---

## OUTPUT FORMAT (Strict JSON)

Return a single JSON object:

{{
  "queries": [
    {{ "reason": "...", "intent": "external_usage|peer_dependency|internal_definition|interface_implementations|removal_cleanup|parameter_adoption|adoption_migration", "query": "..." }}
  ],
  "context_lines_reason": "...",
  "context_lines": 10
}}

Constraints:
- `queries` is an array (possibly empty) of objects each with `reason` and `query`.  
- No additional top-level fields.  
- No markdown.  
- Reasons must be concise and impact-oriented (describe why the usage matters).
- 'intent' MUST be one of the enumerated values. If unsure, pick the closest; do not invent new values.

---

## LANGUAGE-AGNOSTIC EXAMPLES

**Example 1: A JavaScript component's props are changed.**  
PR Theme: Refactoring a UI component.  
Key Entity: `UserProfileComponent`  
Output:
{{
  "queries": [
    {{
      "reason": "Find usages of the modified component to verify new props adoption.",
      "intent": "peer_dependency",
      "query": "<UserProfileComponent"
    }}
  ],
  "context_lines_reason": "10 lines suffice; no helper definitions lie outside the default window.",
  "context_lines": 10
}}

**Example 2: A Java interface method is modified.**  
PR Theme: Updating a service contract.  
Key Entity: `IDataProcessor` interface.  
Output:
{{
  "queries": [
    {{
      "reason": "Locate implementors of IDataProcessor to ensure they reflect the updated method signature.",
      "intent": "interface_implementations",
      "query": "implements IDataProcessor"
    }}
  ],
  "context_lines_reason": "Interface change is fully visible within 10 lines; no expansion needed.",
  "context_lines": 10
}}

---

Analyze the following PR details and generate the search queries.

PR Details:
{pr_details}

---

Repository Primary Language Hint:
The dominant language of this repository is: {primary_language}.
When producing queries, prefer idioms of this language and avoid cross-language inheritance keywords that do not apply.

---

## OPTIONAL DIFF-DERIVED HINTS (You MAY ignore these)

These are heuristically extracted candidate entities and parameter keys. Use your own judgment; you are NOT required to include any of them if they are low impact.

{diff_entities_block}
"""

# ==============================================================================
# Helper Functions
# ==============================================================================

def normalize_github_search_query(raw_query: str) -> str:
    """
    Minimal, GitHub-safe normalization:
    - Strip surrounding symmetric quote / backtick pairs.
    - Strip enclosing /.../ wrapper (common LLM regex hallucination).
    - Do NOT alter internal characters: keep (), <>, [], {}, +, *, ?, :, etc.
    - Do NOT collapse or mutate internal whitespace beyond leading/trailing trim
      (GitHub treats internal spaces as AND separators; preserving user intent).
    - Return the original (minus outer wrapping) as-is for maximum literal fidelity.
    """
    if not isinstance(raw_query, str):
        return ""
    q = raw_query.strip()
    if len(q) >= 2 and (
        (q.startswith('"') and q.endswith('"')) or
        (q.startswith("'") and q.endswith("'")) or
        (q.startswith('`') and q.endswith('`'))
    ):
        q = q[1:-1].strip()
    if len(q) >= 2 and q.startswith('/') and q.endswith('/'):
        # Treat /.../ only as a wrapper if balanced exactly; internal slashes untouched.
        q = q[1:-1].strip()
    return q

def canonical_key(q: str) -> str:
    """
    Canonical form for dedup:
    - Remove a single leading '<'
    - Trim.
    """
    if not q:
        return ""
    cq = q.strip()
    if cq.startswith('<'):
        cq = cq[1:].strip()
    return cq

LOW_SIGNAL_WORDS = {
    "extends", "implement", "implements", "class", "interface", "struct", "enum",
    "type", "function", "fn", "def", "public", "private", "protected", "new",
    "removed", "deleted", "abstract", "final", "static", "return", "void"
}

def reduce_multiword_query(q: str) -> str:
    parts = q.split()
    if len(parts) <= 1:
        return q
    candidates = [p for p in parts if p.lower() not in LOW_SIGNAL_WORDS]
    candidates = [c for c in candidates if len(c) >= 3]
    if not candidates:
        return q
    candidates.sort(key=lambda x: (-len(x), x))
    return candidates[0]

def reduce_multiword_query_with_intent(q: str, intent: Optional[str]) -> str:
    """
    Intent-aware reduction:
    - Skip reduction for parameter_adoption intent.
    - Preserve two-token pattern like "Symbol param" where param looks like a parameter name.
    """
    parts = q.split()
    if len(parts) <= 1:
        return q
    if intent == "parameter_adoption":
        return q
    if len(parts) == 2:
        sym, maybe_param = parts
        param_like = re.match(r'^[a-z_][a-z0-9_]{2,}$', maybe_param)
        sym_callable_like = (
            re.match(r'^[a-z_][a-z0-9_]*$', sym) or
            re.match(r'^[a-z]+[A-Za-z0-9]*$', sym) or
            re.match(r'^[A-Z][A-Za-z0-9]+$', sym)
        )
        if param_like and sym_callable_like:
            return q
    low_signal_count = sum(1 for p in parts if p.lower() in LOW_SIGNAL_WORDS)
    if len(parts) > 3 or low_signal_count >= len(parts) - 1:
        return reduce_multiword_query(q)
    return q

def classify_identifier(name: str) -> Dict[str, bool]:
    """
    Lightweight classification for deciding if we need call-variant.
    """
    is_snake = bool(re.match(r'^[a-z]+(_[a-z0-9]+)+$', name))
    is_lower_camel = bool(re.match(r'^[a-z]+[A-Za-z0-9]*$', name)) and not is_snake
    is_pascal = bool(re.match(r'^[A-Z][A-Za-z0-9]+$', name))
    is_all_upper = name.isupper() and len(name) > 1
    # Likely function-ish if snake_case or lowerCamel (not all uppercase).
    likely_callable = (is_snake or is_lower_camel) and not is_all_upper
    return {
        "is_snake": is_snake,
        "is_lower_camel": is_lower_camel,
        "is_pascal": is_pascal,
        "is_all_upper": is_all_upper,
        "likely_callable": likely_callable
    }

# ==============================================================================
# Query Assembly Pipeline
# ==============================================================================

def extract_core_llm_queries(raw_llm_queries: List[Dict[str, str]],
                             max_core: int,
                             debug: List[str]) -> List[Dict[str, str]]:
    """
    Extract normalized & intent-aware reduced core queries from LLM output,
    preserving order and deduplicating by canonical key.
    """
    seen_canon = set()
    core: List[Dict[str, str]] = []

    for item in raw_llm_queries:
        raw_q = item.get("query", "")
        reason = item.get("reason", "")
        intent = (item.get("intent") or "").strip()
        safe = normalize_github_search_query(raw_q)
        if not safe:
            debug.append(f"drop_llm_empty:{raw_q!r}")
            continue

        reduced = reduce_multiword_query_with_intent(safe, intent)
        final_query = reduced
        tag = "reduced" if reduced != safe else "original"

        canon = canonical_key(final_query)
        if not canon:
            continue
        if canon.lower() in seen_canon:
            debug.append(f"dedup_llm:{final_query}->{canon}")
            continue

        core.append({"reason": reason, "query": final_query, "intent": intent})
        seen_canon.add(canon.lower())
        debug.append(f"add_core[{tag}]:{final_query}")

        if len(core) >= max_core:
            return core

    return core

def ensure_removed_symbols(core: List[Dict[str, str]],
                           removed_symbols: Set[str],
                           max_total: int,
                           debug: List[str]) -> None:
    """
    Guarantee each removed symbol (up to capacity) appears as a core query
    if not already present canonically.
    """
    if not removed_symbols:
        return
    existing = {canonical_key(c["query"]).lower() for c in core}
    for sym in sorted(removed_symbols):
        if len(core) >= max_total:
            break
        sym_safe = normalize_github_search_query(sym)
        if not sym_safe:
            continue
        
        if sym_safe.lower() in RESERVED_KEYWORDS:
            debug.append(f"drop_removed_keyword:{sym_safe}")
            continue

        canon = canonical_key(sym_safe).lower()
        if canon in existing:
            continue
        core.append({
            "reason": f"Detect lingering references to removed symbol {sym}",
            "query": sym_safe,
            "intent": "removal_cleanup"
        })
        existing.add(canon)
        debug.append(f"add_removed:{sym_safe}")

def build_param_adoption_queries(core: List[Dict[str, str]],
                                 added_symbols: Set[str],
                                 added_params: Set[str],
                                 max_total: int,
                                 debug: List[str]) -> List[Dict[str, str]]:
    """
    Construct a small set of "symbol param" adoption queries if room permits.
    Only generate 1 to avoid over-AND saturation.
    """
    # Filter out meaningless parameter names like 'None'
    meaningful_params = {p for p in added_params if p and p.lower() != 'none'}

    if not added_symbols or not meaningful_params:
        return []
    if len(core) >= max_total:
        return []

    # Choose a representative symbol (shortest high-signal) and a param
    sym = sorted(list(added_symbols), key=lambda s: len(s))[0]
    param = sorted(list(meaningful_params), key=lambda s: len(s))[0]

    sym_safe = normalize_github_search_query(sym)
    param_safe = normalize_github_search_query(param)
    if not sym_safe or not param_safe:
        return []

    # Avoid AND if either is too short
    if len(sym_safe) < 3 or len(param_safe) < 3:
        return []

    query_str = f"{sym_safe} {param_safe}"
    canon = canonical_key(query_str).lower()
    existing = {canonical_key(c["query"]).lower() for c in core}
    if canon in existing:
        return []

    debug.append(f"add_adoption:{query_str}")
    return [{
        "reason": f"Check adoption of new parameter {param_safe} in calls to {sym_safe}",
        "query": query_str,
        "intent": "parameter_adoption"
    }]

def build_call_variants(core: List[Dict[str, str]],
                        max_total: int,
                        max_call_variants: int,
                        debug: List[str]) -> List[Dict[str, str]]:
    """
    Add call-form variants 'name(' for a subset of single-token cores.
    Rules:
      - Base (canonical) already present
      - Token single word, no spaces
      - Not already endswith '('
      - likely_callable OR (heuristic fallback: not all upper, length between 3..60)
      - Do not exceed max_call_variants
    """
    variants = []
    existing_canon = {canonical_key(c["query"]).lower() for c in core}
    count_added = 0

    for c in core:
        if len(core) + len(variants) >= max_total:
            break
        base = c["query"]
        if ' ' in base:
            continue
        if base.endswith('('):
            continue
        classified = classify_identifier(base)
        # condition for adding call variant
        if not (classified["likely_callable"] or (not classified["is_all_upper"] and 3 <= len(base) <= 60)):
            continue

        call_form = f"{base}("
        canon_call = canonical_key(call_form).lower()
        if canon_call in existing_canon:
            continue

        variants.append({
            "reason": f"Locate call or instantiation sites of {base}",
            "query": call_form,
            "intent": "external_usage"
        })
        existing_canon.add(canon_call)
        count_added += 1
        debug.append(f"add_call_variant:{call_form}")
        if count_added >= max_call_variants:
            break

    return variants

def assemble_queries(response: str,
                     entities: Dict[str, Set[str]],
                     max_queries: int = 5,
                     max_core_from_llm: int = 5,
                     max_call_variants: int = 2) -> str:
    """
    Master orchestration:
      1. Load LLM response JSON
      2. Extract & reduce core queries (Stage 1)
      3. Ensure removed symbols coverage
      4. Add parameter adoption (optional)
      5. Add call variants
      6. Trim to max_queries
      7. Write back to JSON (replace queries)
    """
    debug_decisions: List[str] = []
    GENERIC_NOISE_TOKENS = {
        "list","string","int","float","double","char","map","array","dict","object",
        "data","value","result"
    }

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        fallback = {"queries": [], "error": "invalid_llm_json"}
        return json.dumps(fallback, ensure_ascii=False)

    raw_llm_queries = data.get("queries", [])
    added_symbols = entities.get("added_symbols", set())
    removed_symbols = entities.get("removed_symbols", set())
    added_params = entities.get("added_params", set())

    # Stage 1: core from LLM
    core = extract_core_llm_queries(
        raw_llm_queries,
        max_core=max_core_from_llm,
        debug=debug_decisions
    )

    # Stage 2: guarantee removed symbols
    ensure_removed_symbols(core, removed_symbols, max_total=max_queries, debug=debug_decisions)

    # Stage 3: add param adoption query
    param_adoption = build_param_adoption_queries(core, added_symbols, added_params,
                                                  max_total=max_queries, debug=debug_decisions)
    core.extend(param_adoption)

    # Stage 4: call variants (only if capacity remains)
    if len(core) < max_queries:
        call_variants = build_call_variants(core, max_total=max_queries,
                                            max_call_variants=max_call_variants,
                                            debug=debug_decisions)
        core.extend(call_variants)

    # Stage 5: final trim
    final_queries = core[:max_queries]

    try:
        pruned = []
        for q in final_queries:
            qt = q.get("query","").strip().lower()
            if qt in GENERIC_NOISE_TOKENS:
                debug_decisions.append(f"drop_noise:{qt}")
                continue
            pruned.append(q)
        final_queries = pruned
    except Exception:
        pass

    logger.info(f"[QueryPipeline] LLM raw={len(raw_llm_queries)} -> final={len(final_queries)} decisions={len(debug_decisions)}")
    logger.debug(f"[QueryPipeline] Decisions trace: {debug_decisions}")

    data["queries"] = final_queries
    return json.dumps(data, ensure_ascii=False)

def build_queries_response(response: str,
                           entities: Dict[str, Set[str]],
                           max_queries: int = 5) -> str:
    """
    Thin wrapper for assemble_queries.
    """
    return assemble_queries(
        response=response,
        entities=entities,
        max_queries=max_queries,
        max_core_from_llm=max_queries,
        max_call_variants=2
    )

def language_noise_filter(queries_json: str,
                           primary_language: str,
                           pr_content: Dict[str, Any]) -> str:
    """
    Remove cross-language noisy queries:
      - 'extends X' when primary language is Python/Go/Rust (no typical 'extends')
      - '<Component' when repo has no UI-like extensions and primary language not JS/TS
    Keep queries if they have explicit adoption/parameter intent even if look UI-like.
    """
    try:
        data = json.loads(queries_json)
    except json.JSONDecodeError:
        return queries_json
    qs = data.get("queries", [])
    filtered = []

    file_changes = pr_content.get("file_changes", []) or []
    ui_exts = (".tsx", ".jsx", ".vue", ".svelte")
    has_ui_files = any(
        isinstance(fc, dict) and str(fc.get("file_path","")).endswith(ext)
        for fc in file_changes for ext in ui_exts
    )

    for q in qs:
        qt = q.get("query","")
        intent = (q.get("intent") or "").lower()
        if not qt:
            continue
        # Drop 'extends ' patterns in languages where it's almost certainly noise
        if ("extends " in qt) and primary_language in ("Python","Go","Rust"):
            continue
        # Drop JSX-like start unless UI context exists or intent strongly suggests parameter adoption
        if qt.startswith("<") and not has_ui_files and primary_language not in ("JavaScript","TypeScript"):
            if intent not in ("parameter_adoption",):
                continue
        filtered.append(q)

    data["queries"] = filtered
    return json.dumps(data, ensure_ascii=False)