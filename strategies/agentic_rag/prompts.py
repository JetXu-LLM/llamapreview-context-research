import re
import logging
from typing import List, Dict, Any, Optional
from .state import CollectionState

logger = logging.getLogger("LlamaPReview")

def build_think_prompt(
    state: CollectionState,
    repo_structure: str,
    accessible_files: List[str],
    phase: str,  # "focus" or "plan"
    dependency_hints: List[str] = None
) -> str:
    """
    Build prompt for THINK phase (supports two-phase thinking: Focus -> Plan).
    
    Args:
        state: Current collection state.
        repo_structure: String representation of the repo tree (global or focused).
        accessible_files: List of file paths available in the current scope.
        phase: "focus" (choose directory) or "plan" (choose files).
        dependency_hints: List of potential dependency paths detected from diffs.
    """
    
    # Static prefix (identical for KV cache optimization)
    static_prefix = f"""You are an expert AI developer analyzing a pull request.

**Pull Request Details:**
{state.pr_details}

**Current Context ({len(state.collected_files)} files collected):**
{_format_collected_files(state) if state.collected_files else "No files collected yet."}
"""

    dynamic_suffix = ""

    if phase == "focus":
        # --- FOCUS PHASE LOGIC ---
        
        # Build refocus context if this is a refocus iteration
        refocus_context = ""
        if state.should_refocus and state.previous_focus_path:
            refocus_context = f"""
**Previous Exploration:**
- **Previous Focus:** `{state.previous_focus_path}`
- **Why Refocus:** {state.refocus_reason or 'Context needs improvement'}
- **Directional Hint:** {state.refocus_hint or 'No specific hint'}
- **Current Quality:** {state.current_quality.overall:.1f}/10 (Completeness: {state.current_quality.completeness:.1f}/10)

The previous focus area has been explored. Based on the analysis above, consider exploring a different area of the codebase.
"""
        elif state.should_refocus:
            # Refocus but no previous focus (shouldn't happen, but handle it)
            refocus_context = f"""
**Refocus Requested:**
- **Directional Hint:** {state.refocus_hint or 'No specific hint'}
- **Reason:** {state.refocus_reason or 'Context needs improvement'}
- **Current Quality:** {state.current_quality.overall:.1f}/10 (Completeness: {state.current_quality.completeness:.1f}/10)
"""

        # Calculate directory counts for annotation
        from collections import defaultdict
        dir_counts = defaultdict(int)
        for file_path in state.accessible_files:
            parts = file_path.split('/')
            for i in range(1, len(parts)):
                dir_path = '/'.join(parts[:i]) + '/'
                dir_counts[dir_path] += 1
        
        # Identify directories containing changed files for highlighting
        changed_files = _extract_changed_files_info(state.pr_content)
        changed_dirs = set()
        for cf in changed_files:
            if '/' in cf['path']:
                # Add all parent directories
                parts = cf['path'].split('/')[:-1]
                for i in range(1, len(parts) + 1):
                    changed_dirs.add('/'.join(parts[:i]) + '/')

        # Annotate repo_structure with file counts and relevance markers
        annotated_structure_lines = []
        for line in repo_structure.split('\n'):
            # Try to extract directory path from tree line
            dir_match = re.search(r'([a-zA-Z0-9_\-./]+/)(?:\s|$)', line)
            if dir_match:
                dir_path = dir_match.group(1)
                count = dir_counts.get(dir_path, 0)
                
                # Check if this directory contains changed files
                is_relevant = dir_path in changed_dirs
                relevance_mark = " ⭐" if is_relevant else ""
                
                if count > 0:
                    # Add file count indicator
                    if count < 50:
                        indicator = f"{relevance_mark} ✅ ({count} files)"
                    elif count < 200:
                        indicator = f"{relevance_mark} 👍 ({count} files)"
                    elif count < 1000:
                        # If relevant, downgrade warning to acceptable
                        icon = "⚠️" if not is_relevant else "👀"
                        indicator = f"{relevance_mark} {icon} ({count} files)"
                    else:
                        indicator = f"{relevance_mark} 🔴 ({count} files)"
                    line = line.rstrip() + indicator
            annotated_structure_lines.append(line)
        
        annotated_structure = '\n'.join(annotated_structure_lines)
        
        dynamic_suffix = f"""
{refocus_context}

**Repository Structure (High-Level View with File Counts):**

**Legend:**
- ⭐ **CONTAINS PR CHANGES (High Priority)**
- ✅ < 50 files (Ideal size)
- 👍 50-200 files (Good size)
- 👀 200-1000 files (Large, but **ACCEPTABLE** if marked with ⭐)
- ⚠️/🔴 > 1000 files (Very large, use only if absolutely necessary)

```
{annotated_structure}
```

**Your Task:**
Analyze the PR and the repository structure above. Identify the most relevant directory to explore in detail.

**Rules:**
1. **PRIMARY GOAL:** Choose the directory that contains the core logic modified in the PR.
2. **PRIORITY:** You **MUST** prioritize directories marked with **⭐**, even if they are large (👀/⚠️).
   - It is better to explore a large directory that contains the actual changes than a small directory that is irrelevant.
3. Only choose a "clean" directory (✅/👍) if the core logic is NOT in the ⭐ directories.
4. If the current context is sufficient, set `new_focus_path` to empty string.
5. If this is a refocus iteration, explore a DIFFERENT area than before.

**Respond in JSON:**
```json
{{
  "reasoning": "Why this directory is most relevant...",
  "new_focus_path": "path/to/directory"
}}
```
"""
    
    else:  # phase == "plan"
        # Plan phase: select specific files
        
        if state.focus_path:
            focus_prefix = state.focus_path.rstrip('/') + '/'
            focused_files = [f for f in accessible_files if f.startswith(focus_prefix)]
            focus_context = f"""
**Current Focus:** `{state.focus_path}` ({len(focused_files)} files in this directory)
**Previous Reasoning:** {state.last_reasoning[:300] if state.last_reasoning else 'N/A'}...
"""
        else:
            focused_files = accessible_files
            focus_context = f"""
**Current Focus:** Global view ({len(focused_files)} files)
"""

        accessible_files_text = _format_accessible_files(focused_files)
        
        # Construct Hints Section
        hints_section = ""
        if dependency_hints:
            hints_section = "\n**💡 AUTOMATED HINTS (Potential Dependencies):**\n"
            hints_section += "The system detected these files might be referenced by the PR changes:\n"
            for hint in dependency_hints:
                hints_section += f"- `{hint}`\n"

        collected_display = "\n".join([f"- {f}" for f in state.collected_files]) if state.collected_files else "(None)"

        dynamic_suffix = f"""
{focus_context}

**Repository Structure (Detailed View):**
```
{repo_structure}
```

**🛡️ Already Collected Files (Context Held):**
{collected_display}

**Available Files in Focus Area:**
```
{accessible_files_text}
```

{hints_section}

**Your Task:**
Act as a **Static Analysis Engine**. Your goal is to resolve the "Symbolic References" in the PR Diff into "Concrete Definitions". You need to select relevant files from the **Available Files in Focus Area** to resolve symbols found in the **Context Held**.

**The Philosophy of "Deep Context":**
A PR Diff is just a set of instructions. To understand if the instructions are correct, we need the **Manual** (the definitions).
- If the Diff calls `AuthService.login()`, the Reviewer needs to see `AuthService.ts` to know if `login()` handles errors correctly.
- If the Diff uses `UserType`, the Reviewer needs `types/user.ts` to check field compatibility.

**Execution Protocol:**

1.  **🚫 ZERO REDUNDANCY:**
    - Check the "Already Collected Files" list above.
    - **NEVER** request a file that is already in that list. It is a waste of tokens and time.
    - If you think a file is missing, double-check the "Already Collected" list first.

2.  **🔗 LINKER LOGIC:**
    - Scan the PR Diff for *imported* classes, functions, and types.
    - Locate their definitions in the "Available Files" list.
    - **Collect them.**

3.  **🔍 HINT VERIFICATION:**
    - Look at the "Automated Hints". These are files the system *thinks* are dependencies.
    - If they seem relevant to the core logic, collect them.

**Respond in JSON:**
```json
{{
"reasoning": "Analysis and strategy...",
"files_to_collect": [
    {{
    "path": "exact/path/from/available/files.ts",
    "reason": "Why this file is needed",
    "priority": "critical|high|medium|low"
    }}
],
"files_to_remove": [
    {{
    "path": "path/to/remove.ts",
    "reason": "Why removing this from Collected Files"
    }}
]
}}
```

**Budget:** {state.remaining_tokens():,} tokens, {state.remaining_time():.0f}s remaining
"""
        
    prompt = static_prefix + dynamic_suffix
    
    if phase == "focus":
        logger.debug(f"📄 FOCUS Prompt ({len(prompt)} chars):\n{prompt[:3000]}...\n...{prompt[-1000:]}")
    elif phase == "plan":
        logger.debug(f"📄 PLAN Prompt ({len(prompt)} chars):\n{prompt[:2000]}...\n...{prompt[-1000:]}")
    
    return prompt

def build_reflect_prompt(state: CollectionState) -> str:
    """
    Build the prompt for the REFLECT phase.
    This prompt provides the LLM with all information needed to evaluate context quality
    and plan the next iteration.
    """

    inventory_list = "\n".join([f"- {f}" for f in state.collected_files]) if state.collected_files else "(None)"
    
    # Build a summary of collected files
    collected_summary = []
    for path, meta in state.file_metadata.items():
        size_kb = meta.size / 1024
        collected_summary.append(
            f"- **{path}** ({meta.priority.value}, {size_kb:.1f}KB): {meta.reason}"
        )
    collected_text = "\n".join(collected_summary) if collected_summary else "No files have been collected yet."

    # Status update on missing files
    status_update = ""
    # [LOGIC ADAPTED FROM ORIGINAL: Check iteration history for missing files]
    if state.current_iteration > 1 and hasattr(state, 'iteration_history') and state.iteration_history:
        # Note: In original code, iteration_history was a list of IterationRecord objects.
        # We assume state.iteration_history is populated in collector.py
        last_iter_record = state.iteration_history[-1]
        if hasattr(last_iter_record, 'missing_critical_files'):
            last_missing = last_iter_record.missing_critical_files
            if last_missing:
                found = [f for f in last_missing if f in state.collected_files]
                still_missing = [f for f in last_missing if f not in state.collected_files]
                
                status_lines = ["\n### 📋 Status Update on Previously Requested Files\n"]
                if found:
                    status_lines.append("**✅ The following files you requested have been collected:**")
                    for f in found:
                        status_lines.append(f"- `{f}`")
                if still_missing:
                    # Filter out files that we know don't exist (state.non_existent_files check)
                    # Assuming non_existent_files is tracked in state (added in original code's REFLECT phase)
                    non_existent = getattr(state, 'non_existent_files', set())
                    actual_missing = [f for f in still_missing if f not in non_existent]
                    
                    if actual_missing:
                        status_lines.append("\n**❌ These files are still missing (please collect them):**")
                        for f in actual_missing:
                            status_lines.append(f"- `{f}`")
                            
                status_update = "\n".join(status_lines) + "\n"

    # Warning about non-existent files
    if hasattr(state, 'non_existent_files') and state.non_existent_files:
        status_update += "\n**⛔ REQUEST FAILED (NON-EXISTENT FILES):**\n"
        status_update += "The following files were requested but **DO NOT EXIST** in the current directory.\n"
        status_update += "**DO NOT request them again.** Check the 'Available Files' list carefully.\n"
        for f in sorted(list(state.non_existent_files))[:5]:
            status_update += f"- `{f}`\n"
        status_update += "\n"
    
    # Failed files summary
    failed_text = ""
    if state.failed_files:
        failed_list = [f"- {path}: {reason}" for path, reason in state.failed_files.items()]
        failed_text = f"\n## Failed to Collect Files\n\n{chr(10).join(failed_list)}"

    # Focus info
    if state.focus_path:
        focus_prefix = state.focus_path.rstrip('/') + '/'
        focused_accessible_files = [f for f in state.accessible_files if f.startswith(focus_prefix)]
        focus_info = f"Current Focus: `{state.focus_path}` ({len(focused_accessible_files)} files)"
        
        focus_context = f"""
### Current Focus Area

The agent is currently focused on: `{state.focus_path}`

**Question for you:** Based on the collected context and the PR's needs, should we:
- **Stay** in this area and collect more files here? (Set `should_explore_elsewhere: false`)
- **Switch** to a different area of the codebase? (Set `should_explore_elsewhere: true` and provide a hint)

If you recommend switching, provide a **directional hint** (e.g., "backend API layer", "shared types", "test utilities") rather than an exact path, since you don't have the full repository structure in this phase.
"""
    else:
        focused_accessible_files = list(state.accessible_files)
        focus_info = f"Global view ({len(focused_accessible_files)} files)"
        focus_context = """
### Current Focus Area
The agent is currently using a global view (no specific focus).
**Question for you:** Should we narrow down to a specific area? If yes, provide a directional hint.
"""

    # Accessible files summary (limit to 500)
    accessible_files_summary = "\n".join(sorted(focused_accessible_files)[:500])
    if len(focused_accessible_files) > 500:
        accessible_files_summary += f"\n... and {len(focused_accessible_files) - 500} more files."

    prompt = f"""You are a **Lead Architect** performing a "Context Audit".
Analyze the **current state of the PR context retrieval session** below to understand the coverage achieved so far.

## 📦 Current Context Inventory
{inventory_list}

----------

{focus_context}

{status_update}

## Core Pull Request Information

This is the full context of the PR you are analyzing. Use this as the "ground truth" to determine what context is necessary.

```
{state.pr_details}
```

---

## Current State of Context Collection

### Automatic Quality Metrics
- **Overall:** {state.current_quality.overall:.2f}/10
- **Completeness:** {state.current_quality.completeness:.2f}/10
- **Relevance:** {state.current_quality.relevance:.2f}/10
- **Sufficiency:** {state.current_quality.sufficiency:.2f}/10
- **Efficiency:** {state.current_quality.efficiency:.2f}/10
- **Confidence:** {state.current_quality.confidence:.2f}

### Collected Files ({len(state.collected_files)})
{collected_text}
--------------------
{failed_text}

### Budget Status
- Tokens: {state.total_tokens:,} / {state.token_budget:,} ({state.remaining_tokens():,} remaining)
- Time: {state.elapsed_time():.1f}s / {state.time_budget}s ({state.remaining_time():.1f}s remaining)
- Iteration: {state.current_iteration} / {state.max_iterations}

---

## Reference: Available Files in Current Focus Area

**{focus_info}**

This is a **COMPLETE** list of files available in the current focus area.
**CRITICAL: If you suggest an "add" action, you MUST choose a file from this list.**
**CRITICAL: If you suggest an "add" action, you MUST choose a file from this list. Any file not in this list does NOT exist in the current focus area.**

```
{accessible_files_summary}
```

**If you need a file that's not in this list:**
- It may exist in a different area of the repository
- Set `exploration_strategy.should_explore_elsewhere: true` and provide a `directional_hint`
- Do NOT request files that are not in the list above

---

## Your Task

Audit the `Collected Files` to ensure they provide the **Structural Knowledge** required to review this PR.

**The "Context Entropy" Test:**
1.  **Low Entropy (Bad):** The collected files are just a copy of the PR's `Changed Files`.
    - *Verdict:* The Agent is being lazy. You MUST reject this and demand the *dependencies*.
2.  **High Entropy (Good):** The collected files contain the *definitions* of types, interfaces, and services used in the PR, even if those files weren't touched.
    - *Verdict:* This is valuable "Dark Matter".

**Critical Validation Steps:**

1.  **Check for Missing Definitions:**
    - Pick a key function call or type in the PR Diff. Do we have its definition file?
    - If NO, set `recommendation: "refine"` and ask for it.

2.  **Hallucination Check:**
    - When recommending new files to add, ask yourself: *Does this file actually exist in the likely file structure?*
    - If you are unsure of the exact path, describe the *kind* of file needed (e.g., "The user service definition") rather than guessing a path like `src/services/impl/user_service_impl.ts` which might not exist.

3.  **Redundancy Check:**
    - Do not ask to add files that are already in the `Collected Files` list.

**Decision Logic:**
- **Refine:** If critical definitions (Base Classes, API Clients, Global Types) are missing.
- **Stop:** If we have enough context to understand *how* the changes interact with the rest of the system.

**Respond in the following JSON format:**

```json
{{
  "completeness_assessment": {{
    "score": 7.5,
    "missing_critical": ["path/to/critical_file.ts"],
    "missing_nice_to_have": ["path/to/optional_file.ts"]
  }},
  "relevance_assessment": {{
    "score": 8.0,
    "redundant_files": ["path/to/redundant_file.tsx"],
    "generic_components": ["path/to/generic_icon.tsx"]
  }},
  "overall_assessment": {{
    "score": 7.8,
    "confidence": 0.85,
    "can_reviewer_understand": true,
    "critical_gaps": ["Missing the validation schema used by the form."],
    "recommendation": "refine|acceptable|stop",
    "reason": "The context is good but is missing the core validation logic from the 'utils' module. We need to add that file to be sure of the change's impact."
  }},
  "exploration_strategy": {{
    "should_explore_elsewhere": false,
    "directional_hint": null,
    "reason": "Current focus area still has relevant files to collect."
  }},
  "priority_actions": [
    {{
      "action": "add",
      "file": "src/app/utils/validators/user-schema.ts",
      "reason": "This file contains the critical validation logic mentioned in the gaps.",
      "priority": "critical"
    }},
    {{
      "action": "remove",
      "file": "src/ui/generic/Icon.tsx",
      "reason": "This is a generic component and not relevant to the PR's logic.",
      "priority": "low"
    }}
  ],
  "estimated_improvement": 1.2
}}
```

**Recommendation values:**
- "refine": Quality needs improvement, continue in current or new area
- "acceptable": Quality is good enough, but can improve if budget allows
- "stop": Quality is sufficient, no need to continue

**Target Quality Score:** 8.0/10
**Current Score:** {state.current_quality.overall:.2f}/10
**Gap to Target:** {max(0, 8.0 - state.current_quality.overall):.2f} points

Be honest and critical. Recommend "stop" if quality is acceptable or if the cost of further improvement outweighs the benefit.
"""

    logger.debug(f"📄 REFLECT Prompt ({len(prompt)} chars):\n{prompt[:2000]}...\n...{prompt[-1000:]}")

    return prompt

def _format_collected_files(state: CollectionState) -> str:
    """Format collected files for prompt"""
    if not state.collected_files:
        return "None"
    
    lines = []
    for path, meta in state.file_metadata.items():
        lines.append(f"- {path} ({meta.priority.value}): {meta.reason}")
    
    return '\n'.join(lines[:20])  # Limit to 20 files for brevity

def _format_accessible_files(accessible_files: List[str]) -> str:
    """Format accessible files list for prompt"""
    if not accessible_files:
        return "No files available"
    
    # Group by directory for better readability
    from collections import defaultdict
    by_dir = defaultdict(list)
    
    for file_path in sorted(accessible_files):
        if '/' in file_path:
            dir_name = '/'.join(file_path.split('/')[:-1])
            file_name = file_path.split('/')[-1]
            by_dir[dir_name].append(file_name)
        else:
            by_dir['(root)'].append(file_path)
    
    lines = []
    for dir_name in sorted(by_dir.keys())[:30]:  # Limit directories
        lines.append(f"\n{dir_name}/")
        for file_name in sorted(by_dir[dir_name])[:15]:  # Limit files per dir
            lines.append(f"  - {file_name}")
        if len(by_dir[dir_name]) > 15:
            lines.append(f"  ... and {len(by_dir[dir_name]) - 15} more")
    
    if len(by_dir) > 30:
        lines.append(f"\n... and {len(by_dir) - 30} more directories")
    
    return '\n'.join(lines)

def _extract_changed_files_info(pr_content: Dict[str, Any]) -> List[Dict]:
    """
    Helper to extract changed files info from pr_content.
    Used to highlight relevant directories in the Focus prompt.
    """
    changed_files = []
    
    if 'file_changes' in pr_content:
        for file_change in pr_content['file_changes']:
            changed_files.append({
                'path': file_change.get('file_path', ''),
                'change_type': file_change.get('change_type', 'modified'),
                'additions': file_change.get('additions', 0),
                'deletions': file_change.get('deletions', 0),
                'diff': file_change.get('diff', '')
            })
    
    if not changed_files and 'commits' in pr_content:
        seen_files = set()
        for commit in pr_content['commits']:
            if 'files' in commit:
                for file_path in commit['files']:
                    if file_path not in seen_files:
                        seen_files.add(file_path)
                        changed_files.append({
                            'path': file_path,
                            'change_type': 'modified',
                            'additions': 0,
                            'deletions': 0,
                            'diff': ''
                        })
    
    logger.debug(f"Extracted {len(changed_files)} changed files")
    
    return changed_files