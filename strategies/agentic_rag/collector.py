import time
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set

from core.llm_client import DeepSeekClient
from core.github_client import GithubClient
from core.code_analysis import CodeContextExtractor
from strategies.base_strategy import ContextStrategy
from strategies.agentic_rag.state import (
    CollectionState, 
    IterationRecord, 
    FileMetadata, 
    Priority, 
    Decision
)
from strategies.agentic_rag.evaluator import QualityEvaluator
from strategies.agentic_rag.prompts import build_think_prompt, build_reflect_prompt

logger = logging.getLogger("LlamaPReview")

class AgenticRAGStrategy(ContextStrategy):
    """
    Solution 3: Agentic ReAct Loop.
    
    Implements a "Think -> Act -> Observe -> Reflect -> Decide" loop to intelligently 
    explore the repository and collect context.
    
    Architecture:
    - THINK: DeepSeek Reasoner (R1) analyzes repo structure and plans file collection.
    - ACT: Collects files using GitHub API, slicing large files via CodeContextExtractor.
    - OBSERVE: QualityEvaluator calculates completeness/relevance metrics.
    - REFLECT: DeepSeek Chat (V3) audits the context and suggests improvements.
    - DECIDE: Determines whether to continue, stop, or rollback based on budget and quality.
    """

    # Configuration Constants
    REASONER_MODEL = "deepseek-reasoner"
    CHAT_MODEL = "deepseek-chat"
    
    MAX_FILE_SIZE = 50000
    MAX_CONTEXT_CHARS = 200000

    # Two-phase thinking depth configuration
    GLOBAL_VIEW_DEPTH = 2  # Shallow depth for initial global view
    FOCUSED_VIEW_DEPTH = 4  # Deeper depth when focusing on specific directory

    def __init__(self, llm_client: DeepSeekClient, github_client: GithubClient):
        """
        Initialize the Agentic Strategy with core clients.
        """
        self.llm = llm_client
        self.github = github_client
        self.code_extractor = CodeContextExtractor()

    def execute(self, pr_details: str, pr_content: Dict[str, Any], repo_full_name: str) -> str:
        """
        Main entry point for ReAct-based context collection.
        """
        # Initialize State
        state = CollectionState(
            pr_details=pr_details,
            pr_content=pr_content,
            repo_full_name=repo_full_name,
            sha=self._get_latest_sha(pr_content)
        )
        
        # Extract changed files info (Diffs, Additions, Deletions)
        changed_files_info = self._extract_changed_files_info(pr_content)
        
        logger.info("="*70)
        logger.info("🚀 ReAct PR Context Collection Started")
        logger.info("="*70)
        logger.info(f"Repository: {repo_full_name}")
        logger.info(f"Changed files: {len(changed_files_info)}")
        logger.info(f"Budget: {state.token_budget} tokens")

        # Establish ground truth: get full file list once
        logger.info("\n🔐 Establishing ground truth for accessible files...")
        try:
            # Call GitHub Client to get full structure
            full_structure_data = self.github.get_repo_structure_for_llm(
                repo_full_name=repo_full_name,
                sha=state.sha,
                max_depth=99,
                include_file_list=True,
                include_summary=False,
                show_files=False
            )
            
            if 'error' in full_structure_data or 'files' not in full_structure_data:
                raise RuntimeError(f"Failed to get full file list: {full_structure_data.get('error', 'Unknown error')}")

            state.accessible_files = {
                item['full_path'] for item in full_structure_data.get('files', [])
            }
            logger.info(f"✓ Ground truth established: {len(state.accessible_files)} accessible files found.")

        except Exception as e:
            logger.critical(f"Could not establish accessible files. Aborting. Error: {e}", exc_info=True)
            return self._generate_partial_context(state, changed_files_info, "Failed to list repository files.")
        
        # ReAct Loop
        try:
            while state.should_continue and state.current_iteration < state.max_iterations:
                state.current_iteration += 1
                
                logger.info("\n" + "="*70)
                logger.info(f"🔄 Iteration {state.current_iteration}/{state.max_iterations}")
                logger.info("="*70)
                
                # Save snapshot for potential rollback
                # (Logic copied from original CollectionState.save_snapshot, implemented here or in state)
                # Assuming state has a method or we do it manually. 
                # Since state.py was defined as dataclass, we might need to handle snapshots manually if method missing.
                # But based on previous context, state has save_snapshot.
                state.save_snapshot() 
                
                iteration_start = time.time()
                
                # --- THINK Phase ---
                # Handles Focus (Directory selection) and Plan (File selection)
                plan = self._phase_think(state, changed_files_info)
                if not plan:
                    logger.warning("Planning failed, stopping")
                    break
                
                # --- ACT Phase ---
                # Collects files based on the plan
                self._phase_act(state, plan)
                
                # --- OBSERVE Phase ---
                # Calculate quality metrics
                quality_before = state.current_quality.overall if hasattr(state, 'quality_history') and state.quality_history else 0.0
                state.current_quality = QualityEvaluator.evaluate(state, changed_files_info)
                if not hasattr(state, 'quality_history'): state.quality_history = []
                state.quality_history.append(state.current_quality)
                
                # --- REFLECT Phase ---
                # LLM evaluates context quality
                reflection = self._phase_reflect(state)
                
                # --- DECIDE Phase ---
                # Determine next step
                decision, reason = self._phase_decide(state, reflection)
                
                # Record iteration statistics
                iteration_duration = time.time() - iteration_start
                record = IterationRecord(
                    iteration=state.current_iteration,
                    action=plan.get('action', 'collect') if plan else 'unknown',
                    files_added=[f.get('path', '') for f in plan.get('files_to_collect', [])] if plan else [],
                    files_removed=[f.get('path', '') for f in plan.get('files_to_remove', [])] if plan else [],
                    quality_before=quality_before,
                    quality_after=state.current_quality.overall,
                    reasoning=getattr(state, 'reasoner_thinking', '')[:500],
                    tokens_used=state.total_tokens,
                    duration=iteration_duration,
                    timestamp=datetime.now(),
                    missing_critical_files=reflection.get('completeness_assessment', {}).get('missing_critical', [])
                )
                if not hasattr(state, 'iteration_history'): state.iteration_history = []
                state.iteration_history.append(record)
                
                logger.info(f"\n📊 Iteration {state.current_iteration} Summary:")
                logger.info(f"   Quality: {quality_before:.2f} → {state.current_quality.overall:.2f}")
                logger.info(f"   Files: {len(state.collected_files)}")
                logger.info(f"   Decision: {decision.value.upper()} - {reason}")
                
                # Handle Focus Logic for next iteration
                if state.should_refocus:
                    logger.info(f"🔄 Next iteration will re-evaluate focus (hint: {state.refocus_hint or 'none'})")
                
                if decision == Decision.STOP:
                    state.should_continue = False
                    break
                elif decision == Decision.ROLLBACK:
                    self._rollback_to_previous(state)
                    break
            
            # Finalize Context
            pr_context = self._finalize_context(state, changed_files_info)
            
            logger.info("\n" + "="*70)
            logger.info("✅ Collection Complete")
            logger.info("="*70)
            
            return pr_context
        
        except Exception as e:
            logger.error(f"Critical error in ReAct loop: {e}", exc_info=True)
            return self._generate_partial_context(state, changed_files_info, str(e))
        
    def _extract_insights(self, reasoning: str) -> List[str]:
        """Extract key insights from reasoning"""
        insights = []

        logger.debug(f"Extracting insights from {len(reasoning)} chars of reasoning")
        
        # Look for insight indicators (expanded patterns)
        patterns = [
            r'(?:I notice|I see|I observe|I find|I identify|We can see|It appears)\s+(.+?)(?:\.|$)',
            r'(?:This suggests|This indicates|This means|This shows)\s+(.+?)(?:\.|$)',
            r'(?:Key insight[s]?:|Important[ly]?:|Critical[ly]?:)\s+(.+?)(?:\.|$)',
            r'(?:The main point is|Notably|Significantly)\s+(.+?)(?:\.|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE | re.DOTALL)
            # Clean up matches (remove newlines, extra spaces)
            cleaned = [re.sub(r'\s+', ' ', m.strip()) for m in matches if len(m.strip()) > 10]
            insights.extend(cleaned[:3])

        logger.debug(f"Extracted {len(insights)} raw insights before deduplication")
        
        # Deduplicate while preserving order
        seen = set()
        unique_insights = []
        for insight in insights:
            insight_lower = insight.lower()
            if insight_lower not in seen and len(insight) < 200:
                seen.add(insight_lower)
                unique_insights.append(insight)
        
        return unique_insights[:5]

    def _phase_think(self, state: CollectionState, changed_files_info: List[Dict]) -> Optional[Dict]:
        """
        THINK: Two-phase thinking (Focus -> Plan).
        Uses DeepSeek Reasoner to analyze structure and plan collection.
        """
        
        # Phase 1: FOCUS - Determine which directory to explore
        should_execute_focus = (not state.focus_path) or state.should_refocus
        
        if should_execute_focus:
            logger.info("🧠 THINK Phase 1: FOCUS - Identifying exploration target...")
            
            # Get shallow global view
            repo_structure_data = self.github.get_repo_structure_for_llm(
                repo_full_name=state.repo_full_name,
                sha=state.sha,
                target_path=None,
                max_depth=self.GLOBAL_VIEW_DEPTH,
                include_file_list=False,
                include_summary=True
            )
            repo_structure = repo_structure_data.get('tree', '')

            # Build Prompt using strategies/agentic_rag/prompts.py
            prompt = build_think_prompt(
                state=state,
                repo_structure=repo_structure,
                accessible_files=[], # No file list in focus phase
                phase="focus"
            )
            
            messages = [
                {"role": "system", "content": "You are an expert code reviewer analyzing a PR to identify the most relevant area to explore."},
                {"role": "user", "content": prompt}
            ]
            
            try:
                # Call LLM (Reasoner)
                response = self.llm.chat(
                    messages=messages,
                    model=self.REASONER_MODEL,
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                # Parse Response
                message = response['choices'][0]['message']
                content = message.get('content', '')
                reasoning = message.get('reasoning_content', '')
                
                # Update State with Insights
                state.reasoner_thinking = reasoning # Store reasoning
                state.key_insights = self._extract_insights(reasoning) # Optional: Extract insights
                
                focus_result = self._parse_json_response(content)
                if not focus_result:
                    logger.error("Failed to parse focus result")
                    return None
                
                new_focus_path = focus_result.get('new_focus_path', '')
                
                if new_focus_path:
                    state.focus_path = new_focus_path
                    logger.info(f"🎯 Focus determined: {new_focus_path}")
                else:
                    logger.info("🎯 No specific focus needed, using global view")
                
                # Reset refocus flags
                state.should_refocus = False
                state.refocus_hint = None
                state.refocus_reason = None
            except Exception as e:
                logger.error(f"FOCUS phase failed: {e}", exc_info=True)
                return None
        else:
            logger.info(f"🧠 THINK: Skipping FOCUS (continuing with current focus: {state.focus_path or 'global'})")

        # Phase 2: PLAN - Creating collection strategy
        logger.info("🧠 THINK Phase 2: PLAN - Creating collection strategy...")
        
        # Get detailed view of focused area
        repo_structure_data = self.github.get_repo_structure_for_llm(
            repo_full_name=state.repo_full_name,
            sha=state.sha,
            target_path=state.focus_path,
            max_depth=self.FOCUSED_VIEW_DEPTH if state.focus_path else self.GLOBAL_VIEW_DEPTH,
            include_file_list=False
        )
        repo_structure = repo_structure_data.get('tree', '')
        
        # Get file list for the focused area
        accessible_files = self._get_accessible_files_for_context(state, state.focus_path)
        accessible_files = [f for f in accessible_files if f not in state.collected_files]

        # Heuristic dependency check
        dependency_hints = self._identify_potential_dependencies(state, changed_files_info)
        if dependency_hints:
            logger.info(f"💡 Detected {len(dependency_hints)} potential external dependencies")

        # Build Prompt
        prompt = build_think_prompt(
            state=state,
            repo_structure=repo_structure,
            accessible_files=accessible_files,
            phase="plan",
            dependency_hints=dependency_hints
        )
        
        messages = [
            {"role": "system", "content": "You are an expert code reviewer creating a detailed file collection plan."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Call LLM (Chat Model is sufficient for Planning if Focus was done by Reasoner, 
            # but original code used Chat model here to save cost/time)
            response = self.llm.chat(
                messages=messages,
                model=self.CHAT_MODEL,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            message = response['choices'][0]['message']
            content = message.get('content', '')
            
            plan = self._parse_json_response(content)
            if not plan:
                logger.error("Failed to parse plan")
                return None
            
            # Validate files against accessible_files
            files_to_collect = plan.get('files_to_collect', [])
            validated_files = []
            
            for file_info in files_to_collect:
                file_path = file_info.get('path', '')
                if file_path in state.accessible_files:
                    validated_files.append(file_info)
                else:
                    logger.warning(f"File not accessible: {file_path}")
            
            plan['files_to_collect'] = validated_files
            return plan
        
        except Exception as e:
            logger.error(f"PLAN phase failed: {e}", exc_info=True)
            return None

    def _phase_act(self, state: CollectionState, plan: Dict) -> None:
        """
        ACT: Collect files based on plan.
        Uses CodeContextExtractor to slice large files.
        """
        logger.info("⚙️  ACT: Collecting files...")
        
        # Extract symbols for smart slicing
        symbols_of_interest = self._extract_symbols_from_pr(state.pr_content)

        # 1. Remove files
        files_to_remove = plan.get('files_to_remove', [])
        if files_to_remove:
            logger.info(f"📦 Removing {len(files_to_remove)} redundant file(s)...")
            for file_info in files_to_remove:
                path = file_info.get('path')
                if path in state.collected_files:
                    del state.collected_files[path]
                    del state.file_metadata[path]
                    logger.info(f"   - Removed: {path} (Reason: {file_info.get('reason', 'N/A')})")
                else:
                    logger.warning(f"   - Already removed or not found: {path}")
        
        # 2. Collect new files
        files_to_collect = plan.get('files_to_collect', [])
        if not files_to_collect:
            return
        
        # Sort by priority
        priority_order = {
            "critical": 0, "high": 1, "medium": 2, "low": 3
        }
        files_to_collect.sort(
            key=lambda x: priority_order.get(x.get('priority', 'medium').lower(), 2)
        )
        
        for i, file_info in enumerate(files_to_collect, 1):
            file_path = file_info['path']
            priority_str = file_info.get('priority', 'medium').upper()
            priority = Priority[priority_str] if priority_str in Priority.__members__ else Priority.MEDIUM
            reason = file_info.get('reason', 'No reason provided')
            
            logger.info(f"   [{i}/{len(files_to_collect)}] {file_path} ({priority.value})")
            
            # Mark as attempted (assuming state has this set)
            if not hasattr(state, 'attempted_files'): state.attempted_files = set()
            state.attempted_files.add(file_path)
            
            # Read file safely
            content = self._read_file_safe(
                state=state,
                file_path=file_path,
                priority=priority,
                symbols_of_interest=symbols_of_interest
            )
            
            if content:
                state.collected_files[file_path] = content
                state.file_metadata[file_path] = FileMetadata(
                    path=file_path,
                    size=len(content),
                    priority=priority,
                    reason=reason,
                    iteration_collected=state.current_iteration
                )
                logger.info(f"      ✓ Collected ({len(content):,} chars)")
            else:
                logger.warning(f"      ✗ Failed to collect")

    def _phase_reflect(self, state: CollectionState) -> Dict:
        """
        REFLECT: LLM evaluates context quality and suggests improvements.
        """
        logger.info("🤔 REFLECT: Evaluating context quality...")
        
        # Skip if budget low
        if state.remaining_tokens() < 3000:
            return {'recommendation': 'stop', 'reason': 'Budget exhausted', 'priority_actions': []}
        
        # Build Prompt
        prompt = build_reflect_prompt(state)
        
        messages = [
            {"role": "system", "content": "You are an expert code reviewer evaluating PR context quality. Respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm.chat(
                messages=messages,
                model=self.CHAT_MODEL,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response['choices'][0]['message']['content']
            reflection = self._parse_json_response(content)
            
            if not reflection:
                return {'recommendation': 'unknown', 'reason': 'Parse failed', 'priority_actions': []}
            
            # Validate actions (Guardrails against hallucinations)
            reflection = self._validate_reflection_actions(state, reflection)
            
            # Handle Exploration Strategy
            exploration = reflection.get('exploration_strategy', {})
            if exploration.get('should_explore_elsewhere', False):
                state.should_refocus = True
                state.refocus_hint = exploration.get('directional_hint')
                state.refocus_reason = exploration.get('reason')
                state.previous_focus_path = state.focus_path
            
            return reflection
            
        except Exception as e:
            logger.error(f"REFLECT phase failed: {e}", exc_info=True)
            return {'recommendation': 'stop', 'reason': f'Error: {e}', 'priority_actions': []}

    def _phase_decide(self, state: CollectionState, reflection: Dict) -> Tuple[Decision, str]:
        """
        DECIDE: Determine whether to continue, stop, or rollback
        
        Strategy: Be moderately aggressive - continue if there's clear improvement potential
        and sufficient budget, but stop if quality is acceptable and gains are marginal.
        """
        
        logger.info("🎯 DECIDE: Evaluating next action...")
        
        logger.debug(f"Decision inputs:")
        logger.debug(f"  Iteration: {state.current_iteration}/{state.max_iterations}")
        logger.debug(f"  Quality: {state.current_quality.overall:.2f} (threshold: {state.quality_threshold})")
        logger.debug(f"  Completeness: {state.current_quality.completeness:.2f}")
        logger.debug(f"  Tokens: {state.remaining_tokens():,} remaining")
        logger.debug(f"  Time: {state.remaining_time():.1f}s remaining")
        
        llm_recommendation = reflection.get('overall_assessment', {}).get('recommendation', 'unknown')
        logger.debug(f"  LLM recommendation: {llm_recommendation}")
        
        # === Hard Constraints (Always Stop) ===
        
        if state.current_iteration >= state.max_iterations:
            return Decision.STOP, "Max iterations reached"
        
        if state.remaining_tokens() < 5000:
            return Decision.STOP, "Token budget nearly exhausted"
        
        if state.remaining_time() < 60:
            return Decision.STOP, "Time budget nearly exhausted"
        
        # === Quality Degradation (Rollback) ===
        
        if len(state.quality_history) >= 2:
            previous_quality = state.quality_history[-2].overall
            current_quality = state.current_quality.overall
            
            if current_quality < previous_quality - 0.5:
                logger.warning(f"⚠️  Quality degraded: {previous_quality:.2f} → {current_quality:.2f}")
                return Decision.ROLLBACK, "Quality degraded significantly"
        
        # === Excellent Quality (Stop) ===
        
        # If quality is excellent (9.0+) and no critical gaps, stop immediately
        missing_critical = reflection.get('completeness_assessment', {}).get('missing_critical', [])
        
        if state.current_quality.overall >= 9.0 and len(missing_critical) == 0:
            return Decision.STOP, f"Excellent quality achieved ({state.current_quality.overall:.2f}/10)"
        
        # === LLM Recommendation: "stop" ===
        
        # Only stop on LLM's "stop" recommendation if quality is genuinely good
        if llm_recommendation == 'stop':
            if state.current_quality.overall >= 7.5:
                return Decision.STOP, "LLM recommends stopping (quality sufficient)"
            else:
                logger.warning(f"⚠️  LLM suggests 'stop' but quality is only {state.current_quality.overall:.2f}/10, continuing anyway")
        
        # === LLM Recommendation: "acceptable" ===
        
        # "acceptable" means quality is OK but can be improved
        # Continue if: (1) budget is sufficient, AND (2) there's clear improvement potential
        if llm_recommendation == 'acceptable':
            estimated_improvement = reflection.get('estimated_improvement', 0.0)
            has_budget = state.remaining_tokens() >= 20000  # Conservative estimate for one more iteration
            
            # If quality is already good (8.0+) and improvement is marginal, stop
            if state.current_quality.overall >= 8.0 and estimated_improvement < 0.5:
                return Decision.STOP, f"Quality acceptable ({state.current_quality.overall:.2f}/10), marginal gains"
            
            # If quality is decent (7.0-8.0) but no budget or low improvement, stop
            if state.current_quality.overall >= 7.0 and (not has_budget or estimated_improvement < 0.3):
                return Decision.STOP, f"Quality acceptable ({state.current_quality.overall:.2f}/10), cost-benefit unfavorable"
            
            # Otherwise, if there's budget and improvement potential, continue
            if has_budget and estimated_improvement >= 0.3:
                return Decision.CONTINUE, f"Quality acceptable but can improve (+{estimated_improvement:.1f} expected)"
            
            # Fallback: stop if we've done at least 2 iterations
            if state.current_iteration >= 2:
                return Decision.STOP, f"Quality acceptable after {state.current_iteration} iterations"
        
        # === LLM Recommendation: "refine" ===
        
        # "refine" means quality needs improvement - continue if budget allows
        if llm_recommendation == 'refine':
            if state.remaining_tokens() >= 15000:
                return Decision.CONTINUE, "Quality needs refinement"
            else:
                return Decision.STOP, "Quality needs refinement but budget insufficient"
        
        # === Diminishing Returns Check ===
        
        if len(state.quality_history) >= 2:
            previous_quality = state.quality_history[-2].overall
            current_quality = state.current_quality.overall
            improvement = current_quality - previous_quality
            
            # If improvement is very small and quality is acceptable, stop
            if improvement < 0.2 and current_quality >= 7.0:
                return Decision.STOP, f"Diminishing returns ({improvement:.2f} improvement)"
        
        # === Critical Files Missing ===
        
        # If critical files are missing and we have budget, continue
        if len(missing_critical) > 0 and state.remaining_tokens() >= 15000:
            return Decision.CONTINUE, f"Critical files missing: {len(missing_critical)}"
        
        # === Quality Below Minimum ===
        
        # If quality is poor (<6.0), continue if possible
        if state.current_quality.overall < 6.0:
            if state.remaining_tokens() >= 10000:
                return Decision.CONTINUE, "Quality below minimum threshold"
            else:
                return Decision.STOP, "Quality poor but budget insufficient"
        
        # === Default: Continue if quality is not yet good ===
        
        if state.current_quality.overall < state.quality_threshold:
            return Decision.CONTINUE, f"Quality below target ({state.current_quality.overall:.2f} < {state.quality_threshold})"
        
        # === Fallback: Stop ===
        
        return Decision.STOP, f"Quality sufficient ({state.current_quality.overall:.2f}/10)"
    
    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _identify_potential_dependencies(self, state: CollectionState, changed_files_info: List[Dict]) -> List[str]:
        """
        Analyze diffs to find potential external dependencies (imports/includes).
        Returns a list of file paths from accessible_files that might be dependencies.
        """
        potential_deps = set()
        
        # Regex to capture import paths from added lines
        # Matches: import ... from 'path', require('path'), include "path", from ... import
        # This is a broad regex to catch JS/TS, Python, Go, C++, etc.
        import_pattern = re.compile(r'(?:from|import|include|require)\s+(?:[\w\s,{}\*]*\s+from\s+)?["\'<]([\w\-\./@]+)["\'>]')
        
        for file_info in changed_files_info:
            diff = file_info.get('diff', '')
            # Only look at added lines
            added_lines = '\n'.join([line for line in diff.split('\n') if line.startswith('+')])
            
            matches = import_pattern.findall(added_lines)
            for match in matches:
                # Clean up the path (remove leading @, etc. if needed, but keep simple for now)
                clean_match = match.strip()
                if not clean_match or clean_match.startswith('.'):
                    continue # Skip relative imports for simple matching (too hard to resolve without full path logic)

                # Simple heuristic: check if the import string appears in any accessible file path
                # e.g. import 'services/user' matches 'app/services/user.ts'
                for accessible_file in state.accessible_files:
                    # Avoid suggesting the file itself or files already collected
                    if accessible_file == file_info['path'] or accessible_file in state.collected_files:
                        continue
                        
                    # Check for suffix match (handling extensions)
                    # e.g. "utils/format" matches "src/utils/format.ts"
                    if accessible_file.endswith(clean_match) or \
                       f"{clean_match}." in accessible_file or \
                       f"{clean_match}/index" in accessible_file:
                        potential_deps.add(accessible_file)
        
        # Limit to top 10 most likely dependencies to avoid overwhelming prompt
        return sorted(list(potential_deps))[:10]

    def _extract_symbols_from_pr(self, pr_content: Dict[str, Any]) -> Set[str]:
        """
        Extract symbols (classes, functions, types) referenced in PR diff.
        Language-agnostic approach focusing on added lines.
        
        Args:
            pr_content: PR content dictionary containing file_changes
        
        Returns:
            Set of symbol names that are likely important for understanding the PR.
            Filtered to remove common noise and limited to top 50 most frequent symbols.
        """
        symbols = set()
        
        # Extract from file changes
        for file_change in pr_content.get('file_changes', []):
            diff = file_change.get('diff', '')
            
            # Pattern 1: Function/method calls - myFunc(, obj.method(
            # Only extract from added lines (lines starting with '+')
            added_lines = [line[1:] for line in diff.split('\n') if line.startswith('+')]
            added_diff = '\n'.join(added_lines)
            
            func_calls = re.findall(r'\b([a-z_][a-zA-Z0-9_]*)\s*\(', added_diff)
            symbols.update(func_calls)
            
            # Pattern 2: Class/Type references - new MyClass, : MyType
            type_refs = re.findall(r'\b([A-Z][a-zA-Z0-9_]*)\b', added_diff)
            symbols.update(type_refs)
            
            # Pattern 3: Import statements
            import_symbols = re.findall(r'import\s+\{?\s*([A-Za-z0-9_, ]+)\s*\}?', added_diff)
            for imp in import_symbols:
                symbols.update(s.strip() for s in imp.split(','))
        
        # Enhanced noise filter
        noise = {
            # JavaScript/TypeScript built-ins
            'console', 'log', 'error', 'warn', 'info', 'debug',
            'JSON', 'parse', 'stringify',
            'Array', 'Object', 'String', 'Number', 'Boolean', 'Date', 'Math',
            'map', 'filter', 'reduce', 'forEach', 'find', 'some', 'every',
            'push', 'pop', 'shift', 'unshift', 'slice', 'splice',
            'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval',
            'Promise', 'async', 'await',
            
            # Python built-ins
            'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
            'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool',
            
            # Common keywords
            'if', 'else', 'for', 'while', 'return', 'const', 'let', 'var',
            'function', 'class', 'import', 'export', 'from', 'as', 'new',
            'this', 'self', 'true', 'false', 'null', 'undefined',
            
            # Common generic names
            'data', 'result', 'value', 'item', 'index', 'key', 'name', 'id',
            'get', 'set', 'add', 'remove', 'update', 'delete', 'create'
        }
        
        # Filter: remove noise, keep only symbols with length > 2
        symbols = {s for s in symbols if s not in noise and len(s) > 2}
        
        # Additional filter: prioritize symbols that appear multiple times
        # (more likely to be important)
        if len(symbols) > 50:  # If too many symbols, filter by frequency
            symbol_counts = {}
            for file_change in pr_content.get('file_changes', []):
                diff = file_change.get('diff', '')
                for symbol in symbols:
                    symbol_counts[symbol] = symbol_counts.get(symbol, 0) + diff.count(symbol)
            
            # Keep top 50 most frequent symbols
            sorted_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)
            symbols = {s for s, _ in sorted_symbols[:50]}
        
        return symbols
    
    def _read_file_safe(
        self,
        state: CollectionState,
        file_path: str,
        priority: Priority,
        symbols_of_interest: Optional[Set[str]] = None
    ) -> Optional[str]:
        """
        Safely read file with error handling and retry.
        If symbols_of_interest is provided, extract only relevant code blocks.
        """
        # Check if already failed
        if not hasattr(state, 'failed_files'): state.failed_files = {}
        if file_path in state.failed_files:
            return None

        try:
            # Use GitHub Client
            content = self.github.get_file_content(state.repo_full_name, file_path)
            
            if content is None:
                raise Exception("File content is None")
            
            # Try to extract relevant blocks if symbols are provided and file is non-trivial
            if symbols_of_interest and len(content) > 3000:
                extracted_blocks = self._extract_relevant_blocks(content, symbols_of_interest)
                
                if extracted_blocks:
                    assembled = self._assemble_code_blocks(extracted_blocks, file_path)
                    # Use assembled if significantly smaller
                    if len(assembled) < len(content) * 0.8:
                        content = assembled
            
            # Truncate if still too large
            if len(content) > self.MAX_FILE_SIZE:
                content = content[:self.MAX_FILE_SIZE] + "\n\n[... file truncated ...]"
            
            return content
            
        except Exception as e:
            state.failed_files[file_path] = str(e)
            logger.error(f"Failed to read {file_path}: {e}")
            return None

    def _extract_relevant_blocks(self, content: str, symbols: Set[str]) -> List[str]:
        """
        Extract code blocks that define or use the given symbols.
        Uses CodeContextExtractor for language-agnostic extraction.
        
        Returns:
            List of code block strings.
        """
        blocks = []
        lines = content.splitlines()
        
        # Find lines that mention any of the symbols
        relevant_line_indices = []
        for i, line in enumerate(lines):
            for symbol in symbols:
                # Match whole word only (avoid partial matches)
                if re.search(rf'\b{re.escape(symbol)}\b', line):
                    relevant_line_indices.append(i)
                    break
        
        if not relevant_line_indices:
            # No symbols found, extract top-level definitions as fallback
            logger.debug(f"   No symbols found, extracting top-level definitions as fallback")
            
            # Use a set to track seen blocks efficiently
            seen_block_ranges = set()
            top_level_blocks = []
            
            for i, line in enumerate(lines):
                # Check if this line starts a definition
                is_def_start = any(
                    pattern.search(line) 
                    for pattern in self.code_extractor._FUNC_CONTEXT_PATTERNS
                )
                
                if is_def_start:
                    block, start, end = self.code_extractor.extract_enclosing_block(
                        content=content,
                        line_index=i,
                        symbol="",
                        max_block_lines=50
                    )
                    
                    if block and start > 0 and end > 0:
                        block_key = (start, end)
                        if block_key not in seen_block_ranges:
                            seen_block_ranges.add(block_key)
                            top_level_blocks.append((start, end))
                            
                            if len(top_level_blocks) >= 5:
                                break
            
            if top_level_blocks:
                blocks = [
                    f"# Lines {start}-{end}\n{chr(10).join(lines[start-1:end])}"
                    for start, end in top_level_blocks
                ]
                return blocks
            else:
                # Last resort: return first 100 lines
                logger.warning(f"   No definitions found, returning first 100 lines")
                return ["\n".join(lines[:100])]
        
        logger.debug(f"   Found {len(relevant_line_indices)} lines mentioning symbols")
        
        # For each relevant line, extract the enclosing block
        seen_blocks = set()
        
        for line_idx in relevant_line_indices:
            # Try to extract enclosing block
            block, start, end = self.code_extractor.extract_enclosing_block(
                content=content,
                line_index=line_idx,
                symbol="",
                max_block_lines=100
            )

            if block and start >= 1 and end >= start:
                block_key = f"{start}-{end}"
                if block_key not in seen_blocks:
                    seen_blocks.add(block_key)
                    blocks.append(f"# Lines {start}-{end}\n{block}")
            else:
                # Fallback: use line window
                window, start, end = self.code_extractor.build_line_window(
                    content=content,
                    line_index=line_idx,
                    window=5
                )
                if start > 0 and end > 0:
                    block_key = f"{start}-{end}"
                    if block_key not in seen_blocks:
                        seen_blocks.add(block_key)
                        blocks.append(f"# Lines {start}-{end}\n{window}")
        
        return blocks

    def _assemble_code_blocks(self, blocks: List[str], file_path: str) -> str:
        """
        Assemble extracted code blocks into a readable format.
        
        Args:
            blocks: List of code block strings (each with line number comments)
            file_path: Original file path (for context)
        
        Returns:
            Assembled content string.
        """
        if not blocks:
            return ""
        
        header = f"# Relevant excerpts from {file_path}\n"
        header += f"# (Extracted {len(blocks)} code blocks based on PR symbols)\n"
        header += f"# Note: This is a filtered view. Some code may be omitted.\n\n"
        
        # Use a clearer separator with context
        separator = "\n\n# " + "="*70 + "\n"
        separator += "# [Next code block]\n"
        separator += "# " + "="*70 + "\n\n"
        
        assembled = header + separator.join(blocks)
        
        return assembled

    def _validate_reflection_actions(self, state: CollectionState, reflection: Dict) -> Dict:
        """
        Filters the 'priority_actions' from the reflection response.
        Records non-existent files to prevent future hallucinations.
        """
        if 'priority_actions' not in reflection:
            return reflection

        validated_actions = []
        original_actions = reflection.get('priority_actions', [])
        accessible_files = state.accessible_files

        logger.debug(f"🛡️  Guardrail checking {len(original_actions)} actions against {len(accessible_files)} accessible files")
        
        for action in original_actions:
            if action.get('action') == 'add':
                file_path = action.get('file')
                if file_path and file_path in accessible_files:
                    validated_actions.append(action)
                else:
                    # Record the hallucination so we can warn the LLM next time
                    if file_path:
                        state.non_existent_files.add(file_path)
                        logger.warning(f"REFLECT Guardrail: Discarding hallucinated 'add' action for non-existent file: {file_path}")
                        filename = file_path.split('/')[-1]
                        similar_files = [f for f in accessible_files if filename in f]
                        if similar_files:
                            logger.info(f"   💡 Similar files exist: {similar_files[:3]}")
                        else:
                            dir_path = '/'.join(file_path.split('/')[:-1])
                            files_in_dir = [f for f in accessible_files if f.startswith(dir_path + '/')]
                            if files_in_dir:
                                logger.info(f"   💡 Directory '{dir_path}/' exists with {len(files_in_dir)} files")
                                logger.info(f"      Files in that directory: {files_in_dir[:5]}")
                            else:
                                logger.error(f"   ❌ Directory '{dir_path}/' does not exist")
            else:
                validated_actions.append(action)
        
        if len(validated_actions) < len(original_actions):
            logger.info(
                f"REFLECT Guardrail: Sanitized reflection actions. "
                f"Original: {len(original_actions)}, Validated: {len(validated_actions)}"
            )
            reflection['priority_actions'] = validated_actions

        return reflection

    def _rollback_to_previous(self, state: CollectionState):
        """Rollback to previous iteration snapshot"""
        
        previous_iteration = state.current_iteration - 1
        
        if previous_iteration not in state.snapshots:
            logger.error("No snapshot available for rollback")
            return
        
        snapshot = state.snapshots[previous_iteration]
        
        state.collected_files = snapshot['collected_files']
        state.file_metadata = snapshot['file_metadata']
        state.current_quality = snapshot['quality']
        
        logger.info(f"✓ Rolled back to iteration {previous_iteration}")
        logger.info(f"   Quality restored: {state.current_quality.overall:.2f}/10")
        logger.info(f"   Files restored: {len(state.collected_files)}")

    @staticmethod
    def _get_key_definitions_summary(content: str) -> List[str]:
        """Lightweight extraction of key definitions (class/function names)"""
        defs = []
        defs.extend(re.findall(r'(?:class|interface|type|enum|struct)\s+([A-Z][a-zA-Z0-9_]*)', content))
        defs.extend(re.findall(r'(?:export\s+)?(?:function|const|fn|def)\s+([a-z_][a-zA-Z0-9_]*)', content)[:10])
        return list(dict.fromkeys(defs))[:10]  # Deduplicate, limit to 10

    def _add_file_section(self, sections: List[str], path: str, content: str, meta: FileMetadata):
        """
        Add a file section to context.
        For large files, extract only relevant code blocks using CodeContextExtractor.
        """
        ext = path.split('.')[-1] if '.' in path else ''
        size_kb = meta.size / 1024
        
        sections.append(f"#### `{path}`")
        sections.append(f"*{meta.reason}* ({size_kb:.1f}KB)\n")
        
        # If file is large, extract key definitions only
        if len(content) > 10000:  # 10KB threshold
            key_defs = self._get_key_definitions_summary(content)
            if key_defs:
                sections.append(f"**Key Definitions:** {', '.join(key_defs[:10])}\n")
                sections.append(f"```{ext}")
                sections.append(content[:5000])  # First 5KB only
                sections.append("\n... (file truncated, showing key definitions above) ...")
                sections.append("```\n")
            else:
                # Fallback: show full content
                sections.append(f"```{ext}")
                sections.append(content)
                sections.append("```\n")
        else:
            # Small file: show full content
            sections.append(f"```{ext}")
            sections.append(content)
            sections.append("```\n")

    def _finalize_context(self, state: CollectionState, changed_files_info: List[Dict]) -> str:
        """Generate final formatted PR context"""
        
        logger.info("✨ Finalizing context...")
        
        sections = []
        
        # Header
        sections.append("# PR Review Context\n")
        sections.append(f"**Repository:** {state.repo_full_name}")
        sections.append(f"**SHA:** {state.sha[:8]}")
        sections.append(f"**Quality Score:** {state.current_quality.overall:.1f}/10")
        sections.append(f"**Files Analyzed:** {len(state.collected_files)}\n")
        
        # Changed Files Summary
        sections.append("## Changed Files Summary\n")
        sections.append("The following files were modified in this PR:\n")
        
        for file_info in changed_files_info:
            path = file_info['path']
            change_type = file_info.get('change_type', 'modified')
            additions = file_info.get('additions', 0)
            deletions = file_info.get('deletions', 0)
            
            sections.append(f"### `{path}` ({change_type})")
            sections.append(f"- **Changes:** +{additions} lines, -{deletions} lines\n")
        
        # Context Files by Priority
        sections.append("## Related Context Files\n")
        
        # Group by priority
        by_priority = {
            Priority.CRITICAL: [],
            Priority.HIGH: [],
            Priority.MEDIUM: [],
            Priority.LOW: []
        }
        
        for path, meta in state.file_metadata.items():
            by_priority[meta.priority].append((path, meta))
        
        # Critical files
        if by_priority[Priority.CRITICAL]:
            sections.append("### Critical Dependencies\n")
            for path, meta in by_priority[Priority.CRITICAL]:
                self._add_file_section(sections, path, state.collected_files[path], meta)
        
        # High priority files
        if by_priority[Priority.HIGH]:
            sections.append("### High Priority Context\n")
            for path, meta in by_priority[Priority.HIGH]:
                self._add_file_section(sections, path, state.collected_files[path], meta)
        
        # Medium priority files
        if by_priority[Priority.MEDIUM]:
            sections.append("### Supporting Context\n")
            for path, meta in by_priority[Priority.MEDIUM]:
                self._add_file_section(sections, path, state.collected_files[path], meta)
        
        # Repository Structure (relevant paths)
        sections.append("## Repository Structure\n")
        sections.append("```")
        sections.append(self._extract_relevant_structure(state.repo_structure, changed_files_info))
        sections.append("```\n")
        
        # Quality Report
        sections.append("## Context Quality Report\n")
        sections.append(f"**Overall Score:** {state.current_quality.overall:.1f}/10\n")
        sections.append("**Metrics:**")
        sections.append(f"- Completeness: {state.current_quality.completeness:.1f}/10")
        sections.append(f"- Relevance: {state.current_quality.relevance:.1f}/10")
        sections.append(f"- Sufficiency: {state.current_quality.sufficiency:.1f}/10")
        sections.append(f"- Efficiency: {state.current_quality.efficiency:.1f}/10")
        sections.append(f"- Confidence: {state.current_quality.confidence:.0%}\n")
        
        sections.append("**Collection Statistics:**")
        sections.append(f"- Files collected: {len(state.collected_files)}")
        sections.append(f"- Files attempted: {len(state.attempted_files)}")
        sections.append(f"- Files failed: {len(state.failed_files)}")
        sections.append(f"- Tokens used: {state.total_tokens:,}")
        sections.append(f"- Time taken: {state.elapsed_time():.1f}s")
        sections.append(f"- Iterations: {state.current_iteration}\n")
        
        # Key Insights
        if state.key_insights:
            sections.append("**Key Insights:**")
            for insight in state.key_insights:
                sections.append(f"- {insight}")
            sections.append("")
        
        # Known Limitations
        if state.failed_files:
            sections.append("**Known Limitations:**")
            for path, reason in list(state.failed_files.items())[:5]:
                sections.append(f"- {path}: {reason}")
            sections.append("")
        
        final_context = '\n'.join(sections)
        
        # Final size check
        if len(final_context) > self.MAX_CONTEXT_CHARS:
            logger.warning(f"Context too large ({len(final_context):,} chars), truncating...")
            final_context = final_context[:self.MAX_CONTEXT_CHARS] + "\n\n[... context truncated ...]"
        
        logger.info(f"✓ Final context: {len(final_context):,} characters")
        
        return final_context

    def _extract_relevant_structure(self, full_structure: str, changed_files_info: List[Dict]) -> str:
        """Extract relevant parts of repo structure"""
        
        changed_dirs = set()
        for file_info in changed_files_info:
            path = file_info['path']
            dir_path = '/'.join(path.split('/')[:-1])
            if dir_path:
                changed_dirs.add(dir_path)
        
        relevant_lines = []
        for line in full_structure.split('\n')[:100]:
            if any(dir_path in line for dir_path in changed_dirs):
                relevant_lines.append(line)
            elif any(line.strip().startswith(dir_path.split('/')[0]) for dir_path in changed_dirs):
                relevant_lines.append(line)
        
        if not relevant_lines:
            relevant_lines = full_structure.split('\n')[:50]
        
        return '\n'.join(relevant_lines)

    def _generate_partial_context(self, state: CollectionState, changed_files_info: List[Dict], error: str) -> Dict[str, Any]:
        """Generate partial context when collection fails"""
        
        logger.warning("Generating partial context due to error")
        
        context = f"""# PR Review Context (Partial)

**Repository:** {state.repo_full_name}
**SHA:** {state.sha[:8]}
**Status:** Incomplete due to error

## Error

{error}

## Changed Files

"""
        
        for file_info in changed_files_info:
            context += f"- {file_info['path']} ({file_info.get('change_type', 'modified')})\n"
        
        if state.collected_files:
            context += f"\n## Partially Collected Files ({len(state.collected_files)})\n\n"
            for path in list(state.collected_files.keys())[:5]:
                context += f"- {path}\n"
        
        context += "\n*Note: Context collection was interrupted. Please review based on available information.*\n"
        
        return {
            'pr_context': context,
            'files_included': list(state.collected_files.keys()),
            'metrics': {
                'quality_score': 0.0,
                'status': 'partial',
                'error': error,
                'iterations': state.current_iteration,
                'total_tokens': state.total_tokens,
                'files_count': len(state.collected_files)
            }
        }

    def _extract_changed_files_info(self, pr_content: Dict[str, Any]) -> List[Dict]:
        """
        Extract changed files info from pr_content.
        Includes diffs and stats which are essential for the Evaluator.
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
                        path_str = file_path if isinstance(file_path, str) else file_path.get('filename', '')
                        if path_str and path_str not in seen_files:
                            seen_files.add(path_str)
                            changed_files.append({
                                'path': path_str,
                                'change_type': 'modified',
                                'additions': 0,
                                'deletions': 0,
                                'diff': ''
                            })
        return changed_files

    def _get_accessible_files_for_context(self, state: CollectionState, target_path: Optional[str]) -> List[str]:
        """Get list of accessible files for a specific path"""
        if not target_path:
            return list(state.accessible_files)
        
        # Filter files under target_path
        prefix = target_path.rstrip('/') + '/'
        filtered = [f for f in state.accessible_files if f.startswith(prefix)]
        
        return filtered

    def _parse_json_response(self, content: str) -> Dict:
        """Parse JSON from LLM response"""
        
        # Try markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try direct JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try entire content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        logger.error(f"Failed to parse JSON. Content preview: {content[:200]}")
        return {}
