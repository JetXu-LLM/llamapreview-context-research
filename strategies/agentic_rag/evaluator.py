import re
import logging
from typing import List, Dict, Tuple
from .state import CollectionState, QualityMetrics

logger = logging.getLogger("LlamaPReview")

class QualityEvaluator:
    """
    Automatic quality metrics calculation for the Agentic RAG strategy.
    Stateless logic that evaluates the current CollectionState against the PR changes.
    """
    
    @staticmethod
    def evaluate(state: CollectionState, changed_files_info: List[Dict]) -> QualityMetrics:
        """
        Calculate quality metrics automatically based on current state.
        
        Args:
            state: The current collection state.
            changed_files_info: List of dicts containing info about changed files (diffs, paths).
            
        Returns:
            QualityMetrics object with calculated scores.
        """
        metrics = QualityMetrics()
        
        # 1. Completeness (30% weight)
        metrics.completeness = QualityEvaluator._calculate_completeness(
            state, changed_files_info
        )
        
        # 2. Relevance (35% weight)
        metrics.relevance = QualityEvaluator._calculate_relevance(
            state, changed_files_info
        )
        
        # 3. Sufficiency (20% weight)
        metrics.sufficiency = QualityEvaluator._calculate_sufficiency(state, changed_files_info)
        
        # 4. Efficiency (15% weight)
        metrics.efficiency = QualityEvaluator._calculate_efficiency(state)
        
        # Overall score
        metrics.overall = (
            metrics.completeness * 0.30 +
            metrics.relevance * 0.35 +
            metrics.sufficiency * 0.20 +
            metrics.efficiency * 0.15
        )
        
        # Confidence based on data quality
        metrics.confidence = QualityEvaluator._estimate_confidence(state)
        
        logger.info(f"Quality metrics calculated: {metrics.overall:.2f}/10")
        logger.debug(f"  Completeness: {metrics.completeness:.2f}")
        logger.debug(f"  Relevance: {metrics.relevance:.2f}")
        logger.debug(f"  Sufficiency: {metrics.sufficiency:.2f}")
        logger.debug(f"  Efficiency: {metrics.efficiency:.2f}")
        logger.debug(f"  Confidence: {metrics.confidence:.2f}")
        
        return metrics

    @staticmethod
    def _calculate_completeness(state: CollectionState, changed_files_info: List[Dict]) -> float:
        """
        Calculate completeness by checking if internal dependencies (imports) are covered.
        Language-agnostic approach supporting: JS/TS, Python, Java, Go, Rust, C/C++, C#, Ruby, PHP, Kotlin, Swift.
        
        Strategy:
        1. Extract import paths from changed files' diffs (multi-language)
        2. Check if these imports are covered by collected files
        3. Give partial credit for partial coverage
        """

        if not state.collected_files:
            return 0.0
        
        # Extract internal imports from changed files
        required_imports = set()
        
        for file_info in changed_files_info:
            diff = file_info.get('diff', '')
            file_path = file_info.get('path', '')
            
            # Detect language from file extension
            ext = file_path.split('.')[-1].lower() if '.' in file_path else ''
            
            # ============================================================
            # JavaScript/TypeScript: import/from statements
            # ============================================================
            if ext in ['js', 'jsx', 'ts', 'tsx', 'mjs', 'cjs']:
                # Relative imports: ./utils, ../shared
                relative_imports = re.findall(r'from\s+["\'](\.[^"\']+)["\']', diff)
                relative_imports.extend(re.findall(r'import\s+["\'](\.[^"\']+)["\']', diff))
                
                # Absolute imports: app-react-v2/..., src/...
                absolute_imports = re.findall(r'from\s+["\']([^.][^"\']*(?:app-|src/)[^"\']+)["\']', diff)
                absolute_imports.extend(re.findall(r'import\s+["\']([^.][^"\']*(?:app-|src/)[^"\']+)["\']', diff))
                
                # Resolve relative imports
                if file_path and relative_imports:
                    base_dir = '/'.join(file_path.split('/')[:-1])
                    for rel_imp in relative_imports:
                        normalized = rel_imp.replace('./', '').replace('../', '').lstrip('/')
                        for ext_suffix in ['', '.ts', '.tsx', '.js', '.jsx', '.mjs']:
                            required_imports.add(f"{base_dir}/{normalized}{ext_suffix}".lstrip('/'))
                
                # Add absolute imports
                for abs_imp in absolute_imports:
                    for ext_suffix in ['', '.ts', '.tsx', '.js', '.jsx']:
                        required_imports.add(f"{abs_imp}{ext_suffix}")
            
            # ============================================================
            # Python: import/from statements
            # ============================================================
            elif ext == 'py':
                # from .module import X, from ..package import Y
                relative_imports = re.findall(r'from\s+(\.+[\w.]*)\s+import', diff)
                
                # from mypackage.module import X
                absolute_imports = re.findall(r'from\s+([\w.]+)\s+import', diff)
                absolute_imports.extend(re.findall(r'import\s+([\w.]+)', diff))
                
                # Resolve relative imports
                if file_path and relative_imports:
                    base_dir = '/'.join(file_path.split('/')[:-1])
                    for rel_imp in relative_imports:
                        # Convert dots to path: .module -> module, ..pkg -> ../pkg
                        depth = len(rel_imp) - len(rel_imp.lstrip('.'))
                        module_path = rel_imp.lstrip('.').replace('.', '/')
                        
                        # Go up 'depth-1' directories
                        parts = base_dir.split('/')
                        if depth > 1:
                            parts = parts[:-(depth-1)] if len(parts) >= depth-1 else []
                        
                        resolved = '/'.join(parts + [module_path]) if parts else module_path
                        for suffix in ['', '.py', '/__init__.py']:
                            required_imports.add(f"{resolved}{suffix}".lstrip('/'))
                
                # Add absolute imports (filter out stdlib)
                stdlib_prefixes = ('os', 'sys', 'json', 'time', 're', 'math', 'datetime', 'collections', 'typing')
                for abs_imp in absolute_imports:
                    if not abs_imp.startswith(stdlib_prefixes):
                        path = abs_imp.replace('.', '/')
                        for suffix in ['', '.py', '/__init__.py']:
                            required_imports.add(f"{path}{suffix}")
            
            # ============================================================
            # Java: import statements
            # ============================================================
            elif ext == 'java':
                # import com.example.MyClass; or import com.example.*;
                java_imports = re.findall(r'import\s+([\w.]+)(?:\.\*)?;', diff)
                
                for java_imp in java_imports:
                    # Filter out java.* and javax.* (standard library)
                    if not java_imp.startswith(('java.', 'javax.', 'android.', 'org.junit')):
                        # Convert com.example.MyClass -> com/example/MyClass.java
                        path = java_imp.replace('.', '/') + '.java'
                        required_imports.add(path)
            
            # ============================================================
            # Go: import statements
            # ============================================================
            elif ext == 'go':
                # import "github.com/user/repo/pkg" or import ( ... )
                go_imports = re.findall(r'import\s+["\']([^"\']+)["\']', diff)
                go_imports.extend(re.findall(r'^\s*["\']([^"\']+)["\']', diff, re.MULTILINE))
                
                for go_imp in go_imports:
                    # Only keep internal imports (not stdlib or external)
                    if '/' in go_imp and not go_imp.startswith('github.com/') and not go_imp.startswith('golang.org/'):
                        # Assume internal: myproject/pkg -> pkg/*.go
                        parts = go_imp.split('/')
                        if len(parts) > 1:
                            internal_path = '/'.join(parts[-2:])  # Last 2 parts
                            required_imports.add(f"{internal_path}.go")
                            required_imports.add(f"{internal_path}/*.go")
            
            # ============================================================
            # Rust: use statements
            # ============================================================
            elif ext == 'rs':
                # use crate::module::MyStruct; or use super::sibling;
                rust_imports = re.findall(r'use\s+((?:crate|super|self)(?:::\w+)+)', diff)
                
                for rust_imp in rust_imports:
                    # crate::module::submodule -> module/submodule.rs or module/submodule/mod.rs
                    if rust_imp.startswith('crate::'):
                        path = rust_imp.replace('crate::', '').replace('::', '/')
                        required_imports.add(f"{path}.rs")
                        required_imports.add(f"{path}/mod.rs")
                    elif rust_imp.startswith('super::'):
                        # Relative import, resolve based on file_path
                        if file_path:
                            base_dir = '/'.join(file_path.split('/')[:-1])
                            parent_dir = '/'.join(base_dir.split('/')[:-1])
                            module_path = rust_imp.replace('super::', '').replace('::', '/')
                            required_imports.add(f"{parent_dir}/{module_path}.rs")
            
            # ============================================================
            # C/C++: #include statements
            # ============================================================
            elif ext in ['c', 'cpp', 'cc', 'cxx', 'h', 'hpp']:
                # #include "myheader.h" (local) or #include <mylib.h> (system, ignore)
                cpp_includes = re.findall(r'#include\s+"([^"]+)"', diff)
                
                for include in cpp_includes:
                    # Resolve relative to file's directory
                    if file_path:
                        base_dir = '/'.join(file_path.split('/')[:-1])
                        required_imports.add(f"{base_dir}/{include}".lstrip('/'))
                    else:
                        required_imports.add(include)
            
            # ============================================================
            # C#: using statements
            # ============================================================
            elif ext == 'cs':
                # using MyNamespace.MyClass; or using MyNamespace;
                csharp_usings = re.findall(r'using\s+([\w.]+);', diff)
                
                for using in csharp_usings:
                    # Filter out System.* (standard library)
                    if not using.startswith(('System', 'Microsoft')):
                        # Convert MyNamespace.MyClass -> MyNamespace/MyClass.cs
                        path = using.replace('.', '/') + '.cs'
                        required_imports.add(path)
            
            # ============================================================
            # Ruby: require/require_relative
            # ============================================================
            elif ext == 'rb':
                # require 'my_module' or require_relative '../other'
                ruby_requires = re.findall(r'require\s+["\']([^"\']+)["\']', diff)
                ruby_requires.extend(re.findall(r'require_relative\s+["\']([^"\']+)["\']', diff))
                
                for req in ruby_requires:
                    if file_path and req.startswith('.'):
                        # Relative require
                        base_dir = '/'.join(file_path.split('/')[:-1])
                        normalized = req.replace('./', '').replace('../', '')
                        required_imports.add(f"{base_dir}/{normalized}.rb".lstrip('/'))
                    else:
                        # Absolute require
                        required_imports.add(f"{req}.rb")
            
            # ============================================================
            # PHP: use/require/include
            # ============================================================
            elif ext == 'php':
                # use MyNamespace\MyClass;
                php_uses = re.findall(r'use\s+([\w\\]+);', diff)
                # require 'file.php' or include 'file.php'
                php_requires = re.findall(r'(?:require|include)(?:_once)?\s+["\']([^"\']+)["\']', diff)
                
                for use in php_uses:
                    # Convert MyNamespace\MyClass -> MyNamespace/MyClass.php
                    path = use.replace('\\', '/') + '.php'
                    required_imports.add(path)
                
                for req in php_requires:
                    if file_path and req.startswith('.'):
                        base_dir = '/'.join(file_path.split('/')[:-1])
                        required_imports.add(f"{base_dir}/{req}".lstrip('/'))
                    else:
                        required_imports.add(req)
            
            # ============================================================
            # Kotlin: import statements
            # ============================================================
            elif ext == 'kt':
                # import com.example.MyClass
                kotlin_imports = re.findall(r'import\s+([\w.]+)', diff)
                
                for kt_imp in kotlin_imports:
                    if not kt_imp.startswith(('java.', 'javax.', 'kotlin.', 'android.')):
                        path = kt_imp.replace('.', '/') + '.kt'
                        required_imports.add(path)
            
            # ============================================================
            # Swift: import statements
            # ============================================================
            elif ext == 'swift':
                # import MyModule
                swift_imports = re.findall(r'import\s+(\w+)', diff)
                
                for swift_imp in swift_imports:
                    # Filter out system frameworks
                    if swift_imp not in ('Foundation', 'UIKit', 'SwiftUI', 'Combine'):
                        required_imports.add(f"{swift_imp}.swift")
                        required_imports.add(f"{swift_imp}/*.swift")
        
        # ============================================================
        # Fallback: No imports found
        # ============================================================
        if not required_imports:
            # Use heuristic based on file count
            num_changed = len(changed_files_info)
            num_collected = len(state.collected_files)
            
            if num_collected >= num_changed * 2:
                return 7.0  # Decent coverage
            elif num_collected >= num_changed:
                return 5.0  # Minimal coverage
            else:
                return 3.0  # Insufficient
        
        # ============================================================
        # Check coverage of required imports
        # ============================================================
        covered_count = 0
        for imp in required_imports:
            # Normalize import path for matching (remove extensions)
            normalized_imp = re.sub(r'\.(ts|tsx|js|jsx|py|java|go|rs|cpp|c|h|hpp|cs|rb|php|kt|swift)$', '', imp)
            
            # Check if any collected file matches this import
            for collected_path in state.collected_files.keys():
                normalized_collected = re.sub(r'\.(ts|tsx|js|jsx|py|java|go|rs|cpp|c|h|hpp|cs|rb|php|kt|swift)$', '', collected_path)
                
                # Match if either path contains the other (handles partial matches)
                if normalized_imp in normalized_collected or normalized_collected in normalized_imp:
                    covered_count += 1
                    break
        
        coverage_ratio = covered_count / len(required_imports) if required_imports else 0.0
        
        # Scoring: 5.0 base + 5.0 for full coverage
        score = 5.0 + (coverage_ratio * 5.0)
        
        return min(10.0, score)
    
    @staticmethod
    def _calculate_relevance(state: CollectionState, changed_files_info: List[Dict]) -> float:
        """Calculate relevance score (0-10)"""
        if not state.collected_files:
            return 0.0
        
        # Check for generic/low-value files
        generic_patterns = ['button', 'icon', 'layout', 'theme', 'style']
        generic_count = sum(
            1 for path in state.collected_files.keys()
            if any(pattern in path.lower() for pattern in generic_patterns)
        )
        
        generic_ratio = generic_count / len(state.collected_files)
        
        # Penalize generic files
        score = 10.0 - (generic_ratio * 5.0)
        
        return min(10.0, max(0.0, score))
    
    @staticmethod
    def _calculate_sufficiency(state: CollectionState, changed_files_info: List[Dict]) -> float:
        """
        Calculate sufficiency with gentler penalties for collecting more files.
        """
        file_count = len(state.collected_files)
        
        min_opt, max_opt = QualityEvaluator._get_optimal_file_count_range(changed_files_info)
        
        if file_count == 0:
            return 0.0
        elif file_count < min_opt:
            return (file_count / min_opt) * 7.5
        elif min_opt <= file_count <= max_opt:
            return 10.0
        else:
            # Much gentler penalty
            overage = file_count - max_opt
            penalty = overage * 0.2  # Reduced from 0.5
            return max(6.0, 10.0 - penalty)  # Raised floor from 4.0
    
    @staticmethod
    def _calculate_efficiency(state: CollectionState) -> float:
        """Calculate efficiency score (0-10)"""
        if not state.collected_files:
            return 10.0
        
        # Check for oversized files
        large_files = sum(
            1 for content in state.collected_files.values()
            if len(content) > 50000
        )
        
        large_ratio = large_files / len(state.collected_files)
        
        # Penalize large files
        score = 10.0 - (large_ratio * 5.0)
        
        return min(10.0, max(0.0, score))
    
    @staticmethod
    def _estimate_confidence(state: CollectionState) -> float:
        """Estimate confidence in quality assessment"""
        confidence = 1.0
        
        # Reduce confidence if many files failed
        if state.failed_files:
            fail_ratio = len(state.failed_files) / max(len(state.attempted_files), 1)
            confidence -= fail_ratio * 0.3
        
        # Reduce confidence if very few files collected
        if len(state.collected_files) < 5:
            confidence -= 0.2
        
        return max(0.5, min(1.0, confidence))

    @staticmethod
    def _get_optimal_file_count_range(changed_files_info: List[Dict]) -> Tuple[int, int]:
        """
        Calculate optimal file count range with more lenient limits.
        Considers PR size and complexity.
        """
        num_changed_files = len(changed_files_info)
        total_line_changes = sum(
            f.get('additions', 0) + f.get('deletions', 0) for f in changed_files_info
        )

        # More generous base range
        min_optimal = 3
        max_optimal = 15  # Increased from 8

        # More generous scaling
        max_optimal += (num_changed_files // 2) * 3  # Changed from //3 * 2
        max_optimal += (total_line_changes // 300)   # Changed from //500

        min_optimal = min(min_optimal, max_optimal)
        max_optimal = min(max_optimal, 40)  # Increased cap from 25

        return min_optimal, max_optimal