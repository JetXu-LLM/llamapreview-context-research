import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger("LlamaPReview")

# ============================================================================
# Enums
# ============================================================================

class Priority(str, Enum):
    """File collection priority levels"""
    CRITICAL = "critical"  # P0: Must collect
    HIGH = "high"          # P1: Should collect
    MEDIUM = "medium"      # P2: Nice to have
    LOW = "low"            # P3: Optional


class Decision(str, Enum):
    """ReAct loop decisions"""
    CONTINUE = "continue"
    STOP = "stop"
    ROLLBACK = "rollback"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FileMetadata:
    """Metadata for collected files"""
    path: str
    size: int
    priority: Priority
    reason: str
    iteration_collected: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IterationRecord:
    """Record of a single iteration"""
    iteration: int
    action: str
    files_added: List[str]
    files_removed: List[str]
    quality_before: float
    quality_after: float
    reasoning: str
    tokens_used: int
    duration: float
    timestamp: datetime
    missing_critical_files: List[str] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    completeness: float = 0.0
    relevance: float = 0.0
    sufficiency: float = 0.0
    efficiency: float = 0.0
    overall: float = 0.0
    confidence: float = 0.0


@dataclass
class CollectionState:
    """
    Enhanced state management for ReAct loop.
    Holds the entire context of the current exploration session.
    """
    # Core state
    pr_details: str
    pr_content: Dict[str, Any]
    repo_full_name: str
    github_token: str
    sha: str
    
    # Repo structure (Lazy loaded or updated dynamically)
    repo_structure: str = ""
    accessible_files: Set[str] = field(default_factory=set)
    
    # Iteration state
    current_iteration: int = 0
    max_iterations: int = 3
    iteration_history: List[IterationRecord] = field(default_factory=list)
    should_continue: bool = True
    
    # Collection state
    collected_files: Dict[str, str] = field(default_factory=dict)
    file_metadata: Dict[str, FileMetadata] = field(default_factory=dict)
    attempted_files: Set[str] = field(default_factory=set)
    failed_files: Dict[str, str] = field(default_factory=dict)
    
    # Quality state
    current_quality: QualityMetrics = field(default_factory=QualityMetrics)
    quality_history: List[QualityMetrics] = field(default_factory=list)
    quality_threshold: float = 8.0
    
    # Budget state
    total_tokens: int = 0
    token_budget: int = 200000
    time_started: datetime = field(default_factory=datetime.now)
    time_budget: int = 2400
    
    # Reasoning state
    reasoner_thinking: str = ""
    key_insights: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)

    # Focus & Exploration state
    focus_path: Optional[str] = None  # The directory path the agent is currently focused on.
    last_reasoning: Optional[str] = None  # To carry over reasoning from Focus to Plan phase.
    should_refocus: bool = False  # If True, next iteration will re-execute FOCUS phase
    refocus_hint: Optional[str] = None  # Directional hint from REFLECT (e.g., "backend", "types")
    refocus_reason: Optional[str] = None  # Why REFLECT suggests refocusing
    previous_focus_path: Optional[str] = None  # Track the last focus path before refocusing
    
    # Guardrails
    # Track files that LLM requested but do not exist (to prevent repeated requests)
    non_existent_files: Set[str] = field(default_factory=set)

    # Snapshots for rollback
    snapshots: Dict[int, Dict] = field(default_factory=dict)
    
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return (datetime.now() - self.time_started).total_seconds()
    
    def remaining_tokens(self) -> int:
        """Get remaining token budget"""
        return self.token_budget - self.total_tokens
    
    def remaining_time(self) -> float:
        """Get remaining time budget"""
        return self.time_budget - self.elapsed_time()
    
    def budget_exhausted(self) -> bool:
        """Check if any budget is exhausted"""
        return self.remaining_tokens() < 5000 or self.remaining_time() < 60
    
    def save_snapshot(self):
        """Save current state snapshot for potential rollback"""
        self.snapshots[self.current_iteration] = {
            'collected_files': self.collected_files.copy(),
            'file_metadata': self.file_metadata.copy(),
            'quality': QualityMetrics(
                completeness=self.current_quality.completeness,
                relevance=self.current_quality.relevance,
                sufficiency=self.current_quality.sufficiency,
                efficiency=self.current_quality.efficiency,
                overall=self.current_quality.overall,
                confidence=self.current_quality.confidence
            ),
            'timestamp': datetime.now()
        }
        logger.debug(f"Snapshot saved for iteration {self.current_iteration}")
