"""
models.py

Data structures that flow between the orchestrator, resolution engine,
and API endpoints. Defined here as dataclasses so they are usable
anywhere in the project without circular imports.

Why dataclasses rather than Pydantic models?
  Dataclasses are plain Python with no external dependency. Pydantic
  models are used at the API boundary (request/response validation).
  Inside the orchestration pipeline, dataclasses are lighter and faster.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PhaseResult:
    """
    The output of one phase (Phase 1 or Phase 2) for one model.

    Stores both the answer and the confidence score so the resolution
    engine can compare isolation scores (Phase 1) against peer-influenced
    scores (Phase 2) and detect social consensus vs genuine conviction.
    """
    provider: str
    model: str
    answer: str
    confidence: int
    reasoning: str
    fault_found: bool
    fault_reason: Optional[str]
    phase: int                    # 1 = isolation check, 2 = peer comparison


@dataclass
class LoopIteration:
    """
    Everything that happened in one complete loop iteration.

    Stores Phase 1 and Phase 2 results separately so the full
    decision trail is preserved and visible in the UI audit log.
    """
    loop_number: int
    phase1_results: List[PhaseResult]
    phase2_results: List[PhaseResult]
    consensus_reached: bool
    consensus_answer: Optional[str]


@dataclass
class OrchestrationResult:
    """
    The complete output of the orchestration pipeline.

    This is what the resolution engine reads to make its final decision,
    and what the API endpoint serialises into the JSON response.
    The full iterations list is the audit trail shown in the UI.
    """
    question: str
    resources: str
    initial_answers: List[PhaseResult]
    iterations: List[LoopIteration]
    loops_completed: int
    consensus_reached: bool
    consensus_answer: Optional[str]
    max_loops: int
