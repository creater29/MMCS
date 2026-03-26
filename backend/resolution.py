"""
resolution.py

Applies the priority chain to an OrchestrationResult and produces
a final, human-readable response with a full explanation of how
the decision was made.

The resolution engine never re-runs any model calls. It only reads
the data the orchestrator already collected and makes a deterministic
decision based on the priority chain. This means the same
OrchestrationResult will always produce the same resolution —
the engine has no randomness or external dependencies.
"""

import logging
from dataclasses import dataclass
from typing import Optional, List

from models import OrchestrationResult, PhaseResult

logger = logging.getLogger(__name__)


@dataclass
class ResolutionResult:
    """
    The final output of the entire pipeline — what gets sent to the user.

    Every field here is designed to support the audit trail shown in the
    UI. The user should be able to read this and understand not just what
    the answer is, but exactly why the system chose it and what the
    alternative answers were.
    """
    final_answer: str
    resolution_rule: str        # Which rule in the priority chain fired
    resolution_explanation: str # Human-readable explanation of the decision
    winning_provider: str       # Which model's answer was selected
    winning_confidence: int     # That model's Phase 1 isolation confidence
    loops_completed: int
    consensus_reached: bool
    resource_conflict: bool     # True if answer conflicts with user resources
    resource_conflict_note: str # Explanation of the conflict if present
    all_answers: List[dict]     # Every model's final answer for the audit trail


def _get_final_answers(result: OrchestrationResult) -> List[PhaseResult]:
    """
    Extract the most recent answer from each model.

    If the loop ran at least once, we use the Phase 2 results from the
    last iteration — these represent each model's final considered position
    after seeing all peers. If somehow no iterations ran, we fall back to
    the initial answers.
    """
    if result.iterations:
        last_iteration = result.iterations[-1]
        return last_iteration.phase2_results
    return result.initial_answers


def _get_phase1_results(result: OrchestrationResult) -> List[PhaseResult]:
    """
    Extract Phase 1 isolation results from the first loop iteration.

    We specifically use the FIRST iteration's Phase 1 results, not the
    last. The reasoning: Phase 1 of the first loop is the most genuinely
    independent data point. By later iterations, models have already seen
    peer answers in previous Phase 2 rounds, so their "isolation" scores
    are not truly isolation scores anymore. The first loop's Phase 1 is
    the only truly uncontaminated reading.
    """
    if result.iterations:
        return result.iterations[0].phase1_results
    return []


def _find_highest_isolation_confidence(
    phase1_results: List[PhaseResult],
    final_answers: List[PhaseResult],
) -> Optional[PhaseResult]:
    """
    Find the model with the highest Phase 1 isolation confidence score
    and return its corresponding final answer.

    We look up the Phase 1 score by provider name, then return that
    provider's Phase 2 (final) answer rather than its Phase 1 answer.
    The distinction matters: a model might have changed its answer during
    the loop, and we want its most considered final position, scored by
    how independently confident it was before seeing peers.

    This is the core of the "isolation score" principle: use the Phase 1
    score to rank trustworthiness, but use the Phase 2 answer as the
    actual response content.
    """
    if not phase1_results:
        return None

    # Build a map from provider name to Phase 1 confidence score.
    phase1_by_provider = {r.provider: r for r in phase1_results}

    # Find the final answer whose provider had the highest Phase 1 score.
    best = None
    best_score = -1
    for final in final_answers:
        phase1 = phase1_by_provider.get(final.provider)
        if phase1 and phase1.confidence > best_score:
            best_score = phase1.confidence
            best = final

    return best


def _find_no_fault_model(final_answers: List[PhaseResult]) -> Optional[PhaseResult]:
    """
    Find a model that declared no fault in its own answer during self-check.

    This is the default rule — used when loops are exhausted without
    consensus and no Phase 1 data is available. A model that actively
    examined its own answer and declared it sound is making a meaningful
    epistemic claim. Prefer the one with the highest confidence among
    those that found no fault.

    If all models found faults (unusual but possible), we fall back to
    the highest final confidence score — at least we pick the model
    that is least uncertain.
    """
    no_fault_models = [r for r in final_answers if not r.fault_found]

    if no_fault_models:
        return max(no_fault_models, key=lambda r: r.confidence)

    # All models found faults — return the highest confidence regardless.
    logger.warning("All models found faults in their answers. Using highest confidence.")
    return max(final_answers, key=lambda r: r.confidence)


def _check_resource_conflict(answer: str, resources: str) -> tuple:
    """
    Check whether the winning answer appears to conflict with the
    user-supplied resources.

    This is a deliberately simple check — we look for key terms from the
    answer in the resources text. A more sophisticated implementation could
    use semantic similarity, but for v1 a keyword check catches the most
    obvious conflicts (e.g. resources say "Canberra" but model answered
    "Sydney") without requiring an additional model call.

    Returns a tuple of (conflict_detected: bool, note: str).
    """
    if not resources or not resources.strip():
        # No resources provided — no conflict possible.
        return False, ""

    # Take the first 10 words of the answer as key terms to search for.
    answer_terms = answer.lower().split()[:10]
    resources_lower = resources.lower()

    matches = sum(1 for term in answer_terms if term in resources_lower)

    # If fewer than 20% of the answer's key terms appear in the resources,
    # flag a potential conflict for the user to review.
    if len(answer_terms) > 0 and matches / len(answer_terms) < 0.2:
        note = (
            "The final answer may not align with your provided resources. "
            "The resources did not appear to contain key terms from the answer. "
            "Please review both and use your judgment."
        )
        return True, note

    return False, ""


def resolve(result: OrchestrationResult) -> ResolutionResult:
    """
    Apply the priority chain to the orchestration result and return
    a final answer with a full explanation.

    This is the only public function in this module. It is called once,
    after run_orchestration() returns, and its output is what gets
    serialised into the JSON response sent to the user.

    Priority chain (applied in order, stops at first match):
      Rule 1 — Consensus reached during loop.
      Rule 2 — Highest Phase 1 isolation confidence score.
      Rule 3 — Default rule: model that found no fault in itself.
      Rule 4 — Resource conflict check (applied on top of any rule above).
    """
    final_answers = _get_final_answers(result)
    phase1_results = _get_phase1_results(result)

    # Format all answers for the audit trail regardless of which rule fires.
    all_answers = [
        {
            "provider": r.provider,
            "model": r.model,
            "answer": r.answer,
            "confidence": r.confidence,
            "reasoning": r.reasoning,
            "fault_found": r.fault_found,
            "fault_reason": r.fault_reason,
        }
        for r in final_answers
    ]

    # ── Rule 1: Consensus reached ─────────────────────────────────────────
    if result.consensus_reached and result.consensus_answer:
        logger.info("Resolution Rule 1 fired: consensus reached during loop.")

        # Find the provider whose answer matches the consensus answer,
        # so we can report a winning provider and confidence score.
        winning = next(
            (r for r in final_answers
             if r.answer.strip().lower() == result.consensus_answer.strip().lower()),
            final_answers[0] if final_answers else None
        )
        provider = winning.provider if winning else "consensus"
        confidence = winning.confidence if winning else 0

        conflict, conflict_note = _check_resource_conflict(
            result.consensus_answer, result.resources
        )

        return ResolutionResult(
            final_answer=result.consensus_answer,
            resolution_rule="consensus",
            resolution_explanation=(
                f"All qualifying models reached agreement after "
                f"{result.loops_completed} loop(s). "
                f"The consensus answer was confirmed by majority vote "
                f"with sufficient confidence."
            ),
            winning_provider=provider,
            winning_confidence=confidence,
            loops_completed=result.loops_completed,
            consensus_reached=True,
            resource_conflict=conflict,
            resource_conflict_note=conflict_note,
            all_answers=all_answers,
        )

    # ── Rule 2: Highest Phase 1 isolation confidence ──────────────────────
    best_isolation = _find_highest_isolation_confidence(phase1_results, final_answers)

    if best_isolation:
        logger.info(
            f"Resolution Rule 2 fired: highest isolation confidence "
            f"({best_isolation.provider}, {best_isolation.confidence})."
        )

        # Find this provider's Phase 1 score for the explanation.
        phase1_score = next(
            (r.confidence for r in phase1_results
             if r.provider == best_isolation.provider),
            best_isolation.confidence
        )

        conflict, conflict_note = _check_resource_conflict(
            best_isolation.answer, result.resources
        )

        return ResolutionResult(
            final_answer=best_isolation.answer,
            resolution_rule="highest_isolation_confidence",
            resolution_explanation=(
                f"No full consensus was reached after {result.loops_completed} loop(s). "
                f"The answer from {best_isolation.provider} was selected because it had "
                f"the highest Phase 1 isolation confidence score ({phase1_score}/100) — "
                f"the score recorded before seeing any peer answers, making it the most "
                f"genuinely independent signal available."
            ),
            winning_provider=best_isolation.provider,
            winning_confidence=phase1_score,
            loops_completed=result.loops_completed,
            consensus_reached=False,
            resource_conflict=conflict,
            resource_conflict_note=conflict_note,
            all_answers=all_answers,
        )

    # ── Rule 3: Default rule — model that found no fault ──────────────────
    no_fault = _find_no_fault_model(final_answers)
    logger.info(
        f"Resolution Rule 3 fired: default rule "
        f"({no_fault.provider if no_fault else 'none'})."
    )

    if no_fault:
        conflict, conflict_note = _check_resource_conflict(
            no_fault.answer, result.resources
        )
        return ResolutionResult(
            final_answer=no_fault.answer,
            resolution_rule="default_no_fault",
            resolution_explanation=(
                f"No consensus and no Phase 1 isolation data available. "
                f"Applying default rule: the answer from {no_fault.provider} "
                f"was selected because it was the only model (or the most confident "
                f"model) that actively examined its own answer and declared it sound. "
                f"A model that cannot find a fault in its own reasoning under scrutiny "
                f"is making a genuine epistemic claim."
            ),
            winning_provider=no_fault.provider,
            winning_confidence=no_fault.confidence,
            loops_completed=result.loops_completed,
            consensus_reached=False,
            resource_conflict=conflict,
            resource_conflict_note=conflict_note,
            all_answers=all_answers,
        )

    # ── Absolute fallback — should never reach this point ─────────────────
    # If somehow all three rules failed (e.g. no answers at all), return
    # a safe error response rather than crashing the entire request.
    logger.error("All resolution rules failed. Returning fallback response.")
    return ResolutionResult(
        final_answer="Unable to determine a confident answer.",
        resolution_rule="fallback",
        resolution_explanation="All resolution rules failed. Please try again.",
        winning_provider="none",
        winning_confidence=0,
        loops_completed=result.loops_completed,
        consensus_reached=False,
        resource_conflict=False,
        resource_conflict_note="",
        all_answers=all_answers,
    )
