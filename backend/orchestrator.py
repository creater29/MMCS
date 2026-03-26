"""
orchestrator.py

Coordinates parallel LLM dispatch and the two-phase self-check loop.

This module has one public function: run_orchestration().
Everything else is a private helper that it calls in sequence.

The orchestrator never decides which answer wins — that is the
resolution engine's responsibility. The orchestrator's job is to
run the process faithfully and record everything that happened.
"""

import asyncio
import logging
from typing import List, Optional

from clients.factory import get_client
from clients.base_client import LLMResponse
from clients.prompt_builder import (
    build_generation_prompt,
    build_phase1_prompt,
    build_phase2_prompt,
)
from models import PhaseResult, LoopIteration, OrchestrationResult
from session_store import SessionKeys

logger = logging.getLogger(__name__)


# ── Private helpers ───────────────────────────────────────────────────────

def _llm_response_to_phase_result(response: LLMResponse, phase: int) -> PhaseResult:
    """
    Convert an LLMResponse (from the client layer) into a PhaseResult
    (for the orchestration layer). This translation keeps the two layers
    decoupled — the orchestrator does not depend on LLMResponse internals,
    and the client layer does not need to know about PhaseResult.
    """
    return PhaseResult(
        provider=response.provider,
        model=response.model,
        answer=response.answer,
        confidence=response.confidence,
        reasoning=response.reasoning,
        fault_found=response.fault_found,
        fault_reason=response.fault_reason,
        phase=phase,
    )


async def _generate_initial_answers(
    clients: list,
    question: str,
    resources: str,
) -> List[PhaseResult]:
    """
    Fire all three models simultaneously and collect their initial answers.

    asyncio.gather() runs all three coroutines concurrently. Total wait
    time is approximately equal to the slowest single provider, not the
    sum of all three. For providers that take ~5 seconds each, this means
    ~5 seconds total instead of ~15 seconds sequential.

    If one provider fails, we log the error and continue with the
    remaining two rather than failing the entire request. Two models
    can still reach consensus, and the failed provider is recorded in
    the audit trail.
    """
    prompt = build_generation_prompt(question, resources)

    async def safe_generate(client, prompt):
        try:
            response = await client.generate(prompt)
            return response
        except Exception as e:
            logger.error(
                f"Provider {client.__class__.__name__} failed during generation. "
                f"Type: {type(e).__name__}. Detail: {e}"
            )
            return None

    raw_responses = await asyncio.gather(
        *[safe_generate(c, prompt) for c in clients]
    )

    results = []
    for response in raw_responses:
        if response is not None:
            results.append(_llm_response_to_phase_result(response, phase=0))

    return results


async def _run_phase1(
    clients: list,
    question: str,
    current_answers: List[PhaseResult],
) -> List[PhaseResult]:
    """
    Phase 1 — isolation self-check.

    Each model receives only its own current answer and is asked to
    re-examine it without knowledge of what the other models said.
    All three checks run in parallel.

    The isolation score produced here is the most trustworthy signal
    in the system. It represents genuine independent conviction rather
    than peer-influenced agreement. The resolution engine weights this
    score more heavily than the Phase 2 revised score.

    We match clients to answers by list position, which works because
    both lists are always ordered [claude, openai, gemini].
    """
    async def safe_phase1(client, answer_result):
        try:
            prompt = build_phase1_prompt(
                question=question,
                your_answer=answer_result.answer,
            )
            response = await client.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"Phase 1 failed for {client.__class__.__name__}: {e}")
            return None

    raw_responses = await asyncio.gather(
        *[safe_phase1(c, a) for c, a in zip(clients, current_answers)]
    )

    results = []
    for i, response in enumerate(raw_responses):
        if response is not None:
            results.append(_llm_response_to_phase_result(response, phase=1))
        else:
            # If Phase 1 failed for a model, carry forward its previous
            # answer with a reduced confidence score so the resolution
            # engine knows this data point is less reliable.
            fallback = current_answers[i]
            results.append(PhaseResult(
                provider=fallback.provider,
                model=fallback.model,
                answer=fallback.answer,
                confidence=max(0, fallback.confidence - 20),
                reasoning="Phase 1 check failed — using previous answer.",
                fault_found=False,
                fault_reason=None,
                phase=1,
            ))

    return results


async def _run_phase2(
    clients: list,
    question: str,
    phase1_results: List[PhaseResult],
) -> List[PhaseResult]:
    """
    Phase 2 — peer comparison.

    Each model now sees all three answers and Phase 1 confidence scores.
    It is asked whether this new information changes its own confidence,
    and if so why.

    We deliberately show Phase 1 scores (not Phase 2, which do not exist
    yet) so each model can judge the strength of each peer's independent
    conviction. A peer with 90 isolation confidence is more persuasive
    than one with 40.

    The delta between Phase 1 and Phase 2 scores is meaningful:
      - Small delta (held firm): genuine independent conviction.
      - Large upward delta after seeing agreement: social consensus.
      - Large downward delta after seeing disagreement: model updated
        based on peer reasoning — worth examining in the audit trail.
    """
    async def safe_phase2(client, own_result, peer_results):
        try:
            peers = [
                {
                    "provider": r.provider,
                    "answer": r.answer,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                }
                for r in peer_results
            ]
            prompt = build_phase2_prompt(
                question=question,
                your_answer=own_result.answer,
                your_phase1_confidence=own_result.confidence,
                peer_answers=peers,
            )
            response = await client.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"Phase 2 failed for {client.__class__.__name__}: {e}")
            return None

    tasks = []
    for i, client in enumerate(clients):
        own = phase1_results[i]
        peers = [r for j, r in enumerate(phase1_results) if j != i]
        tasks.append(safe_phase2(client, own, peers))

    raw_responses = await asyncio.gather(*tasks)

    results = []
    for i, response in enumerate(raw_responses):
        if response is not None:
            results.append(_llm_response_to_phase_result(response, phase=2))
        else:
            fallback = phase1_results[i]
            results.append(PhaseResult(
                provider=fallback.provider,
                model=fallback.model,
                answer=fallback.answer,
                confidence=max(0, fallback.confidence - 10),
                reasoning="Phase 2 check failed — using Phase 1 answer.",
                fault_found=False,
                fault_reason=None,
                phase=2,
            ))

    return results


def _check_consensus(phase2_results: List[PhaseResult]) -> tuple[bool, Optional[str]]:
    """
    Determine whether the models have reached consensus after a loop.

    Consensus requires two conditions to both be true:
      1. At least two models give the same answer (majority agreement).
      2. The agreeing models have a combined average confidence >= 70.

    The confidence threshold prevents low-quality consensus — three
    models agreeing with 30% confidence each is not meaningful agreement,
    it is shared uncertainty. We want confident agreement, not just
    identical guesses.

    Returns a tuple of (consensus_reached: bool, winning_answer: Optional[str]).
    """
    if not phase2_results:
        return False, None

    # Count how many models gave each answer.
    # We normalise answers to lowercase and strip whitespace to avoid
    # treating "Canberra" and "canberra." as different answers.
    answer_groups: dict = {}
    for result in phase2_results:
        normalised = result.answer.strip().lower().rstrip(".")
        if normalised not in answer_groups:
            answer_groups[normalised] = []
        answer_groups[normalised].append(result)

    # Find the answer that the most models agree on.
    best_answer_key = max(answer_groups, key=lambda k: len(answer_groups[k]))
    agreeing = answer_groups[best_answer_key]

    if len(agreeing) < 2:
        # No majority — all three models gave different answers.
        return False, None

    avg_confidence = sum(r.confidence for r in agreeing) / len(agreeing)
    if avg_confidence < 70:
        # Majority agrees but confidence is too low to trust.
        logger.info(
            f"Majority agreement on answer but avg confidence {avg_confidence:.0f} < 70. "
            f"Continuing loop."
        )
        return False, None

    # Use the original (non-normalised) answer from the highest-confidence agreeing model.
    best = max(agreeing, key=lambda r: r.confidence)
    return True, best.answer


# ── Public interface ──────────────────────────────────────────────────────

async def run_orchestration(
    session_keys: SessionKeys,
    question: str,
    resources: str = "",
    max_loops: int = 3,
) -> OrchestrationResult:
    """
    Run the full multi-model orchestration pipeline for a given question.

    This is the only function the API endpoint needs to call. It handles
    client instantiation, parallel dispatch, the self-check loop, and
    result packaging. The resolution engine then reads the returned
    OrchestrationResult to make its final decision.

    Args:
        session_keys : The user's API keys retrieved from the session store.
        question     : The question to answer.
        resources    : Optional user-supplied reference material.
        max_loops    : Maximum self-check iterations before applying default rule.

    Returns:
        OrchestrationResult containing the full audit trail.
    """
    logger.info(f"Starting orchestration for question: {question[:80]}...")

    # Instantiate one client per provider using the user's own keys.
    clients = [
        get_client("claude", session_keys.claude.key, session_keys.claude.model),
        get_client("openai", session_keys.openai.key, session_keys.openai.model),
        get_client("gemini", session_keys.gemini.key, session_keys.gemini.model),
    ]

    # Initial parallel generation — all three models answer simultaneously.
    logger.info("Running initial parallel generation...")
    initial_answers = await _generate_initial_answers(clients, question, resources)

    if not initial_answers:
        raise RuntimeError("All three providers failed during initial generation.")

    # Self-check loop.
    iterations: List[LoopIteration] = []
    current_answers = initial_answers
    consensus_reached = False
    consensus_answer = None

    for loop_num in range(1, max_loops + 1):
        logger.info(f"Loop {loop_num}/{max_loops} — running Phase 1 isolation check...")
        phase1_results = await _run_phase1(clients, question, current_answers)

        logger.info(f"Loop {loop_num}/{max_loops} — running Phase 2 peer comparison...")
        phase2_results = await _run_phase2(clients, question, phase1_results)

        consensus_reached, consensus_answer = _check_consensus(phase2_results)

        iteration = LoopIteration(
            loop_number=loop_num,
            phase1_results=phase1_results,
            phase2_results=phase2_results,
            consensus_reached=consensus_reached,
            consensus_answer=consensus_answer,
        )
        iterations.append(iteration)

        if consensus_reached:
            logger.info(f"Consensus reached on loop {loop_num}: {consensus_answer}")
            break

        # Update current_answers to Phase 2 results for the next iteration.
        # This means each loop builds on the previous one's peer comparison,
        # giving models the opportunity to genuinely update rather than repeat.
        current_answers = phase2_results
        logger.info(f"Loop {loop_num} complete. No consensus yet. Continuing...")

    if not consensus_reached:
        logger.info(f"Max loops ({max_loops}) reached without full consensus.")

    return OrchestrationResult(
        question=question,
        resources=resources,
        initial_answers=initial_answers,
        iterations=iterations,
        loops_completed=len(iterations),
        consensus_reached=consensus_reached,
        consensus_answer=consensus_answer,
        max_loops=max_loops,
    )
