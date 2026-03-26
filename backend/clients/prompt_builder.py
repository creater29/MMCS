"""
prompt_builder.py

Builds the structured prompts used in both phases of the self-check loop.
All three clients use these same prompt templates, which ensures the
confidence scoring and fault-detection logic is consistent across providers.

Why centralise prompt building?
  If ClaudeClient, GPTClient, and GeminiClient each wrote their own prompts,
  a small wording difference could make one model score confidence differently
  from another, making cross-model comparisons meaningless. One shared template
  means the models are responding to identical instructions.

Prompt format design:
  We use clearly labelled sections with uppercase markers (ANSWER:, CONFIDENCE:,
  etc.) because these are distinctive enough that a parser can find them reliably
  even if the model adds extra commentary around them. We avoid XML or JSON inside
  the prompt because models sometimes hallucinate closing tags or misplace brackets,
  which breaks parsers. Plain uppercase labels are more robust.
"""


def build_generation_prompt(question: str, resources: str = "") -> str:
    """
    Phase 0 prompt: ask the model to answer the question.
    
    This is the initial prompt sent to all three models in parallel
    before any self-check loop begins. The model is asked to provide
    its answer along with an initial confidence score.
    
    The resources parameter carries any external documents or context
    the user has provided. When present, the model is instructed to
    treat these as authoritative and prioritise them over its training.
    """
    resource_section = ""
    if resources and resources.strip():
        resource_section = f"""
The user has provided the following reference material.
Treat this as your primary source and prioritise it over your training data:

--- REFERENCE MATERIAL START ---
{resources.strip()}
--- REFERENCE MATERIAL END ---

"""

    return f"""You are one of three AI models answering a question independently.
Answer accurately and concisely.
{resource_section}
QUESTION: {question}

Respond using EXACTLY this format, with no text before ANSWER: or after REASONING:

ANSWER: [your answer here]
CONFIDENCE: [a number from 0 to 100 representing how confident you are]
REASONING: [2-3 sentences explaining why you hold this confidence level]"""


def build_phase1_prompt(question: str, your_answer: str) -> str:
    """
    Phase 1 prompt: isolation self-check.
    
    The model sees only its own answer — not what the other models said.
    It is asked to critically re-examine its answer and report whether
    it can find any fault in its reasoning.
    
    This isolation is the most important property of Phase 1. The score
    produced here is the only truly independent signal in the system.
    A model that cannot find a fault in its own answer under solo scrutiny
    is making a genuine epistemic claim, not just agreeing with peers.
    """
    return f"""You previously answered a question. Review your answer critically and honestly.
Do NOT consider what other models may have said — evaluate your answer on its own merits only.

ORIGINAL QUESTION: {question}
YOUR ANSWER: {your_answer}

Re-examine your answer carefully. Look for factual errors, logical gaps, or missing context.

Respond using EXACTLY this format:

ANSWER: [your answer — keep it if correct, correct it if wrong]
CONFIDENCE: [0-100, your honest confidence after self-review]
REASONING: [2-3 sentences explaining your confidence]
FAULT_FOUND: [YES or NO — did you find an error in your original answer?]
FAULT_REASON: [if YES, briefly describe the fault. If NO, write NONE]"""


def build_phase2_prompt(
    question: str,
    your_answer: str,
    your_phase1_confidence: int,
    peer_answers: list[dict],
) -> str:
    """
    Phase 2 prompt: peer comparison.
    
    Now the model sees all three answers and the other models' Phase 1
    confidence scores. It is asked whether this new information changes
    its own confidence.
    
    We deliberately show the Phase 1 scores (not Phase 2, which don't
    exist yet) so the model can judge the strength of each peer's
    independent conviction. A peer with 90% isolation confidence is
    more persuasive than one with 40%.
    
    The prompt explicitly asks the model to distinguish between updating
    because of new reasoning versus updating because of peer pressure.
    This self-awareness instruction improves the quality of the
    reasoning traces and helps the resolution engine interpret score changes.
    """
    peer_section = ""
    for i, peer in enumerate(peer_answers, 1):
        peer_section += f"""
Peer {i} ({peer["provider"]}):
  Answer: {peer["answer"]}
  Phase 1 confidence: {peer["confidence"]}/100
  Reasoning: {peer["reasoning"]}
"""

    return f"""You are reviewing a question alongside two peer AI models.
You have already evaluated your own answer independently (Phase 1).
Now consider what your peers said.

ORIGINAL QUESTION: {question}
YOUR ANSWER: {your_answer}
YOUR PHASE 1 CONFIDENCE: {your_phase1_confidence}/100

PEER RESPONSES:
{peer_section.strip()}

Consider whether the peer responses reveal any error or gap in your answer.
Important: distinguish between updating because of genuine new reasoning
versus simply agreeing with the majority. If you are updating, explain why
the peer's argument is logically compelling, not just that they disagreed.

Respond using EXACTLY this format:

ANSWER: [your final answer — update only if genuinely convinced]
CONFIDENCE: [0-100, your revised confidence after seeing peers]
REASONING: [2-3 sentences explaining any change or why you held firm]
FAULT_FOUND: [YES or NO]
FAULT_REASON: [describe fault if found, otherwise NONE]"""