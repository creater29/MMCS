"""
response_parser.py

Extracts structured fields from LLM response text.

Language models do not always respond in exactly the format you requested.
This parser is deliberately defensive — if a field is missing or formatted
unexpectedly, it falls back to a safe default rather than crashing.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _extract_field(text: str, label: str) -> Optional[str]:
    """
    Find a labelled field in the response and return its value.
    Searches for LABEL: and returns content until the next label or end of string.
    Returns None if the label does not appear in the text at all.
    """
    # Build the regex pattern using string concatenation to avoid any
    # f-string or raw-string conflicts with the backslashes in the pattern.
    pattern = label + ":" + r"\s*(.*?)(?=[A-Z_]+:|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _parse_confidence(raw: Optional[str]) -> int:
    """
    Extract an integer 0-100 from the raw confidence field.
    Falls back to 50 (neutral) if no number can be found.
    """
    if raw is None:
        logger.warning("Confidence field missing, defaulting to 50.")
        return 50
    numbers = re.findall(r"[0-9]+", raw)
    if numbers:
        return max(0, min(100, int(numbers[0])))
    logger.warning("Could not parse confidence value, defaulting to 50.")
    return 50


def _parse_fault(raw: Optional[str]) -> bool:
    """
    Return True if the model reported finding a fault in its answer.
    Accepts YES, TRUE, or 1 as affirmative. Defaults to False if missing.
    """
    if raw is None:
        return False
    return raw.strip().upper() in ("YES", "TRUE", "1")


def parse_response(raw_text: str, provider: str, model: str) -> dict:
    """
    Parse a raw LLM response string into a structured dictionary.

    Returns a plain dict rather than an LLMResponse dataclass so this
    module has no import dependency on base_client.py. Each client file
    assembles the final LLMResponse from this dict. This separation means
    you can test the parser with raw strings without needing to instantiate
    any client class at all, which makes unit testing much simpler.
    """
    answer       = _extract_field(raw_text, "ANSWER")
    confidence   = _extract_field(raw_text, "CONFIDENCE")
    reasoning    = _extract_field(raw_text, "REASONING")
    fault_raw    = _extract_field(raw_text, "FAULT_FOUND")
    fault_reason = _extract_field(raw_text, "FAULT_REASON")

    # If ANSWER is completely missing the model response was malformed.
    # Use the full raw text as a fallback so the content is at least
    # visible in the audit trail and not silently lost.
    if answer is None:
        logger.error("ANSWER field missing from " + provider + " response. Using full raw text.")
        answer = raw_text.strip()

    if reasoning is None:
        reasoning = "No reasoning provided."

    fault_found = _parse_fault(fault_raw)

    # If the fault reason is literally the word NONE treat it as absent.
    # Also clear fault_found if no reason was actually given, to avoid
    # false positives flowing into the resolution engine downstream.
    if fault_reason and fault_reason.upper() == "NONE":
        fault_reason = None
    if fault_found and not fault_reason:
        fault_found = False

    return {
        "answer":       answer,
        "confidence":   _parse_confidence(confidence),
        "reasoning":    reasoning,
        "fault_found":  fault_found,
        "fault_reason": fault_reason,
        "provider":     provider,
        "model":        model,
        "raw_response": raw_text,
    }
