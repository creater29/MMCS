"""
base_client.py

Defines the shared interface that all three LLM client wrappers must follow.
This is the "contract" that makes the orchestrator provider-agnostic.

Why use an abstract base class?
  If ClaudeClient, GPTClient, and GeminiClient all inherit from LLMBaseClient,
  Python enforces that they each implement generate() and health_check().
  A missing method raises an error at class definition time, not at 3am
  when a real user query hits the missing code path.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """
    The single, unified response shape that every client must return.
    
    The orchestrator only ever works with this shape — it never touches
    provider-specific response objects. This is what makes the system
    swappable at the provider level.
    
    Fields:
        answer        : The model's actual response text.
        confidence    : Self-reported confidence score, 0-100.
                        Extracted from the structured prompt response.
        reasoning     : The model's explanation for its confidence score.
                        This is the "reasoning trace" used by the resolution engine.
        fault_found   : True if the model identified an error in its own answer
                        during the self-check phase.
        fault_reason  : If fault_found is True, what the fault was.
        provider      : Which provider produced this response (claude/openai/gemini).
        model         : Which specific model version was used.
        raw_response  : The full unprocessed text from the model, kept for debugging.
    """
    answer: str
    confidence: int
    reasoning: str
    fault_found: bool
    fault_reason: Optional[str]
    provider: str
    model: str
    raw_response: str


class LLMBaseClient(ABC):
    """
    Abstract base class for all LLM provider clients.
    
    Every concrete client (Claude, GPT, Gemini) inherits from this
    and must implement both methods below. The type annotations on
    generate() ensure the orchestrator always receives an LLMResponse,
    regardless of which provider produced it.
    """

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    async def generate(self, prompt: str) -> LLMResponse:
        """
        Send a prompt to the provider and return a structured LLMResponse.
        Must be async so the orchestrator can call all three in parallel
        using asyncio.gather().
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verify the API key is valid and the provider is reachable.
        Returns True if healthy, False if the key is invalid or the
        provider is down. Used by the /session/status endpoint to
        give users early warning before they run a full query.
        """
        pass