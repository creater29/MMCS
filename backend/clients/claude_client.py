"""
claude_client.py

LLM client wrapper for Anthropic's Claude API.
Inherits from LLMBaseClient and implements generate() and health_check()
using the official anthropic Python SDK.
"""

import logging
import anthropic

from clients.base_client import LLMBaseClient, LLMResponse
from clients.response_parser import parse_response

logger = logging.getLogger(__name__)


class ClaudeClient(LLMBaseClient):

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        # The Anthropic client is initialised once per session and reused.
        # Creating it here (rather than inside generate()) avoids the
        # overhead of re-initialising the HTTP connection pool on every call.
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate(self, prompt: str) -> LLMResponse:
        """
        Send a prompt to Claude and return a structured LLMResponse.
        
        Claude's API uses a "messages" format where each message has a role
        (user or assistant) and content. We send one user message containing
        the full prompt and receive one assistant message in response.
        
        max_tokens=1024 is generous enough for a detailed answer plus
        confidence score and reasoning, but not so large that a runaway
        response costs significantly more than expected.
        """
        try:
            message = await self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = message.content[0].text
            parsed = parse_response(raw_text, provider="claude", model=self.model)
            return LLMResponse(**parsed)

        except anthropic.AuthenticationError:
            logger.error("Claude: API key is invalid or expired.")
            raise
        except anthropic.RateLimitError:
            logger.error("Claude: Rate limit exceeded.")
            raise
        except Exception as e:
            logger.error(f"Claude: Unexpected error during generation: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Verify the Claude API key is valid by sending a minimal test message.
        Uses the smallest available model and 10 tokens to minimise cost.
        """
        try:
            await self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception:
            return False