"""
openai_client.py

LLM client wrapper for OpenAI's GPT API.
Uses the official openai Python SDK with async support.
"""

import logging
from openai import AsyncOpenAI, AuthenticationError, RateLimitError

from clients.base_client import LLMBaseClient, LLMResponse
from clients.response_parser import parse_response

logger = logging.getLogger(__name__)


class GPTClient(LLMBaseClient):

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        # AsyncOpenAI manages its own connection pool internally.
        self._client = AsyncOpenAI(api_key=api_key)

    async def generate(self, prompt: str) -> LLMResponse:
        """
        Send a prompt to GPT and return a structured LLMResponse.
        
        OpenAI's chat completions API uses a "messages" array with roles.
        We use the "user" role for the prompt. The "system" role could be
        used to set persistent instructions, but keeping everything in the
        user message makes the prompt fully visible and debuggable in logs.
        """
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.choices[0].message.content
            parsed = parse_response(raw_text, provider="openai", model=self.model)
            return LLMResponse(**parsed)

        except AuthenticationError:
            logger.error("OpenAI: API key is invalid or expired.")
            raise
        except RateLimitError:
            logger.error("OpenAI: Rate limit exceeded.")
            raise
        except Exception as e:
            logger.error(f"OpenAI: Unexpected error during generation: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Verify the OpenAI API key by sending a minimal test completion.
        """
        try:
            await self._client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception:
            return False