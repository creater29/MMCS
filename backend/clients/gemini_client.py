"""
gemini_client.py

LLM client wrapper for Google Gemini API.
Uses google-generativeai SDK with asyncio executor for async compatibility.
"""

import logging
import asyncio
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from clients.base_client import LLMBaseClient, LLMResponse
from clients.response_parser import parse_response

logger = logging.getLogger(__name__)


class GeminiClient(LLMBaseClient):

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        # Configure the SDK with the API key.
        # This is a global configuration call — it sets the key for all
        # subsequent calls made with this SDK in this process.
        genai.configure(api_key=api_key)
        
        # Safety settings — set to low blocking so the model can
        # answer a wide range of factual questions without refusing.
        self._safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        # The Google SDK sometimes requires the "models/" prefix.
        # Normalise the model name so it always has the prefix.
        normalised_model = model if model.startswith("models/") else f"models/{model}"
        self._model_instance = genai.GenerativeModel(
            model_name=normalised_model,
            safety_settings=self._safety_settings,
        )

    def _sync_generate(self, prompt: str) -> str:
        """
        Synchronous generation call — runs in a thread pool executor
        so it does not block the main asyncio event loop.
        Returns the raw text from the model response.
        """
        response = self._model_instance.generate_content(prompt)
        return response.text

    async def generate(self, prompt: str) -> LLMResponse:
        """
        Async wrapper around the synchronous Gemini SDK call.
        
        asyncio.get_event_loop().run_in_executor() runs _sync_generate
        in a background thread, freeing the event loop to handle other
        tasks (like the parallel Claude and OpenAI calls) while Gemini
        is processing. This is the correct pattern for wrapping
        synchronous I/O in an async context.
        """
        try:
            loop = asyncio.get_event_loop()
            raw_text = await loop.run_in_executor(
                None,                          # None = use default ThreadPoolExecutor
                self._sync_generate,           # the synchronous function to run
                prompt,                        # its argument
            )
            parsed = parse_response(raw_text, provider="gemini", model=self.model)
            return LLMResponse(**parsed)

        except Exception as e:
            logger.error(f"Gemini generate() failed: {type(e).__name__}: {e}")
            raise

    async def health_check(self) -> bool:
        """Quick key validation using a minimal prompt."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._sync_generate,
                "Reply with the word OK only.",
            )
            return True
        except Exception as e:
            logger.warning(f"Gemini health check failed: {e}")
            return False