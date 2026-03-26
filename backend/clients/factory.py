"""
factory.py

Creates the correct LLM client instance for a given provider name.

The orchestrator calls get_client() rather than importing and instantiating
individual client classes directly. This means the orchestrator never has
an import dependency on any specific provider — it only knows about the
shared LLMBaseClient interface.

This also makes it trivial to add a fourth provider later: add one new
client file, add one new entry in the dictionary below, done.
"""

from clients.base_client import LLMBaseClient
from clients.claude_client import ClaudeClient
from clients.openai_client import GPTClient
from clients.gemini_client import GeminiClient

# Registry mapping provider name strings to their client classes.
# Using a dictionary here instead of if/elif chains means adding a
# new provider never requires modifying existing logic.
_CLIENT_REGISTRY = {
    "claude": ClaudeClient,
    "openai": GPTClient,
    "gemini": GeminiClient,
}


def get_client(provider: str, api_key: str, model: str) -> LLMBaseClient:
    """
    Instantiate and return the correct LLM client for the given provider.
    
    Args:
        provider : One of "claude", "openai", "gemini".
        api_key  : The user's API key for that provider.
        model    : The specific model version to use.
    
    Raises:
        ValueError if the provider name is not recognised.
    """
    client_class = _CLIENT_REGISTRY.get(provider.lower())
    if client_class is None:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            f"Valid options are: {list(_CLIENT_REGISTRY.keys())}"
        )
    return client_class(api_key=api_key, model=model)