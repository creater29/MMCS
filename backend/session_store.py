import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ProviderConfig:
    """Holds the API key and model for one provider. Key may be None if not supplied."""
    key: Optional[str]
    model: str


@dataclass
class SessionKeys:
    """
    Complete session credentials. Any provider may be absent if the user
    chose not to supply that key. The orchestrator skips absent providers.
    """
    claude: ProviderConfig
    openai: ProviderConfig
    gemini: ProviderConfig
    created_at: float = field(default_factory=time.time)

    def active_providers(self):
        """Return list of provider names that have real keys."""
        active = []
        if self.claude.key:
            active.append("claude")
        if self.openai.key:
            active.append("openai")
        if self.gemini.key:
            active.append("gemini")
        return active

    def has_enough_providers(self):
        """At least one provider must be configured to run a query."""
        return len(self.active_providers()) >= 1


_sessions: Dict[str, SessionKeys] = {}
SESSION_EXPIRY_SECONDS = 3600


def create_session(
    claude_key: Optional[str], claude_model: str,
    openai_key: Optional[str], openai_model: str,
    gemini_key: Optional[str], gemini_model: str,
) -> str:
    session_id = secrets.token_urlsafe(32)
    _sessions[session_id] = SessionKeys(
        claude=ProviderConfig(key=claude_key or None, model=claude_model),
        openai=ProviderConfig(key=openai_key or None, model=openai_model),
        gemini=ProviderConfig(key=gemini_key or None, model=gemini_model),
    )
    return session_id


def get_session(session_id: str) -> Optional[SessionKeys]:
    session = _sessions.get(session_id)
    if session is None:
        return None
    if time.time() - session.created_at > SESSION_EXPIRY_SECONDS:
        del _sessions[session_id]
        return None
    return session


def delete_session(session_id: str) -> bool:
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


def get_active_session_count() -> int:
    return len(_sessions)