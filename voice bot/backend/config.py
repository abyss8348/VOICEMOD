"""Configuration and environment variable management for FlowVoice."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings read strictly from environment variables."""

    # LiveKit credentials
    livekit_url: str = os.getenv("LIVEKIT_URL", "")
    livekit_api_key: str = os.getenv("LIVEKIT_API_KEY", "")
    livekit_api_secret: str = os.getenv("LIVEKIT_API_SECRET", "")

    # Rime TTS configuration
    rime_api_key: str = os.getenv("RIME_API_KEY", "")
    rime_model: str = os.getenv("RIME_MODEL", "coda")
    rime_speaker: str = os.getenv("RIME_SPEAKER", "astra")

    # OpenAI configuration (STT and LLM)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    def validate(self, require_livekit: bool = True) -> list[str]:
        """Validate required configuration and return a list of missing variables."""
        missing: list[str] = []

        if require_livekit:
            if not self.livekit_url:
                missing.append("LIVEKIT_URL")
            if not self.livekit_api_key:
                missing.append("LIVEKIT_API_KEY")
            if not self.livekit_api_secret:
                missing.append("LIVEKIT_API_SECRET")

        if not self.rime_api_key:
            missing.append("RIME_API_KEY")

        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")

        return missing


settings = Settings()
