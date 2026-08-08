import os

from dotenv import load_dotenv

load_dotenv(override=True)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class Config:
    GROQ_API_KEY: str = _require_env("GROQ_API_KEY")
    OPENAI_API_KEY: str = _require_env("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = _require_env("ANTHROPIC_API_KEY")
