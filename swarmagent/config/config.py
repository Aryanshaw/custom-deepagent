import os

from dotenv import load_dotenv

load_dotenv(override=True)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class _Config:
    """Env vars are read lazily, per-attribute — importing this module (or
    registering one provider) must not require every provider's key to be
    set."""

    @property
    def GROQ_API_KEY(self) -> str:
        return require_env("GROQ_API_KEY")

    @property
    def OPENAI_API_KEY(self) -> str:
        return require_env("OPENAI_API_KEY")

    @property
    def ANTHROPIC_API_KEY(self) -> str:
        return require_env("ANTHROPIC_API_KEY")

    @property
    def OPENROUTER_API_KEY(self) -> str:
        return require_env("OPENROUTER_API_KEY")


Config = _Config()
