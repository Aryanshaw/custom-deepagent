from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from typing import Any, Literal, NotRequired, TypedDict

from app.config.config import Config

Role = Literal["system", "user", "assistant", "tool"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]
ReasoningEffort = Literal["low", "medium", "high"]
providers = Literal["openai", "groq", "anthropic"]

_EFFORT_TO_REASONING_EFFORT: dict[Effort, ReasoningEffort] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def to_reasoning_effort(effort: Effort) -> ReasoningEffort:
    """Map our 5-level Effort onto the 3-level reasoning_effort OpenAI/Groq accept."""
    return _EFFORT_TO_REASONING_EFFORT[effort]


class LoopInterrupted(Exception):
    """Raised when `on_iteration` returns False, stopping the loop early."""


class ChatMessage(TypedDict, total=False):
    role: Role
    content: str | list[dict[str, Any]]
    name: NotRequired[str]
    tool_calls: NotRequired[list[dict[str, Any]]]
    tool_call_id: NotRequired[str]


class LLM(ABC):
    @abstractmethod
    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        history: list[ChatMessage] | None = None,
        image_urls: list[str] | None = None,
        tools: Sequence[Callable[..., Any] | dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        effort: Effort | None = None,
        stream: bool = False,
        max_iterations: int = 20,
        on_iteration: Callable[[], bool] | None = None,
    ) -> str | Generator[str, None, None]:
        pass

    @abstractmethod
    async def a_run(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        history: list[ChatMessage] | None = None,
        image_urls: list[str] | None = None,
        tools: Sequence[Callable[..., Any] | dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        effort: Effort | None = None,
        stream: bool = False,
        max_iterations: int = 20,
        on_iteration: Callable[[], bool] | None = None,
    ) -> str | AsyncGenerator[str, None]:
        pass


class LLMFactory:
    @staticmethod
    def register(provider: providers) -> LLM:
        # imported here, not at module top — anthropic.py/groq.py/openai.py
        # import LLM/ChatMessage/Effort from this module, so importing them
        # back at the top of this file is a circular import
        match provider:
            case "openai":
                from app.factory.openai import OpenAIAgent

                return OpenAIAgent(api_key=Config.OPENAI_API_KEY)
            case "groq":
                from app.factory.groq import GroqAgent

                return GroqAgent(api_key=Config.GROQ_API_KEY)
            case "anthropic":
                from app.factory.anthropic import AnthropicAgent

                return AnthropicAgent(api_key=Config.ANTHROPIC_API_KEY)
            case _:
                raise ValueError(f"Unknown provider: {provider}")
