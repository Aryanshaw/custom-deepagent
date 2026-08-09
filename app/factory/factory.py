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


def tool_schema_correction(body: object, message: str) -> str | None:
    """Turn a 400 into a corrective instruction for the model — or None if this
    400 isn't a tool-schema problem at all (bad model name, auth, etc.), in
    which case the caller should re-raise rather than swallow it.

    `body` is the SDK exception's parsed error body (`error.body` on
    Groq/OpenAI/Anthropic's `APIError` — all three expose it the same way).

    Detection is a hierarchy of confidence, not one universal signal:
    - Groq's body carries `error.code == "tool_use_failed"` (observed directly
      from a live 400 — this is the strongest signal, exact match).
    - OpenAI and Anthropic have no confirmed equivalent code as of writing;
      this falls back to a keyword heuristic on the message text, which is
      weaker — it can misfire on an unrelated 400 that happens to mention
      "tool". Tighten this if either SDK's real error shape gets confirmed.
    """
    code = None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            code = error.get("code")

    is_tool_schema_error = code == "tool_use_failed" or (
        "tool" in message.lower() and ("schema" in message.lower() or "valid" in message.lower())
    )
    if not is_tool_schema_error:
        return None
    return f"Your last tool call was invalid: {message} Retry with argument types matching the tool's schema exactly."


class LoopInterrupted(Exception):
    """Raised when `on_iteration` returns False, stopping the loop early."""


class ToolCall(TypedDict):
    """One tool invocation the model asked for."""

    id: str
    name: str
    arguments: dict[str, Any]


class Turn(TypedDict, total=False):
    """One provider-agnostic conversation turn.

    This is the public shape callers see — for `history` in and for the
    `new_turns` returned from `run`/`a_run`. Each provider translates to/from
    its own wire format internally; callers never see provider-native shapes
    (Anthropic's content blocks, OpenAI/Groq's `tool_calls`/role:"tool") —
    only this. Safe to store one row per `Turn` in a database: every field is
    JSON-serializable as-is.

    - `role: "user"` — `text` set.
    - `role: "assistant"` — `text` set if the model said anything; `tool_calls`
      set if it also (or instead) asked to call tools.
    - `role: "tool"` — `tool_call_id`/`tool_name`/`result` set; this is the
      result of one call from a preceding assistant turn's `tool_calls`.
      `is_error` is set (True) when the tool raised instead of returning —
      `result` then holds the error message, and the model sees this as a
      failed call it can react to, instead of the whole run blowing up.
    """

    role: Role
    text: str | None
    tool_calls: list[ToolCall]
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool


# Legacy provider-native message shape. No longer used in the public
# run/a_run contract (superseded by `Turn`) — kept only in case something
# still imports it.
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
        history: list[Turn] | None = None,
        image_urls: list[str] | None = None,
        tools: Sequence[Callable[..., Any] | dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        effort: Effort | None = None,
        stream: bool = False,
        max_iterations: int = 20,
        on_iteration: Callable[[], bool] | None = None,
        cache: bool = False
    ) -> tuple[str | Generator[str, None, None], list[Turn]]:
        """Returns (response, new_turns).

        `history` is read-only — never mutated, and never provider-specific —
        pass back exactly what a previous call returned as `new_turns` (or
        whatever you reconstructed from your own storage in the same shape).
        `new_turns` is everything generated this call: the user turn, any
        tool-call/tool-result turns, and the final assistant turn — store
        each `Turn` as its own record however you like (e.g. one DB row per
        turn); the shape is the same regardless of which provider you're
        using, so switching providers mid-conversation is safe.

        If `response` is a generator (real token streaming, no tools were
        called), the final assistant `Turn` in `new_turns` is only populated
        once the generator is fully consumed — read `new_turns` after draining.
        """
        pass

    @abstractmethod
    async def a_run(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        history: list[Turn] | None = None,
        image_urls: list[str] | None = None,
        tools: Sequence[Callable[..., Any] | dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        effort: Effort | None = None,
        stream: bool = False,
        max_iterations: int = 20,
        on_iteration: Callable[[], bool] | None = None,
        cache: bool = False
    ) -> tuple[str | AsyncGenerator[str, None], list[Turn]]:
        """Returns (response, new_turns) — see `run` for the contract."""
        pass


class LLMFactory:
    @staticmethod
    def register(provider: providers, prompt_caching:bool = False) -> LLM:
        # imported here, not at module top — anthropic.py/groq.py/openai.py
        # import LLM/Turn/Effort from this module, so importing them
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
