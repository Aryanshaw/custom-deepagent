from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from typing import Any

from app.agent.prompt.deepagent_systemprompt import SYSTEM_PROMPT
from app.agent.tool_registry import discover_tools
from app.config.logger import logger
from app.factory.factory import LLM, Effort, LLMFactory, Turn, providers


class DeepAgent:
    BASE_AGENT_PROMPT = SYSTEM_PROMPT

    def __init__(self, provider: providers, max_iterations: int = 20):
        self.llm: LLM = LLMFactory.register(provider)
        self.max_iterations = max_iterations
        # TODO: pre configured self.subagents
        # TODO: swarm of agents to solve big tasks
        # TODO: skills
        # TODO: memory (self memory / user's memory)
        # TODO: sandbox
        # TODO: mop marking if agent is hallucinating
        # TODO: built in compressor for context window
        # TODO: mem tool implementation to load 1000s of tools
        # TODO: modes: plan / ask / execute / orchestration mode
        # TODO: human in the loop

    def _invoke(
        self,
        prompt: str,
        model: str,
        history: list[Turn] | None = None,
        image_urls: list[str] | None = None,
        tools: Sequence[Callable[..., Any] | dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        effort: Effort | None = None,
        stream: bool = False,
    ) -> tuple[str | Generator[str, None, None], list[Turn]]:
        """Returns (response, new_turns) — `history` is read-only, never mutated.

        `new_turns` is a list of provider-agnostic `Turn` dicts (role
        user/assistant/tool, each JSON-serializable) — store each one as its
        own record however you like, e.g. one DB row per turn. Fold them
        into your own history store: `history.extend(new_turns)`.
        """
        try:
            discover_tools()
            return self.llm.run(
                system_prompt=self.BASE_AGENT_PROMPT,
                user_prompt=prompt,
                history=history,
                image_urls=image_urls,
                model=model,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                effort=effort,
                stream=stream,
                max_iterations=self.max_iterations,
            )
        except Exception as e:
            logger.error(f"Error in DeepAgent._invoke: {e}")
            raise e

    async def _ainvoke(
        self,
        prompt: str,
        model: str,
        history: list[Turn] | None = None,
        image_urls: list[str] | None = None,
        tools: Sequence[Callable[..., Any] | dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        effort: Effort | None = None,
        stream: bool = False,
    ) -> tuple[str | AsyncGenerator[str, None], list[Turn]]:
        """Returns (response, new_turns) — `history` is read-only, never mutated.

        `new_turns` is a list of provider-agnostic `Turn` dicts (role
        user/assistant/tool, each JSON-serializable) — store each one as its
        own record however you like, e.g. one DB row per turn. Fold them
        into your own history store: `history.extend(new_turns)`.
        """
        try:
            discover_tools() 
            return await self.llm.a_run(
                system_prompt=self.BASE_AGENT_PROMPT,
                user_prompt=prompt,
                history=history,
                image_urls=image_urls,
                model=model,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                effort=effort,
                stream=stream,
                max_iterations=self.max_iterations,
            )
        except Exception as e:
            logger.error(f"Error in DeepAgent._ainvoke: {e}")
            raise e
