from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from typing import Any

from swarmagent.agent.prompt.swarmagent_systemprompt import SYSTEM_PROMPT
from swarmagent.agent.tool_registry import discover_tools, set_verbose , default_tools as get_default_tools
from swarmagent.config.logger import logger
from swarmagent.factory.factory import LLM, Effort, LLMFactory, Turn, providers
from swarmagent.utils.pretty import print_output, print_system_prompt


class SwarmAgent:
    BASE_AGENT_PROMPT = SYSTEM_PROMPT

    def __init__(self, provider: providers, max_iterations: int = 20, verbose: bool = False , load_default_tools: bool = True):
        self.llm: LLM = LLMFactory.register(provider)
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.load_default_tools = load_default_tools
        # TODO: built in tools, shell tools for code execution
        # TODO: tool result size limitations
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

    def invoke(
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
        cache: bool =False
    ) -> tuple[str | Generator[str, None, None], list[Turn]]:
        """Returns (response, new_turns) — `history` is read-only, never mutated.

        `new_turns` is a list of provider-agnostic `Turn` dicts (role
        user/assistant/tool, each JSON-serializable) — store each one as its
        own record however you like, e.g. one DB row per turn. Fold them
        into your own history store: `history.extend(new_turns)`.
        """
        try:
            discover_tools()

            # load default tools
            effective_tools = self._resolve_effective_tools(tools)

            set_verbose(self.verbose)
            if self.verbose:
                print_system_prompt(self.BASE_AGENT_PROMPT)
            response, new_turns = self.llm.run(
                system_prompt=self.BASE_AGENT_PROMPT,
                user_prompt=prompt,
                history=history,
                image_urls=image_urls,
                model=model,
                tools=effective_tools,
                temperature=temperature,
                max_tokens=max_tokens,
                effort=effort,
                stream=stream,
                max_iterations=self.max_iterations,
                cache=cache
            )
            if not self.verbose:
                return response, new_turns
            if isinstance(response, str):
                print_output(response)
                return response, new_turns
            return _verbose_generator(response), new_turns
        except Exception as e:
            logger.error(f"Error in SwarmAgent._invoke: {e}")
            raise e

    async def ainvoke(
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
        cache: bool = False
    ) -> tuple[str | AsyncGenerator[str, None], list[Turn]]:
        """Returns (response, new_turns) — `history` is read-only, never mutated.

        `new_turns` is a list of provider-agnostic `Turn` dicts (role
        user/assistant/tool, each JSON-serializable) — store each one as its
        own record however you like, e.g. one DB row per turn. Fold them
        into your own history store: `history.extend(new_turns)`.
        """
        try:
            discover_tools()

            # load default tools
            effective_tools = self._resolve_effective_tools(tools)

            set_verbose(self.verbose)
            if self.verbose:
                print_system_prompt(self.BASE_AGENT_PROMPT)
            response, new_turns = await self.llm.a_run(
                system_prompt=self.BASE_AGENT_PROMPT,
                user_prompt=prompt,
                history=history,
                image_urls=image_urls,
                model=model,
                tools=effective_tools,
                temperature=temperature,
                max_tokens=max_tokens,
                effort=effort,
                stream=stream,
                max_iterations=self.max_iterations,
                cache=cache
            )
            if not self.verbose:
                return response, new_turns
            if isinstance(response, str):
                print_output(response)
                return response, new_turns
            return _a_verbose_generator(response), new_turns
        except Exception as e:
            logger.error(f"Error in SwarmAgent._ainvoke: {e}")
            raise e

    def _resolve_effective_tools(
        self, tools: Sequence[Callable[..., Any] | dict[str, Any]] | None
    ) -> list[Callable[..., Any] | dict[str, Any]]:
        effective_tools = list(tools or [])
        if self.load_default_tools:
            seen = {getattr(t, "__name__", None) for t in effective_tools}
            for t in get_default_tools():
                if getattr(t, "__name__", None) not in seen:
                    effective_tools.append(t)
                    seen.add(getattr(t, "__name__", None))
        return effective_tools

def _verbose_generator(response: Generator[str, None, None]) -> Generator[str, None, None]:
    """Pass tokens through untouched, then print the full output once the stream ends."""
    parts: list[str] = []
    for chunk in response:
        parts.append(chunk)
        yield chunk
    print_output("".join(parts))


async def _a_verbose_generator(response: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    parts: list[str] = []
    async for chunk in response:
        parts.append(chunk)
        yield chunk
    print_output("".join(parts))
