from collections.abc import Callable, Sequence
from typing import Any

from app.agent.prompt.deepagent_systemprompt import SYSTEM_PROMPT
from app.agent.tool_registry import discover_tools
from app.config.logger import logger
from app.factory.factory import LLM, ChatMessage, Effort, LLMFactory, providers


class DeepAgent:
    BASE_AGENT_PROMPT = SYSTEM_PROMPT

    def __init__(self, provider: providers, max_iterations: int = 20):
        self.llm: LLM = LLMFactory.register(provider)
        self.max_iterations = max_iterations
        # TODO: self.subagents
        # TODO: skills
        # TODO: memory
        # TODO: sandbox

    def _invoke(
        self,
        prompt: str,
        model: str,
        history: list[ChatMessage] | None = None,
        image_urls: list[str] | None = None,
        tools: Sequence[Callable[..., Any] | dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        effort: Effort | None = None,
        stream: bool = False,
    ):
        try:
            discover_tools()  # import app/agent/tools/*.py so @tool functions register
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
