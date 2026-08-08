from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from typing import Any, cast

from anthropic import Anthropic, AsyncAnthropic, omit
from anthropic.types import MessageParam, OutputConfigParam, ToolUnionParam

from app.agent.tool_registry import call_tool, resolve_tools, tools_for
from app.config.logger import logger
from app.factory.factory import LLM, ChatMessage, Effort, LoopInterrupted


class AnthropicAgent(LLM):
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.aclient = AsyncAnthropic(api_key=api_key)

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Dispatch a tool call to whatever's registered via @tool.

        Unlike OpenAI/Groq, Anthropic hands you already-parsed args (a dict,
        not a JSON string) — no json.loads needed here.
        """
        return call_tool(name, arguments)

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
        try:
            _ = temperature
            content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]

            if image_urls:
                for image_url in image_urls:
                    content.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": image_url},
                        }
                    )
            history_msgs: list[dict[str, Any]] = [dict(m) for m in history] if history else []
            history_msgs.append({"role": "user", "content": content})
            messages = cast(list[MessageParam], history_msgs)
            resolved_tools = resolve_tools(tools, "anthropic") if tools is not None else tools_for("anthropic")
            tools_param = cast(list[ToolUnionParam], resolved_tools) if resolved_tools else omit
            output_config = cast(OutputConfigParam, {"effort": effort}) if effort is not None else omit

            # tools + true token streaming isn't wired (a tool_use block arrives
            # whole, but reconstructing it requires buffering the stream anyway)
            # — only take the raw-stream shortcut when there are no tools to call.
            effective_stream = stream and not resolved_tools

            if effective_stream:

                def _stream_generator() -> Generator[str, None, None]:
                    with self.client.messages.stream(
                        model=model,
                        max_tokens=max_tokens,
                        system=system_prompt,
                        messages=messages,
                        tools=tools_param,
                        output_config=output_config,
                    ) as s:
                        yield from s.text_stream

                return _stream_generator()

            def _as_result(text: str) -> str | Generator[str, None, None]:
                if stream and not effective_stream:

                    def _one_shot() -> Generator[str, None, None]:
                        yield text

                    return _one_shot()
                return text

            for _ in range(max_iterations):
                if on_iteration is not None and not on_iteration():
                    raise LoopInterrupted

                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=messages,
                    tools=tools_param,
                    output_config=output_config,
                )

                if response.stop_reason != "tool_use":
                    return _as_result(next((b.text for b in response.content if b.type == "text"), ""))

                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
                if not tool_use_blocks:
                    return _as_result(next((b.text for b in response.content if b.type == "text"), ""))

                messages.append(cast(MessageParam, {"role": "assistant", "content": response.content}))
                tool_results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(self._call_tool(block.name, cast(dict[str, Any], block.input))),
                    }
                    for block in tool_use_blocks
                ]
                messages.append(cast(MessageParam, {"role": "user", "content": tool_results}))

            raise RuntimeError("hit max_iterations without the model finishing")
        except LoopInterrupted:
            raise
        except Exception as e:
            logger.error(f"Error in AnthropicAgent.run: {e}")
            raise e

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
        try:
            _ = temperature
            content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]

            if image_urls:
                for image_url in image_urls:
                    content.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": image_url},
                        }
                    )
            history_msgs: list[dict[str, Any]] = [dict(m) for m in history] if history else []
            history_msgs.append({"role": "user", "content": content})
            messages = cast(list[MessageParam], history_msgs)
            resolved_tools = resolve_tools(tools, "anthropic") if tools is not None else tools_for("anthropic")
            tools_param = cast(list[ToolUnionParam], resolved_tools) if resolved_tools else omit
            output_config = cast(OutputConfigParam, {"effort": effort}) if effort is not None else omit

            effective_stream = stream and not resolved_tools

            if effective_stream:

                async def _a_stream_generator() -> AsyncGenerator[str, None]:
                    async with self.aclient.messages.stream(
                        model=model,
                        max_tokens=max_tokens,
                        system=system_prompt,
                        messages=messages,
                        tools=tools_param,
                        output_config=output_config,
                    ) as s:
                        async for text in s.text_stream:
                            yield text

                return _a_stream_generator()

            def _as_result(text: str) -> str | AsyncGenerator[str, None]:
                if stream and not effective_stream:

                    async def _one_shot() -> AsyncGenerator[str, None]:
                        yield text

                    return _one_shot()
                return text

            for _ in range(max_iterations):
                if on_iteration is not None and not on_iteration():
                    raise LoopInterrupted

                response = await self.aclient.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=messages,
                    tools=tools_param,
                    output_config=output_config,
                )

                if response.stop_reason != "tool_use":
                    return _as_result(next((b.text for b in response.content if b.type == "text"), ""))

                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
                if not tool_use_blocks:
                    return _as_result(next((b.text for b in response.content if b.type == "text"), ""))

                messages.append(cast(MessageParam, {"role": "assistant", "content": response.content}))
                tool_results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(self._call_tool(block.name, cast(dict[str, Any], block.input))),
                    }
                    for block in tool_use_blocks
                ]
                messages.append(cast(MessageParam, {"role": "user", "content": tool_results}))

            raise RuntimeError("hit max_iterations without the model finishing")
        except LoopInterrupted:
            raise
        except Exception as e:
            logger.error(f"Error in AnthropicAgent.a_run: {e}")
            raise e
