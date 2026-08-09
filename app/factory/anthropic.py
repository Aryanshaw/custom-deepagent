from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from typing import Any, cast

from anthropic import Anthropic, AsyncAnthropic, BadRequestError, omit
from anthropic.types import CacheControlEphemeralParam, ContentBlock, MessageParam, OutputConfigParam, ToolUnionParam

from app.agent.tool_registry import call_tool, resolve_tools, tools_for
from app.config.logger import logger
from app.factory.factory import LLM, Effort, LoopInterrupted, ToolCall, Turn, tool_schema_correction


def _turn_to_message(turn: Turn) -> MessageParam:
    """Turn -> Anthropic wire format.

    Anthropic has no `role: "tool"` — a tool result rides back as a
    `role: "user"` message with a `tool_result` content block. Callers never
    need to know this; they only ever see `Turn(role="tool", ...)`.
    """
    role = turn["role"]
    if role == "tool":
        return cast(
            MessageParam,
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": turn["tool_call_id"],
                        "content": str(turn.get("result", "")),
                        "is_error": turn.get("is_error", False),
                    }
                ],
            },
        )
    if role == "assistant":
        blocks: list[dict[str, Any]] = []
        if turn.get("text"):
            blocks.append({"type": "text", "text": turn["text"]})
        for tc in turn.get("tool_calls", []):
            blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})
        return cast(MessageParam, {"role": "assistant", "content": blocks})
    # user
    return cast(MessageParam, {"role": "user", "content": [{"type": "text", "text": turn.get("text") or ""}]})


def _turns_to_messages(turns: list[Turn]) -> list[MessageParam]:
    return [_turn_to_message(t) for t in turns]


def _assistant_turn(content_blocks: list[ContentBlock]) -> Turn:
    text = next((b.text for b in content_blocks if b.type == "text"), None)
    tool_use_blocks = [b for b in content_blocks if b.type == "tool_use"]
    turn: Turn = {"role": "assistant", "text": text}
    if tool_use_blocks:
        turn["tool_calls"] = [
            ToolCall(id=b.id, name=b.name, arguments=cast(dict[str, Any], b.input)) for b in tool_use_blocks
        ]
    return turn


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
        history: list[Turn] | None = None,
        image_urls: list[str] | None = None,
        tools: Sequence[Callable[..., Any] | dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        effort: Effort | None = None,
        stream: bool = False,
        max_iterations: int = 20,
        on_iteration: Callable[[], bool] | None = None,
        cache: bool = False,
    ) -> tuple[str | Generator[str, None, None], list[Turn]]:
        try:
            _ = temperature
            content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            if image_urls:
                for image_url in image_urls:
                    content.append({"type": "image", "source": {"type": "url", "url": image_url}})

            messages: list[MessageParam] = []
            if history:
                messages.extend(_turns_to_messages(history))
            messages.append(cast(MessageParam, {"role": "user", "content": content}))

            new_turns: list[Turn] = [{"role": "user", "text": user_prompt}]

            resolved_tools = resolve_tools(tools, "anthropic") if tools is not None else tools_for("anthropic")
            tools_param = cast(list[ToolUnionParam], resolved_tools) if resolved_tools else omit
            output_config = cast(OutputConfigParam, {"effort": effort}) if effort is not None else omit

            # tools + true token streaming isn't wired (a tool_use block arrives
            # whole, but reconstructing it requires buffering the stream anyway)
            # — only take the raw-stream shortcut when there are no tools to call.
            effective_stream = stream and not resolved_tools

            if effective_stream:

                def _stream_generator() -> Generator[str, None, None]:
                    parts: list[str] = []
                    with self.client.messages.stream(
                        model=model,
                        max_tokens=max_tokens,
                        system=system_prompt,
                        messages=messages,
                        tools=tools_param,
                        output_config=output_config,
                        cache_control=CacheControlEphemeralParam(type="ephemeral") if cache else omit,
                    ) as s:
                        for text in s.text_stream:
                            parts.append(text)
                            yield text
                    new_turns.append({"role": "assistant", "text": "".join(parts)})

                return _stream_generator(), new_turns

            def _as_result(text: str) -> str | Generator[str, None, None]:
                if stream and not effective_stream:

                    def _one_shot() -> Generator[str, None, None]:
                        yield text

                    return _one_shot()
                return text

            for _ in range(max_iterations):
                if on_iteration is not None and not on_iteration():
                    raise LoopInterrupted

                try:
                    response = self.client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        system=system_prompt,
                        messages=messages,
                        tools=tools_param,
                        output_config=output_config,
                        cache_control=CacheControlEphemeralParam(type="ephemeral") if cache else omit,
                    )
                except BadRequestError as e:
                    correction = tool_schema_correction(e.body, str(e))
                    if correction is None:
                        raise  # a real error, not a fixable schema mismatch
                    messages.append(cast(MessageParam, {"role": "user", "content": correction}))
                    new_turns.append({"role": "user", "text": correction})
                    continue

                if response.stop_reason != "tool_use":
                    text = next((b.text for b in response.content if b.type == "text"), "")
                    new_turns.append({"role": "assistant", "text": text})
                    return _as_result(text), new_turns

                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
                if not tool_use_blocks:
                    text = next((b.text for b in response.content if b.type == "text"), "")
                    new_turns.append({"role": "assistant", "text": text})
                    return _as_result(text), new_turns

                messages.append(cast(MessageParam, {"role": "assistant", "content": response.content}))
                new_turns.append(_assistant_turn(response.content))

                tool_results = []
                for block in tool_use_blocks:
                    is_error = False
                    try:
                        result = self._call_tool(block.name, cast(dict[str, Any], block.input))
                    except Exception as tool_error:
                        result = str(tool_error)
                        is_error = True
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                            "is_error": is_error,
                        }
                    )
                    new_turns.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.id,
                            "tool_name": block.name,
                            "result": result,
                            "is_error": is_error,
                        }
                    )
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
        history: list[Turn] | None = None,
        image_urls: list[str] | None = None,
        tools: Sequence[Callable[..., Any] | dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        effort: Effort | None = None,
        stream: bool = False,
        max_iterations: int = 20,
        on_iteration: Callable[[], bool] | None = None,
        cache: bool = False,
    ) -> tuple[str | AsyncGenerator[str, None], list[Turn]]:
        try:
            _ = temperature
            content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            if image_urls:
                for image_url in image_urls:
                    content.append({"type": "image", "source": {"type": "url", "url": image_url}})

            messages: list[MessageParam] = []
            if history:
                messages.extend(_turns_to_messages(history))
            messages.append(cast(MessageParam, {"role": "user", "content": content}))

            new_turns: list[Turn] = [{"role": "user", "text": user_prompt}]

            resolved_tools = resolve_tools(tools, "anthropic") if tools is not None else tools_for("anthropic")
            tools_param = cast(list[ToolUnionParam], resolved_tools) if resolved_tools else omit
            output_config = cast(OutputConfigParam, {"effort": effort}) if effort is not None else omit

            effective_stream = stream and not resolved_tools

            if effective_stream:

                async def _a_stream_generator() -> AsyncGenerator[str, None]:
                    parts: list[str] = []
                    async with self.aclient.messages.stream(
                        model=model,
                        max_tokens=max_tokens,
                        system=system_prompt,
                        messages=messages,
                        tools=tools_param,
                        output_config=output_config,
                        cache_control=CacheControlEphemeralParam(type="ephemeral") if cache else omit,
                    ) as s:
                        async for text in s.text_stream:
                            parts.append(text)
                            yield text
                    new_turns.append({"role": "assistant", "text": "".join(parts)})

                return _a_stream_generator(), new_turns

            def _as_result(text: str) -> str | AsyncGenerator[str, None]:
                if stream and not effective_stream:

                    async def _one_shot() -> AsyncGenerator[str, None]:
                        yield text

                    return _one_shot()
                return text

            for _ in range(max_iterations):
                if on_iteration is not None and not on_iteration():
                    raise LoopInterrupted

                try:
                    response = await self.aclient.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        system=system_prompt,
                        messages=messages,
                        tools=tools_param,
                        output_config=output_config,
                        cache_control=CacheControlEphemeralParam(type="ephemeral") if cache else omit,
                    )
                except BadRequestError as e:
                    correction = tool_schema_correction(e.body, str(e))
                    if correction is None:
                        raise
                    messages.append(cast(MessageParam, {"role": "user", "content": correction}))
                    new_turns.append({"role": "user", "text": correction})
                    continue

                if response.stop_reason != "tool_use":
                    text = next((b.text for b in response.content if b.type == "text"), "")
                    new_turns.append({"role": "assistant", "text": text})
                    return _as_result(text), new_turns

                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
                if not tool_use_blocks:
                    text = next((b.text for b in response.content if b.type == "text"), "")
                    new_turns.append({"role": "assistant", "text": text})
                    return _as_result(text), new_turns

                messages.append(cast(MessageParam, {"role": "assistant", "content": response.content}))
                new_turns.append(_assistant_turn(response.content))

                tool_results = []
                for block in tool_use_blocks:
                    is_error = False
                    try:
                        result = self._call_tool(block.name, cast(dict[str, Any], block.input))
                    except Exception as tool_error:
                        result = str(tool_error)
                        is_error = True
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                            "is_error": is_error,
                        }
                    )
                    new_turns.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.id,
                            "tool_name": block.name,
                            "result": result,
                            "is_error": is_error,
                        }
                    )
                messages.append(cast(MessageParam, {"role": "user", "content": tool_results}))

            raise RuntimeError("hit max_iterations without the model finishing")
        except LoopInterrupted:
            raise
        except Exception as e:
            logger.error(f"Error in AnthropicAgent.a_run: {e}")
            raise e
