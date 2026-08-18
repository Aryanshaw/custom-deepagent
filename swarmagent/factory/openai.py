import json
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from typing import Any

from openai import AsyncOpenAI, BadRequestError, OpenAI

from swarmagent.agent.tool_registry import call_tool, resolve_tools, tools_for
from swarmagent.config.logger import logger
from swarmagent.factory.factory import LLM, Effort, LoopInterrupted, Turn, to_reasoning_effort, tool_schema_correction
from swarmagent.utils.chat_completions_compat import (
    a_stream_tool_loop,
    assistant_message_to_turn,
    stream_tool_loop,
    tool_result_turn,
    turns_to_messages,
    user_turn,
)


class OpenAIAgent(LLM):
    def __init__(
        self,
        api_key: str,
    ):
        self.aclient = AsyncOpenAI(api_key=api_key)
        self.client = OpenAI(api_key=api_key)

    def _call_tool(self, name: str, arguments_json: str) -> Any:
        """Dispatch a tool call to whatever's registered via @tool."""
        return call_tool(name, json.loads(arguments_json))

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
        try:
            user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            if image_urls:
                for image_url in image_urls:
                    user_content.append({"type": "image_url", "image_url": {"url": image_url}})

            messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
            if history:
                messages.extend(turns_to_messages(history))
            messages.append({"role": "user", "content": user_content})

            new_turns: list[Turn] = [user_turn(user_prompt)]

            resolved_tools = resolve_tools(tools, "openai") if tools is not None else tools_for("openai")

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
                "stream": stream,
            }
            if resolved_tools:
                kwargs["tools"] = resolved_tools
            if effort is not None:
                kwargs["reasoning_effort"] = to_reasoning_effort(effort)
            if cache:
                kwargs["prompt_cache_key"] = "deep-agent-session"
                kwargs["prompt_cache_retention"] = "24h"

            if stream:
                return (
                    stream_tool_loop(
                        create=lambda: self.client.chat.completions.create(**kwargs),
                        call_tool=self._call_tool,
                        bad_request_error=BadRequestError,
                        messages=messages,
                        new_turns=new_turns,
                        max_iterations=max_iterations,
                        on_iteration=on_iteration,
                        error_context="OpenAIAgent.run",
                    ),
                    new_turns,
                )

            for _ in range(max_iterations):
                if on_iteration is not None and not on_iteration():
                    raise LoopInterrupted

                try:
                    response = self.client.chat.completions.create(**kwargs)
                except BadRequestError as e:
                    correction = tool_schema_correction(e.body, str(e))
                    if correction is None:
                        raise  # a real error, not a fixable schema mismatch
                    messages.append({"role": "user", "content": correction})
                    new_turns.append({"role": "user", "text": correction})
                    kwargs["messages"] = messages
                    continue

                message = response.choices[0].message

                # if response is not a tool call return the message content
                if response.choices[0].finish_reason != "tool_calls":
                    new_turns.append({"role": "assistant", "text": message.content or ""})
                    return message.content or "", new_turns

                tool_calls = message.tool_calls
                if not tool_calls:
                    new_turns.append({"role": "assistant", "text": message.content or ""})
                    return message.content or "", new_turns

                # narrow to function-type calls once — a checker can't carry
                # a type check across two separate `for` loops, so build the
                # validated list up front and use it everywhere below.
                function_calls = [tc for tc in tool_calls if tc.type == "function"]
                if len(function_calls) != len(tool_calls):
                    bad = next(tc for tc in tool_calls if tc.type != "function")
                    # custom (freeform) tool calls aren't wired up yet
                    raise NotImplementedError(f"unsupported tool_call type: {bad.type}")

                messages.append(message.model_dump(exclude_none=True))
                new_turns.append(assistant_message_to_turn(message))

                for tool_call in function_calls:
                    is_error = False
                    try:
                        result = self._call_tool(tool_call.function.name, tool_call.function.arguments)
                    except Exception as tool_error:
                        result = str(tool_error)
                        is_error = True
                    content = f"Error: {result}" if is_error else str(result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": content,
                        }
                    )
                    new_turns.append(
                        tool_result_turn(tool_call.id, tool_call.function.name, result, is_error=is_error)
                    )
                kwargs["messages"] = messages

            raise RuntimeError("hit max_iterations without the model finishing")
        except LoopInterrupted:
            raise
        except Exception as e:
            logger.error(f"Error in OpenAIAgent.run: {e}")
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
        cache: bool = False
    ) -> tuple[str | AsyncGenerator[str, None], list[Turn]]:
        try:
            user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            if image_urls:
                for image_url in image_urls:
                    user_content.append({"type": "image_url", "image_url": {"url": image_url}})

            messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
            if history:
                messages.extend(turns_to_messages(history))
            messages.append({"role": "user", "content": user_content})

            new_turns: list[Turn] = [user_turn(user_prompt)]

            resolved_tools = resolve_tools(tools, "openai") if tools is not None else tools_for("openai")

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
                "stream": stream,
            }
            if resolved_tools:
                kwargs["tools"] = resolved_tools
            if effort is not None:
                kwargs["reasoning_effort"] = to_reasoning_effort(effort)
            if cache:
                kwargs["prompt_cache_key"] = "deep-agent-session"
                kwargs["prompt_cache_retention"] = "24h"

            if stream:
                return (
                    a_stream_tool_loop(
                        create=lambda: self.aclient.chat.completions.create(**kwargs),
                        call_tool=self._call_tool,
                        bad_request_error=BadRequestError,
                        messages=messages,
                        new_turns=new_turns,
                        max_iterations=max_iterations,
                        on_iteration=on_iteration,
                        error_context="OpenAIAgent.a_run",
                    ),
                    new_turns,
                )

            for _ in range(max_iterations):
                if on_iteration is not None and not on_iteration():
                    raise LoopInterrupted

                try:
                    response = await self.aclient.chat.completions.create(**kwargs)
                except BadRequestError as e:
                    correction = tool_schema_correction(e.body, str(e))
                    if correction is None:
                        raise
                    messages.append({"role": "user", "content": correction})
                    new_turns.append({"role": "user", "text": correction})
                    kwargs["messages"] = messages
                    continue

                message = response.choices[0].message

                if response.choices[0].finish_reason != "tool_calls":
                    new_turns.append({"role": "assistant", "text": message.content or ""})
                    return message.content or "", new_turns

                tool_calls = message.tool_calls
                if not tool_calls:
                    new_turns.append({"role": "assistant", "text": message.content or ""})
                    return message.content or "", new_turns

                function_calls = [tc for tc in tool_calls if tc.type == "function"]
                if len(function_calls) != len(tool_calls):
                    bad = next(tc for tc in tool_calls if tc.type != "function")
                    # custom (freeform) tool calls aren't wired up yet
                    raise NotImplementedError(f"unsupported tool_call type: {bad.type}")

                messages.append(message.model_dump(exclude_none=True))
                new_turns.append(assistant_message_to_turn(message))

                for tool_call in function_calls:
                    is_error = False
                    try:
                        result = self._call_tool(tool_call.function.name, tool_call.function.arguments)
                    except Exception as tool_error:
                        result = str(tool_error)
                        is_error = True
                    content = f"Error: {result}" if is_error else str(result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": content,
                        }
                    )
                    new_turns.append(
                        tool_result_turn(tool_call.id, tool_call.function.name, result, is_error=is_error)
                    )
                kwargs["messages"] = messages

            raise RuntimeError("hit max_iterations without the model finishing")
        except LoopInterrupted:
            raise
        except Exception as e:
            logger.error(f"Error in OpenAIAgent.a_run: {e}")
            raise e
