import json
from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any, cast

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream

from app.agent.tool_registry import call_tool, tools_for
from app.config.logger import logger
from app.factory.factory import LLM, ChatMessage, Effort, LoopInterrupted, to_reasoning_effort


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
        history: list[ChatMessage] | None = None,
        image_urls: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        effort: Effort | None = None,
        stream: bool = False,
        max_iterations: int = 20,
        on_iteration: Callable[[], bool] | None = None,
    ) -> str | Generator[str, None, None]:
        try:
            content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]

            if image_urls:
                for image_url in image_urls:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        }
                    )
            messages: list[ChatMessage] = [
                {"role": "system", "content": system_prompt},
            ]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": content})

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }
            resolved_tools = tools if tools is not None else tools_for("openai")
            if resolved_tools:
                kwargs["tools"] = resolved_tools
            if effort is not None:
                kwargs["reasoning_effort"] = to_reasoning_effort(effort)

            for _ in range(max_iterations):
                if on_iteration is not None and not on_iteration():
                    raise LoopInterrupted

                response = self.client.chat.completions.create(**kwargs)

                # streaming + tool loop is out of scope for now — return raw text stream
                if isinstance(response, Stream):

                    def _stream_generator(
                        response: Stream[Any] = response,
                    ) -> Generator[str, None, None]:
                        for chunk in response:
                            if chunk.choices and chunk.choices[0].delta.content:
                                yield chunk.choices[0].delta.content

                    return _stream_generator()

                message = response.choices[0].message

                # if response is not a tool call return the message content
                if response.choices[0].finish_reason != "tool_calls":
                    return message.content or ""

                tool_calls = message.tool_calls
                if not tool_calls:
                    return message.content or ""

                messages.append(cast(ChatMessage, message.model_dump()))

                # call tools for the model
                for tool_call in tool_calls:
                    if tool_call.type != "function":
                        # custom (freeform) tool calls aren't wired up yet
                        raise NotImplementedError(f"unsupported tool_call type: {tool_call.type}")

                    result = self._call_tool(tool_call.function.name, tool_call.function.arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
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
        history: list[ChatMessage] | None = None,
        image_urls: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        effort: Effort | None = None,
        stream: bool = False,
        max_iterations: int = 20,
        on_iteration: Callable[[], bool] | None = None,
    ) -> str | AsyncGenerator[str, None]:
        try:
            content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]

            if image_urls:
                for image_url in image_urls:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        }
                    )
            messages: list[ChatMessage] = [
                {"role": "system", "content": system_prompt},
            ]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": content})

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }
            resolved_tools = tools if tools is not None else tools_for("openai")
            if resolved_tools:
                kwargs["tools"] = resolved_tools
            if effort is not None:
                kwargs["reasoning_effort"] = to_reasoning_effort(effort)

            for _ in range(max_iterations):
                if on_iteration is not None and not on_iteration():
                    raise LoopInterrupted

                response = await self.aclient.chat.completions.create(**kwargs)

                # streaming + tool loop is out of scope for now — return raw text stream
                if isinstance(response, AsyncStream):

                    async def _a_stream_generator(
                        response: AsyncStream[Any] = response,
                    ) -> AsyncGenerator[str, None]:
                        async for chunk in response:
                            if chunk.choices and chunk.choices[0].delta.content:
                                yield chunk.choices[0].delta.content

                    return _a_stream_generator()

                message = response.choices[0].message

                if response.choices[0].finish_reason != "tool_calls":
                    return message.content or ""

                tool_calls = message.tool_calls
                if not tool_calls:
                    return message.content or ""

                messages.append(cast(ChatMessage, message.model_dump()))
                for tool_call in tool_calls:
                    if tool_call.type != "function":
                        # custom (freeform) tool calls aren't wired up yet
                        raise NotImplementedError(f"unsupported tool_call type: {tool_call.type}")
                    result = self._call_tool(tool_call.function.name, tool_call.function.arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )
                kwargs["messages"] = messages

            raise RuntimeError("hit max_iterations without the model finishing")
        except LoopInterrupted:
            raise
        except Exception as e:
            logger.error(f"Error in OpenAIAgent.a_run: {e}")
            raise e
