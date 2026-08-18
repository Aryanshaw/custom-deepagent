"""Turn <-> wire-format conversion shared by OpenAI, Groq, and OpenRouter (all Chat Completions).

Keeps `Turn` (the public, provider-agnostic shape) separate from what
actually goes over the wire — `role: "tool"` messages, `tool_calls` with
JSON-string arguments, etc. Neither provider file should build or read those
dicts directly; go through here instead.
"""

import json
from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any

from swarmagent.config.logger import logger
from swarmagent.factory.factory import LoopInterrupted, ToolCall, Turn, tool_schema_correction


def turn_to_message(turn: Turn) -> dict[str, Any]:
    role = turn["role"]
    if role == "tool":
        # OpenAI/Groq's tool-result message has no native error flag — the
        # model only reads `content` as a string, so prefix it on failure.
        content = str(turn.get("result", ""))
        if turn.get("is_error"):
            content = f"Error: {content}"
        return {
            "role": "tool",
            "tool_call_id": turn["tool_call_id"],
            "content": content,
        }
    if role == "assistant":
        message: dict[str, Any] = {"role": "assistant", "content": turn.get("text")}
        tool_calls = turn.get("tool_calls")
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                }
                for tc in tool_calls
            ]
        return message
    # user
    return {"role": "user", "content": [{"type": "text", "text": turn.get("text") or ""}]}


def turns_to_messages(turns: list[Turn]) -> list[dict[str, Any]]:
    return [turn_to_message(t) for t in turns]


def user_turn(text: str) -> Turn:
    return {"role": "user", "text": text}


def assistant_message_to_turn(message: Any) -> Turn:
    """Build a `Turn` from an SDK `ChatCompletionMessage`.

    Callers must have already rejected any non-"function" tool_calls
    (custom/freeform tool calls) before reaching here — this assumes every
    entry in `message.tool_calls` has a `.function`.
    """
    turn: Turn = {"role": "assistant", "text": message.content}
    if message.tool_calls:
        turn["tool_calls"] = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
            for tc in message.tool_calls
        ]
    return turn


def tool_result_turn(tool_call_id: str, tool_name: str, result: Any, is_error: bool = False) -> Turn:
    turn: Turn = {"role": "tool", "tool_call_id": tool_call_id, "tool_name": tool_name, "result": result}
    if is_error:
        turn["is_error"] = True
    return turn


def _accumulate_tool_call_delta(deltas: list[Any], acc: dict[int, dict[str, Any]]) -> None:
    """Merge one chunk's `delta.tool_calls` fragments into `acc`, keyed by index.

    OpenAI-compatible streaming splits each tool call across chunks: only the
    first fragment for an index carries `id`/`function.name`, and every
    fragment (including the first) carries a slice of `function.arguments`
    to concatenate in order.
    """
    for tc in deltas:
        entry = acc.setdefault(tc.index, {"id": None, "name": None, "arguments": ""})
        if tc.id:
            entry["id"] = tc.id
        if tc.function and tc.function.name:
            entry["name"] = tc.function.name
        if tc.function and tc.function.arguments:
            entry["arguments"] += tc.function.arguments


def _finalize_tool_calls(acc: dict[int, dict[str, Any]]) -> list[ToolCall]:
    return [
        ToolCall(id=entry["id"], name=entry["name"], arguments=json.loads(entry["arguments"] or "{}"))
        for _, entry in sorted(acc.items())
    ]


def _tool_call_message(text: str | None, tool_calls: list[ToolCall]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
            }
            for tc in tool_calls
        ],
    }


def stream_tool_loop(
    create: Callable[[], Any],
    call_tool: Callable[[str, str], Any],
    bad_request_error: type[BaseException],
    messages: list[dict[str, Any]],
    new_turns: list[Turn],
    max_iterations: int,
    on_iteration: Callable[[], bool] | None,
    error_context: str,
) -> Generator[str, None, None]:
    """Drive a streamed chat-completions tool loop, yielding text tokens live.

    Runs `create()` (a no-arg thunk closing over the provider's `kwargs`
    dict, whose `messages` key is this same `messages` list — mutating it
    in place is enough to feed the next iteration). Each iteration streams
    one response: text deltas are yielded as they arrive, tool-call deltas
    are reassembled across chunks. If the model asks for tools, they're
    dispatched and the loop re-streams with the results appended; otherwise
    the final assistant `Turn` is appended and the generator ends.
    """
    try:
        for _ in range(max_iterations):
            if on_iteration is not None and not on_iteration():
                raise LoopInterrupted

            try:
                response = create()
            except bad_request_error as e:
                correction = tool_schema_correction(e.body, str(e))
                if correction is None:
                    raise
                messages.append({"role": "user", "content": correction})
                new_turns.append({"role": "user", "text": correction})
                continue

            tool_call_acc: dict[int, dict[str, Any]] = {}
            text_parts: list[str] = []
            finish_reason = None
            for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta.content:
                    text_parts.append(choice.delta.content)
                    yield choice.delta.content
                if choice.delta.tool_calls:
                    _accumulate_tool_call_delta(choice.delta.tool_calls, tool_call_acc)
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            text = "".join(text_parts)
            if finish_reason != "tool_calls" or not tool_call_acc:
                new_turns.append({"role": "assistant", "text": text})
                return

            tool_calls = _finalize_tool_calls(tool_call_acc)
            messages.append(_tool_call_message(text or None, tool_calls))
            new_turns.append({"role": "assistant", "text": text or None, "tool_calls": tool_calls})

            for tool_call in tool_calls:
                is_error = False
                try:
                    result = call_tool(tool_call["name"], json.dumps(tool_call["arguments"]))
                except Exception as tool_error:
                    result = str(tool_error)
                    is_error = True
                content = f"Error: {result}" if is_error else str(result)
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": content})
                new_turns.append(tool_result_turn(tool_call["id"], tool_call["name"], result, is_error=is_error))

        raise RuntimeError("hit max_iterations without the model finishing")
    except LoopInterrupted:
        raise
    except Exception as e:
        logger.error(f"Error in {error_context}: {e}")
        raise


async def a_stream_tool_loop(
    create: Callable[[], Any],
    call_tool: Callable[[str, str], Any],
    bad_request_error: type[BaseException],
    messages: list[dict[str, Any]],
    new_turns: list[Turn],
    max_iterations: int,
    on_iteration: Callable[[], bool] | None,
    error_context: str,
) -> AsyncGenerator[str, None]:
    """Async twin of `stream_tool_loop` — see there for the contract."""
    try:
        for _ in range(max_iterations):
            if on_iteration is not None and not on_iteration():
                raise LoopInterrupted

            try:
                response = await create()
            except bad_request_error as e:
                correction = tool_schema_correction(e.body, str(e))
                if correction is None:
                    raise
                messages.append({"role": "user", "content": correction})
                new_turns.append({"role": "user", "text": correction})
                continue

            tool_call_acc: dict[int, dict[str, Any]] = {}
            text_parts: list[str] = []
            finish_reason = None
            async for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta.content:
                    text_parts.append(choice.delta.content)
                    yield choice.delta.content
                if choice.delta.tool_calls:
                    _accumulate_tool_call_delta(choice.delta.tool_calls, tool_call_acc)
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            text = "".join(text_parts)
            if finish_reason != "tool_calls" or not tool_call_acc:
                new_turns.append({"role": "assistant", "text": text})
                return

            tool_calls = _finalize_tool_calls(tool_call_acc)
            messages.append(_tool_call_message(text or None, tool_calls))
            new_turns.append({"role": "assistant", "text": text or None, "tool_calls": tool_calls})

            for tool_call in tool_calls:
                is_error = False
                try:
                    result = call_tool(tool_call["name"], json.dumps(tool_call["arguments"]))
                except Exception as tool_error:
                    result = str(tool_error)
                    is_error = True
                content = f"Error: {result}" if is_error else str(result)
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": content})
                new_turns.append(tool_result_turn(tool_call["id"], tool_call["name"], result, is_error=is_error))

        raise RuntimeError("hit max_iterations without the model finishing")
    except LoopInterrupted:
        raise
    except Exception as e:
        logger.error(f"Error in {error_context}: {e}")
        raise
