"""Turn <-> wire-format conversion shared by OpenAI and Groq (both Chat Completions).

Keeps `Turn` (the public, provider-agnostic shape) separate from what
actually goes over the wire — `role: "tool"` messages, `tool_calls` with
JSON-string arguments, etc. Neither provider file should build or read those
dicts directly; go through here instead.
"""

import json
from typing import Any

from swarmagent.factory.factory import ToolCall, Turn


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
