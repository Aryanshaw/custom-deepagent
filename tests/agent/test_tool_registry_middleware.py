"""Tests that tool_registry.call_tool() actually invokes registered
middlewares, and that default_tools() filters correctly by the `default`
flag. Uses locally-defined tools registered under unique names so this
doesn't collide with the real tool registry (which is a module-level
global shared across the process)."""

from __future__ import annotations

import pytest

from swarmagent.agent.middleware.base import ToolMiddleware
from swarmagent.agent.tool_registry import (
    call_tool,
    default_tools,
    set_middlewares,
    tool,
)


@pytest.fixture(autouse=True)
def _reset_middlewares():
    """_middlewares is module-level global state — don't leak between tests."""
    set_middlewares([])
    yield
    set_middlewares([])


class _RecordingMiddleware(ToolMiddleware):
    """No-op-by-default base means this only needs to implement the one
    hook it cares about."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def after_tool_call(self, name, args, result):
        self.calls.append((name, args))
        return result


class _UppercaseMiddleware(ToolMiddleware):
    def after_tool_call(self, name, args, result):
        return result.upper() if isinstance(result, str) else result


def test_call_tool_runs_no_middleware_by_default():
    @tool(name="registry_test_plain")
    def plain_tool() -> str:
        """A tool with no side effects."""
        return "ok"

    assert call_tool("registry_test_plain", {}) == "ok"


def test_call_tool_invokes_registered_middleware():
    @tool(name="registry_test_recorded")
    def recorded_tool(x: int) -> int:
        """A tool that gets watched by a middleware."""
        return x * 2

    recorder = _RecordingMiddleware()
    set_middlewares([recorder])

    result = call_tool("registry_test_recorded", {"x": 21})

    assert result == 42
    assert recorder.calls == [("registry_test_recorded", {"x": 21})]


def test_middlewares_chain_in_order():
    @tool(name="registry_test_chained")
    def chained_tool() -> str:
        """A tool whose result passes through multiple middlewares."""
        return "hello"

    recorder = _RecordingMiddleware()
    set_middlewares([recorder, _UppercaseMiddleware()])

    result = call_tool("registry_test_chained", {})

    assert result == "HELLO"  # recorder ran first (no mutation), then uppercase
    assert recorder.calls == [("registry_test_chained", {})]


def test_middleware_does_not_run_on_exception():
    @tool(name="registry_test_raises")
    def raising_tool() -> str:
        """A tool that always raises."""
        raise ValueError("boom")

    recorder = _RecordingMiddleware()
    set_middlewares([recorder])

    with pytest.raises(ValueError):
        call_tool("registry_test_raises", {})

    assert recorder.calls == []  # never reached — exception happens before the hook


def test_default_tools_filters_by_default_flag():
    @tool(name="registry_test_default_true", default=True)
    def default_tool() -> str:
        """Marked default=True."""
        return "d"

    @tool(name="registry_test_default_false")
    def non_default_tool() -> str:
        """Not marked default."""
        return "nd"

    names = {fn.__name__ for fn in default_tools()}
    assert "default_tool" in names
    assert "non_default_tool" not in names
