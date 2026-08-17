"""Provider-agnostic tool registry.

Decorate a plain function with `@tool`; it gets introspected into a JSON
Schema (via pydantic) once, then rendered into whichever shape a given
provider's SDK wants. Anthropic's `input_schema` and OpenAI/Groq's
`parameters` are both plain JSON Schema — same content, different envelope —
so one schema build serves every provider.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_type_hints

from pydantic import create_model

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

Provider = Literal["anthropic", "openai", "groq"]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    func: Callable[..., Any]

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_openai(self) -> dict[str, Any]:
        """Shared shape for OpenAI and Groq — both use Chat Completions function-calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def call(self, arguments: dict[str, Any]) -> Any:
        return self.func(**arguments)


_REGISTRY: dict[str, ToolSpec] = {}
TOOL_SPEC_ATTR = "__tool_spec__"


def _build_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """JSON Schema for func's parameters, via a pydantic model built from its signature.

    Use `Annotated[type, Field(description="...")]` on a parameter for a
    per-argument description; plain type hints still work, just without one.
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)
    fields: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        annotation = hints.get(name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (annotation, default)

    model = create_model(f"{func.__name__}_Params", **fields)
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., Any]:
    """Register a function as a tool. Use bare (`@tool`) or with overrides (`@tool(name=...)`)."""

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        spec = ToolSpec(
            name=name or f.__name__,
            description=description or (inspect.getdoc(f) or "").strip(),
            parameters=_build_schema(f),
            func=f,
        )
        if not spec.description:
            raise ValueError(f"tool '{spec.name}' needs a docstring or description=...")
        _REGISTRY[spec.name] = spec
        setattr(f, TOOL_SPEC_ATTR, spec)  # so the function itself carries its spec
        return f

    return decorator(func) if func is not None else decorator


def resolve_tools(
    tools: Sequence[Callable[..., Any] | dict[str, Any]] | None,
    provider: Provider,
) -> list[dict[str, Any]]:
    """Convert a mixed list of @tool-decorated functions and/or raw provider
    dicts into the shape `provider` expects. A bare dict passes through
    unchanged (for tools you built by hand or got from elsewhere, e.g. MCP).
    """
    if not tools:
        return []
    resolved: list[dict[str, Any]] = []
    for item in tools:
        if isinstance(item, dict):
            resolved.append(item)
            continue
        spec = getattr(item, TOOL_SPEC_ATTR, None)
        if spec is None:
            raise TypeError(
                f"{item!r} isn't a @tool-decorated function and isn't a provider dict either"
            )
        resolved.append(spec.to_anthropic() if provider == "anthropic" else spec.to_openai())
    return resolved


def tools_for(provider: Provider) -> list[dict[str, Any]]:
    specs = _REGISTRY.values()
    if provider == "anthropic":
        return [s.to_anthropic() for s in specs]
    return [s.to_openai() for s in specs]  # openai + groq share the function-calling shape


_verbose = False


def set_verbose(value: bool) -> None:
    global _verbose
    _verbose = value


def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    from swarmagent.utils.pretty import print_tool_call, print_tool_result

    spec = _REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"unknown tool: {name}")

    if _verbose:
        print_tool_call(name, arguments)
    try:
        result = spec.call(arguments)
    except Exception as e:
        if _verbose:
            print_tool_result(name, e, is_error=True)
        raise
    if _verbose:
        print_tool_result(name, result, is_error=False)
    return result


_discovered: set[str] = set()


def discover_tools(package: str = "swarmagent.agent.tools") -> None:
    """Import every module in `package` so its @tool-decorated functions register.

    Safe to call more than once (e.g. once per SwarmAgent instance) — each
    package is only walked and imported the first time.
    """
    if package in _discovered:
        return
    _discovered.add(package)

    pkg = importlib.import_module(package)
    if not hasattr(pkg, "__path__"):
        return  # not a package (no __init__.py) — nothing to walk
    for _, module_name, _ in pkgutil.walk_packages(pkg.__path__, prefix=f"{package}."):
        importlib.import_module(module_name)
