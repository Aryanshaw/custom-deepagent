# swarmagent

Pluggable multi-provider LLM agent (Anthropic, Groq, OpenAI, OpenRouter) built from
scratch around the core agentic tool-use loop: call model → check stop
reason → run tool calls → append results → repeat.

Published on TestPyPI as `swarmagent` (dry run — not yet on real PyPI).

## Install

```bash
pip install -i https://test.pypi.org/simple/ swarmagent
```

The `-i` flag is required because this package currently only exists on
TestPyPI, a separate index from real `pypi.org`. Once published to real
PyPI, plain `pip install swarmagent` will work.

Set the env vars for whichever providers you use — `swarmagent/config/config.py`
reads all three at import time:

```
GROQ_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
```

## Usage

```python
from swarmagent import SwarmAgent

agent = SwarmAgent("groq")

response_stream, new_turns = agent.invoke(
    "hello",
    model="qwen/qwen3.6-27b",
    history=[],
    max_tokens=2000,
    stream=True,
)

for chunk in response_stream:
    print(chunk, end="", flush=True)
```

`history` is a list of `Turn` objects you own — `invoke`/`ainvoke` read it but
never mutate it; append `new_turns` yourself to carry conversation state
forward.

## Built-in terminal tools

Every `SwarmAgent` loads a default toolset on construction — `run_bash`,
`read_file`, `write_file`, `find_files`, `grep`, `list_dir`, `edit_file`
(`swarmagent/agent/tools/terminal_tools.py`) — without you having to pass
them explicitly:

```python
agent = SwarmAgent("groq")                            # default tools loaded
agent = SwarmAgent("groq", load_default_tools=False)   # opt out entirely

# extra tools alongside the defaults — `tools` is a per-call argument on
# invoke()/ainvoke(), merged with the defaults (deduped by name)
agent.invoke("...", model="...", tools=[my_tool])
```

`run_bash` runs a command with a 30s timeout, caps stdout/stderr at 200k
characters each, and — because a `shell=True` pipeline forks children the
shell process doesn't own — kills the whole process group (not just the
shell) once a stream crosses that cap, so a runaway `yes` or unbounded `find /`
can't hang the agent or exhaust memory.

Any function decorated `@tool(default=True)` in `swarmagent/agent/tools/`
joins this default set automatically — no changes needed elsewhere.

## Tool-call middleware

`SwarmAgent(middlewares=[...])` lets you intercept every tool result before
it reaches the LLM. `ToolResultSizeLimiter` is on by default:

```python
from swarmagent.agent.middleware.tool_result_limiter import ToolResultSizeLimiter

agent = SwarmAgent("groq")                                   # default limiter, 75k-token cap
agent = SwarmAgent("groq", middlewares=[])                   # opt out entirely
agent = SwarmAgent("groq", middlewares=[
    ToolResultSizeLimiter(max_tokens=20_000),
])
```

If a tool result exceeds the token limit (measured with `tiktoken`,
`cl100k_base`), the full result is written to `~/.swarmagent/tmp/` and the
LLM gets a short message pointing at the file instead — keeping one huge
`run_bash`/`grep` dump from blowing out the context window, while the LLM
can still `read_file`/`grep` the spilled file if it needs specifics.

Write your own middleware by subclassing `ToolMiddleware`
(`swarmagent/agent/middleware/base.py`) and overriding `after_tool_call(name,
arguments, result)` — unused hooks stay no-ops, so a middleware only
implements what it needs.

## Local development

```bash
uv sync
python -m swarmagent.agent.cli   # interactive REPL, type 'exit' to quit
uv run pytest tests/ -v           # run the test suite
```

## Project layout

```
swarmagent/
  agent/
    swarm_agent.py      SwarmAgent — invoke/ainvoke, default-tool + middleware wiring
    cli.py               interactive REPL entrypoint
    tool_registry.py     @tool decorator, provider-agnostic schema build, call_tool dispatch
    middleware/          tool-call middleware (ToolMiddleware base, ToolResultSizeLimiter)
    tools/                built-in tools (terminal_tools.py, testing_tool.py)
    prompt/               system prompt
  config/                env config + logging
  factory/               LLM abstract interface + Anthropic/Groq/OpenAI/OpenRouter implementations
  utils/
tests/                   pytest suite (mirrors swarmagent/ layout)
assets/        shared CSS for lessons/reference docs
lessons/       interactive HTML lessons on agent internals
reference/     cheat sheets (multi-provider agentic loop, etc.)
```

## Architecture

`LLMFactory.register(provider)` returns a concrete `LLM` implementation
(`AnthropicAgent` / `GroqAgent` / `OpenAIAgent` / `OpenRouterAgent`), all implementing the same
`run` / `a_run` interface defined in `swarmagent/factory/factory.py`:

- Manual tool-calling loop with a `max_iterations` safety cap
- `on_iteration` hook so callers can interrupt the loop mid-run without the
  provider needing to know the interruption policy
- Provider-agnostic tool registry (`swarmagent/agent/tool_registry.py`) that
  generates per-provider tool schemas and dispatches calls

## Releasing

Publishing runs via `.github/workflows/publish.yml` on GitHub Actions,
triggered by a GitHub Release (OIDC trusted publishing, no token needed):

1. Bump `version` in `pyproject.toml`
2. Commit and push
3. `gh release create vX.Y.Z`

## Status

Actively under development — see `MISSION.md` for current learning goals and
scope boundaries.
