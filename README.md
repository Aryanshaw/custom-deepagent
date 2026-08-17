# swarmagent

Pluggable multi-provider LLM agent (Anthropic, Groq, OpenAI) built from
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
```

## Usage

```python
from swarmagent import SwarmAgent

agent = SwarmAgent("groq")

response_stream, new_turns = agent._invoke(
    "hello",
    model="qwen/qwen3.6-27b",
    history=[],
    max_tokens=2000,
    stream=True,
)

for chunk in response_stream:
    print(chunk, end="", flush=True)
```

`history` is a list of `Turn` objects you own — `_invoke` reads it but never
mutates it; append `new_turns` yourself to carry conversation state forward.

## Local development

```bash
uv sync
python -m swarmagent.agent.cli   # interactive REPL, type 'exit' to quit
```

## Project layout

```
swarmagent/
  agent/       SwarmAgent, CLI entrypoint, system prompt, tool registry, tools
  config/      env config + logging
  factory/     LLM abstract interface + Anthropic/Groq/OpenAI implementations
  utils/
assets/        shared CSS for lessons/reference docs
lessons/       interactive HTML lessons on agent internals
reference/     cheat sheets (multi-provider agentic loop, etc.)
```

## Architecture

`LLMFactory.register(provider)` returns a concrete `LLM` implementation
(`AnthropicAgent` / `GroqAgent` / `OpenAIAgent`), all implementing the same
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
