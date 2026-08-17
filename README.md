# deeperagent-setup

Custom terminal agent built from scratch on top of a pluggable `LLM` factory over
Anthropic, Groq, and OpenAI. Learning-by-building project for understanding the
agentic tool-use loop that every agent framework sits on top of: call model →
check stop reason → run tool calls → append results → repeat.

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- API keys for the providers you use (see below)

## Setup

```bash
uv sync
cp .env.example .env   # if present, otherwise create manually
```

`.env` must define whichever keys the providers you register need:

```
GROQ_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

`app/config/config.py` loads these at import time and raises if a required key
is missing.

## Run

```bash
python -m app.agent.cli
```

Starts an interactive REPL backed by `DeepAgent`. Type `exit` to quit.

## Use as a library

Published as `swarmagent` (currently on TestPyPI as a dry run — not yet on
real PyPI).

```bash
pip install -i https://test.pypi.org/simple/ swarmagent
```

The `-i` flag is required because this package only exists on TestPyPI right
now, a separate index from the real `pypi.org` that `pip install swarmagent`
checks by default. Once published to real PyPI, plain `pip install
swarmagent` will work.

The required env vars (`GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
must be set wherever you install it — `app/config/config.py` reads them at
import time.

```python
from app.agent.deep_agent import DeepAgent

agent = DeepAgent("groq")

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
mutates it; append `new_turns` yourself to carry conversation state forward
(see `main.py` for a full REPL loop example).

## Project layout

```
app/
  agent/       DeepAgent, CLI entrypoint, system prompt, tool registry, tools
  config/      env config + logging
  factory/     LLM abstract interface + Anthropic/Groq/OpenAI implementations
  utils/
assets/        shared CSS for lessons/reference docs
lessons/       interactive HTML lessons on agent internals
reference/     cheat sheets (multi-provider agentic loop, etc.)
learning-records/
```

## Architecture

`LLMFactory.register(provider)` returns a concrete `LLM` implementation
(`AnthropicAgent` / `GroqAgent` / `OpenAIAgent`), all implementing the same
`run` / `a_run` interface defined in `app/factory/factory.py`:

- Manual tool-calling loop with a `max_iterations` safety cap
- `on_iteration` hook so callers can interrupt the loop mid-run without the
  provider needing to know the interruption policy
- Provider-agnostic tool registry (`app/agent/tool_registry.py`) that
  generates per-provider tool schemas and dispatches calls

## Status

Actively under development — see `MISSION.md` for current learning goals and
scope boundaries.
