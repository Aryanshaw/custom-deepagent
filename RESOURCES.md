# Agentic Loops — Resources

## Knowledge

- [Building Effective Agents — Anthropic Engineering (Dec 2024)](https://www.anthropic.com/engineering/building-effective-agents)
  The primary source for the whole mission. Defines the workflow-vs-agent distinction and states the core loop plainly: "Agents ... are typically just LLMs using tools based on environmental feedback in a loop." Use for: conceptual grounding, deciding whether a feature needs an agent or a fixed workflow, stopping-condition guidance.
- [Anthropic Tool Use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
  Official mechanics: tool definitions, `tool_choice`, `stop_reason` values, how `tool_result` blocks must be returned. Use for: exact request/response shapes when implementing the loop against the Messages API.
- [Anthropic — Handling stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)
  Use for: the full `stop_reason` state machine (`end_turn`, `tool_use`, `max_tokens`, `pause_turn`, `refusal`) and what each demands from your loop.
- `./reference/agentic-loop-cheatsheet.html` (this workspace)
  Compressed skeleton loop + stop/finish-reason table for all three providers you're wiring up (Anthropic, OpenAI, Groq). Built from the two sources above plus the OpenAI/Groq `tool_calls` shape. Reach for this while coding.

## Wisdom (Communities)

## Gaps
- No community picked yet. Anthropic runs a developer Discord and there's activity on r/ClaudeAI and r/LocalLLaMA for agent-loop patterns generally — not yet verified as high-signal for this specific topic. Revisit once you hit a real design question (e.g. "how do people cap runaway tool-call loops in production") — that's the moment a community actually pays off, not before.
