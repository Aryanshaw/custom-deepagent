# Mission: Agentic Tool-Use Loops

## Why
You're building `deeperagent-setup` — a terminal agent with a pluggable `DeepAgent`/`LLM` factory over Anthropic, Groq, and OpenAI. Right now each provider's `run`/`a_run` does exactly one model call: send a prompt, get one response back, done. To make this an actual *agent* — one that can look something up, act on what it finds, and keep going — it needs to loop: call the model, notice it wants a tool, run the tool, feed the result back, and repeat until the model is done. That loop is the one piece of machinery every agent framework is built on top of.

## Success looks like
- You can explain, from memory, the four things every iteration of the loop does (call model → check why it stopped → run any tool calls → append results and repeat).
- You can read a response's stop/finish reason and correctly decide "loop again" vs "return to caller" for Anthropic (`stop_reason`) and for the OpenAI-shaped APIs Groq/OpenAI use (`finish_reason` + `tool_calls`).
- You've implemented a working manual loop inside `AnthropicAgent.run` (or a shared helper), with a max-iteration safety cap, and traced through what happens when a tool call fails.
- You can name the difference between "workflow" (fixed code path) and "agent" (model decides the next step) and say which one a given feature of your project actually needs.

## Constraints
- Learning happens inside the real `deeperagent-setup` codebase — exercises should produce code you can keep, not throwaway snippets.
- Keep sessions short; you're mid-build on this project, not doing a course.
- Prefer terse, technical explanations (caveman mode) over long prose.

## Out of scope
- Managed Agents / hosted agent platforms (Anthropic's server-run sessions) — you're building your own loop, not using theirs.
- Multi-agent orchestration / subagents.
- Streaming token-by-token UX polish — get the non-streaming loop correct first.
