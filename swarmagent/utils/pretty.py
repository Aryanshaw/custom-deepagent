"""ANSI box-drawing for verbose trace output. Stdlib only — no color lib."""

import json
import shutil
import textwrap
from typing import Any

_RESET = "\033[0m"
_BOLD = "\033[1m"
_COLORS = {
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "red": "\033[31m",
    "magenta": "\033[35m",
}


def _term_width() -> int:
    return min(shutil.get_terminal_size(fallback=(100, 24)).columns, 100)


def print_box(title: str, body: str, color: str = "cyan") -> None:
    width = _term_width()
    inner = width - 4
    c = _COLORS.get(color, "")

    print(f"{c}{_BOLD}┌─ {title} " + "─" * max(0, inner - len(title) - 1) + f"┐{_RESET}")
    for raw_line in body.splitlines() or [""]:
        for line in textwrap.wrap(raw_line, inner) or [""]:
            print(f"{c}│{_RESET} {line.ljust(inner)} {c}│{_RESET}")
    print(f"{c}{_BOLD}└" + "─" * (width - 2) + f"┘{_RESET}")


def print_system_prompt(prompt: str) -> None:
    print_box("SYSTEM PROMPT", prompt, color="cyan")


def print_tool_call(name: str, arguments: dict[str, Any]) -> None:
    print_box(f"TOOL CALL · {name}", json.dumps(arguments, indent=2, default=str), color="yellow")


def print_tool_result(name: str, result: Any, is_error: bool) -> None:
    title = f"TOOL RESULT · {name}" + (" (error)" if is_error else "")
    print_box(title, str(result), color="red" if is_error else "green")


def print_output(text: str) -> None:
    print_box("OUTPUT", text, color="magenta")
