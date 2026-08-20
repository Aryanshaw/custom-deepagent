import json
import time
import uuid
from pathlib import Path
from typing import Any

import tiktoken

from swarmagent.agent.middleware.base import ToolMiddleware

DEFAULT_TMP_DIR = Path.home() / ".swarmagent" / "tmp"


class ToolResultSizeLimiter(ToolMiddleware):
    def __init__(self, max_tokens: int = 75_000, tmp_dir: Path = DEFAULT_TMP_DIR):
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_tokens
        self.tmp_dir = tmp_dir

    def after_tool_call(self, name: str, args: dict[str, Any], result: Any) -> Any:
        return self.check_result_tokens(name, result)

    def check_result_tokens(self, tool_name: str, tool_output: Any) -> Any:
        """
        Check the number of tokens in the tool output and spill it to a tmp
        file if it exceeds `self.max_tokens`, returning a short message that
        points the LLM at the file instead.
        """
        tool_result_string = json.dumps(tool_output, default=str, indent=2)
        token_count = len(self.encoder.encode(tool_result_string))

        if token_count <= self.max_tokens:
            return tool_output

        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{tool_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}.txt"
        tmp_file = self.tmp_dir / filename
        tmp_file.write_text(tool_result_string)

        return (
            f"Tool result too large ({token_count} tokens > {self.max_tokens} limit). "
            f"Full output saved to: {tmp_file}. "
            f'Use read_file/grep to inspect it, e.g. read_file(path="{tmp_file}").'
        )
