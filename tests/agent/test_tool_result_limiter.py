"""Tests for ToolResultSizeLimiter: passthrough under the token limit,
spill-to-file + short message over it."""

from __future__ import annotations

import json

from swarmagent.agent.middleware.tool_result_limiter import ToolResultSizeLimiter


class TestToolResultSizeLimiter:
    def test_passes_through_small_result_unchanged(self, tmp_path):
        limiter = ToolResultSizeLimiter(max_tokens=1000, tmp_dir=tmp_path)
        result = {"stdout": "hello", "stderr": "", "returncode": 0, "success": True}

        out = limiter.after_tool_call("run_bash", {}, result)

        assert out == result  # unchanged, no spill
        assert list(tmp_path.iterdir()) == []

    def test_spills_oversized_result_and_returns_pointer_message(self, tmp_path):
        limiter = ToolResultSizeLimiter(max_tokens=10, tmp_dir=tmp_path)
        result = {"stdout": "word " * 2000, "stderr": "", "returncode": 0, "success": True}

        out = limiter.after_tool_call("run_bash", {}, result)

        assert isinstance(out, str)
        assert "too large" in out
        assert "10" in out  # reports the configured limit
        assert str(tmp_path) in out

        spilled = list(tmp_path.iterdir())
        assert len(spilled) == 1
        written = json.loads(spilled[0].read_text())
        assert written == result  # full, untruncated original result on disk

    def test_token_count_not_char_count(self, tmp_path):
        # Regression: an earlier version compared len(json_string) (chars)
        # to max_tokens instead of len(encoded_tokens). Derive the real
        # token count from the same encoder the limiter uses, then set the
        # limit just above it — a char-count check (chars >> tokens for
        # any real text) would falsely trip here; a correct token check
        # won't.
        limiter = ToolResultSizeLimiter(max_tokens=1, tmp_dir=tmp_path)  # placeholder, fixed below
        result = {"stdout": "word " * 5000, "stderr": "", "returncode": 0, "success": True}
        char_count = len(json.dumps(result, default=str, indent=2))
        real_token_count = len(limiter.encoder.encode(json.dumps(result, default=str, indent=2)))
        assert real_token_count < char_count, "test fixture assumption broke — text must tokenize to fewer units than chars"

        limiter = ToolResultSizeLimiter(max_tokens=real_token_count + 100, tmp_dir=tmp_path)
        out = limiter.after_tool_call("run_bash", {}, result)

        assert out == result  # passes through: real token count is under the limit
        assert list(tmp_path.iterdir()) == []

    def test_filenames_are_unique_per_call(self, tmp_path):
        limiter = ToolResultSizeLimiter(max_tokens=1, tmp_dir=tmp_path)
        result = {"stdout": "x" * 100, "stderr": "", "returncode": 0, "success": True}

        limiter.after_tool_call("run_bash", {}, result)
        limiter.after_tool_call("run_bash", {}, result)

        assert len(list(tmp_path.iterdir())) == 2

    def test_default_tmp_dir_is_home_scoped(self):
        limiter = ToolResultSizeLimiter()
        assert str(limiter.tmp_dir).endswith(".swarmagent/tmp")
