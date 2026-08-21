"""Tests for the default terminal tools (run_bash, read_file, find_files, grep,
write_file, list_dir, edit_file). Each is a plain function under @tool, so we
call the underlying function directly — no LLM/registry involved."""

from __future__ import annotations

import time

from swarmagent.agent.tools.terminal_tools import (
    MAX_GREP_MATCHES,
    MAX_LINE_CHARS,
    MAX_OUTPUT_CHARS,
    MAX_READ_LINES,
    edit_file,
    find_files,
    grep,
    list_dir,
    read_file,
    run_bash,
    write_file,
)


class TestRunBash:
    def test_basic_stdout_and_stderr(self):
        result = run_bash("echo hello && echo world 1>&2")
        assert result["stdout"] == "hello"
        assert result["stderr"] == "world"
        assert result["returncode"] == 0
        assert result["success"] is True

    def test_empty_command_rejected(self):
        result = run_bash("   ")
        assert result["success"] is False
        assert "empty" in result["stderr"].lower()

    def test_nonzero_exit_code(self):
        result = run_bash("exit 3")
        assert result["returncode"] == 3
        assert result["success"] is False

    def test_runaway_output_is_capped_and_killed_promptly(self):
        # `yes` would print forever; without the process-group kill this
        # hangs indefinitely (killing only the shell leaves orphaned
        # pipeline children holding the pipe open).
        start = time.monotonic()
        result = run_bash("yes | head -c 100000000")
        elapsed = time.monotonic() - start

        assert elapsed < 10, "runaway command wasn't killed promptly"
        assert len(result["stdout"]) <= MAX_OUTPUT_CHARS + 200
        assert "truncated" in result["stdout"]
        assert result["success"] is False  # killed => nonzero/negative returncode


class TestReadFile:
    def test_reads_content_line_numbered_and_size(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("line one\nline two\n")

        result = read_file(str(f))

        assert result["success"] is True
        assert result["content"] == "1\tline one\n2\tline two"
        assert result["start_line"] == 1
        assert result["end_line"] == 2
        assert result["truncated"] is False
        assert result["size"] == f.stat().st_size

    def test_missing_file_reports_error(self, tmp_path):
        result = read_file(str(tmp_path / "does_not_exist.txt"))

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_window_is_bounded_and_pageable(self, tmp_path):
        # Regression: read_file used to read the whole file unconditionally
        # — including a file the size-limiter itself spilled, causing an
        # infinite spill -> read -> re-spill loop. A bounded window fixes
        # that structurally: a single call can never return more than
        # MAX_READ_LINES lines.
        f = tmp_path / "big.txt"
        total_lines = MAX_READ_LINES + 500
        f.write_text("\n".join(f"line {i}" for i in range(total_lines)))

        first = read_file(str(f))
        assert first["truncated"] is True
        assert first["start_line"] == 1
        assert first["end_line"] == MAX_READ_LINES
        assert first["content"].count("\n") == MAX_READ_LINES - 1
        assert "note" in first

        second = read_file(str(f), offset=first["end_line"])
        assert second["start_line"] == MAX_READ_LINES + 1
        assert second["end_line"] == total_lines
        assert second["truncated"] is False

    def test_single_long_line_is_truncated(self, tmp_path):
        f = tmp_path / "minified.txt"
        f.write_text("x" * (MAX_LINE_CHARS * 3))

        result = read_file(str(f))

        assert "truncated" in result["content"]
        assert len(result["content"]) < MAX_LINE_CHARS * 3

    def test_limit_argument_is_capped(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line {i}" for i in range(MAX_READ_LINES + 500)))

        result = read_file(str(f), limit=999_999)  # try to request way more than the cap

        assert result["end_line"] == MAX_READ_LINES


class TestFindFiles:
    def test_finds_matching_files_recursively(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("")
        (tmp_path / "notes.txt").write_text("")

        result = find_files(str(tmp_path), "*.py")

        names = {p.split("/")[-1] for p in result["files"]}
        assert names == {"a.py", "b.py"}
        assert result["count"] == 2

    def test_no_matches_returns_empty(self, tmp_path):
        result = find_files(str(tmp_path), "*.missing")
        assert result["files"] == []
        assert result["count"] == 0

    def test_results_are_capped(self, tmp_path):
        for i in range(20):
            (tmp_path / f"f{i}.txt").write_text("")

        result = find_files(str(tmp_path), "*.txt", limit=5)

        assert result["count"] == 5
        assert result["truncated"] is True


class TestGrep:
    def test_finds_pattern_with_line_numbers(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def foo():\n    return needle\n")

        result = grep("needle", str(tmp_path))

        assert result["returncode"] == 0
        assert any("needle" in line for line in result["lines"])

    def test_no_match_returns_nonzero(self, tmp_path):
        (tmp_path / "code.py").write_text("nothing interesting here\n")

        result = grep("needle_not_present", str(tmp_path))

        assert result["returncode"] != 0
        assert result["lines"] == []

    def test_ignore_case(self, tmp_path):
        (tmp_path / "code.py").write_text("NEEDLE\n")

        result = grep("needle", str(tmp_path), ignore_case=True)

        assert any("NEEDLE" in line for line in result["lines"])

    def test_matches_are_capped(self, tmp_path):
        f = tmp_path / "many.txt"
        f.write_text("\n".join("needle" for _ in range(MAX_GREP_MATCHES + 50)))

        result = grep("needle", str(tmp_path), head_limit=10)

        assert result["match_count"] == 10
        assert result["truncated"] is True

    def test_no_unbounded_raw_field(self, tmp_path):
        # Regression: grep used to also return a "raw" field holding the
        # full, uncapped stdout — capping "lines" alone didn't actually
        # bound the result.
        f = tmp_path / "many.txt"
        f.write_text("\n".join("needle" for _ in range(MAX_GREP_MATCHES + 50)))

        result = grep("needle", str(tmp_path), head_limit=10)

        assert "raw" not in result


class TestWriteFile:
    def test_writes_content(self, tmp_path):
        f = tmp_path / "out.txt"

        result = write_file(str(f), "hello world")

        assert result["success"] is True
        assert f.read_text() == "hello world"


class TestListDir:
    def test_lists_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("")
        (tmp_path / "b.txt").write_text("")

        result = list_dir(str(tmp_path))

        assert result["success"] is True
        assert set(result["files"]) == {"a.txt", "b.txt"}
        assert result["count"] == 2

    def test_missing_dir_reports_error(self, tmp_path):
        result = list_dir(str(tmp_path / "nope"))
        assert result["success"] is False


class TestEditFile:
    def test_replaces_first_occurrence(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("x = 1\nx = 1\n")

        result = edit_file(str(f), "x = 1", "x = 2")

        assert result["success"] is True
        assert f.read_text() == "x = 2\nx = 1\n"

    def test_missing_target_leaves_file_untouched(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("unchanged\n")

        result = edit_file(str(f), "not present", "replacement")

        assert result["success"] is False
        assert f.read_text() == "unchanged\n"
