"""Tests for the default terminal tools (run_bash, read_file, find_files, grep,
write_file, list_dir, edit_file). Each is a plain function under @tool, so we
call the underlying function directly — no LLM/registry involved."""

from __future__ import annotations

import time

from swarmagent.agent.tools.terminal_tools import (
    MAX_OUTPUT_CHARS,
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
    def test_reads_content_lines_and_size(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("line one\nline two\n")

        result = read_file(str(f))

        assert result["success"] is True
        assert result["content"] == "line one\nline two\n"
        assert result["lines"] == ["line one", "line two"]
        assert result["size"] == f.stat().st_size

    def test_missing_file_reports_error(self, tmp_path):
        result = read_file(str(tmp_path / "does_not_exist.txt"))

        assert result["success"] is False
        assert "not found" in result["error"].lower()


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
