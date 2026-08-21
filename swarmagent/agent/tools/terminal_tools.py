import subprocess
import os , re , signal
import threading
from pathlib import Path

from swarmagent.config.logger import logger
from swarmagent.agent.tool_registry import tool

MAX_OUTPUT_CHARS = 200_000  # per-stream cap — bounds memory at the source, before the result ever reaches the size-limiter middleware
MAX_READ_LINES = 2000       # read_file's default+max window — a bounded read can never re-trigger the size-limiter on itself
MAX_LINE_CHARS = 2000       # per-line cap within that window (one giant minified line would blow the budget otherwise)
MAX_FIND_RESULTS = 500
MAX_GREP_MATCHES = 200


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the whole process group, not just the shell. With shell=True a
    pipeline (`yes | head`) forks children the shell itself doesn't own —
    proc.kill() alone leaves them running as orphans still holding the pipe
    open, so the reader thread never sees EOF and hangs forever."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass  # already dead


def _read_capped(stream, cap: int, proc: subprocess.Popen) -> tuple[str, bool]:
    """Read a process stream line-by-line, capping collected chars. Kills
    the process group the moment the cap is crossed so a runaway command
    (e.g. `yes`, `find /`) can't buffer unboundedly in memory before we get
    to it."""
    chunks: list[str] = []
    total = 0
    truncated = False
    for line in stream:
        chunks.append(line)
        total += len(line)
        if total >= cap:
            truncated = True
            _kill_process_group(proc)
            break
    return "".join(chunks), truncated


@tool(default=True)
def run_bash(cmd: str):
    """
    Run a bash command and return the output

    Args:
        cmd (str): The bash command to run.

    Returns:
        dict: A dictionary containing the output of the command.
    """
    if len(cmd.strip()) == 0:
        return {
            "stdout": "",
            "stderr": "Command cannot be empty",
            "returncode": 1,
            "success": False
        }

    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.getcwd(),
        env=os.environ,
        start_new_session=True,  # own process group, so we can kill the whole pipeline, not just the shell
    )

    results: dict[str, dict] = {"stdout": {}, "stderr": {}}

    def _collect(stream, target: dict) -> None:
        text, truncated = _read_capped(stream, MAX_OUTPUT_CHARS, proc)
        target["text"] = text
        target["truncated"] = truncated

    stdout_thread = threading.Thread(target=_collect, args=(proc.stdout, results["stdout"]))
    stderr_thread = threading.Thread(target=_collect, args=(proc.stderr, results["stderr"]))
    stdout_thread.start()
    stderr_thread.start()

    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        proc.wait()

    stdout_thread.join()
    stderr_thread.join()

    stdout = results["stdout"]["text"].strip()
    stderr = results["stderr"]["text"].strip()
    if results["stdout"]["truncated"]:
        stdout += f"\n[stdout truncated at {MAX_OUTPUT_CHARS} chars, process killed]"
    if results["stderr"]["truncated"]:
        stderr += f"\n[stderr truncated at {MAX_OUTPUT_CHARS} chars, process killed]"

    return {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": proc.returncode,
        "success": proc.returncode == 0
    }

@tool(default=True)
def find_files(directory: str, pattern: str = "*", limit: int = MAX_FIND_RESULTS):
    """
    Find files matching a pattern in a directory (recursively), capped at
    `limit` results so a broad pattern over a huge tree can't return an
    unbounded list.

    Args:
        directory (str): The directory to search in.
        pattern (str, optional): The pattern to search for. Defaults to "*".
        limit (int, optional): Max results to return (capped at 500). Defaults to 500.

    Returns:
        dict: files found, count, and whether the result was truncated.
    """
    limit = min(max(limit, 1), MAX_FIND_RESULTS)

    results: list[str] = []
    truncated = False
    for p in Path(directory).rglob(pattern):  # rglob = recursive glob
        if not p.is_file():
            continue
        if len(results) >= limit:
            truncated = True
            break
        results.append(str(p))

    return {
        "files": results,
        "count": len(results),
        "truncated": truncated,
    }

@tool(default=True)
def read_file(path: str, offset: int = 0, limit: int = MAX_READ_LINES, encoding: str = "utf-8"):
    """
    Read a file's contents, line-numbered, in a bounded window. Defaults to
    the first 2000 lines; pass `offset` to page through the rest. A single
    call can never return more than `limit` lines (capped at 2000) or more
    than 2000 chars per line — this is what lets it safely re-read a file
    that was itself spilled by the tool-result size limiter, without
    re-triggering that same limiter.

    Args:
        path (str): The path to the file.
        offset (int, optional): 0-indexed line to start reading from. Defaults to 0.
        limit (int, optional): Max lines to return (capped at 2000). Defaults to 2000.
        encoding (str, optional): The encoding of the file. Defaults to "utf-8".

    Returns:
        dict: line-numbered content for the window, start/end line, whether
        more lines remain, and (if so) the offset to pass next.
    """
    limit = min(max(limit, 1), MAX_READ_LINES)
    try:
        lines: list[str] = []
        has_more = False
        with open(path, "r", encoding=encoding, errors="replace") as f:
            for i, raw_line in enumerate(f):
                if i < offset:
                    continue
                if len(lines) >= limit:
                    has_more = True
                    break
                line = raw_line.rstrip("\n")
                if len(line) > MAX_LINE_CHARS:
                    line = line[:MAX_LINE_CHARS] + f"... [line truncated, {len(line)} chars total]"
                lines.append(f"{i + 1}\t{line}")

        start_line = offset + 1
        end_line = offset + len(lines)
        result = {
            "content": "\n".join(lines),
            "start_line": start_line,
            "end_line": end_line,
            "truncated": has_more,
            "size": os.path.getsize(path),
            "success": True,
        }
        if has_more:
            result["note"] = (
                f"Showing lines {start_line}-{end_line}. More lines follow — "
                f"pass offset={end_line} to continue."
            )
        return result
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return {"success": False, "error": f"File not found: {path}"}
    except PermissionError:
        logger.error(f"Permission denied: {path}")
        return {"success": False, "error": f"Permission denied: {path}"}
    except Exception as e:
        logger.error(f"Error in read_file: {e}")
        return {
            "content": "",
            "error": str(e),
            "success": False
        }

@tool(default=True)
def grep(pattern: str, directory: str, ignore_case: bool = False, head_limit: int = MAX_GREP_MATCHES):
    """
    Search for pattern in all files under directory, capped at `head_limit`
    matches so a broad pattern over a huge tree can't return an unbounded
    result.

    Args:
        pattern (str): The pattern to search for.
        directory (str): The directory to search in.
        ignore_case (bool, optional): Whether to ignore case. Defaults to False.
        head_limit (int, optional): Max matching lines to return (capped at 200). Defaults to 200.

    Returns:
        dict: A dictionary containing the search results.
    """
    limit = min(max(head_limit, 1), MAX_GREP_MATCHES)
    try:
        cmd = ["grep", "-rn"]           # -r recursive, -n line numbers
        if ignore_case:
            cmd.append("-i")
        cmd += [pattern, directory]

        result = subprocess.run(cmd, capture_output=True, text=True)
        all_lines = result.stdout.splitlines()
        lines = all_lines[:limit]
        return {
            "lines": lines,
            "match_count": len(lines),
            "truncated": len(all_lines) > limit,
            "returncode": result.returncode
        }
    except (PermissionError, IsADirectoryError,FileNotFoundError, Exception) as e:
        logger.error(f"Error in grep: {e}")
        return {"success": False, "error": str(e)}
    
@tool(default=True)
def write_file(path: str, content: str, encoding: str = "utf-8"):
    """
    Write content to a file

    Args:
        path (str): The path to the file.
        content (str): The content to write to the file.
        encoding (str, optional): The encoding of the file. Defaults to "utf-8".
    
    Returns:
        dict: A dictionary containing the result.
    """
    try:
        with open(path , "w" , encoding=encoding) as f:
            f.write(content)
        logger.info(f"File written successfully: {path}")
        return {"success": True}
    except Exception as e:
        logger.error(f"Error in write_file: {e}")
        return {"success": False, "error": str(e)}

@tool(default=True)
def list_dir(path: str = "."):
    """
    list directory

    Args:
        path (str): The path to the directory to list. Defaults to ".".

    Returns:
        dict: A dictionary containing the list of files and directories.
    """
    try:
        files = os.listdir(path)
        return {
            "files": files,
            "count": len(files),
            "success": True
        }
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return {"success": False, "error": f"File not found: {path}"}
    except PermissionError:
        logger.error(f"Permission denied: {path}")
        return {"success": False, "error": f"Permission denied: {path}"}
    except Exception as e:
        logger.error(f"Error in list_dir: {e}")
        return {
            "files": [],
            "error": str(e),
            "success": False
        }

@tool(default=True)
def edit_file(path: str, old_content: str, new_content: str):
    """
    Replace a specific string/block in a file without rewriting the whole thing

    Args:
        path (str): The path to the file.
        old_content (str): The content to replace.
        new_content (str): The new content.

    Returns:
        dict: A dictionary containing the result.
    """
    try:
        with open(path, "r") as f:
            original = f.read()
    
        if old_content not in original:
            return {"success": False, "error": "Target content not found in file"}

        updated = original.replace(old_content, new_content, 1)  # replace only first occurrence

        with open(path, "w") as f:
            f.write(updated)

        return {"success": True}
    except Exception as e:
        logger.error(f"Error in edit_file: {e}")
        return {"success": False, "error": str(e)}
