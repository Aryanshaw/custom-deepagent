import subprocess
import os , re , signal
import threading
from pathlib import Path

from swarmagent.config.logger import logger
from swarmagent.agent.tool_registry import tool

MAX_OUTPUT_CHARS = 200_000  # per-stream cap — bounds memory at the source, before the result ever reaches the size-limiter middleware


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
def find_files(directory: str, pattern: str = "*"):
    """
    Find files matching a pattern in a directory (recursively)
    
    Args:
        directory (str): The directory to search in.
        pattern (str, optional): The pattern to search for. Defaults to "*".
    
    Returns:
        dict: A dictionary containing the list of files.
    """

    # pathlib appraoch (modern, clean)
    results = list(Path(directory).rglob(pattern)) # rglob = recursive glob 

    results = [str(p) for p in results if p.is_file()]
    
    return {
        "files": results,
        "count": len(results)
    }

@tool(default=True)
def read_file(path: str, encoding: str = "utf-8"):
    """
    Read a file and return its contents

    Args:
        path (str): The path to the file.
        encoding (str, optional): The encoding of the file. Defaults to "utf-8".
    
    Returns:
        dict: A dictionary containing the file contents.
    """
    try:
        with open(path, "r", encoding=encoding) as f:
            content = f.read()
        # detect encoding of the file if not provided

        return {
            "content": content,
            "lines": content.splitlines(),  # list of lines, no \n
            "size": os.path.getsize(path),
            "success": True
        }
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
def grep(pattern: str, directory: str, ignore_case: bool = False):
    """
    Search for pattern in all files under directory

    Args:
        pattern (str): The pattern to search for.
        directory (str): The directory to search in.
        ignore_case (bool, optional): Whether to ignore case. Defaults to False.
    
    Returns:
        dict: A dictionary containing the search results.
    """

    # flags = re.IGNORECASE if ignore_case else 0
    # matches = []

    # for root , _, files in os.walk(directory):
    #     for file in files:
    #         filepath = os.path.join(root , file)
    #         try:
    #             with open(filepath , "r", encoding="utf-8", errors="ignore") as f:
    #                 for line_num , line in enumerate(f , start=1):
    #                     if re.search(pattern , line , flags):
    #                         matches.append({
    #                             "file": filepath,
    #                             "line_num": line_num,
    #                             "line": line.strip()
    #                         })
    #         except (PermissionError, IsADirectoryError):
    #             continue
    
    # return {"matches": matches , "count" : len(matches)}
    try:
        cmd = ["grep", "-rn"]           # -r recursive, -n line numbers
        if ignore_case:
            cmd.append("-i")
        cmd += [pattern, directory]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "raw": result.stdout,
            "lines": result.stdout.splitlines(),
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
