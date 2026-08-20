import subprocess
import os , re
from pathlib import Path

from swarmagent.config.logger import logger
from swarmagent.agent.tool_registry import tool

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
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=os.getcwd(),
        env=os.environ
    )
    return {
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "returncode": result.returncode,
        "success": result.returncode == 0
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
