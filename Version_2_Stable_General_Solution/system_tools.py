"""
System-level tools and helpers factored out of the debugging notebooks.

These are treated as infrastructure and should NOT themselves be debugged
by the automated agent (they are "system_*" files for easy filtering).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os
import platform
import re
import subprocess
import sys

from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain.tools import tool as lc_tool


# ---------------------------------------------------------------------------
# Output recording
# ---------------------------------------------------------------------------


class RecordOutputs(BaseModel):
    """Save script execution results (stdout, stderr, exit code)."""

    stdout: str = Field(..., description="Standard output text from the script")
    stderror: str = Field(..., description="Error output text from the script")
    exitcode: int = Field(..., description="Exit code integer (0=success, non-zero=error)")


@tool(args_schema=RecordOutputs)
def record_outputs(stdout: str = "", stderror: str = "", exitcode: int = 0) -> str:
    """Records and appends the current outputs to a JSON tracking file."""
    file_path = "current_outputs.json"

    outputs_to_save = {
        "stdout": stdout,
        "stderror": stderror,
        "exitcode": exitcode,
    }

    all_records: List[Dict[str, Any]] = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as file:
                all_records = json.load(file)
        except Exception:
            all_records = []

    all_records.append(outputs_to_save)

    with open(file_path, "w") as file:
        json.dump(all_records, file, indent=4)

    return f"Success: Recorded entry. File now has {len(all_records)} items."


# ---------------------------------------------------------------------------
# System configuration snapshot
# ---------------------------------------------------------------------------


class ConfigInput(BaseModel):
    """Record system configuration ONLY at startup."""

    config_name: str = Field(
        default="startup_config",
        description="Name identifying this config snapshot, e.g. 'startup_config'",
    )
    config_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="System configuration key-value pairs",
    )


@tool(args_schema=ConfigInput)
def record_system_config(
    config_name: str = "startup_config", config_data: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Records system configuration (OS, Python, Torch, CUDA) to system_config.json."""

    if config_data is None:
        config_data = {}

    file_path = "system_config.json"
    all_records: Dict[str, Any] = {}

    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as file:
                all_records = json.load(file)
        except Exception:
            all_records = {}

    all_records.update(config_data)

    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": "Not installed",
        "cuda_available": False,
        "nvcc_version": "Not found",
        "current_dir": os.getcwd(),
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = str(torch.cuda.get_device_capability(0))
    except ImportError:
        pass

    try:
        nvcc_result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=False
        )
        if nvcc_result.returncode == 0:
            info["nvcc_version"] = nvcc_result.stdout.strip().split("\n")[0]
    except FileNotFoundError:
        pass

    all_records.update(info)

    with open(file_path, "w") as file:
        json.dump(all_records, file, indent=4)

    return all_records


# ---------------------------------------------------------------------------
# File listing / type detection
# ---------------------------------------------------------------------------


class ListFolderInput(BaseModel):
    """List files in the current directory and their format."""

    pass


@tool(args_schema=ListFolderInput)
def list_files() -> Dict[str, str]:
    """List all files in the current directory with basic type info."""

    import magic  # type: ignore[import-not-found]

    file_list: Dict[str, str] = {}
    m = magic.Magic()
    for f in os.listdir("."):
        try:
            file_list[f] = m.from_file(f)
        except IsADirectoryError:
            file_list[f] = "Folder"

    return file_list


# ---------------------------------------------------------------------------
# Terminal command executor
# ---------------------------------------------------------------------------


class ExecuteTerminalCommand(BaseModel):
    """Execute a general terminal command and capture the output."""

    command: str = Field(..., description="Command to execute")


@tool(args_schema=ExecuteTerminalCommand)
def terminal_command_executor(command: str) -> Dict[str, Any]:
    """Execute a terminal command and return stdout, stderr and exit code."""

    output_dict: Dict[str, Any] = {}
    parts = command.split()
    output_dict["command"] = parts

    try:
        result = subprocess.run(
            parts, capture_output=True, text=True, check=True
        )
        output_dict["stdout"] = result.stdout
        output_dict["stderror"] = result.stderr
        output_dict["exit_code"] = result.returncode
    except subprocess.CalledProcessError as e:
        output_dict["stdout"] = e.stdout
        output_dict["stderror"] = e.stderr
        output_dict["exit_code"] = e.returncode

    file_path = "current_outputs.json"
    all_records: List[Dict[str, Any]] = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as file:
                all_records = json.load(file)
        except Exception:
            all_records = []

    all_records.append(output_dict)

    with open(file_path, "w") as file:
        json.dump(all_records, file, indent=4)

    return output_dict


# ---------------------------------------------------------------------------
# File reader / writer tools
# ---------------------------------------------------------------------------


class ReadFileInput(BaseModel):
    """Read file contents (code/data aware) with optional error-line focus."""

    file_path: str = Field(
        ..., description="Path of the file to read, e.g. 'bug.py'"
    )
    file_type: Optional[str] = Field(
        default=None,
        description="Optional file type hint like 'Python', 'JSON', 'CSV'",
    )
    error_line: Optional[int] = Field(
        default=0,
        description="Error line (0-based) to focus context around for long files",
    )
    window_before: int = Field(description="Number of lines to show before the error line")
    window_after: int = Field(description="Number of lines to show after the error line")

@lc_tool(args_schema=ReadFileInput)
@lc_tool(args_schema=ReadFileInput)
def read_file(
    file_path: str, 
    file_type: Optional[str] = None, 
    error_line: Optional[int] = None,
    window_before: int = 20,  # ADD THIS
    window_after: int = 30,   # ADD THIS
) -> Dict[str, Any]:
    """Read a file and return structured content based on type."""

    result: Dict[str, Any] = {
        "success": False,
        "file_path": file_path,
        "file_type": file_type,
        "content": None,
        "error": None,
    }

    try:
        if not os.path.exists(file_path):
            result["error"] = f"File not found: {file_path}"
            return result

        if os.path.isdir(file_path):
            result["error"] = f"{file_path} is a directory, not a file"
            return result

        result["size_bytes"] = os.path.getsize(file_path)

        if file_type is None:
            import magic  # type: ignore[import-not-found]

            m = magic.Magic()
            file_type = m.from_file(file_path)
            result["file_type"] = file_type

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == ".json" or (file_type and "JSON" in file_type):
            with open(file_path, "r", encoding="utf-8") as f:
                result["content"] = json.load(f)
            result["file_type"] = "JSON"
            result["success"] = True

        elif ext == ".csv" or (file_type and "CSV" in file_type):
            import csv

            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            result["content"] = {
                "headers": reader.fieldnames if reader.fieldnames else [],
                "rows": rows,
                "row_count": len(rows),
            }
            result["file_type"] = "CSV"
            result["success"] = True

        elif ext in [
            ".py",
            ".txt",
            ".md",
            ".js",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".sh",
            ".yaml",
            ".yml",
            ".xml",
            ".html",
            ".css",
        ]:
            encodings = ["utf-8", "latin-1", "cp1252"]
            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                    result["file_type"] = file_type or f"Text ({ext})"
                    lines = content.split("\n")  # Raw lines WITHOUT numbering
                    
                    start_val = 0
                    end_val = len(lines)

                    if error_line is not None and error_line > 0 and len(lines) > 100:
                        print("using smaller portion")
                        
                        # Convert to 0-indexed
                        err_idx = error_line - 1
                        
                        # Ensure err_idx is within bounds
                        err_idx = max(0, min(err_idx, len(lines) - 1))
                        
                        # Start: search upward for function/class start, but not more than 50 lines
                        start_val = max(0, err_idx - window_before - 30)
                        for i in range(err_idx, start_val - 1, -1):
                            # Search on RAW lines, not numbered
                            line_content = lines[i].lstrip()
                            if line_content.startswith("def ") or line_content.startswith("class "):
                                start_val = i
                                break
                        
                        # CRITICAL: Ensure we show at least 10 lines BEFORE error
                        # This ensures context is visible even if function def is far away
                        start_val = min(start_val, max(0, err_idx - window_before))
                        
                        # End: at least 20 lines AFTER error_line, or next function
                        end_val = min(len(lines), err_idx + window_after)
                        for i in range(err_idx + 1, end_val):
                            line_content = lines[i].lstrip()
                            if line_content.startswith("def ") or line_content.startswith("class "):
                                end_val = i
                                break
                        
                        # CRITICAL: Always ensure error_line is included with context
                        if err_idx < start_val:
                            start_val = max(0, err_idx - 10)
                        if err_idx >= end_val:
                            end_val = min(len(lines), err_idx + 20)
                        
                        # Add line numbers AFTER slicing
                        numbered = [f"{i+1:5d} | {lines[i]}" for i in range(start_val, end_val)]
                        result["content"] = "\n".join(numbered)
                        
                        # Add metadata about what we're showing
                        result["shown_lines"] = f"{start_val + 1}-{end_val}"
                        result["error_line_in_snippet"] = error_line
                    else:
                        # Full file with line numbers
                        numbered = [f"{i+1:5d} | {line}" for i, line in enumerate(lines)]
                        result["content"] = "\n".join(numbered)

                    result["success"] = True
                    break
                except UnicodeDecodeError:
                    continue

            if not result["success"]:
                result["error"] = "Could not decode file with any standard encoding"

        elif ext in [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".pdf",
            ".zip",
            ".tar",
            ".gz",
        ]:
            result["error"] = (
                f"Binary file type {ext} cannot be read as text. File type: {file_type}"
            )
            result["file_type"] = file_type

        else:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                result["content"] = content
                result["file_type"] = file_type or "Text"
                result["success"] = True
            except UnicodeDecodeError:
                result["error"] = (
                    f"File appears to be binary or uses unsupported encoding. "
                    f"File type: {file_type}"
                )

    except Exception as e:
        result["error"] = f"Error reading file: {e}"

    return result

class WriteFileInput(BaseModel):
    """Edit or create a file by replacing specific lines with new content."""

    file_path: str = Field(..., description="Path to the file to edit, e.g. 'bug.py'")
    start_line: Optional[int] = Field(
        default=None,
        description="Starting line number (1-indexed). None = replace entire file.",
    )
    end_line: Optional[int] = Field(
        default=None,
        description="Ending line number (1-indexed, inclusive). None = entire file.",
    )
    new_content: str = Field(
        ...,
        description="New code/text to insert. Can be multi-line.",
    )
    create_backup: bool = Field(
        default=True,
        description="Whether to create a backup file before editing.",
    )


@lc_tool(args_schema=WriteFileInput)
def write_file(
    file_path: str,
    new_content: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    create_backup: bool = True,
) -> Dict[str, Any]:
    """Edit a file by replacing specific lines or entire content."""

    result: Dict[str, Any] = {
        "success": False,
        "file_path": file_path,
        "action": None,
        "lines_replaced": 0,
        "new_total_lines": 0,
        "backup_path": None,
        "error": None,
        "preview": None,
    }

    try:
        file_exists = os.path.exists(file_path)
        if not file_exists:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            result["success"] = True
            result["action"] = "created"
            result["new_total_lines"] = len(new_content.split("\n"))
            result["preview"] = (
                f"Created new file with {result['new_total_lines']} lines"
            )
            return result

        with open(file_path, "r", encoding="utf-8") as f:
            original_lines = f.readlines()

        original_line_count = len(original_lines)

        if create_backup:
            backup_path = f"{file_path}.backup"
            counter = 1
            while os.path.exists(backup_path):
                backup_path = f"{file_path}.backup{counter}"
                counter += 1
            with open(backup_path, "w", encoding="utf-8") as f:
                f.writelines(original_lines)
            result["backup_path"] = backup_path

        # Split and handle newlines carefully to avoid creating empty lines
        new_lines = new_content.split("\n")
        
        # Remove trailing empty string if content ended with \n
        if new_lines and new_lines[-1] == "":
            new_lines = new_lines[:-1]
        
        # Add newlines back to each line
        new_lines = [line + "\n" for line in new_lines]
        
        # Ensure last line matches original file's ending
        if new_lines and original_lines:
            if not original_lines[-1].endswith("\n"):
                new_lines[-1] = new_lines[-1].rstrip("\n")

        if start_line is None and end_line is None:
            modified_lines = new_lines
            result["action"] = "replaced_entire_file"
            result["lines_replaced"] = original_line_count
        else:
            if start_line is not None and start_line < 1:
                result["error"] = f"start_line must be >= 1, got {start_line}"
                return result
            if end_line is not None and end_line < 1:
                result["error"] = f"end_line must be >= 1, got {end_line}"
                return result
            if (
                start_line is not None
                and end_line is not None
                and start_line > end_line
            ):
                result["error"] = (
                    f"start_line ({start_line}) must be <= end_line ({end_line})"
                )
                return result

            start_idx = (start_line - 1) if start_line is not None else 0
            end_idx = end_line if end_line is not None else original_line_count

            if start_idx >= original_line_count:
                result["error"] = (
                    f"start_line {start_line} is beyond file length "
                    f"({original_line_count} lines)"
                )
                return result
            if end_idx > original_line_count:
                result["error"] = (
                    f"end_line {end_line} is beyond file length "
                    f"({original_line_count} lines)"
                )
                return result

            modified_lines = (
                original_lines[:start_idx] + new_lines + original_lines[end_idx:]
            )
            result["action"] = "replaced_lines"
            result["lines_replaced"] = end_idx - start_idx

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(modified_lines)

        result["success"] = True
        result["new_total_lines"] = len(modified_lines)

        if start_line is not None and end_line is not None:
            start_idx = (start_line - 1) if start_line is not None else 0
            preview_start = max(0, start_idx - 2)
            preview_end = min(len(modified_lines), start_idx + len(new_lines) + 2)
            preview_lines: List[str] = []
            for i in range(preview_start, preview_end):
                line_num = i + 1
                line_content = modified_lines[i].rstrip("\n")
                if start_idx <= i < start_idx + len(new_lines):
                    preview_lines.append(f"  {line_num:4d} + | {line_content}")
                else:
                    preview_lines.append(f"  {line_num:4d}   | {line_content}")
            result["preview"] = "\n".join(preview_lines)
        else:
            preview_lines = [
                f"  {i+1:4d} | {line.rstrip()}"
                for i, line in enumerate(modified_lines[:10])
            ]
            if len(modified_lines) > 10:
                preview_lines.append(
                    f"  ... ({len(modified_lines) - 10} more lines)"
                )
            result["preview"] = "\n".join(preview_lines)

    except Exception as e:
        result["error"] = f"Error writing file: {e}"

    return result


# ---------------------------------------------------------------------------
# Script runner and error helpers
# ---------------------------------------------------------------------------


class RunScriptInput(BaseModel):
    """Execute a Python script and return output and exit code."""

    script_path: str = Field(..., description="Path to Python file")
    args: List[str] = Field(default_factory=list, description="Command-line arguments")


@tool(args_schema=RunScriptInput)
def run_script(script_path: str, args: List[str] | None = None) -> Dict[str, Any]:
    """Run a Python script and capture stdout/stderr/exit code."""

    if args is None:
        args = []

    print(f"--- Starting Execution: {script_path} ---")
    command = [sys.executable, script_path] + args
    outputs: Dict[str, Any] = {
        "command": command,
        "stderror": None,
        "exitcode": None,
        "stdout": None,
    }

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
        outputs["stdout"] = result.stdout
        outputs["exitcode"] = result.returncode
        outputs["stderror"] = result.stderr

        if result.returncode == 0:
            print(f"✅ Success! Exit Code: {result.returncode}")
        else:
            print(f"❌ Script failed! Exit Code: {result.returncode}")
    except FileNotFoundError:
        error_msg = f"Error: The file '{script_path}' was not found."
        print(error_msg)
        outputs = {"exitcode": 1, "stderror": error_msg, "stdout": None}
    except Exception as e:  # pragma: no cover - defensive
        error_msg = f"An unexpected error occurred: {e}"
        print(error_msg)
        outputs = {"exitcode": 1, "stderror": error_msg, "stdout": None}

    file_path = "current_outputs.json"
    all_records: List[Dict[str, Any]] = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as file:
                all_records = json.load(file)
        except Exception:
            all_records = []

    all_records.append(outputs)

    with open(file_path, "w") as file:
        json.dump(all_records, file, indent=4)

    return outputs

def error_analyzer(error: str, error_history: List[Dict] = None, current_depth: int = 0) -> Dict[str, Any]:
    """
    Analyze Python traceback to locate error and file/line context.
    
    If error_history shows we've seen this error location before,
    go back through the call stack to find the real source.
    """

    counts = error.count("File")
    my_pattern = r'File "([^"]+)", line (\d+)'
    matches = re.findall(my_pattern, error)
    current_cwd = os.getcwd()

    # Clean up paths
    cleaned_matches = []
    for file_path, line_num in matches:
        clean_path = file_path.replace(current_cwd, "").lstrip("/")
        cleaned_matches.append((clean_path, int(line_num)))

    error_flow = cleaned_matches[0][0] if cleaned_matches else ""
    for file_path, _ in cleaned_matches[1:]:
        if not error_flow.endswith(file_path):
            error_flow += f"->{file_path}"

    single_file_error = len(cleaned_matches) == len(set(f for f, _ in cleaned_matches))

    # Determine which stack frame to use
    # Default: last frame (deepest in stack)
    # If we've seen this error before: go up the stack
    target_index = -1  # Default to last
    
    if error_history and cleaned_matches:
        # Check if the last error location matches current last frame
        last_file, last_line = cleaned_matches[-1]
        
        # Count how many times we've hit this exact location
        repeat_count = 0
        for prev_error in error_history:
            prev_file = prev_error.get("file", "")
            prev_line = prev_error.get("line", 0)
            if prev_file == last_file and prev_line == last_line:
                repeat_count += 1
        
        # If repeated, go back in the stack
        if repeat_count > 0 and len(cleaned_matches) > 1:
            # Go back one frame for each repeat, but don't exceed stack depth
            target_index = max(-len(cleaned_matches), -(repeat_count + 1))
            print(f"   🔄 Error repeated {repeat_count}x at {last_file}:{last_line}, checking caller (frame {target_index})")
    
    # Get the target frame
    if cleaned_matches:
        target_file, target_line = cleaned_matches[target_index]
    else:
        target_file, target_line = "", 0

    # Parse the error message
    parsed = re.split("File", error)
    last_error = "File " + parsed[-1] if len(parsed) > 1 else error
    lines = last_error.split("\n")
    last_line_msg = lines[-2] if len(lines) >= 2 else ""

    # Extract error type and message
    error_type = ""
    error_message = ""
    for line in reversed(error.split("\n")):
        line = line.strip()
        if line and ":" in line and not line.startswith("File"):
            parts = line.split(":", 1)
            error_type = parts[0].strip()
            error_message = parts[1].strip() if len(parts) > 1 else ""
            break

    module_error_type = ""
    missing_module = ""
    
    if "ModuleNotFoundError" in error:
        module_pattern = r"No module named ['\"]([^'\"]+)['\"]"
        module_match = re.search(module_pattern, error)
        if module_match:
            missing_module = module_match.group(1)
            top_level_module = missing_module.split(".")[0]
            
            files = list_files.invoke({})
            is_local_file = False
            for fname in files.keys():
                if fname == f"{top_level_module}.py" or fname == top_level_module:
                    is_local_file = True
                    break
            
            if is_local_file:
                module_error_type = "File"
            else:
                module_error_type = "Lib"
            
            print(f"   📦 Detected missing module: {missing_module} (type: {module_error_type})")

    # Check if target line is a 'raise' statement
    is_raise_statement = False
    caller_info = None
    try:
        with open(target_file, 'r') as f:
            file_lines = f.readlines()
            if 0 < target_line <= len(file_lines):
                target_code = file_lines[target_line - 1].strip()
                is_raise_statement = target_code.startswith("raise ")
                
                # If it's a raise, get the caller info
                if is_raise_statement and len(cleaned_matches) > 1:
                    caller_idx = target_index - 1 if target_index > -len(cleaned_matches) else target_index
                    caller_file, caller_line = cleaned_matches[caller_idx]
                    caller_info = {"file": caller_file, "line": caller_line}
    except:
        pass

    return {
        "error_counts": counts,
        "error_line": target_line,
        "file": target_file,
        "last_error": last_error,
        "is_single_error_file": single_file_error,
        "error_flow": error_flow,
        "module_error": module_error_type,
        "missing_module": missing_module,
        "error_type": error_type,
        "error_message": error_message,
        "is_raise_statement": is_raise_statement,
        "caller_info": caller_info,
        "stack_depth_used": abs(target_index),
        "full_stack": cleaned_matches,
    }

def verify_python_syntax(code_string: str) -> Optional[str]:
    """Check if a string is valid Python syntax."""

    import ast

    try:
        ast.parse(code_string)
        return None
    except SyntaxError as e:
        return f"Syntax Error on line {e.lineno}: {e.msg}"
    except Exception as e:  # pragma: no cover - defensive
        return str(e)


def check_syntax(file_path: str) -> str:
    """Use pyflakes to check for syntax/name errors in a file."""

    result = subprocess.run(
        ["pyflakes", file_path], capture_output=True, text=True
    )
    if result.returncode != 0:
        return f"Syntax/Name Error detected: {result.stdout}"
    return "OK"


class NoAction(BaseModel):
    task_complete: bool = True



