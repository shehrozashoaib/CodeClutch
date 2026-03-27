"""
LLM-driven debugging agent extracted from the original debugging notebook.

This module uses the action schema:
- RunScriptInput
- ReadFileInput
- WriteFileInput
- ConfigInput
- ExecuteTerminalCommand
- NoAction

and the tools from system_tools to iteratively:
- run a script
- inspect errors
- read files
- write minimal fixes
- optionally run terminal commands (e.g. pip install)
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
from typing import Annotated, Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from llama_cpp import Llama  # type: ignore[import-not-found]

from system_tools import (
    RunScriptInput,
    ReadFileInput,
    WriteFileInput,
    ConfigInput,
    NoAction,
    ExecuteTerminalCommand,
    run_script,
    read_file,
    write_file,
    record_system_config,
    terminal_command_executor,
    error_analyzer,
    check_syntax,
)
from system_analysis import (
    autonomous_analysis,  # ✅ Correct module
)
from system_agent import (
    _decide_strategy,
    _filter_to_entry_flow,
    _promote_helpers_to_callers,
    apply_minimal_inputs,
    mock_function,
    MOCK_CACHE,
)

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[38;5;51m"
BLUE = "\033[38;5;39m"
MAGENTA = "\033[38;5;213m"
YELLOW = "\033[38;5;228m"
RED = "\033[38;5;203m"
GREEN = "\033[38;5;120m"


def _supports_ansi() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def _style(text: str, *codes: str) -> str:
    if not _supports_ansi():
        return text
    return "".join(codes) + text + RESET


def _console_rule(char: str = "─") -> str:
    return char * 80


def _print_console_panel(title: str, tone: str = "info") -> None:
    palette = {
        "info": BLUE,
        "action": MAGENTA,
        "review": CYAN,
        "warn": YELLOW,
        "error": RED,
        "success": GREEN,
    }
    color = palette.get(tone, BLUE)
    rule = _style(_console_rule(), color) if _supports_ansi() else ("-" * 80)
    label = _style(title, color, BOLD) if _supports_ansi() else title
    print("\n" + rule)
    print(label)
    print(rule)


def _print_labeled_value(label: str, value: str, tone: str = "info") -> None:
    palette = {"info": CYAN, "action": MAGENTA, "warn": YELLOW, "error": RED, "success": GREEN, "review": CYAN}
    color = palette.get(tone, CYAN)
    bullet = _style("◆", color, BOLD) if _supports_ansi() else "-"
    print(f"  {bullet} {label:<14} {value}")


def _format_step_header(step_number: int, max_steps: int, phase: str, mocking_phase: str) -> str:
    step_text = f"Step {step_number}/{max_steps}"
    meta = f"phase={phase} | mocking={mocking_phase}"
    if _supports_ansi():
        return f"{_style(step_text, BOLD, MAGENTA)}  {_style(meta, DIM, CYAN)}"
    return f"[{step_text}] {meta}"


def _print_step_header(step_number: int, max_steps: int, phase: str, mocking_phase: str) -> None:
    _print_console_panel(_format_step_header(step_number, max_steps, phase, mocking_phase), tone="action")


def preserve_multiline_indentation(
    new_content: str,
    original_lines: list,
    start_line: int,
    end_line: int
) -> str:
    """
    Properly handle indentation for multiline replacements.
    
    The problem: When LLM outputs multiline content with \\n escapes,
    each line needs proper indentation based on the ORIGINAL code structure.
    
    Args:
        new_content: The content from LLM (already has \\n converted to newlines)
        original_lines: List of lines from original file
        start_line: 1-indexed start line
        end_line: 1-indexed end line
    
    Returns:
        Properly indented content
    """
    if not new_content or not original_lines:
        return new_content
    
    # Get the indentation of the original line being replaced
    start_idx = max(0, start_line - 1)
    if start_idx >= len(original_lines):
        return new_content
    
    original_line = original_lines[start_idx]
    original_indent = len(original_line) - len(original_line.lstrip())
    indent_str = " " * original_indent
    
    # Split the new content into lines
    new_lines = new_content.split('\n')
    
    # Check if it's effectively a single line (with optional trailing newline)
    non_empty_lines = [line for line in new_lines if line.strip()]
    
    if len(non_empty_lines) <= 1:
        # Single line (or empty) - just add indent if needed
        if non_empty_lines:
            stripped = non_empty_lines[0].lstrip()
            current_indent = len(non_empty_lines[0]) - len(stripped)
            if current_indent < original_indent:
                # Rebuild with proper indent, preserving trailing newline if present
                result = indent_str + stripped
                if new_content.endswith('\n'):
                    result += '\n'
                return result
        return new_content
    
    # Multiline content - need to preserve relative indentation
    # Find the minimum indentation in the new content (excluding empty lines)
    min_new_indent = float('inf')
    for line in new_lines:
        if line.strip():  # Non-empty line
            line_indent = len(line) - len(line.lstrip())
            min_new_indent = min(min_new_indent, line_indent)
    
    if min_new_indent == float('inf'):
        min_new_indent = 0
    
    # Reindent all lines
    result_lines = []
    for i, line in enumerate(new_lines):
        if not line.strip():
            # Empty line - preserve it
            result_lines.append(line)
        else:
            # Calculate relative indent from the first significant line
            current_indent = len(line) - len(line.lstrip())
            relative_indent = current_indent - min_new_indent
            
            # Apply: original_indent + relative_indent
            new_indent = original_indent + relative_indent
            result_lines.append(" " * new_indent + line.lstrip())
    
    return '\n'.join(result_lines)


def detect_structural_break(
    file_path: str,
    new_content: str,
    start_line: int,
    end_line: int
) -> tuple:
    """
    Detect if a proposed edit would break code structure.
    
    Returns: (is_safe, warning_message)
    """
    try:
        with open(file_path, 'r') as f:
            original_lines = f.readlines()
    except:
        return True, ""  # Can't read file, let it proceed
    
    if start_line < 1 or start_line > len(original_lines):
        return True, ""
    
    start_idx = start_line - 1
    original_line = original_lines[start_idx]
    
    # Check 1: If original line is inside an if/for/with block, 
    # the replacement must maintain structure
    
    # Look backwards for block start
    block_indent = None
    block_type = None
    for i in range(start_idx - 1, -1, -1):
        line = original_lines[i]
        stripped = line.lstrip()
        if not stripped or stripped.startswith('#'):
            continue
        
        indent = len(line) - len(stripped)
        
        # Found a block header
        if stripped.endswith(':'):
            block_indent = indent
            for kw in ['if ', 'elif ', 'else', 'for ', 'while ', 'with ', 'try', 'except', 'finally', 'def ', 'class ']:
                if stripped.startswith(kw):
                    block_type = kw.strip().rstrip(':')
                    break
            break
        
        # Found a line at same or lower indent - no enclosing block
        orig_line_indent = len(original_line) - len(original_line.lstrip())
        if indent <= orig_line_indent:
            break
    
    if block_type:
        # We're inside a block - check if new content would break it
        new_lines = new_content.split('\n')
        orig_line_indent = len(original_line) - len(original_line.lstrip())
        
        for new_line in new_lines:
            if not new_line.strip():
                continue
            new_indent = len(new_line) - len(new_line.lstrip())
            
            # If new line has LESS indent than original, it might break the block
            if new_indent < orig_line_indent:
                return False, (
                    f"Your fix would break a '{block_type}' block. "
                    f"Original line has {orig_line_indent} spaces indent, "
                    f"but your replacement has only {new_indent} spaces. "
                    f"Ensure proper indentation is maintained."
                )
    
    # Check 2: Detect undefined variable placeholders
    placeholder_patterns = [
        r'\bdefault_value\b',
        r'\bNone\b\s*#\s*TODO',
        r'\bpass\b\s*#\s*TODO',
        r'\b(PLACEHOLDER|FIXME_VALUE|UNKNOWN)\b',
    ]
    
    for pattern in placeholder_patterns:
        if re.search(pattern, new_content, re.IGNORECASE):
            # Check if this placeholder exists in the original code
            original_text = ''.join(original_lines[start_idx:end_line])
            if not re.search(pattern, original_text, re.IGNORECASE):
                cleaned_pattern = pattern.replace("\\b", "")
                return False, (
                    f"Your fix contains an undefined placeholder "
                    f"(detected: {cleaned_pattern}). "
                    f"Use actual values from the codebase, not placeholders like 'default_value'."
                )
    
    return True, ""


def is_safe_write_path(file_path: str, working_dir: str) -> tuple:
    """
    Check if a file path is safe to write to.
    Returns (is_safe, reason)
    """
    abs_path = os.path.abspath(file_path)
    abs_working = os.path.abspath(working_dir)
    
    # BLOCKED PATTERNS - Never write to these
    blocked_patterns = [
        "site-packages",
        "dist-packages", 
        "/usr/lib",
        "/usr/local/lib",
        "anaconda3/envs",
        "anaconda3/lib",
        "miniconda3/envs",
        "miniconda3/lib",
        "/opt/",
        "/.local/lib",
        "__pycache__",
    ]
    
    for pattern in blocked_patterns:
        if pattern in abs_path:
            return False, f"Cannot write to system/library path containing '{pattern}'"
    
    # Must be within working directory
    if not abs_path.startswith(abs_working):
        return False, f"Path '{abs_path}' is outside working directory '{abs_working}'"
    
    return True, "OK"

# =====================================================================
# FIX 4: STRONGER LIBRARY FILE DETECTION
# =====================================================================

def _is_runtime_library_file(file_path: str) -> bool:
    abs_path = os.path.abspath(file_path or "")
    workspace_root = os.path.abspath(os.getcwd())
    normalized = abs_path.replace("\\", "/")
    library_patterns = [
        "site-packages",
        "dist-packages",
        "/usr/lib/python",
        "/usr/local/lib/python",
        "anaconda3/lib",
        "miniconda3/lib",
        ".local/lib/python",
        "/opt/",
        "gguf-py/",
        "gguf/",
    ]
    if any(pattern in normalized for pattern in library_patterns):
        return True
    if "/lib/python" in normalized and not normalized.startswith(workspace_root + "/"):
        return True
    if "/.venvs/" in normalized and "/lib/python" in normalized and "/site-packages/" not in normalized:
        return True
    return False


def _find_first_user_stack_frame(full_stack: List[tuple]) -> Dict[str, Any] | None:
    for stack_file, stack_line in reversed(full_stack or []):
        if not _is_runtime_library_file(stack_file):
            return {"file": stack_file, "line": stack_line}
    return None


def _parse_flag_arguments(run_args: List[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    args = list(run_args or [])
    i = 0
    while i < len(args):
        item = str(args[i])
        if item.startswith("--"):
            key = item[2:]
            value = "true"
            if i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                value = str(args[i + 1])
                i += 1
            parsed[key] = value
        i += 1
    return parsed


def _resolve_candidate_data_path(raw_value: str, base_file: str) -> str:
    value = str(raw_value or "").strip().strip("\"'")
    if not value:
        return ""
    candidates = [value]
    if not os.path.isabs(value):
        candidates.append(os.path.join(os.getcwd(), value))
        if base_file:
            candidates.append(os.path.join(os.path.dirname(os.path.abspath(base_file)), value))
    for candidate in candidates:
        if candidate and os.path.exists(candidate) and os.path.isfile(candidate):
            return os.path.relpath(candidate, os.getcwd()) if os.path.isabs(candidate) and candidate.startswith(os.getcwd()) else candidate
    return ""


def _infer_loaded_artifact_from_user_frame(state: Dict[str, Any], user_frame: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not user_frame:
        return None
    frame_file = user_frame.get("file", "")
    frame_line = int(user_frame.get("line", 0) or 0)
    if not frame_file or frame_line <= 0 or not os.path.exists(frame_file):
        return None
    try:
        with open(frame_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return None
    if frame_line > len(lines):
        return None
    source_line = lines[frame_line - 1].strip()

    for literal in re.findall(r"[\"']([^\"']+\.(?:json|ya?ml|toml|ini|cfg|csv|txt))[\"']", source_line, re.IGNORECASE):
        resolved = _resolve_candidate_data_path(literal, frame_file)
        if resolved:
            return {"file": resolved, "line": 1, "reason": f"literal artifact path in {frame_file}:{frame_line}"}

    cli_args = _parse_flag_arguments(state.get("run_args", []))
    variable_names = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*(?:file|path|name|json|yaml|yml|toml|config|cfg))\b', source_line)
    for var_name in variable_names:
        cli_key = var_name.lstrip('_')
        cli_value = cli_args.get(cli_key)
        if not cli_value:
            continue
        resolved = _resolve_candidate_data_path(cli_value, frame_file)
        if resolved:
            return {"file": resolved, "line": 1, "reason": f"CLI argument --{cli_key} resolved from {frame_file}:{frame_line}"}
    return None


def is_library_or_utility_file(file_path: str) -> tuple:
    """
    Check if a file is part of a library or shared utility that shouldn't be edited.
    
    Returns: (is_library: bool, reason: str)
    """
    abs_path = os.path.abspath(file_path)
    if _is_runtime_library_file(abs_path):
        return True, "File is in runtime/library path"
    return False, "OK"



DEFAULT_MODEL_CANDIDATES = [
    os.environ.get("DEBUG_AGENT_MODEL_PATH"),
    os.environ.get("LLM_MODEL_PATH"),
    "/workspace/llama.cpp/IQuest_Coder_V1_40B_Q4_K_M.gguf",
    "llama.cpp/IQuest_Coder_V1_40B_Q4_K_M.gguf",
    "./IQuest_Coder_V1_40B_Q4_K_M.gguf",
    "/home/shehroz/cgcnn_edge/Qwen2.5_Coder/Qwen2.5-Coder-8B-Q4_K_M.gguf",
]


def resolve_model_path(model_path: str | None = None) -> str:
    """Resolve the preferred GGUF path, prioritizing IQuest and env overrides."""
    candidates: List[str] = []
    if model_path:
        candidates.append(model_path)
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    return candidates[0] if candidates else "llama.cpp/IQuest_Coder_V1_40B_Q4_K_M.gguf"


LLM_MODEL_PATH = resolve_model_path()
_LLM_RUNTIME_CONFIGS: Dict[int, Dict[str, Any]] = {}


def create_llm(
    model_path: str | None = None,
    n_ctx: int = 10000,
    n_gpu_layers: int = -1,
    n_threads: int = 4,
    n_batch: int = 512,
    chat_format: str = "chatml",
    verbose: bool = False,
) -> Llama:
    """Factory to create and return a Llama instance."""
    resolved_model_path = resolve_model_path(model_path)
    if not os.path.exists(resolved_model_path):
        searched = "\n  - ".join([c for c in [model_path, *DEFAULT_MODEL_CANDIDATES] if c])
        raise FileNotFoundError(
            "Could not find a GGUF model. Checked:\n  - " + searched
        )

    instance = Llama(
        model_path=resolved_model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        n_batch=n_batch,
        chat_format=chat_format,
        verbose=verbose,
    )
    _LLM_RUNTIME_CONFIGS[id(instance)] = {
        "model_path": resolved_model_path,
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "n_threads": n_threads,
        "n_batch": n_batch,
        "chat_format": chat_format,
        "verbose": verbose,
    }
    return instance


def _get_llm_runtime_config(llm_obj: Llama | None) -> Dict[str, Any]:
    if llm_obj is None:
        return {}
    return dict(_LLM_RUNTIME_CONFIGS.get(id(llm_obj), {}))


def _install_llm_references(llm_obj: Llama | None) -> None:
    globals()["llm"] = llm_obj
    set_llm_instance(llm_obj)
    set_rag_llm_instance(llm_obj)


def get_active_llm_instance() -> Llama | None:
    return globals().get("llm")


def close_llm_instance(llm_obj: Llama | None) -> None:
    if llm_obj is None:
        return
    try:
        llm_obj.close()
    except Exception:
        pass
    finally:
        _LLM_RUNTIME_CONFIGS.pop(id(llm_obj), None)


def _release_llm_for_user_run(active_llm: Llama | None, state: Dict[str, Any]) -> None:
    runtime_cfg = _get_llm_runtime_config(active_llm)
    state["llm_runtime_config"] = runtime_cfg
    if runtime_cfg and not state.get("llm_original_runtime_config"):
        state["llm_original_runtime_config"] = dict(runtime_cfg)
    uses_gpu = bool(runtime_cfg) and int(runtime_cfg.get("n_gpu_layers", 0) or 0) != 0
    state["llm_uses_gpu"] = uses_gpu
    if not active_llm or not uses_gpu:
        return

    print("   💤 Releasing agent model from GPU before user run")
    _install_llm_references(None)
    close_llm_instance(active_llm)
    del active_llm
    gc.collect()


def _restore_llm_after_user_run(state: Dict[str, Any]) -> Llama | None:
    runtime_cfg = dict(state.get("llm_original_runtime_config") or state.get("llm_runtime_config") or {})
    if not runtime_cfg:
        return globals().get("llm")
    if globals().get("llm") is not None:
        return globals().get("llm")

    preferred_layers = int(runtime_cfg.get("n_gpu_layers", 0) or 0)
    attempts = [runtime_cfg]
    if preferred_layers != 0:
        cpu_cfg = dict(runtime_cfg)
        cpu_cfg["n_gpu_layers"] = 0
        attempts.append(cpu_cfg)

    last_error = None
    for index, cfg in enumerate(attempts):
        try:
            if index == 0:
                print("   ♻️ Restoring agent model after user run")
            else:
                print("   ⚠️ GPU reload failed; restoring agent model on CPU")
            restored = create_llm(**cfg)
            _install_llm_references(restored)
            state["llm_runtime_config"] = dict(cfg)
            state.setdefault("llm_original_runtime_config", dict(runtime_cfg))
            state["llm_uses_gpu"] = int(cfg.get("n_gpu_layers", 0) or 0) != 0
            return restored
        except Exception as exc:
            last_error = exc
            _install_llm_references(None)
            gc.collect()

    if last_error is not None:
        raise last_error
    return None


# Optional: leave a module-level llm = None to avoid creating the model at import time
llm = None

from typing import Dict, Any, List, Optional

# ----------------------------
# 1) AgentAction schema
# ----------------------------
def validate_write_safety(
    file_path: str,
    new_content: str,
    start_line: Optional[int],
    end_line: Optional[int],
    state: Dict[str, Any],
) -> tuple:
    """
    Comprehensive validation before any file write.
    Returns (is_safe, reason, should_continue)
    
    - is_safe: True if write should proceed
    - reason: Explanation if blocked
    - should_continue: True if agent should continue (vs abort)
    """
    from system_tools import verify_python_syntax
    
    working_dir = os.getcwd()
    
    # Check 1: Safe path
    path_safe, path_reason = is_safe_write_path(file_path, working_dir)
    if not path_safe:
        return False, f"⛔ PATH BLOCKED: {path_reason}", True

    # Check 1b: runtime user-protected paths
    if _is_user_protected_path(state, file_path):
        return False, (
            f"⛔ USER-PROTECTED PATH: {file_path} is in the user no-touch list. "
            f"Choose another file/tool."
        ), True
    
    # Check 2: File exists
    if not os.path.exists(file_path):
        # Allow creating new files in working directory
        return True, "OK - new file", True
    
    # Check 3: Detect undefined placeholder variables FIRST (before syntax check)
    placeholder_patterns = [
        (r'\bdefault_value\b', 'default_value'),
        (r'\bPLACEHOLDER\b', 'PLACEHOLDER'),
        (r'\bFIXME_VALUE\b', 'FIXME_VALUE'),
        (r'\bUNKNOWN_VALUE\b', 'UNKNOWN_VALUE'),
        (r'\bTODO_VALUE\b', 'TODO_VALUE'),
    ]
    
    for pattern, name in placeholder_patterns:
        if re.search(pattern, new_content, re.IGNORECASE):
            return False, (
                f"⛔ UNDEFINED PLACEHOLDER: Your fix uses '{name}' which is not defined. "
                f"Use an ACTUAL value from the codebase, not a placeholder."
            ), True
    # Check if we're currently fixing a KeyError - if so, block ALL .get() patterns
    # This forces the LLM to find the CORRECT key rather than using fallback values
    current_error_type = None
    current_error_msg = ""
    if state.get("full_error_history"):
        last_error = state["full_error_history"][-1]
        current_error_type = last_error.get("error_type", "")
        current_error_msg = last_error.get("message", "")
    
    if current_error_type == "KeyError":
        # Block ANY .get() pattern when fixing KeyError - must find correct key
        get_match = re.search(r'\.get\s*\(\s*["\']([^"\']+)["\']', new_content)
        if get_match:
            attempted_key = get_match.group(1)
            
            # Try to find similar keys in the file
            similar_keys = []
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    # Find dict access patterns near the error
                    key_pattern = re.findall(r'\[[\"\']([^\"\']+)[\"\']\]', content)
                    # Find keys that share parts with the attempted key
                    attempted_parts = set(attempted_key.lower().split('_'))
                    for key in set(key_pattern):
                        key_parts = set(key.lower().split('_'))
                        if attempted_parts & key_parts:  # Any common parts
                            similar_keys.append(key)
                    similar_keys = list(set(similar_keys))[:5]  # Top 5 unique
                except:
                    pass
            
            suggestion = ""
            if similar_keys:
                suggestion = f"\n\n💡 SIMILAR KEYS FOUND IN CODEBASE:\n"
                state["last_similar_keys"] = similar_keys  # Store for later use
                for key in similar_keys:
                    suggestion += f"   - \"{key}\"\n"
                suggestion += f"\nTry one of these with DIRECT ACCESS: [\"{similar_keys[0]}\"]"
            
            return False, (
                f"⛔ BAD PATTERN: Using .get(\"{attempted_key}\", ...) does NOT fix KeyError.\n"
                f"The key doesn't exist - you must find the CORRECT key name.\n"
                f"Use DIRECT ACCESS syntax: [\"correct_key\"] not .get(\"key\", fallback)"
                f"{suggestion}"
            ), True
    else:
        # For non-KeyError, only block .get(..., None) which causes TypeError
        if re.search(r'\.get\s*\([^,]+,\s*None\s*\)', new_content):
            return False, (
                f"⛔ BAD PATTERN: Using .get(..., None) will cause TypeError later. "
                f"Find the CORRECT KEY that exists, don't use None as fallback."
            ), True
    
    # Check 4: Python syntax validation (for .py files)
    if file_path.endswith('.py'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            original_lines = original_content.split('\n')
            
            # Apply the proposed change
            if start_line is not None and end_line is not None:
                # Partial replacement
                start_idx = max(0, start_line - 1)
                end_idx = end_line
                new_lines = new_content.split('\n')
                # Remove trailing empty string if content ended with \n
                if new_lines and new_lines[-1] == '' and new_content.endswith('\n'):
                    new_lines = new_lines[:-1]
                result_lines = original_lines[:start_idx] + new_lines + original_lines[end_idx:]
            else:
                # Full file replacement
                result_lines = new_content.split('\n')
            
            test_content = '\n'.join(result_lines)
            
            # Validate syntax
            syntax_error = verify_python_syntax(test_content)
            if syntax_error:
                return False, f"⛔ SYNTAX ERROR: {syntax_error}", True
                
        except Exception as e:
            return False, f"⛔ VALIDATION ERROR: {e}", True
    
    # Check 4: Prevent writing mocks to wrong file
    if state.get("strategy") == "mocking":
        expensive_funcs = state.get("expensive_functions", [])
        
        # Check if content contains a function definition
        func_def_match = re.search(r'def\s+(\w+)\s*\(', new_content)
        if func_def_match:
            func_name = func_def_match.group(1)
            
            # Check if this function belongs in a different file
            for func in expensive_funcs:
                if func["function"] == func_name:
                    correct_file = os.path.abspath(func["file"])
                    target_file = os.path.abspath(file_path)
                    
                    if correct_file != target_file:
                        return False, (
                            f"⛔ WRONG FILE: Function '{func_name}' is defined in "
                            f"'{func['file']}', not '{file_path}'"
                        ), True
    
    # Check 5: Prevent duplicate mocking
    if "MOCK" in new_content or "mocked" in new_content.lower():
        mocked_funcs = state.get("mocked_functions", [])
        for mocked in mocked_funcs:
            if os.path.abspath(mocked["file"]) == os.path.abspath(file_path):
                func_match = re.search(r'def\s+' + mocked["function"] + r'\s*\(', new_content)
                if func_match:
                    return False, (
                        f"⛔ ALREADY MOCKED: Function '{mocked['function']}' "
                        f"is already mocked in '{file_path}'"
                    ), True
    
    # Check 6: Prevent writing garbage (common patterns from bad LLM output)
    garbage_patterns = [
        r'def\s+\w+\([^)]*\):\s*\n[a-zA-Z]',  # Function with no indented body
        r'parser\.add_argument\(\s*\n\s*figlet',  # Mixed argparse and figlet
        r'""".*"""\s*\n\s*"""',  # Adjacent docstrings
    ]
    
    for pattern in garbage_patterns:
        if re.search(pattern, new_content):
            return False, "⛔ INVALID CONTENT: Write appears to contain malformed code", True
    
    return True, "OK", True

class RunAction(BaseModel):
    action_type: Literal["RunScriptInput"]
    action: RunScriptInput


class ReadAction(BaseModel):
    action_type: Literal["ReadFileInput"]
    action: ReadFileInput


class WriteAction(BaseModel):
    action_type: Literal["WriteFileInput"]
    action: WriteFileInput


class RAGSearchInput(BaseModel):
    """Trigger online/contextual retrieval for the current debugging error."""

    error_type: str = Field(default="", description="Error class, e.g. KeyError/TypeError")
    error_message: str = Field(default="", description="Error message text")
    file_path: str = Field(default="", description="File where error originated")
    error_line: int = Field(default=0, description="Line number of error in file_path")
    code_context: str = Field(default="", description="Optional code snippet around the error")


class RAGSearchAction(BaseModel):
    action_type: Literal["RAGSearchInput"]
    action: RAGSearchInput


class TerminalInstructionAction(BaseModel):
    action_type: Literal["ExecuteTerminalCommand"]
    action: ExecuteTerminalCommand


class ConfigAction(BaseModel):
    action_type: Literal["ConfigInput"]
    action: ConfigInput


class NoActionAction(BaseModel):
    action_type: Literal["NoAction"]
    action: NoAction


AgentAction = Annotated[
    Union[
        RunAction,
        ReadAction,
        WriteAction,
        RAGSearchAction,
        ConfigAction,
        TerminalInstructionAction,
        NoActionAction,
    ],
    Field(discriminator="action_type"),
]

AGENT_ACTION_ADAPTER = TypeAdapter(AgentAction)

ACTION_SCHEMAS_TEXT = """ACTION SCHEMAS (exact keys, copy this format):

- RunScriptInput:
  {"action_type":"RunScriptInput","action":{"script_path":"<path>","args":[]}}

- ReadFileInput:
  {"action_type":"ReadFileInput","action":{"file_path":"<path>","error_line":12,"window_before":20,"window_after":30}}

- ExecuteTerminalCommand:
  {"action_type":"ExecuteTerminalCommand","action":{"command":"python -m pip install numpy"}}

IMPORTANT: For dependency installation, default to the target runtime environment and prefer `python -m pip ...` so installs land in the same interpreter that will run the code. If the user explicitly asks for a different installer or environment, preserve that request instead of rewriting it.

- WriteFileInput:
  {"action_type":"WriteFileInput","action":{
    "file_path":"<path>",
    "start_line": 10,
    "end_line": 12,
    "new_content":"line1\\nline2\\nline3\\n",
    "create_backup": true
  }}

- RAGSearchInput:
  {"action_type":"RAGSearchInput","action":{
    "error_type":"TypeError",
    "error_message":"object of type 'float' has no len()",
    "file_path":"train.py",
    "error_line":42,
    "code_context":"optional snippet"
  }}

- ConfigInput:
  {"action_type":"ConfigInput","action":{"config_name":"startup_config","config_data":{}}}

- NoAction:
  {"action_type":"NoAction","action":{"task_complete":true}}

"""

def sanitize_jsonish(s: str) -> str:
    """Normalize some Python-ish JSON into strict JSON."""

    s = s.strip()
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)

    def _triple_to_json(m: "re.Match[str]") -> str:
        inner = m.group(1)
        inner = inner.replace("\\", "\\\\").replace('"', '\\"')
        inner = inner.replace("\r\n", "\n").replace("\r", "\n")
        inner = inner.replace("\n", "\\n")
        return f'"new_content":"{inner}"'

    s = re.sub(
        r'"new_content"\s*:\s*"""\s*(.*?)\s*"""', _triple_to_json, s, flags=re.DOTALL
    )
    
    # Fix unescaped quotes inside new_content string
    # Pattern: "new_content": "...unescaped["stuff"]..." 
    def _fix_inner_quotes(m: "re.Match[str]") -> str:
        prefix = m.group(1)  # "new_content": "
        inner = m.group(2)   # the content
        suffix = m.group(3)  # ", "create_backup" or similar
        
        # Escape any unescaped double quotes inside (but not the outer ones)
        # This handles: self.hparams["key"] -> self.hparams[\"key\"]
        fixed_inner = re.sub(r'(?<!\\)"', '\\"', inner)
        return f'{prefix}{fixed_inner}{suffix}'
    
    # Match new_content field and fix inner quotes
    s = re.sub(
        r'("new_content"\s*:\s*")([^"]*(?:\\"[^"]*)*[^\\])(",\s*")',
        _fix_inner_quotes,
        s
    )
    
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s

def allowed_actions(state: Dict[str, Any]) -> List[str]:
    """Simple phase-based guardrail policy."""
    rag_allowed = bool(
        state.get("rag_eligible", False)
        or state.get("force_rag", False)
        or state.get("user_requested_rag", False)
    )

    phase = state.get("phase", "START")
    if phase == "START":
        return ["RunScriptInput"]
    if phase == "RAN":
        if state.get("last_exitcode") == 0:
            return ["NoAction"]
        if state.get("artifact_read_target"):
            return ["ReadFileInput"]
        actions = ["ReadFileInput", "ExecuteTerminalCommand"]
        if rag_allowed:
            actions.append("RAGSearchInput")
        return actions
    if phase == "READ":
        actions = ["WriteFileInput", "ReadFileInput"]
        if rag_allowed:
            actions.append("RAGSearchInput")
        return actions
    if phase == "RAG":
        return ["ReadFileInput", "WriteFileInput", "RunScriptInput"]
    if phase == "TERMINAL_COMMAND":
        if state.get("env_resolution_active"):
            actions = ["ExecuteTerminalCommand"]
            if state.get("env_ready_to_rerun"):
                actions.append("RunScriptInput")
            if rag_allowed:
                actions.append("RAGSearchInput")
            return actions
        return ["RunScriptInput"]
    if phase == "WROTE":
        return ["WriteFileInput", "RunScriptInput"]
    return ["RunScriptInput"]


def build_state_summary(state: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.append(f"phase={state.get('phase','')}")
    if state.get("target_file"):
        parts.append(f"target_file={state['target_file']}")
    if "last_exitcode" in state:
        parts.append(f"last_exitcode={state['last_exitcode']}")
    if state.get("last_stderr"):
        parts.append(
            "last_stderr_snippet="
            + json.dumps(str(state["last_stderr"])[:1200], ensure_ascii=False)
        )
    if state.get("last_stdout"):
        parts.append(
            "last_stdout_snippet="
            + json.dumps(str(state["last_stdout"])[:600], ensure_ascii=False)
        )
    if state.get("user_protected_paths"):
        parts.append(
            "user_protected_paths="
            + json.dumps(state.get("user_protected_paths", [])[:20], ensure_ascii=False)
        )
    if state.get("env_resolution_active"):
        parts.append("env_resolution_active=true")
    if state.get("env_resolution_packages"):
        parts.append(
            "env_resolution_packages="
            + json.dumps(state.get("env_resolution_packages", [])[:10], ensure_ascii=False)
        )
    if state.get("last_env_diagnostic_summary"):
        parts.append(
            "last_env_diagnostic_summary="
            + json.dumps(state.get("last_env_diagnostic_summary", ""), ensure_ascii=False)
        )
    if state.get("env_diag_blocked_count", 0):
        parts.append(f"env_diag_blocked_count={state['env_diag_blocked_count']}")
    return "\n".join([p for p in parts if p])

def track_tried_fix_and_get_alternatives(
    state: Dict[str, Any],
    error_location: str,  # e.g., "file.py:120"
    attempted_value: str,  # e.g., "ffn_config"
    all_candidates: List[str]
) -> Tuple[List[str], str]:
    """
    Track which fix values have been tried and return remaining alternatives.
    
    Returns: (remaining_candidates, message)
    """
    # Initialize tracking dict if needed
    if "tried_fixes" not in state:
        state["tried_fixes"] = {}
    
    # Get or create the set of tried values for this error location
    if error_location not in state["tried_fixes"]:
        state["tried_fixes"][error_location] = set()
    
    tried = state["tried_fixes"][error_location]
    tried.add(attempted_value)
    
    # Find remaining candidates
    remaining = [c for c in all_candidates if c not in tried]
    
    if remaining:
        message = (
            f"You've tried {len(tried)} keys that didn't work: {list(tried)}\n"
            f"Keys you HAVEN'T tried yet: {remaining}\n"
            f"Try one of these UNTRIED keys."
        )
    else:
        message = (
            f"You've tried ALL suggested keys: {list(tried)}\n"
            f"None of them worked. You need to READ the file more carefully "
            f"to find what keys actually exist in the hparams/config."
        )
    
    return remaining, message


import re
import json

def extract_json(text: str, default_script_path: str = "train_alignn.py") -> str:
    """Extract JSON from LLM response with fallback handling."""
    text = text.strip()

    def fix_unescaped_brackets(match):
        # Find content between "new_content": " and the next ", "
        full = match.group(0)
        # Replace ["..."] with [\"...\"] inside the string
        fixed = re.sub(r'\[(["\'])([^"\']+)\1\]', r'[\"\2\"]', full)
        return fixed
    
    # Look for patterns like: ["key"] or ['key'] that should be [\"key\"]
    text = re.sub(
        r'"new_content"\s*:\s*"[^"]*\[["\'][^\]]+["\']\][^"]*"',
        fix_unescaped_brackets,
        text
    )
    
    # Try to find JSON object
    start = text.find("{")
    if start == -1:
        # No JSON found - try to infer action from text
        return _infer_action_from_text(text, default_script_path=default_script_path)
    
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    
    # Unbalanced braces - try to fix or infer
    # Option 1: Try adding missing closing braces
    partial_json = text[start:]
    if depth > 0:
        partial_json += "}" * depth
        try:
            json.loads(partial_json)  # Validate it works
            print(f"   ⚠️ Fixed incomplete JSON (added {depth} closing braces)")
            return partial_json
        except json.JSONDecodeError:
            pass
    
    # Option 2: Infer action from text content
    return _infer_action_from_text(text, default_script_path=default_script_path)


def _infer_action_from_text(text: str, default_script_path: str = "train_alignn.py") -> str:
    """Fallback: infer action type from text when JSON parsing fails."""
    text_lower = text.lower()
    
    # Try to detect intended action
    if "runscript" in text_lower or "run the script" in text_lower or "execute" in text_lower:
        print("   ⚠️ Inferred RunScriptInput from malformed response")
        return (
            '{"action_type": "RunScriptInput", "action": '
            f'{{"script_path": "{default_script_path}"}}'
            "}"
        )
    
    if "readfile" in text_lower or "read the file" in text_lower or "look at" in text_lower:
        # Try to extract a file path
        file_match = re.search(r'["\']?([a-zA-Z0-9_/]+\.py)["\']?', text)
        file_path = file_match.group(1) if file_match else default_script_path
        print(f"   ⚠️ Inferred ReadFileInput for {file_path} from malformed response")
        return (
            '{"action_type": "ReadFileInput", "action": '
            f'{{"file_path": "{file_path}", "error_line": 0, "window_before": 20, "window_after": 30}}'
            "}"
        )
    
    if "writefile" in text_lower or "write" in text_lower or "fix" in text_lower:
        print("   ⚠️ Inferred ReadFileInput (need to read before write) from malformed response")
        return (
            '{"action_type": "ReadFileInput", "action": '
            f'{{"file_path": "{default_script_path}", "error_line": 0, "window_before": 20, "window_after": 30}}'
            "}"
        )
    
    if "conda install" in text_lower:
        conda_match = re.search(r'(conda install\s+.+)', text, re.IGNORECASE)
        command = conda_match.group(1).strip() if conda_match else "conda install"
        print(f"   ⚠️ Preserving explicit conda command: {command}")
        return f'{{"action_type": "ExecuteTerminalCommand", "action": {{"command": "{command}"}}}}'

    if "pip install" in text_lower or "terminal" in text_lower:
        package_match = re.search(r'pip install(?:\s+-[^\s]+)*(?:\s+[a-zA-Z0-9_-]+)?\s+([a-zA-Z0-9_.-]+)', text)
        package = package_match.group(1) if package_match else "unknown"
        print(f"   ⚠️ Inferred terminal command: python -m pip install {package}")
        return f'{{"action_type": "ExecuteTerminalCommand", "action": {{"command": "python -m pip install {package}"}}}}'
    
    if "noaction" in text_lower or "finish" in text_lower or "done" in text_lower:
        print("   ⚠️ Inferred NoAction from malformed response")
        return '{"action_type": "NoAction", "action": {}}'
    
    # Default fallback: run script again
    print("   ⚠️ Could not parse LLM response, defaulting to RunScriptInput")
    return (
        '{"action_type": "RunScriptInput", "action": '
        f'{{"script_path": "{default_script_path}"}}'
        "}"
    )


def call_llama_for_action(
    llm_obj: Llama, messages: List[Dict[str, str]], max_tokens: int = 512
) -> str:
    """Call llama_cpp in chat-completion mode."""

    resp = llm_obj.create_chat_completion(
        messages=messages,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        max_tokens=max_tokens,
        repeat_penalty=1.05,
    )
    return resp["choices"][0]["message"]["content"]


def _safe_file_line_count(file_path: str) -> int:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return len(f.readlines())
    except Exception:
        return 0


def _fallback_read_target_for_invalid_write(state: Dict[str, Any], proposed_file: str) -> Dict[str, Any]:
    last_error = state.get("full_error_history", [])[-1] if state.get("full_error_history") else {}
    user_frame = _find_first_user_stack_frame(last_error.get("full_stack", []))
    fallback_file = proposed_file if proposed_file and os.path.exists(proposed_file) else ""
    fallback_line = int(state.get("error_line", 0) or 0)

    if user_frame:
        fallback_file = user_frame.get("file") or fallback_file
        fallback_line = int(user_frame.get("line", fallback_line) or fallback_line or 1)

    if not fallback_file:
        fallback_file = state.get("target_file", proposed_file or "")

    line_count = _safe_file_line_count(fallback_file)
    if line_count:
        fallback_line = min(max(1, fallback_line or 1), line_count)
    else:
        fallback_line = max(1, fallback_line or 1)

    return {
        "action_type": "ReadFileInput",
        "action": {
            "file_path": fallback_file,
            "error_line": fallback_line,
            "window_before": 20,
            "window_after": 30,
        },
    }


def _normalize_write_action_bounds(obj: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    action = obj.get("action") if isinstance(obj, dict) else None
    if not isinstance(action, dict):
        return obj

    file_path = str(action.get("file_path", "") or "")
    if not file_path or not os.path.exists(file_path):
        return obj

    start_line = action.get("start_line")
    end_line = action.get("end_line")
    if start_line is None or end_line is None:
        return obj

    try:
        start_line = int(start_line)
        end_line = int(end_line)
    except Exception:
        return _fallback_read_target_for_invalid_write(state, file_path)

    line_count = _safe_file_line_count(file_path)
    if line_count <= 0:
        return obj

    if start_line < 1 or end_line < 1 or start_line > end_line:
        return _fallback_read_target_for_invalid_write(state, file_path)
    if start_line > line_count or end_line > line_count:
        return _fallback_read_target_for_invalid_write(state, file_path)

    action["start_line"] = start_line
    action["end_line"] = end_line
    return obj

def normalize_action_payload(obj: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fill required action fields when the model omits them.
    Keeps the agent resilient to partially correct JSON.
    """
    if not isinstance(obj, dict):
        return obj

    action_type = obj.get("action_type")
    action = obj.get("action")
    if not isinstance(action, dict):
        return obj

    if action_type == "ReadFileInput":
        action.setdefault("error_line", state.get("error_line", 0) or 0)
        action.setdefault("window_before", 20)
        action.setdefault("window_after", 30)
    elif action_type == "WriteFileInput":
        obj = _normalize_write_action_bounds(obj, state)
        if obj.get("action_type") != "WriteFileInput":
            return obj
    elif action_type == "RAGSearchInput":
        last_error = state.get("full_error_history", [])[-1] if state.get("full_error_history") else {}
        action.setdefault("error_type", last_error.get("error_type", ""))
        action.setdefault("error_message", last_error.get("message", ""))
        action.setdefault("file_path", last_error.get("file", state.get("target_file", "")))
        action.setdefault("error_line", last_error.get("line", state.get("error_line", 0) or 0))
        action.setdefault("code_context", "")
    elif action_type == "ExecuteTerminalCommand":
        command = str(action.get("command", "") or "").strip()
        normalized = " ".join(command.split())
        lower_command = normalized.lower()
        runtime_context = _load_runtime_cuda_context()
        normalized_torch_command = _normalize_torch_install_command(normalized, runtime_context)
        if normalized_torch_command:
            action["command"] = normalized_torch_command
        else:
            last_error = state.get("full_error_history", [])[-1] if state.get("full_error_history") else {}
            missing_import = str(last_error.get("message", "") or "").strip().strip("\'\"")
            package_candidates = state.get("last_rag_results", {}).get("package_candidates", []) or []
            if missing_import and package_candidates and _is_generic_import_install_command(normalized, missing_import):
                best_command = package_candidates[0].get("recommended_command")
                if best_command:
                    action["command"] = best_command

    return obj


def _append_user_protected_path(state: Dict[str, Any], path: str) -> None:
    if not path:
        return
    paths = state.setdefault("user_protected_paths", [])
    norm = os.path.abspath(path)
    if norm not in paths:
        paths.append(norm)


def _is_user_protected_path(state: Dict[str, Any], file_path: str) -> bool:
    norm = os.path.abspath(file_path)
    for protected in state.get("user_protected_paths", []):
        protected_norm = os.path.abspath(protected)
        if norm == protected_norm or norm.startswith(protected_norm + os.sep):
            return True
    return False


def _summarize_action_for_approval(action: AgentAction) -> str:
    if action.action_type == "WriteFileInput":
        a = action.action
        snippet = (a.new_content or "").strip().splitlines()
        preview = snippet[0][:120] if snippet else ""
        return (
            f"Write {a.file_path}:{a.start_line}-{a.end_line}. "
            f"Preview: {preview}"
        )
    if action.action_type == "ExecuteTerminalCommand":
        return f"Execute terminal command: {action.action.command}"
    if action.action_type == "ReadFileInput":
        return f"Read file: {action.action.file_path}:{action.action.error_line or 0}"
    if action.action_type == "RAGSearchInput":
        a = action.action
        return f"RAG search: {a.error_type or 'unknown'} at {a.file_path or '<current error>'}:{a.error_line or 0}"
    if action.action_type == "ConfigInput":
        return f"Record config: {action.action.config_name}"
    return f"{action.action_type}"

def is_comment_out_fix(original_lines: list, new_content: str) -> bool:
    """
    Detect if the proposed fix is just commenting out existing code.
    """
    new_lines = new_content.strip().split('\n')
    for new_line in new_lines:
        stripped_new = new_line.strip()
        if not stripped_new:
            continue
        if stripped_new.startswith('#'):
            commented_content = stripped_new.lstrip('#').strip()
            for orig_line in original_lines:
                orig_stripped = orig_line.strip()
                if commented_content == orig_stripped:
                    return True
                if orig_stripped and stripped_new == '#' + orig_stripped:
                    return True
                if orig_stripped and stripped_new == '# ' + orig_stripped:
                    return True
    return False

def safe_input(prompt: str) -> str:
    """Flush pending output before waiting for terminal/notebook input.

    Some notebook or wrapped terminal frontends may hand back non-string values
    such as booleans. Normalize those here so downstream prompt handlers can
    safely call .strip() / .lower() without crashing the debugger itself.
    """
    sys.stdout.flush()
    value = input(prompt)
    if isinstance(value, bool):
        return "y" if value else "n"
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _preview_text_block(text: str, max_chars: int = 1600) -> str:
    text = (text or "").strip()
    if not text:
        return "<empty>"
    if len(text) <= max_chars:
        return text
    tail = max_chars // 3
    head = max_chars - tail - 32
    return text[:head] + "\n... [truncated] ...\n" + text[-tail:]


def _load_execution_history() -> List[Dict[str, Any]]:
    try:
        with open("current_outputs.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _summarize_terminal_result_for_prompt(command: str, result: Dict[str, Any]) -> str:
    exit_code = result.get("exit_code")
    stdout = result.get("stdout", "") or ""
    stderr = result.get("stderror", "") or ""
    parts = [f"TERMINAL COMMAND: {command}", f"exit_code={exit_code}"]
    if stdout.strip():
        parts.append("stdout_tail=\n" + _preview_text_block(stdout, max_chars=1200))
    if stderr.strip():
        parts.append("stderr_tail=\n" + _preview_text_block(stderr, max_chars=1600))
    return "\n\n".join(parts)


def review_latest_run_output(state: Dict[str, Any], label: str = "script run") -> None:
    """Offer an interactive terminal menu for reviewing the latest run output."""
    if not state.get("interactive_output_review", True):
        return

    stdout = state.get("last_stdout", "") or ""
    stderr = state.get("last_stderr", "") or ""
    exitcode = state.get("last_exitcode")

    _print_console_panel(f"Output Review · {label}", tone="review")
    _print_labeled_value("Exit", str(exitcode), tone="review")
    _print_labeled_value("Stdout chars", str(len(stdout)), tone="review")
    _print_labeled_value("Stderr chars", str(len(stderr)), tone="review")

    while True:
        choice = safe_input(
            "Output options [Enter continue, 1 stdout, 2 stderr, 3 combined tail, 4 saved log, 5 disable prompts]: "
        ).strip().lower()
        if choice in {"", "c", "continue"}:
            return
        if choice == "1":
            _print_console_panel("STDOUT", tone="review")
            print(_preview_text_block(stdout, max_chars=6000))
            continue
        if choice == "2":
            _print_console_panel("STDERR", tone="review")
            print(_preview_text_block(stderr, max_chars=6000))
            continue
        if choice == "3":
            combined = "\n".join(part for part in [stdout, stderr] if part)
            _print_console_panel("Combined Output", tone="review")
            print(_preview_text_block(combined, max_chars=7000))
            continue
        if choice == "4":
            history = _load_execution_history()
            latest = history[-1] if history else {}
            _print_console_panel("Saved Execution Log", tone="review")
            print(_preview_text_block(json.dumps(latest, indent=2, ensure_ascii=False), max_chars=7000))
            continue
        if choice == "5":
            state["interactive_output_review"] = False
            print("   Output review prompts disabled for the rest of this session.")
            return
        print("   Invalid choice.")


def _build_write_diff_preview(action: AgentAction, max_lines: int = 40) -> str:
    """Create a compact unified-diff preview for write approval."""
    if action.action_type != "WriteFileInput":
        return ""
    a = action.action
    if not a.file_path or a.start_line is None or a.end_line is None:
        return ""
    if not os.path.exists(a.file_path):
        return ""
    try:
        import difflib
        with open(a.file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        start_idx = max(0, a.start_line - 1)
        end_idx = min(len(lines), a.end_line)
        old_chunk = lines[start_idx:end_idx]
        new_chunk = a.new_content.splitlines(keepends=True)
        if old_chunk and not old_chunk[-1].endswith("\n"):
            old_chunk[-1] += "\n"
        if new_chunk and not new_chunk[-1].endswith("\n"):
            new_chunk[-1] += "\n"
        diff_lines = list(
            difflib.unified_diff(
                old_chunk,
                new_chunk,
                fromfile=f"{a.file_path}:{a.start_line}-{a.end_line}",
                tofile=f"{a.file_path}:{a.start_line}-{a.end_line} (proposed)",
                lineterm="",
            )
        )
        if len(diff_lines) > max_lines:
            diff_lines = diff_lines[:max_lines] + ["... (diff truncated)"]
        return "\n".join(diff_lines)
    except Exception as e:
        return f"(diff preview unavailable: {e})"


def _extract_terminal_command_from_text(text: str) -> str:
    compact = " ".join((text or "").split())
    patterns = [
        r'((?:python3? -m pip|pip3?|conda)\s+(?:install|uninstall|update|upgrade)\b.+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, compact, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def get_user_approval_for_action(action: AgentAction, state: Dict[str, Any]) -> tuple[bool, str]:
    """
    Interactive user gate for agent actions after an error is observed.
    Returns (approved, feedback_message_for_llm_if_rejected).
    """
    mutating_action_types = {"WriteFileInput", "ExecuteTerminalCommand"}
    requires_review = bool(state.get("awaiting_user_feedback", False) or action.action_type in mutating_action_types)

    if not requires_review:
        return True, ""

    if not state.get("interactive_action_review", True):
        return True, ""

    if action.action_type == "NoAction":
        return True, ""

    is_library_write = False
    if action.action_type == "WriteFileInput":
        is_library_write = is_library_or_utility_file(action.action.file_path)[0]

    _print_console_panel("Approval Required", tone="warn")
    _print_labeled_value("Proposed", _summarize_action_for_approval(action), tone="warn")
    if action.action_type == "WriteFileInput":
        diff_preview = _build_write_diff_preview(action)
        if diff_preview:
            print("\nProposed patch preview:")
            print(diff_preview)
    print("Options:")
    print("  [y] approve and execute")
    print("  [n] reject and provide guidance")
    print("  [p] reject, add suggested path to user-protected list, and re-plan")
    print("  [r] reject and force RAG / external investigation")
    if is_library_write:
        print("  [l] allow this exact library patch once")

    prompt = "Select option [y/n/p/r"
    if is_library_write:
        prompt += "/l"
    prompt += "]: "
    choice = safe_input(prompt).strip().lower() or "y"

    if choice in {"y", "yes"}:
        # Respect explicit user approval for comment-out fixes (one-shot bypass).
        if action.action_type == "WriteFileInput":
            a = action.action
            try:
                if a.start_line and a.end_line and os.path.exists(a.file_path):
                    with open(a.file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    start_idx = max(0, a.start_line - 1)
                    end_idx = min(len(lines), a.end_line)
                    if is_comment_out_fix(lines[start_idx:end_idx], (a.new_content or "")):
                        state["allow_comment_out_once"] = True
            except Exception:
                pass
        state["awaiting_user_feedback"] = False
        return True, ""

    if choice in {"r", "rag"}:
        state["user_requested_rag"] = True
        return False, "USER_FEEDBACK: Do not execute that action yet. Investigate externally with RAG first, then re-plan."

    if choice in {"l", "lib", "library"} and is_library_write:
        _allow_library_patch_once(state, action.action.file_path)
        state["awaiting_user_feedback"] = False
        return True, ""

    if choice in {"p", "protect"}:
        proposed_path = ""
        if action.action_type == "WriteFileInput":
            proposed_path = action.action.file_path
        path = safe_input(
            f"Path to protect [{proposed_path or 'enter path'}]: "
        ).strip() or proposed_path
        if path:
            _append_user_protected_path(state, path)
            feedback = (
                f"USER_FEEDBACK: Do not modify '{path}'. "
                f"It is now user-protected. Re-plan using another file or tool."
            )
        else:
            feedback = "USER_FEEDBACK: Action rejected. Re-plan using another tool."
        return False, feedback

    guidance = safe_input("Guidance for the agent: ").strip()
    guidance_lower = guidance.lower()
    if (
        "rag" in guidance_lower
        and any(k in guidance_lower for k in ["use", "call", "run", "trigger"])
    ):
        state["user_requested_rag"] = True
    if "comment out" in guidance_lower:
        state["prefer_comment_out"] = True
    preferred_command = _extract_terminal_command_from_text(guidance)
    if preferred_command:
        state["preferred_next_command"] = preferred_command
        guidance = f"{guidance}\nHonor the user's explicit requested command next: {preferred_command}"
    if not guidance:
        guidance = "Action rejected by user. Re-plan with a different fix strategy."
    feedback = f"USER_FEEDBACK: {guidance}"
    return False, feedback


RECENT_ACTIONS: list = []
MAX_REPEATED_ACTIONS = 3

def ask_action(
    llm_obj: Llama,
    goal: str,
    chat_history: List[Dict[str, str]],
    state: Dict[str, Any]
) -> AgentAction:
    """
    Ask the LLM which tool action to take next.

    IMPORTANT: Only the last 10 messages from chat_history are sent to the LLM
    to keep context manageable for smaller models (~15B parameters).
    """
    global RECENT_ACTIONS

    allowed = allowed_actions(state)
    state_summary = build_state_summary(state)
    debug_enabled = bool(state.get("debug", False))

    # ===== MESSAGE WINDOWING: Use only last 10 messages =====
    MAX_HISTORY_WINDOW = 10
    windowed_history = chat_history[-MAX_HISTORY_WINDOW:] if len(chat_history) > MAX_HISTORY_WINDOW else chat_history

    if debug_enabled:
        print(f"   🔍 DEBUG: Total history: {len(chat_history)} messages, sending last {len(windowed_history)} to LLM")
        if windowed_history:
            print(f"   🔍 DEBUG: Last user message: {windowed_history[-1]['content'][:200]}...")
    system = f"""You are an automated Python debugging agent.

GOAL:
{goal}

The main file is: {state.get("target_file","<unknown>")}

HARD CONSTRAINTS:
- You MUST choose an action whose action_type is in ALLOWED_ACTIONS.
- Output MUST be a single JSON object and MUST match the exact keys shown below.
- Do NOT use null for writing Python Code.
- Return ONLY JSON. No markdown. No extra text.
- Prefer small, reversible edits over large rewrites.
- If context is insufficient, choose ReadFileInput before WriteFileInput.
- Choose RAGSearchInput only after cyclic/repeated error detection, unless the user explicitly requests immediate RAG.
- Respect user-protected paths from STATE SUMMARY; never propose writing them.
- Human approval is required before mutating actions are executed. Keep proposals explicit and minimal.
- If STATE SUMMARY shows environment-resolution mode or a cached environment diagnostic summary, do not repeat the same successful pip show or conda list command for the same failure class. Choose a targeted repair command or a more specific diagnosis instead.

ALLOWED_ACTIONS:
{allowed}

{ACTION_SCHEMAS_TEXT}

STATE SUMMARY:
{state_summary}
"""

    # Use windowed history (last 10 messages) instead of full history
    base_messages = (
        [{"role": "system", "content": system}]
        + windowed_history
        + [{"role": "user", "content": state["current_input"]}]
    )

    def attempt(messages: List[Dict[str, str]]) -> AgentAction:
        raw = call_llama_for_action(llm_obj, messages)
        if debug_enabled:
            print(f"   🔍 DEBUG: Raw LLM response: {raw[:500]}...")
        
        # Fix unescaped quotes in new_content (common LLM error)
        # self.hparams["key"] -> self.hparams[\"key\"]
        # Fix unescaped quotes in new_content (common LLM error)
        # Pattern: self.hparams["key"] inside JSON string -> self.hparams[\"key\"]
        def fix_inner_quotes(text):
            # Find new_content field and fix quotes inside it
            match = re.search(r'"new_content"\s*:\s*"', text)
            if not match:
                return text
            
            start = match.end()
            # Find the closing quote (not escaped)
            depth = 0
            i = start
            while i < len(text):
                if text[i] == '\\' and i + 1 < len(text):
                    i += 2  # Skip escaped char
                    continue
                if text[i] == '"':
                    # Check if this is inside brackets (unescaped quote we need to fix)
                    # Look back for [ and forward for ]
                    break
                i += 1
            
            # Extract the content part
            content = text[start:i]
            # Fix: ["something"] -> [\"something\"]
            fixed = re.sub(r'\[(["\'])([^"\']*)\1\]', r'[\\"\2\\"]', content)
            
            return text[:start] + fixed + text[i:]
        
        raw = fix_inner_quotes(raw)

        
        js = extract_json(raw, default_script_path=state.get("target_file", "train_alignn.py"))
        js = sanitize_jsonish(js)
        js = js.replace("\\'", "'")
        obj = json.loads(js)
        obj = normalize_action_payload(obj, state)
        return AGENT_ACTION_ADAPTER.validate_python(obj)

    try:
        action = attempt(base_messages)
    except ValidationError:
        correction = (
            "Your JSON did not match the required schema keys. "
            f"ALLOWED_ACTIONS={allowed}. "
            "Return ONLY JSON that exactly matches one of the ACTION SCHEMAS."
        )
        try:
            action = attempt(base_messages + [{"role": "user", "content": correction}])
        except ValidationError:
            print("   ⚠️ Action schema validation failed twice; using safe fallback action")
            if "ReadFileInput" in allowed:
                fallback_obj = {
                    "action_type": "ReadFileInput",
                    "action": {
                        "file_path": state.get("target_file", ""),
                        "error_line": state.get("error_line", 0) or 0,
                        "window_before": 20,
                        "window_after": 30,
                    },
                }
            elif "RunScriptInput" in allowed:
                fallback_obj = {
                    "action_type": "RunScriptInput",
                    "action": {
                        "script_path": state.get("target_file", ""),
                        "args": state.get("run_args", []),
                    },
                }
            else:
                fallback_obj = {"action_type": "NoAction", "action": {"task_complete": False}}
            action = AGENT_ACTION_ADAPTER.validate_python(fallback_obj)

    if action.action_type not in allowed:
        bad = action.action_type
        correction = (
            f"Invalid action_type '{bad}'. You MUST choose one of {allowed}. "
            "Return ONLY JSON matching the ACTION SCHEMAS."
        )
        action = attempt(base_messages + [{"role": "user", "content": correction}])

    # === LOOP DETECTION ===
    action_signature = json.dumps(action.model_dump(), sort_keys=True)
    
    if len(RECENT_ACTIONS) >= MAX_REPEATED_ACTIONS:
        recent_sigs = [json.dumps(a, sort_keys=True) for a in RECENT_ACTIONS[-MAX_REPEATED_ACTIONS:]]
        
        if all(sig == action_signature for sig in recent_sigs):
            print(f"   ⚠️ LOOP DETECTED: Same action repeated {MAX_REPEATED_ACTIONS} times")
            
            # If stuck on WriteFileInput, the file is probably corrupted
            if action.action_type == "WriteFileInput":
                file_path = action.action.file_path
                
                # Try to restore from backup
                for backup_suffix in [".backup", ".backup1", ".backup2", ".backup3"]:
                    backup_path = f"{file_path}{backup_suffix}"
                    if os.path.exists(backup_path):
                        print(f"   🔄 Restoring {file_path} from {backup_path}")
                        import shutil
                        shutil.copy(backup_path, file_path)
                        break
                
                # Force a fresh run
                print(f"   🏃 Switching to RunScriptInput to get fresh error state")
                RECENT_ACTIONS.clear()
                
                return AGENT_ACTION_ADAPTER.validate_python({
                    "action_type": "RunScriptInput",
                    "action": {
                        "script_path": state.get("target_file", ""),
                        "args": state.get("run_args", [])
                    }
                })
            
            # If stuck on ReadFileInput, try running script
            elif action.action_type == "ReadFileInput":
                print(f"   🏃 Switching to RunScriptInput")
                RECENT_ACTIONS.clear()
                
                return AGENT_ACTION_ADAPTER.validate_python({
                    "action_type": "RunScriptInput",
                    "action": {
                        "script_path": state.get("target_file", ""),
                        "args": state.get("run_args", [])
                    }
                })

            elif action.action_type == "ExecuteTerminalCommand":
                RECENT_ACTIONS.clear()
                if state.get("env_ready_to_rerun") and "RunScriptInput" in allowed:
                    print("   🏃 Switching to RunScriptInput after repeated terminal command loop")
                    return AGENT_ACTION_ADAPTER.validate_python({
                        "action_type": "RunScriptInput",
                        "action": {
                            "script_path": state.get("target_file", ""),
                            "args": state.get("run_args", [])
                        }
                    })
                if "RAGSearchInput" in allowed:
                    last_error = state.get("full_error_history", [])[-1] if state.get("full_error_history") else {}
                    print("   🌐 Switching to RAGSearchInput after repeated terminal command loop")
                    return AGENT_ACTION_ADAPTER.validate_python({
                        "action_type": "RAGSearchInput",
                        "action": {
                            "error_type": last_error.get("error_type", ""),
                            "error_message": last_error.get("message", ""),
                            "file_path": last_error.get("file", state.get("target_file", "")),
                            "error_line": last_error.get("line", state.get("error_line", 0) or 0),
                            "code_context": "",
                        }
                    })
    
    RECENT_ACTIONS.append(action.model_dump())
    if len(RECENT_ACTIONS) > 10:
        RECENT_ACTIONS.pop(0)
    # === END LOOP DETECTION ===

    return action


from system_analysis import set_llm_instance

# Global reference to LLM instance for RAG pipeline
_rag_llm_instance = None

def set_rag_llm_instance(llm):
    """Set the LLM instance for RAG pipeline to use."""
    global _rag_llm_instance
    _rag_llm_instance = llm

def get_rag_llm_instance():
    """Get the LLM instance for RAG pipeline."""
    global _rag_llm_instance
    return _rag_llm_instance


# ===================================================================
# LLM-BASED FIX SUGGESTION (Separate reasoning session)
# ===================================================================

def llm_suggest_fix(
    error_type: str,
    error_message: str,
    code_context: str,
    similar_keys: List[str],
    class_name: str = "",
    file_path: str = "",
    runtime_context: Dict[str, Any] | None = None,
) -> str:
    """
    Use a separate LLM call to analyze the error context and suggest a fix.
    This provides flexible reasoning instead of hardcoded pattern matching.

    Returns:
        A string containing the LLM's suggested fix explanation.
    """
    llm = get_rag_llm_instance()
    if llm is None:
        return "LLM not available for fix suggestion."

    runtime_context = runtime_context or {}

    # Build a focused prompt for fix suggestion
    prompt = f"""You are a debugging assistant. Analyze this error and suggest a specific fix.

ERROR: {error_type}: {error_message}
FILE: {file_path}
CLASS: {class_name or "N/A"}

CODE CONTEXT (around the error):
```python
{code_context}
```

"""

    # Add similar keys info for KeyError
    if error_type == "KeyError" and similar_keys:
        missing_key = error_message.strip("'\"")
        prompt += f"""
IMPORTANT CONTEXT - Valid keys found in nearby code:
{similar_keys}

Notice any naming patterns in these valid keys (common prefixes, suffixes, conventions).
The missing key '{missing_key}' is probably a typo or uses the wrong naming convention.
"""

    if runtime_context:
        ctx_items = []
        for key in ["cuda_available", "torch_version", "CUDA_HOME", "CUDAToolkit_ROOT", "nvcc_version", "cuda_device_name", "cuda_capability"]:
            value = runtime_context.get(key)
            if value not in (None, ""):
                ctx_items.append(f"{key}={value}")
        if ctx_items:
            prompt += "\nRUNTIME CONTEXT:\n" + "\n".join(ctx_items) + "\n"

    prompt += """
TASK: Suggest a SPECIFIC fix. Be concise.
- For KeyError: What should the correct key name be? Look at the pattern in valid keys.
- For NameError: What is the correct variable name?
- For ImportError/ModuleNotFoundError on runtime libraries like torch: suggest an install that matches the recorded runtime/toolkit context instead of defaulting to the plain CPU package.
- For other errors: What code change would fix this?

Reply with ONLY the fix suggestion in 1-3 sentences. Start with "FIX:" """

    try:
        print(f"   🤖 Asking LLM to analyze error and suggest fix...")
        messages = [
            {"role": "system", "content": "You are a concise debugging assistant. Suggest specific fixes."},
            {"role": "user", "content": prompt}
        ]

        resp = llm.create_chat_completion(
            messages=messages,
            temperature=0.1,  # Low temperature for focused answers
            max_tokens=200,
            top_k=1,
            top_p=1.0,
        )
        suggestion = resp["choices"][0]["message"]["content"].strip()
        print(f"   💡 LLM Suggestion: {suggestion[:100]}...")
        return suggestion

    except Exception as e:
        print(f"   ⚠️ LLM fix suggestion failed: {e}")
        return f"Could not generate suggestion: {e}"


# ===================================================================
# ENHANCED RAG SEARCH WITH ACTIONABLE FIXES
# ===================================================================

def enhanced_rag_search(
    error_type: str,
    error_message: str,
    file_path: str,
    error_line: int,
    similar_keys: list = None
) -> dict:
    """
    Enhanced RAG that actually provides actionable fixes, not just hints.
    
    The key insight: For KeyError, we already have the valid keys from
    code analysis. We don't need external search - we need to USE the
    internal analysis results more effectively.
    """
    import difflib
    
    results = {
        "actionable_fix": None,
        "confidence": "low",
        "similar_keys": similar_keys or [],
        "suggested_replacement": None,
    }
    
    if error_type == "KeyError" and similar_keys:
        missing_key = error_message.strip("'\"")
        
        # Strategy 1: Look for key with similar suffix
        missing_parts = missing_key.split('_')
        
        for key in similar_keys:
            key_parts = key.split('_')
            # Check if suffix matches (e.g., "key_name" vs "prefix_key_name")
            if len(missing_parts) >= 2 and len(key_parts) >= 2:
                if missing_parts[-1] == key_parts[-1] and missing_parts[-2] == key_parts[-2]:
                    results["suggested_replacement"] = key
                    results["confidence"] = "high"
                    results["actionable_fix"] = (
                        f"Replace '{missing_key}' with '{key}'. "
                        f"The key uses a different prefix convention."
                    )
                    break
        
        # Strategy 2: Look for common prefix pattern
        if not results["suggested_replacement"]:
            prefix_counts = {}
            for key in similar_keys:
                parts = key.split('_')
                if parts:
                    prefix = parts[0]
                    prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
            
            if prefix_counts:
                common_prefix = max(prefix_counts, key=prefix_counts.get)
                if not missing_key.startswith(common_prefix + '_'):
                    # Missing key doesn't have the common prefix
                    suggested = common_prefix + '_' + missing_key
                    # Check if this suggested key exists
                    if suggested in similar_keys:
                        results["suggested_replacement"] = suggested
                        results["confidence"] = "high"
                        results["actionable_fix"] = (
                            f"Add prefix '{common_prefix}_' to make '{suggested}'"
                        )
                    else:
                        # Find closest match with this prefix
                        for key in similar_keys:
                            if key.startswith(common_prefix + '_'):
                                # Check if rest is similar
                                key_rest = key[len(common_prefix)+1:]
                                if any(p in key_rest for p in missing_parts[1:] if len(p) > 2):
                                    results["suggested_replacement"] = key
                                    results["confidence"] = "medium"
                                    results["actionable_fix"] = (
                                        f"Similar key found: '{key}'"
                                    )
                                    break
        
        # Strategy 3: Fuzzy match on key names
        if not results["suggested_replacement"]:
            matches = difflib.get_close_matches(missing_key, similar_keys, n=1, cutoff=0.4)
            if matches:
                results["suggested_replacement"] = matches[0]
                results["confidence"] = "medium"
                results["actionable_fix"] = (
                    f"Closest matching key: '{matches[0]}'"
                )
    
    return results


def build_keyerror_fix_prompt(
    error_message: str,
    similar_keys: list,
    code_context: str,
    file_path: str,
    error_line: int,
    suggested_replacement: str = None
) -> str:
    """
    Build a strongly structured prompt that FORCES the LLM to use valid keys.
    """
    missing_key = error_message.strip("'\"")
    
    # Get the original line's indentation from code context
    indent = "        "  # Default 8 spaces for method body
    for line in code_context.split('\n'):
        if missing_key in line:
            indent = " " * (len(line) - len(line.lstrip()))
            break
    
    prompt = f"""⚠️ KEYERROR FIX REQUIRED ⚠️

ERROR: KeyError: '{missing_key}'
FILE: {file_path}
LINE: {error_line}

CODE CONTEXT:
```python
{code_context}
```

═══════════════════════════════════════════════════════════════════
VALID KEYS THAT EXIST IN THIS CODEBASE:
{similar_keys}
═══════════════════════════════════════════════════════════════════

"""
    
    if suggested_replacement:
        prompt += f"""
💡 SUGGESTED FIX:

The correct key is likely: '{suggested_replacement}'

Replace the incorrect key '{missing_key}' with '{suggested_replacement}' on line {error_line}.

⚠️ BANNED PATTERNS (ALL will be REJECTED):
- .get("{missing_key}", ANY_VALUE)  ← REJECTED: .get() does not fix KeyError
- .get("{suggested_replacement}", ANY_VALUE)  ← REJECTED: .get() does not fix KeyError
- ANY use of .get() ← REJECTED: must use direct key access

✅ CORRECT APPROACH:
- Use direct key access with the correct key: ["{suggested_replacement}"]
- PRESERVE the original line structure and indentation exactly
- Only change the key name, nothing else

"""
    else:
        prompt += f"""
⚠️ CRITICAL INSTRUCTIONS:
1. The key '{missing_key}' does NOT exist
2. You MUST use one of the VALID KEYS listed above
3. Do NOT use placeholder values like 'default_value'
4. Do NOT use .get() with any fallback - use direct key access: obj["key"]
5. PRESERVE the original indentation exactly

"""
    
    return prompt


# ===================================================================
# RAG PIPELINE FOR CYCLIC ERROR RESOLUTION
# ===================================================================

def extract_context_from_file(file_path: str, error_line: int, error_message: str, error_type: str = "") -> Dict[str, Any]:
    """
    Extract useful context from the file for smarter RAG queries.
    Returns library hints, class context, and similar patterns found.

    Args:
        file_path: Path to the file with the error
        error_line: Line number where the error occurred
        error_message: The error message (e.g., "'some_key'")
        error_type: The error type (e.g., "KeyError", "NameError")
    """
    context = {
        "library_hints": [],
        "class_name": None,
        "similar_keys": [],  # For KeyError: keys used successfully nearby
        "similar_patterns": [],
        "imports": []
    }

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Extract imports for library context
        # Exclude standard library modules that aren't useful for search
        stdlib_exclude = [
            'os', 'sys', 're', 'json', 'typing', '__future__', 'collections',
            'functools', 'itertools', 'pathlib', 'io', 'copy', 'math', 'time',
            'datetime', 'logging', 'warnings', 'abc', 'dataclasses', 'enum'
        ]
        for line in lines[:100]:  # Check first 100 lines for imports
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                context["imports"].append(line.strip())
                # Extract library names
                if 'import ' in line:
                    parts = line.replace('from ', '').replace('import ', '').split()
                    if parts:
                        lib = parts[0].split('.')[0]
                        if lib not in stdlib_exclude:
                            context["library_hints"].append(lib)

        # Find class context (search backwards from error line)
        for i in range(min(error_line - 1, len(lines) - 1), -1, -1):
            line = lines[i]
            if line.strip().startswith('class '):
                match = re.match(r'class\s+(\w+)', line.strip())
                if match:
                    context["class_name"] = match.group(1)
                    break

        # For KeyError: find similar dictionary key accesses nearby
        if error_type == "KeyError" or "KeyError" in str(error_message):
            # Get the dict/object name from error line
            if error_line <= len(lines):
                error_code = lines[error_line - 1]
                # Find dict access patterns like: self.hparams["key"], obj.config["key"], dict["key"]
                # The (?:\w+\.)? handles optional prefix like "self." or "obj."
                dict_match = re.search(r'(?:\w+\.)?(\w+)\[[\'"](.*?)[\'"]\]', error_code)
                if dict_match:
                    dict_name = dict_match.group(1)  # e.g., "hparams" from "self.hparams"
                    missing_key = dict_match.group(2)  # The key that caused the error
                    print(f"   🔍 Analyzing dict '{dict_name}' for similar keys...")

                    # Search nearby lines for other accesses to same dict
                    search_start = max(0, error_line - 30)  # Look further back
                    search_end = min(len(lines), error_line + 10)

                    for i in range(search_start, search_end):
                        line = lines[i]
                        # Find all key accesses to the same dict (with optional self./obj. prefix)
                        pattern = rf'(?:\w+\.)?{dict_name}\[[\'"](\w+)[\'"]\]'
                        matches = re.findall(pattern, line)
                        for key in matches:
                            if key not in context["similar_keys"] and key != missing_key:
                                context["similar_keys"].append(key)

                    # Print what we found for debugging
                    if context["similar_keys"]:
                        print(f"   📋 Found {len(context['similar_keys'])} valid keys: {context['similar_keys'][:10]}")
    except Exception as e:
        print(f"   ⚠️ Context extraction failed: {e}")

    return context


def _load_runtime_cuda_context() -> Dict[str, Any]:
    """Load a compact CUDA/toolchain snapshot for CUDA-related RAG queries."""
    context: Dict[str, Any] = {}
    try:
        if os.path.exists("system_config.json"):
            with open("system_config.json", "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for key in [
                    "torch_version",
                    "cuda_available",
                    "cuda_device_name",
                    "cuda_capability",
                    "nvcc_version",
                ]:
                    if key in raw:
                        context[key] = raw[key]
    except Exception:
        pass

    for env_key in ["CUDA_HOME", "CUDAToolkit_ROOT", "LLAMA_CUDA_ARCH"]:
        value = os.environ.get(env_key)
        if value:
            context[env_key] = value
    return context


def _extract_import_statement(code_context: str, missing_import: str) -> str:
    missing_root = (missing_import or "").strip().split(".")[0]
    for line in (code_context or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        if missing_root and missing_root in stripped:
            return stripped
    return ""


def _extract_missing_import_name(error_message: str) -> str:
    message = str(error_message or "").strip()
    patterns = [
        r"No module named '([A-Za-z0-9_.-]+)'",
        r'No module named "([A-Za-z0-9_.-]+)"',
        r"cannot import name '([A-Za-z0-9_.-]+)' from '([A-Za-z0-9_.-]+)'",
        r'cannot import name "([A-Za-z0-9_.-]+)" from "([A-Za-z0-9_.-]+)"',
        r"from '([A-Za-z0-9_.-]+)'",
        r'from "([A-Za-z0-9_.-]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            groups = [g for g in match.groups() if g]
            if not groups:
                continue
            if len(groups) >= 2:
                return groups[-1].strip().lower()
            return groups[0].strip().lower()
    return message.strip().strip("\'\"").lower()


def _normalize_torch_install_command(command: str, runtime_context: Dict[str, Any]) -> str | None:
    normalized = " ".join((command or "").split()).strip()
    lower_command = normalized.lower()
    if "torch" not in lower_command and "pytorch" not in lower_command:
        return None
    if not any(token in lower_command for token in ["pip install", "pip3 install", "python -m pip install", "python3 -m pip install", "conda install"]):
        return None
    recommended_command, _ = _recommended_install_command_for_import("torch", runtime_context)
    return recommended_command


def _extract_import_path_from_statement(import_statement: str, missing_import: str) -> str:
    statement = (import_statement or "").strip()
    if statement.startswith("from "):
        match = re.match(r"from\s+([A-Za-z0-9_\.]+)\s+import\s+", statement)
        if match:
            return match.group(1)
    if statement.startswith("import "):
        match = re.match(r"import\s+([A-Za-z0-9_\.]+)", statement)
        if match:
            return match.group(1)
    return (missing_import or "").strip()


def _clean_package_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "", (name or "").strip())


def _is_generic_import_install_command(command: str, missing_import: str) -> bool:
    normalized = " ".join((command or "").split()).strip()
    lower_command = normalized.lower()
    target = (missing_import or "").strip().lower().split(".")[0]
    if not target:
        return False
    generic_variants = {
        f"pip install {target}",
        f"pip3 install {target}",
        f"python -m pip install {target}",
        f"python3 -m pip install {target}",
        f"conda install {target}",
        f"conda install -c conda-forge {target}",
    }
    return lower_command in generic_variants


def _env_repair_command_mentions_allowed_packages(command: str, state: Dict[str, Any]) -> bool:
    normalized = _normalize_command_text(command).lower()
    allowed: List[str] = []
    for pkg in state.get("env_resolution_packages", []) or []:
        clean = _clean_package_name(pkg).lower()
        if clean and clean not in allowed:
            allowed.append(clean)
    root = _extract_site_package_root((state.get("last_env_issue", {}) or {}).get("error_file", ""))
    for candidate in _build_namespace_family_candidates(root):
        clean = _clean_package_name(candidate).lower()
        if clean and clean not in allowed:
            allowed.append(clean)
    return any(token and token in normalized for token in allowed)


def _is_allowed_env_repair_command(command: str, state: Dict[str, Any]) -> bool:
    normalized = _normalize_command_text(command).lower()
    namespace_hint = state.get("env_namespace_shadow_hint", {}) or {}
    root = (namespace_hint.get("root") or "").lower()
    if namespace_hint.get("suspicious") and root:
        uninstall_patterns = [
            f"python -m pip uninstall -y {root}",
            f"python3 -m pip uninstall -y {root}",
            f"pip uninstall -y {root}",
            f"pip3 uninstall -y {root}",
        ]
        return any(pattern in normalized for pattern in uninstall_patterns)
    return _env_repair_command_mentions_allowed_packages(normalized, state)


def _infer_package_candidates_for_import(
    missing_import: str,
    import_statement: str,
    retrieved_snippets: List[Dict[str, str]],
    env_context: Dict[str, Any] | None = None,
) -> List[Dict[str, str]]:
    import_root = (missing_import or "").strip().lower().split(".")[0]
    import_path = _extract_import_path_from_statement(import_statement, missing_import)
    env_context = env_context or {}
    env_packages = env_context.get("packages", []) or []
    package_scores: Dict[tuple[str, str], Dict[str, Any]] = {}

    def add_candidate(package: str, ecosystem: str, source_url: str, evidence: str, score: int) -> None:
        package = _clean_package_name(package)
        if not package:
            return
        key = (package.lower(), ecosystem)
        entry = package_scores.setdefault(
            key,
            {
                "package": package,
                "ecosystem": ecosystem,
                "score": 0,
                "source_url": source_url,
                "evidence": [],
            },
        )
        entry["score"] += score
        if source_url and not entry.get("source_url"):
            entry["source_url"] = source_url
        if evidence and evidence not in entry["evidence"]:
            entry["evidence"].append(evidence)

    for pkg in env_packages:
        pkg_clean = _clean_package_name(pkg)
        pkg_lower = pkg_clean.lower()
        normalized_pkg = pkg_lower.replace("-", "_")
        if not import_root:
            continue
        if normalized_pkg == import_root or import_root in normalized_pkg or normalized_pkg.startswith(import_root + "_"):
            ecosystem = "conda" if ("-" in pkg_clean and pkg_lower != import_root) else "pip"
            score = 10 if pkg_lower != import_root else 7
            evidence = "Installed distribution in current environment matches import root"
            if pkg_lower.endswith(("-tools", "-python", "_tools", "_python")):
                score += 4
                evidence = "Installed distribution in current environment likely provides the import namespace"
            add_candidate(pkg_clean, ecosystem, "", evidence, score)

    for item in retrieved_snippets:
        url = item.get("url", "") or ""
        title = item.get("title", "") or ""
        excerpt = item.get("excerpt", "") or ""
        combined = " ".join([title, excerpt, url])
        combined_lower = combined.lower()
        source_bonus = 0

        if "pypi.org/project/" in url:
            match = re.search(r"pypi\.org/project/([A-Za-z0-9_.-]+)", url, re.IGNORECASE)
            if match:
                add_candidate(match.group(1), "pip", url, "PyPI project page", 7)
        if "anaconda.org/" in url:
            match = re.search(r"anaconda\.org/([^/]+)/([A-Za-z0-9_.-]+)", url, re.IGNORECASE)
            if match:
                channel = match.group(1)
                pkg = match.group(2)
                add_candidate(pkg, "conda", url, f"Anaconda package page ({channel})", 7 if channel == "conda-forge" else 5)
        if "github.com/" in url:
            source_bonus = 1

        for match in re.findall(r"(?:python\s+-m\s+pip|pip3?|uv\s+pip)\s+install\s+([A-Za-z0-9_.-]+)", combined, re.IGNORECASE):
            add_candidate(match, "pip", url, "Install command in search result", 4 + source_bonus)
        for match in re.findall(r"conda\s+install(?:\s+-c\s+[A-Za-z0-9_.-]+)*(?:\s+--[A-Za-z0-9-]+(?:\s+\S+)*)*\s+([A-Za-z0-9_.-]+)", combined, re.IGNORECASE):
            add_candidate(match, "conda", url, "Conda install command in search result", 4 + source_bonus)

        if import_root:
            if import_root in combined_lower:
                source_bonus += 1
            if import_path and import_path.lower() in combined_lower:
                source_bonus += 2

        for word in re.findall(r"\b[A-Za-z][A-Za-z0-9_.-]{2,}\b", title):
            lower_word = word.lower()
            if import_root and import_root not in lower_word and lower_word not in import_root:
                continue
            if lower_word in {"python", "github", "conda", "forge", "module", "import", "error"}:
                continue
            add_candidate(word, "unknown", url, "Title token related to import", 1 + source_bonus)

    candidates: List[Dict[str, str]] = []
    for entry in sorted(package_scores.values(), key=lambda item: (-item["score"], item["package"].lower())):
        package = entry["package"]
        ecosystem = entry["ecosystem"]
        if ecosystem == "conda":
            recommended_command = f"conda install -c conda-forge {package}"
        else:
            recommended_command = f"python -m pip install {package}"
        candidates.append(
            {
                "package": package,
                "ecosystem": ecosystem,
                "recommended_command": recommended_command,
                "source_url": entry.get("source_url", ""),
                "evidence": "; ".join(entry.get("evidence", [])[:3]),
                "score": str(entry["score"]),
            }
        )
    return candidates[:5]


def _is_cuda_related_issue(error_type: str, error_message: str, code_context: str = "", file_path: str = "") -> bool:
    haystack = " ".join([
        error_type or "",
        error_message or "",
        code_context or "",
        file_path or "",
    ]).lower()
    tokens = [
        "cuda", "cudnn", "cublas", "nvcc", "nvidia", "cudatoolkit",
        "cudart", "ggml_cuda", "cublas", "compute_", "sm_", "gpu architecture",
    ]
    return any(tok in haystack for tok in tokens)


def _recommended_install_command_for_import(missing_import: str, runtime_context: Dict[str, Any]) -> tuple[str | None, str | None]:
    missing = (missing_import or "").strip().lower()
    runtime_context = runtime_context or {}
    if missing not in {"torch", "pytorch"}:
        return None, None

    torch_version = str(runtime_context.get("torch_version", "") or "")
    cuda_available = bool(runtime_context.get("cuda_available"))
    cuda_spec = None
    match = re.search(r"\+cu(\d{2,3})", torch_version)
    if match:
        digits = match.group(1)
        if len(digits) == 3:
            cuda_spec = f"{digits[:2]}.{digits[2:]}"
        elif len(digits) == 2:
            cuda_spec = f"{digits[0]}.{digits[1:]}"

    if cuda_available and cuda_spec:
        return (
            f"conda install -c pytorch -c nvidia pytorch pytorch-cuda={cuda_spec}",
            f"local runtime shows torch_version={torch_version} and cuda_available={cuda_available}",
        )
    if cuda_available:
        return (
            "conda install -c pytorch -c nvidia pytorch pytorch-cuda",
            "local runtime reports CUDA is available, so prefer a CUDA-enabled PyTorch build",
        )
    return (
        "conda install -c pytorch pytorch cpuonly",
        "local runtime does not report CUDA availability, so prefer a CPU-only PyTorch build",
    )


def _extract_import_error_details(error_message: str) -> Dict[str, str]:
    message = str(error_message or "")
    patterns = [
        r"cannot import name '([A-Za-z0-9_.-]+)' from '([A-Za-z0-9_.-]+)'",
        r'cannot import name "([A-Za-z0-9_.-]+)" from "([A-Za-z0-9_.-]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return {
                "imported_symbol": match.group(1),
                "provider_package": match.group(2),
            }
    return {"imported_symbol": "", "provider_package": ""}


def _parse_pip_show_output(text: str) -> Dict[str, Dict[str, str]]:
    parsed: Dict[str, Dict[str, str]] = {}
    current: Dict[str, str] = {}
    current_name = ""
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            if current_name:
                parsed[current_name.lower()] = current
            current = {}
            current_name = ""
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        current[key] = value
        if key == "name":
            current_name = value
    if current_name:
        parsed[current_name.lower()] = current
    return parsed


def _summarize_installed_packages(package_info: Dict[str, Dict[str, str]]) -> str:
    if not package_info:
        return ""
    parts: List[str] = []
    for name in sorted(package_info):
        info = package_info[name]
        version = info.get("version", "?")
        location = info.get("location", "")
        label = f"{info.get('name', name)}=={version}"
        if location:
            label += f" @ {location}"
        parts.append(label)
    return "; ".join(parts[:6])


def _normalize_command_text(command: Any) -> str:
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return " ".join(str(command or "").split())


def _is_install_command(command: Any) -> bool:
    text = f" {_normalize_command_text(command).lower()} "
    return " install " in text or " uninstall " in text or " update " in text or " upgrade " in text


def _is_env_diagnostic_command(command: Any) -> bool:
    normalized = _normalize_command_text(command).lower()
    diagnostic_markers = [
        "python -m pip show",
        "python3 -m pip show",
        "pip show",
        "pip3 show",
        "python -m pip list",
        "python3 -m pip list",
        "pip list",
        "pip3 list",
        "conda list",
    ]
    return any(marker in normalized for marker in diagnostic_markers)


def _normalize_error_signature(error_info: Dict[str, Any], state: Dict[str, Any]) -> str:
    env_issue = state.get("last_env_issue", {}) or {}
    packages = env_issue.get("packages", []) or state.get("env_resolution_packages", []) or []
    normalized_message = " ".join(str(error_info.get("message", "") or "").split())
    return json.dumps(
        {
            "error_type": error_info.get("error_type", ""),
            "error_file": error_info.get("file", ""),
            "error_line": error_info.get("line", 0),
            "error_message": normalized_message[:240],
            "env_packages": packages[:8],
        },
        sort_keys=True,
    )


def _record_env_diagnostic_result(
    state: Dict[str, Any],
    error_info: Dict[str, Any],
    command: Any,
    result: Dict[str, Any],
) -> None:
    command_text = _normalize_command_text(command)
    stdout = str(result.get("stdout", "") or "")
    stderr = str(result.get("stderror", "") or "")
    package_info = _parse_pip_show_output(stdout)
    summary = _summarize_installed_packages(package_info)
    if not summary:
        compact = " ".join(part for part in [stdout.strip(), stderr.strip()] if part)
        summary = compact[:400]

    cache_key = f"{_normalize_error_signature(error_info, state)}::{command_text}"
    cache = state.setdefault("env_diagnostic_cache", {})
    cache[cache_key] = {
        "command": command_text,
        "summary": summary,
        "package_info": package_info,
        "step": state.get("current_step", 0),
        "exit_code": result.get("exit_code"),
    }
    state["last_env_diagnostic_summary"] = summary
    if package_info:
        state["env_package_versions"] = {
            name: info.get("version", "")
            for name, info in package_info.items()
            if info.get("version")
        }
    state["env_namespace_shadow_hint"] = _detect_namespace_shadowing(state, package_info)


def _build_redundant_env_diagnostic_block_message(
    state: Dict[str, Any],
    command: Any,
    cached_entry: Dict[str, Any],
) -> str:
    summary = cached_entry.get("summary", "") or "previous diagnostic had no compact summary"
    state["env_diag_blocked_count"] = state.get("env_diag_blocked_count", 0) + 1
    return (
        "Repeated successful environment diagnostic blocked.\n"
        f"The command {_normalize_command_text(command)} already succeeded for this same failure class.\n"
        f"Cached result: {summary}\n"
        "Do not rerun the same generic inspection.\n"
        "Choose one of these next:\n"
        "1. Execute a targeted repair command in the USER environment.\n"
        "2. Run a more specific diagnosis that tests a concrete compatibility hypothesis.\n"
        "3. Use RAG to look up package/version compatibility before proposing the repair."
    )


def _check_redundant_env_diagnostic(
    state: Dict[str, Any],
    error_info: Dict[str, Any],
    command: Any,
) -> tuple[bool, str]:
    cache = state.get("env_diagnostic_cache", {}) or {}
    command_text = _normalize_command_text(command)
    cache_key = f"{_normalize_error_signature(error_info, state)}::{command_text}"
    cached_entry = cache.get(cache_key)
    if not cached_entry:
        return False, ""
    if cached_entry.get("exit_code") != 0:
        return False, ""
    if state.get("last_env_repair_step", -1) > cached_entry.get("step", -1):
        return False, ""
    return True, _build_redundant_env_diagnostic_block_message(state, command, cached_entry)


def search_online_for_solution(
    error_type: str,
    error_message: str,
    code_context: str = "",
    file_path: str = "",
    error_line: int = 0,
    env_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Search for likely fixes using lightweight web retrieval plus local code context."""
    from html import unescape
    from urllib.parse import quote, urlparse
    from urllib.request import Request, urlopen

    print(f"\n🔍 RAG PIPELINE: Searching online for solutions to {error_type}")

    results = {
        "solutions": [],
        "sources": [],
        "search_queries": [],
        "code_analysis": {},
        "retrieved_snippets": [],
        "runtime_context": {},
        "package_candidates": [],
        "environment_facts": {},
    }
    env_context = env_context or {}

    cuda_related = _is_cuda_related_issue(error_type, error_message, code_context, file_path)
    missing_import_name = ""
    if error_type in {"ImportError", "ModuleNotFoundError"}:
        missing_import_name = _extract_missing_import_name(error_message)
    import_error_details = _extract_import_error_details(error_message)
    runtime_sensitive_import = any(token in missing_import_name for token in ["torch", "pytorch", "cuda", "cudnn", "cublas", "nvidia"])
    if cuda_related or runtime_sensitive_import:
        results["runtime_context"] = _load_runtime_cuda_context()
    if env_context:
        results["environment_facts"] = env_context

    file_context = {}
    if file_path and error_line > 0:
        file_context = extract_context_from_file(file_path, error_line, error_message, error_type)
        results["code_analysis"] = file_context
        if file_context.get("similar_keys"):
            print(f"   📋 Found similar keys in code: {file_context['similar_keys'][:10]}")

    library_hint = ""
    if file_path:
        fname = os.path.basename(file_path).lower()
        if "gguf" in fname or "llama" in fname:
            library_hint = "llama.cpp gguf"
        elif any(token in fname for token in ["torch", "pytorch"]):
            library_hint = "pytorch"
        elif any(token in fname for token in ["hf", "huggingface", "transformers"]):
            library_hint = "huggingface transformers"
        elif "tensorflow" in fname or fname.startswith("tf"):
            library_hint = "tensorflow"

    if not library_hint and file_context.get("library_hints"):
        library_hint = " ".join(file_context["library_hints"][:2])

    identifiers: List[str] = []
    for match in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", code_context or ""):
        if match not in identifiers and match.lower() not in {"self", "true", "false", "none"}:
            identifiers.append(match)
    identifiers = identifiers[:5]

    class_hint = file_context.get("class_name", "")
    basename = os.path.basename(file_path) if file_path else ""
    base_error = f"{error_type}: {error_message}".strip()
    key_name = error_message.strip("'\"") if error_type == "KeyError" else ""
    import_statement = ""
    import_path = ""
    import_root = ""
    provider_package = import_error_details.get("provider_package", "")
    imported_symbol = import_error_details.get("imported_symbol", "")
    if error_type in {"ImportError", "ModuleNotFoundError"}:
        import_statement = _extract_import_statement(code_context, missing_import_name)
        import_path = _extract_import_path_from_statement(import_statement, missing_import_name)
        import_root = (import_path or missing_import_name).split(".")[0]

    queries: List[str] = []

    def add_query(query: str) -> None:
        query = " ".join(query.split())
        if query and query not in queries:
            queries.append(query)

    add_query(f"python {base_error}")
    if library_hint:
        add_query(f"python {library_hint} {base_error}")
    if basename:
        add_query(f"python {basename} {error_type} {error_message[:80]}")
    if class_hint:
        add_query(f"python {class_hint} {error_type} {error_message[:80]}")
    if identifiers:
        add_query(f"python {' '.join(identifiers[:3])} {error_type} {error_message[:60]}")
    if key_name:
        add_query(f"python KeyError {key_name} config")
        if library_hint:
            add_query(f"{library_hint} KeyError {key_name}")
    if error_type in {"ImportError", "ModuleNotFoundError"}:
        missing_import = (error_message or "").strip().strip("\'\"")
        add_query(f"python package for import {missing_import}")
        if error_type == "ModuleNotFoundError":
            add_query(f"ModuleNotFoundError {missing_import} pip install")
        if import_statement:
            add_query(f"python {import_statement}")
            add_query(f"site:pypi.org/project \"{import_statement}\"")
            add_query(f"site:anaconda.org \"{import_statement}\"")
            add_query(f"site:github.com \"{import_statement}\"")
        if import_path:
            add_query(f"python package for import path {import_path}")
            add_query(f"site:pypi.org/project \"{import_path}\"")
            add_query(f"site:anaconda.org \"{import_path}\"")
            add_query(f"site:github.com \"{import_path}\"")
        if import_root:
            add_query(f"site:pypi.org/project {import_root}")
            add_query(f"site:anaconda.org/conda-forge {import_root}")
        if provider_package:
            add_query(f'python "cannot import name" "{imported_symbol}" "{provider_package}"')
            add_query(f'site:github.com "{provider_package}" "{imported_symbol}" importerror')
            add_query(f'site:pypi.org/project "{provider_package}"')
        if any(token in missing_import.lower() for token in ["torch", "pytorch"]):
            add_query(f"pytorch install selector {missing_import}")
            add_query(f"pytorch cuda install {missing_import}")
        if file_path:
            add_query(f"python {os.path.basename(file_path)} import {missing_import}")
    env_packages = env_context.get("packages", []) or []
    env_versions = env_context.get("package_versions", {}) or {}
    for pkg in env_packages[:4]:
        version = env_versions.get(pkg.lower()) or env_versions.get(pkg) or ""
        if version:
            add_query(f"python {pkg} {version} {error_type} {error_message[:80]}")
        if provider_package and pkg.lower() != provider_package.lower():
            add_query(f"python {pkg} {provider_package} compatibility {error_message[:80]}")

    if cuda_related or runtime_sensitive_import:
        cuda_ctx = results.get("runtime_context", {})
        nvcc = cuda_ctx.get("nvcc_version", "")
        arch = cuda_ctx.get("LLAMA_CUDA_ARCH", "")
        home = cuda_ctx.get("CUDA_HOME", "")
        if nvcc:
            add_query(f"{base_error} {nvcc}")
        if arch:
            add_query(f"{base_error} CMAKE_CUDA_ARCHITECTURES {arch}")
        if home:
            add_query(f"{base_error} CUDA_HOME {home}")

    results["search_queries"] = queries[:8]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0 Safari/537.36"
        )
    }

    def _fetch_text(url: str, timeout: int = 6) -> str:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")

    def _strip_html(html: str) -> str:
        html = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(r"<style.*?>.*?</style>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(r"<[^>]+>", " ", html)
        html = unescape(html)
        return " ".join(html.split())

    def _fetch_search_snippets(query: str, limit: int = 3) -> List[Dict[str, str]]:
        snippets: List[Dict[str, str]] = []
        encoded = quote(query)
        search_urls = [
            f"https://duckduckgo.com/html/?q={encoded}",
            f"https://lite.duckduckgo.com/lite/?q={encoded}",
        ]
        seen: set[str] = set()
        for search_url in search_urls:
            try:
                html = _fetch_text(search_url)
            except Exception:
                continue

            anchor_matches = re.findall(
                r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                html,
                flags=re.IGNORECASE | re.DOTALL,
            )
            for href, title_html in anchor_matches:
                href = href.replace("&amp;", "&")
                if href.startswith("/") or "duckduckgo.com" in href:
                    continue
                if href in seen:
                    continue
                seen.add(href)
                title = _strip_html(title_html)
                if not title:
                    continue
                domain = urlparse(href).netloc
                snippets.append({
                    "title": title,
                    "url": href,
                    "domain": domain,
                    "query": query,
                })
                if len(snippets) >= limit:
                    return snippets
        return snippets

    def _fetch_page_excerpt(url: str) -> str:
        try:
            html = _fetch_text(url, timeout=5)
        except Exception:
            return ""
        text_only = _strip_html(html)
        return text_only[:500]

    for query in results["search_queries"]:
        print(f"   🔎 Query: {query}")
        encoded_query = quote(query)
        results["sources"].extend([
            f"https://duckduckgo.com/?q={encoded_query}",
            f"https://stackoverflow.com/search?q={encoded_query}",
            f"https://github.com/search?q={encoded_query}&type=issues",
        ])
        snippets = _fetch_search_snippets(query, limit=3)
        for item in snippets:
            if len(results["retrieved_snippets"]) >= 6:
                break
            item["excerpt"] = _fetch_page_excerpt(item["url"])
            results["retrieved_snippets"].append(item)

    import_is_missing_module = error_type == "ModuleNotFoundError" or "no module named" in str(error_message or "").lower()
    if error_type in {"ImportError", "ModuleNotFoundError"} and import_is_missing_module:
        results["package_candidates"] = _infer_package_candidates_for_import(
            missing_import_name,
            import_statement,
            results["retrieved_snippets"],
            env_context,
        )
        for candidate in results["package_candidates"][:3]:
            results["solutions"].append(
                "Evidence-backed package candidate: "
                f"{candidate['package']} via {candidate['ecosystem']}. "
                f"Suggested command: {candidate['recommended_command']}. "
                f"Evidence: {candidate['evidence'] or 'search result match'}"
            )
            if candidate.get("source_url") and candidate["source_url"] not in results["sources"]:
                results["sources"].append(candidate["source_url"])
    elif provider_package:
        package_versions = env_context.get("package_versions", {}) or {}
        provider_version = package_versions.get(provider_package.lower(), "")
        version_text = f" installed version={provider_version}." if provider_version else ""
        results["solutions"].append(
            f"Compatibility signal: import is failing inside installed package '{provider_package}', not because the module is missing.{version_text} "
            "Prefer version-compatibility investigation or a targeted pin/upgrade over a generic reinstall."
        )

    if library_hint:
        docs_links: List[str] = []
        hint = library_hint.lower()
        if "llama" in hint or "gguf" in hint:
            docs_links = [
                "https://github.com/ggerganov/llama.cpp",
                "https://github.com/abetlen/llama-cpp-python",
            ]
        elif "huggingface" in hint or "transformers" in hint:
            docs_links = ["https://huggingface.co/docs/transformers"]
        elif "pytorch" in hint:
            docs_links = ["https://pytorch.org/docs/stable/"]
        elif "tensorflow" in hint:
            docs_links = ["https://www.tensorflow.org/api_docs"]
        for link in docs_links:
            if link not in results["sources"]:
                results["sources"].append(link)

    results["solutions"] = generate_common_solution_hints(
        error_type,
        error_message,
        code_context,
        file_context,
        runtime_context=results.get("runtime_context", {}),
    )

    for item in results["retrieved_snippets"][:4]:
        title = item.get("title", "")
        domain = item.get("domain", "")
        excerpt = item.get("excerpt", "")
        if title:
            hint = f"Web result from {domain or 'unknown source'}: {title}"
            if excerpt:
                hint += f". Excerpt: {excerpt[:220]}"
            results["solutions"].append(hint)
        url = item.get("url", "")
        if url and url not in results["sources"]:
            results["sources"].append(url)

    return results


def generate_common_solution_hints(
    error_type: str,
    error_message: str,
    code_context: str,
    file_context: Dict[str, Any] = None,
    runtime_context: Dict[str, Any] = None,
) -> List[str]:
    """
    Generate solution hints based on error patterns AND actual code analysis.

    Args:
        error_type: Type of error
        error_message: Error message
        code_context: Code snippet around error
        file_context: Dict with 'similar_keys', 'class_name', 'library_hints' from code analysis
    """
    hints = []
    file_context = file_context or {}
    runtime_context = runtime_context or {}

    if error_type == "NameError":
        if "is not defined" in error_message:
            hints.append("Variable or function not defined. Check for typos or missing imports.")
            hints.append("Ensure the variable is defined before use, or check if it's in scope.")

    elif error_type == "AttributeError":
        if "has no attribute" in error_message:
            hints.append("Check if the object type is correct. Use dir(obj) to see available attributes.")
            hints.append("Verify that the library version matches the API being used.")

    elif error_type == "TypeError":
        if "argument" in error_message:
            hints.append("Check function signature and ensure correct number/type of arguments.")
        if "NoneType" in error_message:
            hints.append("A variable is None when it shouldn't be. Add None checks or verify initialization.")

    elif error_type == "ImportError" or error_type == "ModuleNotFoundError":
        missing_import = (error_message or "").strip().strip("\'\"")
        hints.append("Missing import detected. Determine whether this is a local module, optional dependency, or third-party package before changing code.")
        hints.append("Do not assume the pip package name exactly matches the import name. Inspect project requirements, docs, and import conventions first.")
        if missing_import:
            hints.append(f"Investigate the package providing import '{missing_import}' and update the environment or requirements accordingly.")
            if any(token in missing_import.lower() for token in ["torch", "pytorch"]):
                hints.append("For torch/pytorch imports, choose the build based on the local runtime and CUDA/toolkit context instead of defaulting to the generic CPU wheel.")
                if runtime_context:
                    pieces = []
                    for key in ["cuda_available", "torch_version", "CUDA_HOME", "CUDAToolkit_ROOT", "nvcc_version", "cuda_device_name", "cuda_capability"]:
                        value = runtime_context.get(key)
                        if value not in (None, ""):
                            pieces.append(f"{key}={value}")
                    if pieces:
                        hints.append("Runtime context for selecting the torch build: " + ", ".join(pieces))

    elif error_type == "IndentationError" or error_type == "SyntaxError":
        hints.append("Check indentation levels - Python requires consistent spacing.")
        hints.append("Ensure all brackets, parentheses, and quotes are balanced.")
        if "multiline" in code_context.lower() or "(" in code_context:
            hints.append("For multi-line statements, ensure ALL lines are commented/uncommented together.")

    elif _is_cuda_related_issue(error_type, error_message, code_context):
        hints.append("CUDA-related failure detected. Compare the requested GPU architecture or backend feature against the installed toolkit/compiler version before changing application code.")
        if runtime_context:
            pieces = []
            for key in ["CUDA_HOME", "CUDAToolkit_ROOT", "LLAMA_CUDA_ARCH", "nvcc_version", "cuda_device_name", "cuda_capability"]:
                value = runtime_context.get(key)
                if value not in (None, ""):
                    pieces.append(f"{key}={value}")
            if pieces:
                hints.append("Runtime CUDA context: " + ", ".join(pieces))
        hints.append("Prefer fixing toolkit/arch mismatches via build configuration, compiler selection, or dependency compatibility rather than patching unrelated source files.")

    elif error_type == "KeyError":
        missing_key = error_message.strip("'\"")
        hints.append(f"❌ Key '{missing_key}' does NOT exist in the dictionary/config.")

        # =====================================================================
        # KEY IMPROVEMENT: Show actual keys found in the code!
        # =====================================================================
        similar_keys = file_context.get("similar_keys", [])
        if similar_keys:
            hints.append(f"✅ VALID KEYS found in nearby code: {similar_keys}")
            hints.append(f"🔧 FIX: Replace '{missing_key}' with one of these valid keys that matches your intent.")

            # Try to suggest the most likely correct key
            missing_parts = missing_key.split('_')

            # =====================================================================
            # PATTERN DETECTION: Find common prefix in valid keys
            # e.g., if valid keys share a common prefix that the missing key lacks,
            # suggest adding that prefix to the missing key
            # =====================================================================
            if len(similar_keys) >= 2:
                # Find common prefix among valid keys
                prefixes = [k.split('_')[0] for k in similar_keys]
                common_prefix = max(set(prefixes), key=prefixes.count) if prefixes else None

                if common_prefix and not missing_key.startswith(common_prefix + '_'):
                    # Check if all/most valid keys share this prefix
                    prefix_count = prefixes.count(common_prefix)
                    if prefix_count >= len(similar_keys) * 0.5:  # At least 50% share prefix
                        # Suggest the corrected key with the common prefix
                        suggested_key = f"{common_prefix}_" + '_'.join(missing_parts[1:]) if len(missing_parts) > 1 else f"{common_prefix}_{missing_key}"
                        hints.append(f"💡 PATTERN DETECTED: Valid keys use '{common_prefix}_' prefix")
                        hints.append(f"💡 LIKELY FIX: '{missing_key}' → '{suggested_key}'")

            # Also look for keys with similar suffix patterns
            best_match = None
            best_score = 0
            for key in similar_keys:
                key_parts = key.split('_')
                score = 0
                for i in range(1, min(len(missing_parts), len(key_parts)) + 1):
                    if missing_parts[-i] == key_parts[-i]:
                        score += 1
                    else:
                        break
                if score > best_score:
                    best_score = score
                    best_match = key

            if best_match and best_score >= 1:
                hints.append(f"💡 Similar key: '{best_match}' (shares suffix pattern)")
        else:
            hints.append("READ the surrounding code to see what keys ARE being used successfully.")
            hints.append("Keys in the same dict/config often share a naming convention (prefix/suffix).")

        hints.append("⚠️ DO NOT use .get() with placeholder defaults - find the CORRECT key name.")

    elif error_type == "IndexError":
        hints.append("List index out of range. Check list length before accessing elements.")

    elif error_type == "ValueError":
        hints.append("Invalid value for the operation. Check input data types and ranges.")

    # Generic fallback
    if not hints:
        hints.append(f"Common {error_type} causes: check variable types, function arguments, and object states.")
        hints.append("Review recent code changes that might have introduced this error.")

    return hints


def detect_cyclic_error(full_error_history: List[Dict[str, Any]], threshold: int = 3) -> Optional[Dict[str, Any]]:
    """
    Detect if the same error is occurring repeatedly (cyclic error).

    Args:
        full_error_history: List of error records with type, message, file, line
        threshold: Number of repetitions to consider as cyclic (default: 3)

    Returns:
        Dict with cycle info if detected, None otherwise
    """
    if len(full_error_history) < threshold:
        return None

    # Get last N errors
    recent_errors = full_error_history[-threshold:]

    # Check if they're all the same error
    first_error = recent_errors[0]
    is_cyclic = all(
        err.get("error_type") == first_error.get("error_type") and
        err.get("file") == first_error.get("file") and
        err.get("line") == first_error.get("line")
        for err in recent_errors
    )

    if is_cyclic:
        return {
            "error_type": first_error.get("error_type"),
            "error_message": first_error.get("message"),
            "file": first_error.get("file"),
            "line": first_error.get("line"),
            "repetitions": len([e for e in full_error_history if
                                e.get("error_type") == first_error.get("error_type") and
                                e.get("file") == first_error.get("file") and
                                e.get("line") == first_error.get("line")])
        }

    return None


def should_offer_rag(
    state: Dict[str, Any],
    error_info: Dict[str, Any],
    cyclic_error: Optional[Dict[str, Any]],
) -> tuple[bool, str]:
    """Decide when RAG is worth triggering before the agent gets stuck."""
    if state.get("user_requested_rag") or state.get("force_rag"):
        return True, "explicit_request"

    history = state.get("full_error_history", [])
    if not history:
        return False, ""

    signature = (
        error_info.get("error_type", ""),
        error_info.get("file", ""),
        error_info.get("line", 0),
        error_info.get("message", ""),
    )
    same_signature = sum(
        1
        for err in history
        if (
            err.get("error_type", ""),
            err.get("file", ""),
            err.get("line", 0),
            err.get("message", ""),
        ) == signature
    )
    same_location = sum(
        1
        for err in history
        if err.get("file") == error_info.get("file")
        and err.get("line") == error_info.get("line")
    )

    if cyclic_error:
        return True, "cyclic_error"
    if same_signature >= 2:
        return True, "repeated_signature"
    if same_location >= 2:
        return True, "repeated_location"
    if error_info.get("error_type") in {"ModuleNotFoundError", "ImportError"}:
        return True, "import_failure"
    if any(v >= 2 for v in state.get("blocked_pattern_counts", {}).values()):
        return True, "repeated_blocked_pattern"
    if state.get("get_block_count", 0) >= 1:
        return True, "repeated_key_fix_failure"

    is_library, _ = is_library_or_utility_file(error_info.get("file", ""))
    if is_library and len(history) >= 2:
        return True, "library_traceback"

    return False, ""


def interactive_mock_selection(
    expensive_functions: List[Dict[str, Any]],
    categories: Dict[str, List[Dict[str, Any]]],
    target_file: str
) -> List[Dict[str, Any]]:
    """
    Interactive selection of functions to mock.
    Shows user the detected expensive functions and lets them choose.
    """
    print("\n" + "=" * 80)
    print("MOCK SELECTION")
    print("=" * 80)
    print(f"Entry file: {target_file}")
    print(f"Auto-selected expensive functions: {len(expensive_functions)}")
    print("Press Enter to accept the current selection.")
    print("\nCurrent selection:")

    if not expensive_functions:
        print("   (none selected)")
        print("\n⚠️  No functions auto-selected for mocking!")
        
        # Ask if user wants to manually add any
        while True:
            response = safe_input("\nWould you like to manually add functions to mock? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                return _manual_function_selection(categories)
            elif response in ['no', 'n']:
                print("   Proceeding with no mocking...")
                return []
            else:
                print("   Invalid input. Please enter 'yes' or 'no'.")
    
    # Show auto-selected functions
    for i, func in enumerate(expensive_functions, 1):
        print(f"\n  {i}. {func['file']}::{func['function']}")
        print(f"     Score: {func['score']}")
        print(f"     Purpose: {func.get('purpose', 'unknown')}")
        if func.get('expensive_ops'):
            print(f"     Expensive ops: {', '.join(func['expensive_ops'])}")
        if func.get('side_effects'):
            print(f"     Side effects: {', '.join(func['side_effects'])}")
        if func.get('promoted_from'):
            print(f"     ⚠️  Promoted from helper: {func['promoted_from']}")
    
    # Ask for confirmation
    print("\nOptions: [Enter/1] accept current selection, [2] customize, [3] add more, [4] skip mocking")

    while True:
        choice = safe_input("\nYour choice [Enter=1]: ").strip()
        if not choice:
            choice = "1"
        
        if choice == "1":
            print("\n✅ Proceeding with all auto-selected functions...")
            return expensive_functions
        
        elif choice == "2":
            return _customize_selection(expensive_functions)
        
        elif choice == "3":
            return _add_more_functions(expensive_functions, categories)
        
        elif choice == "4":
            print("\n⏭️  Skipping mocking strategy...")
            return []
        
        else:
            print("   Invalid choice. Enter 1, 2, 3, or 4.")


def _customize_selection(expensive_functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Let user choose which functions to keep."""
    print("\n" + "=" * 80)
    print("CUSTOMIZE FUNCTION SELECTION")
    print("=" * 80)
    print("Enter the numbers of functions to KEEP (comma-separated)")
    print("Example: 1,3,4  (keeps functions 1, 3, and 4)")
    print("Or enter 'all' to keep all, 'none' to skip all")
    
    for i, func in enumerate(expensive_functions, 1):
        print(f"  {i}. {func['file']}::{func['function']} (score={func['score']})")
    
    while True:
        selection = safe_input("\nKeep functions: ").strip().lower()
        
        if selection == 'all':
            return expensive_functions
        
        if selection == 'none':
            return []
        
        try:
            # Parse comma-separated numbers
            indices = [int(x.strip()) for x in selection.split(',')]
            
            # Validate indices
            if all(1 <= i <= len(expensive_functions) for i in indices):
                selected = [expensive_functions[i-1] for i in indices]
                print(f"\n✅ Selected {len(selected)} function(s):")
                for func in selected:
                    print(f"   - {func['file']}::{func['function']}")
                return selected
            else:
                print(f"   Error: Numbers must be between 1 and {len(expensive_functions)}")
        
        except ValueError:
            print("   Error: Invalid input. Use comma-separated numbers (e.g., 1,2,3)")


# In system_llm_agent.py
RECENT_ACTIONS: list = []
MAX_REPEATED_ACTIONS = 3

# File protection set - prevents LLM from editing mocked files
PROTECTED_FILES: set = set()


def protect_file(file_path: str):
    """Mark a file as protected from LLM edits."""
    PROTECTED_FILES.add(os.path.abspath(file_path))


def unprotect_file(file_path: str):
    """Remove protection from a file."""
    abs_path = os.path.abspath(file_path)
    if abs_path in PROTECTED_FILES:
        PROTECTED_FILES.remove(abs_path)


def is_file_protected(file_path: str) -> bool:
    """Check if file is protected from edits."""
    return os.path.abspath(file_path) in PROTECTED_FILES


def get_multiline_call_range(file_path: str, error_line: int) -> tuple:
    """
    Detect if error_line is part of a multi-line function call.
    
    Example:
        train_dgl(       # line 786 <- error here  
            model=model, # line 787
            epochs=100,  # line 788
        )                # line 789
    
    Returns: (start_line, end_line, is_multiline_call)
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if error_line < 1 or error_line > len(lines):
            return error_line, error_line, False
        
        line = lines[error_line - 1]
        
        # Check if this line has unmatched opening paren
        open_parens = line.count('(') - line.count(')')
        open_brackets = line.count('[') - line.count(']')
        
        if open_parens <= 0 and open_brackets <= 0:
            # Not the start - might be IN a multiline expression
            paren_depth = 0
            start_line = error_line
            
            for i in range(error_line - 1, -1, -1):
                check_line = lines[i]
                for char in reversed(check_line):
                    if char == ')':
                        paren_depth += 1
                    elif char == '(':
                        paren_depth -= 1
                
                if paren_depth < 0:
                    start_line = i + 1
                    break
            
            if start_line == error_line:
                return error_line, error_line, False
            
            # Find end from start_line
            paren_depth = 0
            end_line = start_line
            
            for i in range(start_line - 1, len(lines)):
                check_line = lines[i]
                for char in check_line:
                    if char == '(':
                        paren_depth += 1
                    elif char == ')':
                        paren_depth -= 1
                        if paren_depth == 0:
                            end_line = i + 1
                            return start_line, end_line, True
            
            return start_line, len(lines), True
        
        # This line starts a multiline call - find where it ends
        paren_depth = open_parens
        bracket_depth = open_brackets
        end_line = error_line
        
        for i in range(error_line, len(lines)):
            check_line = lines[i]
            for char in check_line:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == '[':
                    bracket_depth += 1
                elif char == ']':
                    bracket_depth -= 1
                
                if paren_depth <= 0 and bracket_depth <= 0:
                    end_line = i + 1
                    return error_line, end_line, True
        
        return error_line, len(lines), True
        
    except Exception as e:
        return error_line, error_line, False


def get_statement_range(file_path: str, error_line: int) -> tuple:
    """
    UNIFIED function that detects:
    1. Multi-line function calls
    2. Block statements (if/for/with/def/etc)
    3. Single line statements
    
    Returns: (start_line, end_line, statement_type)
    """
    # First check for multi-line call
    start, end, is_call = get_multiline_call_range(file_path, error_line)
    if is_call and end > start:
        return start, end, "multiline_call"
    
    # Then check for block statement
    start, end, block_type = get_block_range(file_path, error_line)
    if block_type:
        return start, end, block_type
    
    # Single line
    return error_line, error_line, None



def _add_more_functions(
    current_selection: List[Dict[str, Any]],
    categories: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Let user add more functions from categories."""
    print("\n" + "=" * 80)
    print("ADD MORE FUNCTIONS")
    print("=" * 80)
    
    # Build list of all available functions
    all_funcs = []
    for cat_name, cat_funcs in categories.items():
        for func in cat_funcs:
            # Skip if already selected
            if not any(
                f['file'] == func['file'] and f['function'] == func['function']
                for f in current_selection
            ):
                all_funcs.append(func)
    
    if not all_funcs:
        print("   No additional functions available.")
        return current_selection
    
    print(f"Available functions ({len(all_funcs)}):")
    for i, func in enumerate(all_funcs, 1):
        print(f"  {i}. {func['file']}::{func['function']} (score={func['score']})")
    
    print("\nEnter numbers to add (comma-separated), or 'done' to finish:")
    
    while True:
        selection = safe_input("Add: ").strip().lower()
        
        if selection == 'done':
            break
        
        try:
            indices = [int(x.strip()) for x in selection.split(',')]
            
            if all(1 <= i <= len(all_funcs) for i in indices):
                for i in indices:
                    func = all_funcs[i-1]
                    current_selection.append(func)
                    print(f"   ✅ Added: {func['file']}::{func['function']}")
            else:
                print(f"   Error: Numbers must be between 1 and {len(all_funcs)}")
        
        except ValueError:
            print("   Error: Invalid input. Use comma-separated numbers or 'done'")
    
    return current_selection



def get_block_range(file_path: str, error_line: int) -> tuple:
    """
    If error_line is a block statement (with/if/for/try/while/def/class),
    return the range that includes the entire block.
    Otherwise return just the single line.
    
    Returns: (start_line, end_line, block_type)
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if error_line < 1 or error_line > len(lines):
            return error_line, error_line, None
        
        line = lines[error_line - 1]
        stripped = line.lstrip()
        line_indent = len(line) - len(stripped)
        
        # Check if this is a block statement
        block_keywords = ['with ', 'if ', 'for ', 'while ', 'try:', 'else:', 'elif ', 'except', 'finally:', 'def ', 'class ']
        is_block = any(stripped.startswith(kw) for kw in block_keywords)
        
        if not is_block:
            return error_line, error_line, None
        
        # Find the block type
        block_type = None
        for kw in block_keywords:
            if stripped.startswith(kw):
                block_type = kw.strip().rstrip(':')
                break
        
        # Find end of block (next line at same or lower indentation that's not empty/comment)
        end_line = error_line
        for i in range(error_line, len(lines)):  # Start from line AFTER the block statement
            next_line = lines[i]
            next_stripped = next_line.lstrip()
            
            # Skip empty lines and comments
            if next_stripped == '' or next_stripped == '\n' or next_stripped.startswith('#'):
                end_line = i + 1
                continue
            
            next_indent = len(next_line) - len(next_stripped)
            
            # If we find a line at same or lower indentation, block ends before it
            if next_indent <= line_indent and i > error_line - 1:
                end_line = i  # Don't include this line
                break
            
            end_line = i + 1
        
        return error_line, end_line, block_type
        
    except Exception as e:
        return error_line, error_line, None


def _manual_function_selection(
    categories: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Manually select functions when none were auto-selected."""
    print("\n" + "=" * 80)
    print("MANUAL FUNCTION SELECTION")
    print("=" * 80)
    
    # Build list of ALL functions
    all_funcs = []
    for cat_name, cat_funcs in categories.items():
        all_funcs.extend(cat_funcs)
    
    if not all_funcs:
        print("   No functions detected at all!")
        return []
    
    print(f"All detected functions ({len(all_funcs)}):")
    for i, func in enumerate(all_funcs, 1):
        print(f"  {i}. {func['file']}::{func['function']} (score={func['score']})")
        print(f"     Category: {_get_func_category(func, categories)}")
        if func.get('expensive_ops'):
            print(f"     Ops: {', '.join(func['expensive_ops'])}")
    
    print("\nEnter numbers to mock (comma-separated), or 'none' to skip:")
    
    while True:
        selection = safe_input("Mock: ").strip().lower()
        
        if selection == 'none':
            return []
        
        try:
            indices = [int(x.strip()) for x in selection.split(',')]
            
            if all(1 <= i <= len(all_funcs) for i in indices):
                selected = [all_funcs[i-1] for i in indices]
                print(f"\n✅ Selected {len(selected)} function(s):")
                for func in selected:
                    print(f"   - {func['file']}::{func['function']}")
                return selected
            else:
                print(f"   Error: Numbers must be between 1 and {len(all_funcs)}")
        
        except ValueError:
            print("   Error: Invalid input. Use comma-separated numbers or 'none'")


def _get_func_category(func: Dict[str, Any], categories: Dict[str, List[Dict[str, Any]]]) -> str:
    """Find which category a function belongs to."""
    for cat_name, cat_funcs in categories.items():
        if any(
            f['file'] == func['file'] and f['function'] == func['function']
            for f in cat_funcs
        ):
            return cat_name
    return "unknown"


def find_similar_keys_in_context(
    file_path: str, 
    error_line: int, 
    missing_key: str,
    access_pattern: str = r'\["([^"]+)"\]'  # Default: dict["key"] pattern
) -> List[str]:
    """
    Find dictionary keys in the same function that are similar to the missing key.
    
    GENERAL APPROACH:
    1. Find the function containing the error line
    2. Extract all keys used in that function
    3. Score by similarity to missing key (shared words, prefix patterns)
    4. Return sorted by relevance
    
    Args:
        file_path: Path to the source file
        error_line: Line number where error occurred (1-indexed)
        missing_key: The key that caused KeyError
        access_pattern: Regex pattern to extract keys (default: ["key"] syntax)
    
    Returns:
        List of candidate keys, sorted by relevance (best first)
    """
    import re
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except:
        return []
    
    err_idx = error_line - 1
    if err_idx < 0 or err_idx >= len(lines):
        return []
    
    # Step 1: Find function boundaries containing the error line
    func_start = 0
    func_end = len(lines)
    
    # Search upward for function/method definition
    for i in range(err_idx, -1, -1):
        line = lines[i].lstrip()
        if line.startswith("def ") or line.startswith("class "):
            func_start = i
            break
    
    # Search downward for next function/method (end of current function)
    for i in range(err_idx + 1, len(lines)):
        line = lines[i].lstrip()
        if line.startswith("def ") or line.startswith("class "):
            func_end = i
            break
    
    # Step 2: Extract all keys from this function
    func_content = ''.join(lines[func_start:func_end])
    found_keys = re.findall(access_pattern, func_content)
    unique_keys = list(dict.fromkeys(found_keys))  # Preserve order, remove dupes
    
    # Step 3: Score keys by similarity to missing key
    def score_key(candidate: str) -> float:
        """Higher score = more similar to missing key."""
        score = 0.0
        
        # Exact prefix match (e.g., both start with "block_")
        missing_parts = missing_key.lower().replace('_', ' ').split()
        candidate_parts = candidate.lower().replace('_', ' ').split()
        
        # Check for shared prefix
        if missing_parts and candidate_parts:
            if missing_parts[0] == candidate_parts[0]:
                score += 5.0  # Strong signal: same prefix
        
        # Check for shared word parts
        shared_parts = set(missing_parts) & set(candidate_parts)
        score += len(shared_parts) * 2.0
        
        # Check for similar length
        len_diff = abs(len(candidate) - len(missing_key))
        if len_diff <= 3:
            score += 1.0
        
        # Penalize if candidate was the missing key itself (shouldn't suggest same key)
        if candidate.lower() == missing_key.lower():
            score = -100
        
        return score
    
    # Step 4: Sort by score (highest first) and filter out the missing key
    scored = [(key, score_key(key)) for key in unique_keys]
    scored = [(k, s) for k, s in scored if s > 0]  # Only positive scores
    scored.sort(key=lambda x: -x[1])  # Descending by score
    
    # Return top candidates
    return [k for k, s in scored[:10]]

def get_escalated_read_range(
    state: Dict[str, Any],
    file_path: str,
    error_line: int,
    default_window: int = 20
) -> Tuple[int, int, bool]:
    """
    Get read range, escalating wider if we've read this location multiple times.
    
    Returns: (start_offset, end_offset, was_escalated)
    """
    read_key = f"{file_path}:{error_line}"
    read_counts = state.get("read_attempt_counts", {})
    count = read_counts.get(read_key, 0) + 1
    read_counts[read_key] = count
    state["read_attempt_counts"] = read_counts
    
    if count == 1:
        return (default_window, default_window + 10, False)
    elif count == 2:
        return (default_window * 2, default_window * 2, True)
    elif count <= 5:
        return (default_window * 3, default_window * 3, True)
    else:
        # After 5+ attempts, we're stuck - try reading a DIFFERENT section
        # Read from start of file or a completely different area
        return (error_line - 10, 100, True)  # Show first ~100 lines of relevant section




def track_code_modification(
    state: Dict[str, Any],
    file_path: str,
    start_line: int,
    end_line: int,
    new_content: str
) -> None:
    """
    Track what code the agent modifies for self-inflicted damage detection.
    
    GENERAL CASE: Works for any file/modification, not specific to any codebase.
    """
    if "modifications_history" not in state:
        state["modifications_history"] = []
    
    # Extract function/class names being modified
    modified_names = set()
    
    # Check if we're modifying a function definition
    func_match = re.search(r'def\s+(\w+)\s*\(', new_content)
    if func_match:
        modified_names.add(func_match.group(1))
    
    # Check if we're modifying a class definition  
    class_match = re.search(r'class\s+(\w+)', new_content)
    if class_match:
        modified_names.add(class_match.group(1))
    
    # Also track what was ORIGINALLY at those lines
    original_names = set()
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for i in range(max(0, start_line - 1), min(len(lines), end_line)):
            line = lines[i]
            func_match = re.search(r'def\s+(\w+)\s*\(', line)
            if func_match:
                original_names.add(func_match.group(1))
    except:
        pass
    
    state["modifications_history"].append({
        "file": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "modified_names": modified_names,
        "original_names": original_names,
        "step": state.get("current_step", 0),
    })

from typing import Tuple
def detect_self_inflicted_damage(
    state: Dict[str, Any],
    error_info: Dict[str, Any]
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Check if the current error was caused by a recent agent modification.
    
    GENERAL CASE: 
    - AttributeError for missing method -> check if agent deleted/renamed it
    - NameError for undefined -> check if agent removed the definition
    - Any error in recently modified lines -> likely self-inflicted
    
    Returns: (is_self_inflicted, modification_info, explanation)
    """
    error_type = error_info.get("error_type", "")
    error_message = error_info.get("message", "")
    error_file = error_info.get("file", "")
    error_line = error_info.get("line", 0)
    
    modifications = state.get("modifications_history", [])
    if not modifications:
        return False, None, ""
    
    # Case 1: AttributeError for missing method/attribute
    if error_type == "AttributeError" and "has no attribute" in error_message:
        # Extract the missing attribute name
        attr_match = re.search(r"has no attribute '(\w+)'", error_message)
        if attr_match:
            missing_attr = attr_match.group(1)
            
            # Check if agent recently modified something with this name
            for mod in reversed(modifications[-10:]):  # Check last 10 modifications
                if missing_attr in mod.get("original_names", set()):
                    return True, mod, (
                        f"SELF-INFLICTED DAMAGE DETECTED!\n"
                        f"The missing attribute '{missing_attr}' was in code you modified "
                        f"at step {mod['step']}.\n"
                        f"File: {mod['file']}, Lines: {mod['start_line']}-{mod['end_line']}\n"
                        f"Your modification likely renamed or deleted this method."
                    )
    
    # Case 2: Error occurs in lines we recently modified
    for mod in reversed(modifications[-10:]):
        if (mod["file"] == error_file and 
            mod["start_line"] <= error_line <= mod["end_line"] + 5):  # +5 for line shifts
            return True, mod, (
                f"SELF-INFLICTED DAMAGE DETECTED!\n"
                f"Error at line {error_line} is in code you modified at step {mod['step']}.\n"
                f"Your previous fix likely introduced this error."
            )
    
    # Case 3: NameError for something we might have removed
    if error_type == "NameError" and "is not defined" in error_message:
        name_match = re.search(r"name '(\w+)' is not defined", error_message)
        if name_match:
            missing_name = name_match.group(1)
            for mod in reversed(modifications[-10:]):
                if missing_name in mod.get("original_names", set()):
                    return True, mod, (
                        f"SELF-INFLICTED DAMAGE DETECTED!\n"
                        f"'{missing_name}' was in code you modified at step {mod['step']}.\n"
                        f"Your modification likely removed or renamed this definition."
                    )
    
    return False, None, ""


# =============================================================================
# PATCH 7: Prevent Method Renaming/Corruption
# =============================================================================
# Problem: Agent renamed _is_vision_tensor to _add_feed_forward_length
# This breaks callers and is almost never the right fix.
#
# General Case: Detect when a write would rename a function/method and block it.
# =============================================================================

def detect_function_rename(
    file_path: str,
    start_line: int,
    end_line: int,
    new_content: str
) -> Tuple[bool, str, str, str]:
    """
    Detect if a write operation would rename a function/method.
    
    GENERAL CASE: Any function rename is suspicious and should be blocked.
    
    Returns: (is_rename, original_name, new_name, warning)
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except:
        return False, "", "", ""
    
    # Find function definitions in original lines
    original_funcs = set()
    for i in range(max(0, start_line - 1), min(len(lines), end_line)):
        line = lines[i]
        func_match = re.search(r'def\s+(\w+)\s*\(', line)
        if func_match:
            original_funcs.add(func_match.group(1))
    
    # Find function definitions in new content
    new_funcs = set()
    for func_match in re.finditer(r'def\s+(\w+)\s*\(', new_content):
        new_funcs.add(func_match.group(1))
    
    # If there's a function in original but different name in new, it's a rename
    if original_funcs and new_funcs:
        removed = original_funcs - new_funcs
        added = new_funcs - original_funcs
        
        if removed and added:
            # Likely a rename
            original_name = list(removed)[0]
            new_name = list(added)[0]
            return True, original_name, new_name, (
                f"⛔ BLOCKED: Your edit would RENAME method '{original_name}' to '{new_name}'.\n"
                f"This will break all code that calls '{original_name}()'.\n"
                f"If you need to add '{new_name}', do it SEPARATELY without removing '{original_name}'."
            )
    
    # Also check for function deletion
    if original_funcs and not new_funcs:
        deleted = list(original_funcs)[0]
        return True, deleted, "", (
            f"⛔ BLOCKED: Your edit would DELETE method '{deleted}'.\n"
            f"This will break all code that calls '{deleted}()'.\n"
            f"Deleting methods is almost never the correct fix."
        )
    
    return False, "", "", ""


# =============================================================================
# PATCH 8: Better AttributeError Handling with Class Analysis
# =============================================================================
# Problem: LFM2Model inherits from a parent class that HAS _is_vision_tensor.
# The agent didn't understand inheritance and kept trying to "add" the method.
#
# General Case: For AttributeError, analyze class hierarchy to find if the
# attribute exists in parent classes and suggest proper fixes.
# =============================================================================
def enhanced_rag_for_attributeerror(
    error_message: str,
    file_path: str,
    error_line: int,
    class_name: str = ""
) -> Dict[str, Any]:
    """
    Enhanced RAG specifically for AttributeError that provides actionable fixes.
    
    GENERAL CASE: Combines code analysis with search results to provide
    specific, implementable fixes rather than vague suggestions.
    """
    result = {
        "analysis": {},
        "suggested_fixes": [],
        "code_to_add": None,
        "do_not": [],
    }
    
    # Extract missing attribute
    attr_match = re.search(r"has no attribute '(\w+)'", error_message)
    if not attr_match:
        return result
    
    missing_attr = attr_match.group(1)
    
    # Analyze the class
    if file_path and class_name:
        result["analysis"] = analyze_class_for_missing_attribute(
            file_path, class_name, missing_attr
        )
    
    # Generate specific fixes based on analysis
    analysis = result["analysis"]
    
    if analysis.get("parent_classes"):
        # Method might be in parent - check inheritance
        result["suggested_fixes"].append({
            "type": "check_parent",
            "description": f"The method '{missing_attr}' may be defined in parent class {analysis['parent_classes'][0]}",
            "action": f"Search for 'def {missing_attr}' in the parent class file"
        })
        
        if not analysis.get("has_super_call"):
            result["suggested_fixes"].append({
                "type": "add_super_init",
                "description": "The class may not be calling parent's __init__",
                "action": "Add 'super().__init__(...)' to the class's __init__ method"
            })
    
    if analysis.get("similar_methods"):
        result["suggested_fixes"].append({
            "type": "check_rename",
            "description": f"Similar methods exist: {analysis['similar_methods']}",
            "action": "Check if the method was accidentally renamed"
        })
        
        result["do_not"].append(
            f"Do NOT rename '{analysis['similar_methods'][0]}' - it will break other code"
        )
    
    # Always add these warnings
    result["do_not"].extend([
        "Do NOT delete the line that calls the missing method",
        "Do NOT try to 'add' the method at the call site",
        "Do NOT modify multiple methods at once - fix one thing at a time"
    ])
    
    return result


# =============================================================================
# PATCH 10: Recovery from Corrupted State
# =============================================================================
# Problem: After agent corrupted the file, it kept trying the same wrong fix.
#
# General Case: Detect when we're in a corrupted state and reset to a known good.
# =============================================================================

def should_restore_from_backup(
    state: Dict[str, Any],
    error_count: int,
    same_error_threshold: int = 5
) -> Tuple[bool, str]:
    """
    Determine if we should restore from backup due to repeated failures.
    
    GENERAL CASE: If the same error keeps occurring despite fixes,
    the agent has likely corrupted something and should restore.
    
    Returns: (should_restore, reason)
    """
    full_error_history = state.get("full_error_history", [])
    
    if len(full_error_history) < same_error_threshold:
        return False, ""
    
    # Check for repeated same error
    recent_errors = full_error_history[-same_error_threshold:]
    first_error = recent_errors[0]
    
    all_same = all(
        err.get("error_type") == first_error.get("error_type") and
        err.get("file") == first_error.get("file") and
        abs(err.get("line", 0) - first_error.get("line", 0)) < 10  # Within 10 lines
        for err in recent_errors
    )
    
    if all_same:
        # Check if agent has been modifying the same area
        modifications = state.get("modifications_history", [])
        recent_mods = [m for m in modifications if m.get("step", 0) > state.get("current_step", 0) - same_error_threshold]
        
        if len(recent_mods) >= same_error_threshold - 1:
            return True, (
                f"RESTORATION TRIGGERED: Same error occurred {same_error_threshold} times "
                f"despite {len(recent_mods)} fix attempts.\n"
                f"The file is likely in a corrupted state. Restoring from backup."
            )
    
    return False, ""


def find_earliest_valid_backup(file_path: str) -> Optional[str]:
    """
    Find the earliest backup that likely represents uncorrupted state.
    
    GENERAL CASE: Look for .backup files with lowest number suffix.
    """
    base_name = file_path
    
    # Check for numbered backups
    for i in range(20):  # Check backup1, backup2, ... backup20
        backup_path = f"{base_name}.backup{i}" if i > 0 else f"{base_name}.backup"
        if os.path.exists(backup_path):
            return backup_path
    
    # Fallback to .backup
    if os.path.exists(f"{base_name}.backup"):
        return f"{base_name}.backup"
    
    return None


def analyze_class_for_missing_attribute(
    file_path: str,
    class_name: str,
    missing_attr: str
) -> Dict[str, Any]:
    """
    Analyze a class to understand why an attribute might be missing.
    
    GENERAL CASE: Works for any class/attribute, provides actionable guidance.
    
    Returns dict with:
    - parent_classes: List of parent class names
    - has_super_call: Whether __init__ calls super().__init__()
    - similar_methods: Methods with similar names in the class
    - suggested_fix: Actionable fix suggestion
    """
    result = {
        "parent_classes": [],
        "has_super_call": False,
        "similar_methods": [],
        "suggested_fix": None,
        "class_definition_line": None,
    }
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                result["class_definition_line"] = node.lineno
                
                # Get parent classes
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        result["parent_classes"].append(base.id)
                    elif isinstance(base, ast.Attribute):
                        result["parent_classes"].append(base.attr)
                
                # Check for super().__init__() call
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Attribute):
                            if child.func.attr == "__init__":
                                if isinstance(child.func.value, ast.Call):
                                    if isinstance(child.func.value.func, ast.Name):
                                        if child.func.value.func.id == "super":
                                            result["has_super_call"] = True
                
                # Find similar methods in this class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        # Check for similar names (e.g., _is_audio_tensor vs _is_vision_tensor)
                        if item.name.startswith(missing_attr[:5]) or \
                           any(part in item.name for part in missing_attr.split('_') if len(part) > 3):
                            result["similar_methods"].append(item.name)
                
                break
    except Exception as e:
        pass
    
    # Generate suggested fix based on analysis
    if result["parent_classes"]:
        if missing_attr.startswith("_"):
            # Private method - likely should be in parent or defined here
            result["suggested_fix"] = (
                f"The class '{class_name}' inherits from {result['parent_classes']}.\n"
                f"The method '{missing_attr}' might be:\n"
                f"1. Defined in a parent class - check if you need to call super().{missing_attr}()\n"
                f"2. A method that should exist but was accidentally deleted/renamed\n"
                f"3. A method from a mixin class that's missing from inheritance\n\n"
                f"Check the parent class definitions for '{missing_attr}'."
            )
        else:
            result["suggested_fix"] = (
                f"Check if '{missing_attr}' is defined in parent class {result['parent_classes'][0]}."
            )
    
    if result["similar_methods"]:
        result["suggested_fix"] = (result.get("suggested_fix", "") + 
            f"\n\nSimilar methods found in class: {result['similar_methods']}\n"
            f"You may have renamed the wrong method."
        )
    
    return result


def get_attributeerror_fix_guidance(
    error_message: str,
    file_path: str,
    class_name: str = ""
) -> str:
    """
    Provide detailed guidance for AttributeError based on class analysis.
    
    GENERAL CASE: Analyzes the actual class structure, not hardcoded fixes.
    """
    # Extract the missing attribute
    attr_match = re.search(r"has no attribute '(\w+)'", error_message)
    if not attr_match:
        return "Check that the object type is correct and has the expected attribute."
    
    missing_attr = attr_match.group(1)
    
    guidance = [
        f"🔍 ROOT CAUSE: '{class_name or 'Object'}' doesn't have attribute '{missing_attr}'",
        ""
    ]
    
    # If we have the file and class, do deeper analysis
    if file_path and class_name:
        analysis = analyze_class_for_missing_attribute(file_path, class_name, missing_attr)
        
        if analysis["parent_classes"]:
            guidance.append(f"📊 CLASS HIERARCHY: {class_name} inherits from {analysis['parent_classes']}")
            guidance.append("")
        
        if analysis["similar_methods"]:
            guidance.append(f"🔎 SIMILAR METHODS FOUND: {analysis['similar_methods']}")
            guidance.append("   ⚠️ Did you accidentally rename one of these?")
            guidance.append("")
        
        if analysis["suggested_fix"]:
            guidance.append("💡 SUGGESTED FIX:")
            guidance.append(analysis["suggested_fix"])
            guidance.append("")
    
    guidance.extend([
        "✅ FIX OPTIONS:",
        "   1. If method should exist: check if you accidentally deleted/renamed it",
        "   2. If method is in parent class: ensure super().__init__() is called",
        "   3. If method needs to be added: add it to the class definition",
        "   4. If wrong object type: trace where the object is created",
        "",
        "⚠️ DO NOT:",
        "   • Try to 'add' the method at the call site - that won't work",
        "   • Delete the line that uses the method - fix the underlying issue",
        "   • Rename other methods - you'll break more code"
    ])
    
    return "\n".join(guidance)

def _is_site_packages_path(file_path: str) -> bool:
    path_lower = (file_path or "").replace("\\", "/").lower()
    return "site-packages/" in path_lower or "/dist-packages/" in path_lower


def _extract_error_named_packages(error_info: Dict[str, Any], stderr: str) -> List[str]:
    names: List[str] = []
    message = str(error_info.get("message", "") or "")
    patterns = [
        r"from '([A-Za-z0-9_.-]+)'",
        r'from "([A-Za-z0-9_.-]+)"',
        r"No module named '([A-Za-z0-9_.-]+)'",
        r'No module named "([A-Za-z0-9_.-]+)"',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, message, re.IGNORECASE):
            name = (match.group(1) or "").strip()
            if not name:
                continue
            root = name.split(".")[0]
            if root.startswith("_"):
                continue
            if os.path.exists(root) or os.path.exists(f"{root}.py"):
                continue
            if root and root not in names:
                names.append(root)
    return names


def _extract_site_package_names_from_stderr(stderr: str) -> List[str]:
    packages: List[str] = []
    if not stderr:
        return packages
    patterns = [
        r'site-packages/([^/\\]+)/',
        r'dist-packages/([^/\\]+)/',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, stderr):
            name = (match.group(1) or "").strip()
            if not name:
                continue
            if name.endswith('.py'):
                name = name[:-3]
            if name not in packages:
                packages.append(name)
    return packages


def _classify_environment_compatibility_issue(error_info: Dict[str, Any], stderr: str) -> Dict[str, Any]:
    error_type = str(error_info.get("error_type", "") or "")
    error_message = str(error_info.get("message", "") or "")
    error_file = str(error_info.get("file", "") or "")
    if error_type not in {"ImportError", "ModuleNotFoundError", "AttributeError", "TypeError", "SyntaxError"}:
        return {"is_env_compatibility": False}

    packages = _extract_site_package_names_from_stderr(stderr)
    for name in _extract_error_named_packages(error_info, stderr):
        if name not in packages:
            packages.append(name)
    in_third_party = _is_site_packages_path(error_file) or len(packages) >= 2
    compatibility_markers = [
        "cannot import name",
        "did you mean:",
        "has no attribute",
        "version",
        "api",
        "undefined symbol",
    ]
    missing_module_markers = ["no module named"]
    is_compat = False
    if in_third_party and any(marker in error_message.lower() for marker in compatibility_markers):
        is_compat = True
    if in_third_party and error_type == "ModuleNotFoundError" and len(packages) >= 1:
        is_compat = True
    if in_third_party and any(marker in error_message.lower() for marker in missing_module_markers) and len(packages) >= 2:
        is_compat = True
    if in_third_party and error_type == "SyntaxError":
        is_compat = True

    return {
        "is_env_compatibility": is_compat,
        "packages": packages,
        "error_file": error_file,
        "error_message": error_message,
        "error_type": error_type,
    }


def _extract_site_package_root(error_file: str) -> str:
    normalized = str(error_file or "").replace("\\", "/")
    match = re.search(r"(?:site|dist)-packages/([^/]+)/", normalized)
    if not match:
        return ""
    root = (match.group(1) or "").strip()
    if root.endswith(".py"):
        root = root[:-3]
    return root


def _build_namespace_family_candidates(root: str) -> List[str]:
    root = _clean_package_name(root).lower()
    if not root:
        return []
    variants = [
        root,
        f"{root}-tools",
        f"{root}_tools",
        f"{root}-python",
        f"{root}_python",
        f"python-{root}",
        f"{root}-core",
        f"{root}_core",
    ]
    seen: List[str] = []
    for item in variants:
        if item and item not in seen:
            seen.append(item)
    return seen


def _build_env_diagnosis_command(package_names: List[str], error_file: str = "", error_type: str = "") -> str:
    packages: List[str] = []
    for pkg in package_names:
        cleaned = _clean_package_name(pkg)
        if cleaned and cleaned not in packages:
            packages.append(cleaned)
    root = _extract_site_package_root(error_file)
    if root and (error_type == "SyntaxError" or len(packages) <= 2):
        for candidate in _build_namespace_family_candidates(root):
            if candidate not in packages:
                packages.append(candidate)
    packages = packages[:8]
    if packages:
        joined = " ".join(packages)
        return f"python -m pip show {joined}"
    return "python -m pip list"


def _detect_namespace_shadowing(
    state: Dict[str, Any],
    package_info: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    error_file = (state.get("last_env_issue", {}) or {}).get("error_file", "")
    root = _extract_site_package_root(error_file)
    if not root:
        return {"suspicious": False}
    root_key = root.lower()
    normalized_root = root_key.replace("_", "-")
    installed = {name.lower(): info for name, info in (package_info or {}).items()}
    root_info = installed.get(root_key, {})
    required_by = (root_info.get("required-by", "") or "").strip()

    siblings: List[str] = []
    for name in installed:
        if name == root_key:
            continue
        normalized_name = name.replace("_", "-")
        if normalized_name.startswith(normalized_root + "-") or normalized_name.startswith("python-" + normalized_root):
            siblings.append(name)

    suspicious = False
    reason = ""
    if root_info and siblings and not required_by:
        suspicious = True
        reason = "Exact-root distribution is installed alongside sibling namespace providers and is not required by other packages"
    elif root_info and not required_by and (state.get("last_env_issue", {}) or {}).get("error_type") == "SyntaxError":
        suspicious = True
        reason = "Exact-root distribution failed during import with a SyntaxError and is not required by other packages"

    family_command = ""
    family = _build_namespace_family_candidates(root)
    if family:
        family_command = f"python -m pip show {' '.join(family[:8])}"

    return {
        "suspicious": suspicious,
        "root": root_key,
        "siblings": siblings,
        "family_command": family_command,
        "reason": reason,
        "uninstall_command": f"python -m pip uninstall -y {root_key}" if suspicious else "",
    }


def _build_env_resolution_followup(result: Dict[str, Any], state: Dict[str, Any]) -> str:
    command = result.get("command", "")
    if isinstance(command, list):
        command_text = " ".join(str(part) for part in command)
    else:
        command_text = str(command or "")
    stdout = str(result.get("stdout", "") or "")
    stderr = str(result.get("stderror", "") or "")
    combined = "\n".join([stdout, stderr])
    packages = state.get("env_resolution_packages", []) or []

    missing_distributions: List[str] = []
    missing_match = re.search(r"Package\(s\) not found:\s*(.+)", combined)
    if missing_match:
        missing_distributions = [part.strip() for part in missing_match.group(1).split(",") if part.strip()]

    namespace_hint = state.get("env_namespace_shadow_hint", {}) or {}
    if "pip show" in command_text and namespace_hint.get("suspicious"):
        root = namespace_hint.get("root", "the exact-root distribution")
        siblings = namespace_hint.get("siblings", []) or []
        sibling_text = ", ".join(siblings) if siblings else "related namespace-family distributions"
        return (
            "Dependency inspection completed.\n"
            f"Namespace shadowing is suspected around '{root}'.\n"
            f"Sibling providers detected: {sibling_text}.\n"
            f"Reason: {namespace_hint.get('reason', 'multiple distributions may provide the same import namespace')}.\n"
            f"Do not blindly upgrade or reinstall '{root}'. That often reinforces the wrong package.\n"
            f"Recommended next action: Execute terminal command: {namespace_hint.get('uninstall_command') or namespace_hint.get('family_command')}\n"
            "After that targeted repair, rerun once and compare the traceback."
        )

    if "pip show" in command_text and missing_distributions:
        missing_text = ", ".join(missing_distributions)
        package_text = ", ".join(packages) if packages else "the packages in the traceback"
        return (
            "Dependency inspection completed.\n"
            f"Some distribution names were not found via pip show: {missing_text}.\n"
            "Do not assume that means the import is still missing; import names and distribution names can differ.\n"
            f"Focus on the packages that were found and on the leaf package referenced by the current traceback. Current package set: {package_text}.\n"
            "Do not rerun yet. Choose a targeted repair, a more specific compatibility diagnosis, or a RAG-backed lookup before running the script again."
        )

    if " install " in f" {command_text} ":
        return (
            "Environment install command completed.\n"
            "Rerun the script once to compare the failure chain.\n"
            "If the failure changes but remains inside third-party packages, treat that as progress and diagnose the newly surfaced dependency compatibility issue rather than reinstalling the original package blindly.\n"
            "If the failure is unchanged, do not repeat broad installs; switch to focused version diagnosis or RAG-backed compatibility lookup."
        )

    return "Environment inspection completed. Do not rerun yet. Compare the current traceback against the previous one, then choose the next targeted diagnosis, focused repair, or RAG-backed compatibility lookup based on the dependency chain."

def _build_env_context_for_rag(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "packages": state.get("env_resolution_packages", []) or [],
        "package_versions": state.get("env_package_versions", {}) or {},
        "last_diagnostic_summary": state.get("last_env_diagnostic_summary", ""),
    }


def _build_rag_signature(error_type: str, error_message: str, file_path: str, error_line: int) -> str:
    return json.dumps(
        {
            "error_type": error_type or "",
            "error_message": (error_message or "")[:240],
            "file": file_path or "",
            "line": int(error_line or 0),
        },
        sort_keys=True,
    )


def _allow_library_patch_once(state: Dict[str, Any], file_path: str) -> None:
    state["allow_library_patch_once"] = os.path.abspath(file_path)


def _consume_library_patch_allowance(state: Dict[str, Any], file_path: str) -> bool:
    allowed = os.path.abspath(state.get("allow_library_patch_once", "") or "")
    target = os.path.abspath(file_path)
    if allowed and allowed == target:
        state["allow_library_patch_once"] = ""
        return True
    return False


def _build_env_resolution_guidance(classification: Dict[str, Any], state: Dict[str, Any]) -> str:
    packages = classification.get("packages", []) or []
    error_file = classification.get("error_file", "") or ""
    error_type = classification.get("error_type", "") or ""
    diag_command = _build_env_diagnosis_command(packages, error_file=error_file, error_type=error_type)
    state["env_resolution_active"] = True
    state["env_resolution_packages"] = packages
    signature = classification.get("signature", "") or json.dumps(
        {
            "error_file": classification.get("error_file", ""),
            "error_message": classification.get("error_message", "")[:240],
            "packages": packages[:8],
        },
        sort_keys=True,
    )
    classification["signature"] = signature
    state["env_resolution_command"] = diag_command
    state["env_issue_signature"] = signature
    package_text = ", ".join(packages) if packages else "the third-party packages in the traceback"
    return (
        "Third-party dependency compatibility failure detected.\n"
        f"The traceback indicates an environment/package mismatch involving: {package_text}.\n"
        "Do NOT read or edit library files in site-packages.\n"
        "Behave like a software engineer debugging dependency drift: inspect installed package versions first, then repair the environment, then rerun.\n"
        f"Recommended next action: Execute terminal command: {diag_command}\n"
        "After inspecting versions, choose a targeted upgrade/downgrade/pin command in the USER environment."
    )

def extract_error_info(stderr: str) -> dict:
    """
    Extract structured error information from Python stderr.
    Returns dict with: error_type, message, file, line
    """
    import re
    
    result = {
        "error_type": "Unknown",
        "message": "",
        "file": "unknown",
        "line": 0,
    }
    
    if not stderr:
        return result
    
    lines = stderr.strip().split("\n")
    
    # Find the last "File" reference (where error occurred)
    file_pattern = r'File "([^"]+)", line (\d+)'
    file_matches = list(re.finditer(file_pattern, stderr))
    
    if file_matches:
        last_match = file_matches[-1]
        result["file"] = last_match.group(1)
        result["line"] = int(last_match.group(2))
    
    # Find the error type and message (usually last line or second-to-last)
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        
        # Match patterns like "NameError: name 'x' is not defined"
        error_match = re.match(r'^([A-Za-z_][\w.]*(?:Error|Exception)|AssertionError): (.+)$', line)
        if error_match:
            result["error_type"] = error_match.group(1)
            result["message"] = error_match.group(2)
            break
        
        # Match patterns like "SyntaxError: invalid syntax"
        syntax_match = re.match(r'^(SyntaxError): (.+)$', line)
        if syntax_match:
            result["error_type"] = syntax_match.group(1)
            result["message"] = syntax_match.group(2)
            break
        
        # Match standalone error types
        if re.match(r'^[A-Za-z_][\w.]*(?:Error|Exception):', line):
            parts = line.split(":", 1)
            result["error_type"] = parts[0]
            result["message"] = parts[1].strip() if len(parts) > 1 else ""
            break
    
    # Clean up file path - make it relative if possible
    if result["file"] != "unknown":
        import os
        cwd = os.getcwd()
        if result["file"].startswith(cwd):
            result["file"] = os.path.relpath(result["file"], cwd)
    
    return result


PROTECTED_FILES: set = set()

def protect_file(file_path: str):
    """Mark a file as protected from LLM edits."""
    PROTECTED_FILES.add(os.path.abspath(file_path))

def is_file_protected(file_path: str) -> bool:
    """Check if file is protected."""
    return os.path.abspath(file_path) in PROTECTED_FILES

def clean_stderr(stderr: str) -> str:
    """Remove non-error warnings from stderr before analysis."""
    lines = stderr.split('\n')
    cleaned = []
    for line in lines:
        # Skip git warnings
        if line.startswith("fatal: not a git repository"):
            continue
        # Skip other common warnings that aren't errors
        if "SyntaxWarning:" in line and "invalid escape sequence" in line:
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


def _classify_runtime_resource_issue(error_info: Dict[str, Any], stderr: str, full_stack: List[Any] | None = None) -> Dict[str, Any]:
    error_type = str(error_info.get("error_type", "") or "")
    message = str(error_info.get("message", "") or "")
    combined = f"{error_type}\n{message}\n{stderr or ''}".lower()

    if not any(token in combined for token in [
        "outofmemoryerror",
        "out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
        "memoryerror",
    ]):
        return {"is_resource_issue": False}

    user_frame = _find_first_user_stack_frame(full_stack or [])
    classification = "runtime memory/resource exhaustion"
    if "cuda" in combined:
        classification = "cuda memory/resource exhaustion"

    return {
        "is_resource_issue": True,
        "classification": classification,
        "user_frame": user_frame,
        "error_file": error_info.get("file", ""),
        "error_line": error_info.get("line", 0),
        "error_type": error_type,
        "error_message": message,
        "is_cuda": "cuda" in combined,
    }


def _build_resource_resolution_guidance(resource_issue: Dict[str, Any], state: Dict[str, Any]) -> str:
    user_frame = resource_issue.get("user_frame") or {}
    target_file = user_frame.get("file") or state.get("target_file", "")
    target_line = int(user_frame.get("line", state.get("error_line", 0) or 0) or 0)
    guidance = [
        f"Runtime resource issue detected: {resource_issue.get('classification', 'resource exhaustion')}.",
        "Do NOT edit torch, CUDA, or other library source files for this failure class.",
        "Treat this as a runtime/configuration problem first: reduce runtime pressure, inspect the user-side call site, or gather targeted resource diagnostics.",
    ]
    if target_file:
        guidance.append(
            f"Nearest actionable user frame: {target_file}:{target_line or 1}. Read that location or the controlling config before proposing source edits."
        )
    guidance.append(
        "Preferred next actions: ReadFileInput on the user-side caller/config, or ExecuteTerminalCommand for a focused resource diagnostic. Avoid arbitrary package upgrades."
    )
    if resource_issue.get("is_cuda") and state.get("llm_uses_gpu"):
        guidance.append(
            "The agent runtime will release its GPU-resident model before user-code runs so the script gets that VRAM back on the next rerun."
        )
    return "\n".join(guidance)


def run_llm_debug_agent(
    target_file: str,
    args: List[str] = None,
    max_steps: int = 15,
    llm: Llama = None,
    debug: bool = False,
    use_mocking_strategy: Optional[bool] = None,
    interactive_output_review: bool = True,
    interactive_action_review: bool = True,
    show_agent_payload: bool = False,
) -> Dict[str, Any]:
    """
    Unified LLM-driven debugging agent with enhanced state tracking.
    """
    if args is None:
        args = []
    if llm is None:
        llm = globals()["llm"]
    active_llm = llm
    _install_llm_references(active_llm)
    debug_mode = bool(debug or os.getenv("LLM_DEBUG_AGENT_VERBOSE", "0") == "1")


    # ========================================================================
    # IMPORTANT: Auxiliary Task Message Isolation
    # ========================================================================
    # All auxiliary tasks (static analysis, minimal input generation, etc.)
    # use call_llm_raw() which creates ONE-OFF message contexts.
    # They do NOT append to the main chat_history, keeping it clean for
    # bug-fixing iterations only.
    # ========================================================================

    # ========================================================================
    # EARLY STRATEGY SELECTION: Ask user BEFORE expensive codebase analysis
    # ========================================================================
    if use_mocking_strategy is None:
        print("\n" + "=" * 80)
        print("STRATEGY SELECTION")
        print("=" * 80)
        print("\nFor large codebases, this agent can use a MINIMAL INPUTS + MOCKING strategy that:")
        print("  - Analyzes the codebase to find expensive functions (ML training, data loading, etc.)")
        print("  - Applies minimal inputs first, then can cache expensive function outputs")
        print("  - Takes longer up front, but is better for slow pipelines")
        print("\nAlternatively, you can use FULL RUN mode which:")
        print("  - Skips the codebase analysis entirely")
        print("  - Runs your code as-is and debugs errors directly")
        print("  - Starts faster, but each iteration may be expensive")
        print("\n" + "-" * 80)
        while True:
            response = safe_input(
                "\nChoose strategy [1=full run, 2=minimal inputs + mocking]: "
            ).strip().lower()
            if response in {"1", "full", "full run", "f", "no", "n"}:
                use_mocking_strategy = False
                print("\n   ✅ Full run strategy selected. Skipping codebase analysis...")
                break
            if response in {"2", "minimal", "mocking", "m", "yes", "y"}:
                use_mocking_strategy = True
                print("\n   ✅ Minimal inputs + mocking strategy selected. Starting codebase analysis...")
                break
            print("   Invalid input. Enter 1 for full run or 2 for minimal inputs + mocking.")
    else:
        selected = "MINIMAL INPUTS + MOCKING" if use_mocking_strategy else "FULL RUN"
        print("\nPreflight passed. Starting agent loop.")

    # ========================================================================
    # CONDITIONAL ANALYSIS: Only analyze if mocking strategy was selected
    # ========================================================================
    if use_mocking_strategy:
        # Step 1: Analyze and decide strategy (uses isolated LLM calls)
        print("\n[Step 1] Analyzing codebase...")
        analysis_results, flows = autonomous_analysis(
            llm_instance=active_llm,
            directory=".",
            entry_file=target_file,
        )
    else:
        # Skip analysis entirely - use full_run strategy
        print("\n[Step 1] Skipping codebase analysis (full_run strategy)")
        analysis_results = {}
        flows = {target_file: {}}

    # ========================================================================
    # STRATEGY DECISION & EXPENSIVE FUNCTION DETECTION
    # ========================================================================
    if use_mocking_strategy:
        # User selected mocking - run full analysis pipeline
        strategy, categories = _decide_strategy(analysis_results)

        target_flow = flows.get(target_file, {})

        # Get expensive functions
        expensive_functions = []
        if strategy == "mocking":
            for key in ["critical_ml", "compute_heavy", "io_ops"]:
                expensive_functions.extend(categories.get(key, []))
            expensive_functions = _filter_to_entry_flow(target_file, flows, expensive_functions)
        entry_flow = flows.get(target_file, {})
        expensive_functions = _promote_helpers_to_callers(expensive_functions, entry_flow)


        # Interactive selection
        expensive_functions = interactive_mock_selection(
            expensive_functions,
            categories,
            target_file
        )

        if not expensive_functions:
            strategy = "full_run"
            print("\n   📌 Strategy changed to: FULL_RUN (no mocking)")
        else:
            print(f"\n   📌 Final selection: {len(expensive_functions)} functions to mock")
            for func in expensive_functions:
                print(f"      - {func['file']}::{func['function']}")
    else:
        # User selected full_run - skip all mocking-related logic
        strategy = "full_run"
        categories = {}
        expensive_functions = []
        print("\n   📌 Strategy: FULL_RUN (user selected, no codebase analysis performed)")
    
    print(f"\n   Strategy: {strategy.upper()}")
    if expensive_functions:
        if debug_mode:
            print(f"   Expensive functions: {len(expensive_functions)}", expensive_functions)
        else:
            print(f"   Expensive functions selected: {len(expensive_functions)}")
    
    # Step 2: Apply minimal inputs
    config_changes = {}
    mocked_functions = []
    run_args = args.copy()
    
    if strategy == "mocking":
        # Step 2: Apply minimal inputs (uses isolated LLM calls, not chat_history)
        print("\n[Step 2] Applying mocking strategy setup...")
        entry_only_flows = {target_file: flows.get(target_file, {})}
        config_changes = apply_minimal_inputs(
            target_file,
            expensive_functions,
            entry_only_flows,
            active_llm,
            original_args=args
        )

        minimal_args = config_changes.get("entry_args", [])
        if minimal_args:
            run_args = minimal_args
        print(f"   ✓ Minimal configs applied: {len(config_changes.get('config_swaps', []))}")
    
    # Step 3: Initialize state
    goal = f"Run and fix errors in {target_file}"
    state: Dict[str, Any] = {
        "phase": "START",
        "target_file": target_file,
        "current_input": goal,
        "strategy": strategy,
        "mocking_phase": "setup" if strategy == "mocking" else None,
        "expensive_functions": expensive_functions,
        "mocked_functions": mocked_functions,
        "expensive_funcs_executed": set(),
        "func_outputs": {},
        "config_changes": config_changes,
        "use_mocks": False,
        "run_args": run_args,
        "backups_created": {},
        # NEW: Full error history for cyclic detection
        "full_error_history": [],
        "rag_searches_performed": 0,
        "last_rag_search_step": -1,
        "error_location_history": [],
        "modifications_history": [],
        "current_step": 0,
        "awaiting_user_feedback": False,
        "user_protected_paths": [],
        "rag_eligible": False,
        "rag_reason": "",
        "user_requested_rag": False,
        "force_rag": False,
        "debug": debug_mode,
        "interactive_output_review": interactive_output_review,
        "interactive_action_review": interactive_action_review,
        "show_agent_payload": show_agent_payload,
        "action_history": [],
        "last_rag_results": {},
        "last_terminal_result": {},
        "env_diagnostic_cache": {},
        "env_package_versions": {},
        "last_env_diagnostic_summary": "",
        "last_env_repair_step": -1,
        "env_diag_blocked_count": 0,
        "env_issue_counts": {},
        "env_ready_to_rerun": False,
        "env_repair_history": [],
        "preferred_next_command": "",
        "env_issue_signature": "",
        "last_rag_signature": "",
        "pending_rag_signature": "",
        "allow_library_patch_once": "",
        "artifact_read_target": {},
        "llm_original_runtime_config": _get_llm_runtime_config(active_llm),
    }
    chat_history: List[Dict[str, str]] = [{"role": "user", "content": goal}]
    state["blocked_pattern_counts"] = {}  # Track repeated blocked patterns
    state["escalation_level"] = 0
    state["recent_write_attempts"] = []
    
    strategy_context = ""
    if strategy == "mocking":
        strategy_context = (
            f"\\nMOCKING STRATEGY: Minimal inputs applied. "
            f"Expensive functions detected: {len(expensive_functions)}. "
            f"If errors occur AFTER expensive functions execute, consider mocking them."
        )
    
    # Step 4: Main loop
    for step in range(max_steps):
        state["current_step"] = step
        _print_step_header(step + 1, max_steps, state["phase"], state.get("mocking_phase", "N/A"))
        
        active_llm = active_llm or _restore_llm_after_user_run(state)
        action = ask_action(active_llm, goal + strategy_context, chat_history, state)
        _print_labeled_value("Next action", _summarize_action_for_approval(action), tone="action")
        action_payload = action.model_dump()
        state.setdefault("action_history", []).append(action_payload)
        if state.get("show_agent_payload"):
            print(json.dumps(action_payload, indent=2, ensure_ascii=False))
        chat_history.append({
            "role": "assistant",
            "content": json.dumps(action_payload, ensure_ascii=False),
        }) 

        approved, feedback = get_user_approval_for_action(action, state)
        if not approved:
            state["current_input"] = feedback
            chat_history.append({"role": "user", "content": feedback})
            continue

        # =====================================================================
        # RunScriptInput
        # =====================================================================
        if action.action_type == "RunScriptInput":
            act = action.action
            args_to_use = state["run_args"] if state.get("run_args") is not None else act.args

            if state["strategy"] == "mocking" and state["use_mocks"]:
                print("   🔨 Using mocked functions for this run")
                from system_agent import MOCK_CACHE
                print(f"   Mock cache size: {len(MOCK_CACHE)} entries")

            _release_llm_for_user_run(active_llm, state)
            active_llm = None
            out = run_script.invoke({"script_path": act.script_path, "args": args_to_use})
            state["last_stdout"] = out.get("stdout", "")
            state["last_stderr"] = out.get("stderror", "")
            state["last_exitcode"] = out.get("exitcode", None)
            state["phase"] = "RAN"
            review_latest_run_output(state, label="script run")
            
            if state["strategy"] == "mocking":
                state["mocking_phase"] = "running"

            # === SUCCESS PATH ===
            if state["last_exitcode"] == 0:
                _print_labeled_value("Run result", "script completed successfully", tone="success")
                state["awaiting_user_feedback"] = False
                state["rag_eligible"] = False
                state["env_resolution_active"] = False
                state["env_ready_to_rerun"] = False
                
                if state["strategy"] == "mocking" and not state.get("use_mocks"):
                    for func in state["expensive_functions"]:
                        func_key = f"{func['file']}::{func['function']}"
                        
                        if func_key not in state["expensive_funcs_executed"]:
                            state["expensive_funcs_executed"].add(func_key)
                            state["func_outputs"][func_key] = state["last_stdout"]
                            print(f"   ✅ Recorded output for {func['function']}()")
                            
                            from system_agent import mock_function
                            mock_res = mock_function(
                                file_path=func["file"],
                                func_name=func["function"],
                                cached_value=state["last_stdout"],
                            )
                            
                            if mock_res.get("success"):
                                state["mocked_functions"].append({
                                    "file": func["file"],
                                    "function": func["function"],
                                })
                                print(f"   ✅ Mocked {func['function']}()")
                    
                    state["use_mocks"] = True
                    state["mocking_phase"] = "complete"
                
                state["current_input"] = "Script completed successfully. Output NoAction if done."
                chat_history.append({"role": "user", "content": state["current_input"]})
                continue

            # === FAILURE PATH ===
            else:
                # =====================================================================
                # FIX 1: TRACE TYPEERROR TO CALLER WHEN ERROR IS IN LIBRARY/UTILITY CODE
                # =====================================================================
                def should_trace_to_caller(error_info: dict, error_analysis: dict, file_path: str) -> tuple:
                    """
                    Determine if we should trace the error back to the calling code.

                    Returns: (should_trace: bool, caller_info: dict or None, reason: str)

                    GENERAL CASE TRIGGERS:
                    1. Error is in a library/site-packages file
                    2. Error is a TypeError in a function that validates input types
                    3. Error is in a utility function that doesn't know the business logic
                    4. Error line contains a type-checking pattern (len(), isinstance(), etc.)
                    """
                    import re
                    error_type = error_info.get("error_type", "")
                    error_line_content = ""

                    # Read the error line content
                    try:
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                        if 0 < error_info.get("line", 0) <= len(lines):
                            error_line_content = lines[error_info["line"] - 1].strip()
                    except:
                        pass
                    
                    # Patterns that indicate the error is a TYPE VALIDATION failure
                    # The fix should be at the CALLER, not at this validation code
                    type_validation_patterns = [
                        r'if\s+len\s*\(',           # if len(x) - checking sequence length
                        r'if\s+isinstance\s*\(',    # if isinstance(x, ...) - type check
                        r'if\s+not\s+isinstance',   # if not isinstance(...)
                        r'if\s+type\s*\(',          # if type(x) == ...
                        r'assert\s+isinstance',     # assert isinstance(...)
                        r'if\s+hasattr\s*\(',       # if hasattr(x, ...) - attribute check
                    ]

                    # Check if error line matches type validation pattern
                    is_type_validation = False
                    for pattern in type_validation_patterns:
                        if re.search(pattern, error_line_content):
                            is_type_validation = True
                            break
                        
                    # Check if file is a library or utility file
                    # Check if file is a library or utility file
                    is_library_code = any(p in file_path for p in [
                        'site-packages', 'lib/python', 'anaconda', 'miniconda',
                        '_util', 'utils/', 'helpers/', 'common/', 'base/',
                        'gguf-py/', 'gguf/',  # GGUF library subdirectories
                    ])

                    # GENERAL RULE: TypeError in type validation code = fix the caller
                    if error_type == "TypeError" and (is_type_validation or is_library_code):
                        # Get caller from traceback
                        full_stack = error_analysis.get("full_stack", [])

                        # Find first non-library file in stack (the actual caller)
                        caller_info = None
                        for stack_file, stack_line in reversed(full_stack):
                            if not any(p in stack_file for p in ['site-packages', 'lib/python', 'gguf-py/', 'gguf/']):
                                caller_info = {"file": stack_file, "line": stack_line}
                                break
                            
                        if caller_info:
                            return True, caller_info, (
                                f"TypeError in type validation code. "
                                f"The error is correctly catching bad input - fix the CALLER at "
                                f"{caller_info['file']}:{caller_info['line']} that passed wrong type."
                            )

                    return False, None, ""
                error_analysis = error_analyzer(state["last_stderr"], state.get("error_location_history", []))
                state.setdefault("error_location_history", []).append({
                    "file": error_analysis.get("file", ""),
                    "line": error_analysis.get("error_line", 0),
                    "type": error_analysis.get("error_type", ""),
                })
                state["error_line"] = error_analysis.get("error_line", 0) or 0
                error_info = extract_error_info(state["last_stderr"])
                env_issue = _classify_environment_compatibility_issue(error_info, state["last_stderr"])
                resource_issue = _classify_runtime_resource_issue(
                    error_info,
                    state["last_stderr"],
                    error_analysis.get("full_stack", []),
                )
                state["last_env_issue"] = env_issue
                state["last_resource_issue"] = resource_issue
                if env_issue.get("is_env_compatibility"):
                    signature = env_issue.get("signature", "") or json.dumps(
                        {
                            "error_file": env_issue.get("error_file", ""),
                            "error_message": env_issue.get("error_message", "")[:240],
                            "packages": env_issue.get("packages", [])[:8],
                        },
                        sort_keys=True,
                    )
                    env_issue["signature"] = signature
                    state["env_issue_signature"] = signature
                    counts = state.setdefault("env_issue_counts", {})
                    counts[signature] = counts.get(signature, 0) + 1
                if not env_issue.get("is_env_compatibility"):
                    state["env_resolution_active"] = False
                    state["env_ready_to_rerun"] = False
                state["last_error_file"] = error_info.get("file")
                state["awaiting_user_feedback"] = True

                # ===== TRACK ERROR IN FULL HISTORY =====
                state["full_error_history"].append({
                    "error_type": error_info.get("error_type"),
                    "message": error_info.get("message"),
                    "file": error_info.get("file"),
                    "line": error_info.get("line"),
                    "step": step,
                    "full_stack": error_analysis.get("full_stack", []),
                })

                # Compute cyclic/RAG state once so downstream branches stay consistent.
                cyclic_error = detect_cyclic_error(state["full_error_history"], threshold=2)
                rag_ready, rag_reason = should_offer_rag(state, error_info, cyclic_error)
                can_run_rag = bool(rag_ready and (step - state.get("last_rag_search_step", -99)) > 0)
                state["rag_eligible"] = can_run_rag
                state["rag_reason"] = rag_reason
                if resource_issue.get("is_resource_issue"):
                    _print_console_panel("Runtime Resource Failure", tone="warn")
                    _print_labeled_value("Classification", resource_issue.get("classification", "resource exhaustion"), tone="warn")
                    _print_labeled_value("Policy", "treat as runtime/config issue, not library source edit", tone="warn")
                    state["env_resolution_active"] = False
                    state["env_ready_to_rerun"] = False
                    state["current_input"] = _build_resource_resolution_guidance(resource_issue, state)
                    chat_history.append({"role": "user", "content": state["current_input"]})
                    continue

                if can_run_rag:
                    active_error = dict(cyclic_error) if cyclic_error else {
                        "error_type": error_info.get("error_type", "unknown"),
                        "error_message": error_info.get("message", ""),
                        "message": error_info.get("message", ""),
                        "file": error_info.get("file", state.get("target_file", "")),
                        "line": error_info.get("line", 0),
                        "repetitions": 1,
                    }

                    if env_issue.get("is_env_compatibility"):
                        _print_console_panel("Environment Compatibility Failure", tone="warn")
                        _print_labeled_value("Classification", "third-party package/version mismatch", tone="warn")
                        _print_labeled_value("Policy", "treat as environment repair, not source edit", tone="warn")
                        state["env_ready_to_rerun"] = False
                        state["current_input"] = _build_env_resolution_guidance(env_issue, state)
                        chat_history.append({"role": "user", "content": state["current_input"]})
                        continue

                    rag_signature = _build_rag_signature(
                        active_error.get("error_type", "unknown"),
                        active_error.get("error_message", active_error.get("message", "")),
                        active_error.get("file", ""),
                        active_error.get("line", 0),
                    )
                    if rag_signature == state.get("last_rag_signature", "") and state.get("last_rag_results"):
                        state["current_input"] = (
                            "RAG evidence for this same error signature already exists. Reuse it and choose a concrete repair or diagnosis step instead of running RAG again."
                        )
                    else:
                        state["pending_rag_signature"] = rag_signature
                        state["force_rag"] = True
                        state["current_input"] = (
                            f"RAG investigation is recommended for {active_error.get('error_type', 'unknown')} at "
                            f"{active_error.get('file', '')}:{active_error.get('line', 0)}.\n"
                            f"Reason: {rag_reason or 'general_investigation'}.\n"
                            "Choose RAGSearchInput next and wait for approval before executing any install or write action.\n"
                            "Do not run a generic package install based only on the import name."
                        )
                    state["awaiting_user_feedback"] = True
                    chat_history.append({"role": "user", "content": state["current_input"]})
                    continue

                    if not cyclic_error:
                        print(
                            f"\n🌐 RAG investigation triggered for {active_error.get('error_type', 'unknown')} "
                            f"at {active_error.get('file', 'unknown')}:{active_error.get('line', 0)}"
                        )
                    else:
                        modified_files = state.get("files_modified_for_error", {})
                        error_key = f"{cyclic_error['error_type']}:{cyclic_error['file']}"
                        modified_files[error_key] = modified_files.get(error_key, 0) + 1
                        state["files_modified_for_error"] = modified_files

                        if modified_files[error_key] >= 3:
                            full_stack = error_analysis.get("full_stack", [])
                            tried_file = cyclic_error["file"]

                            alternative_file = None
                            alternative_line = None
                            for stack_file, stack_line in full_stack:
                                if stack_file != tried_file and "site-packages" not in stack_file:
                                    alternative_file = stack_file
                                    alternative_line = stack_line
                                    break

                            if alternative_file:
                                print(f"   🔄 Modified {tried_file} {modified_files[error_key]}x without success")
                                print(f"   🔍 Looking at caller: {alternative_file}:{alternative_line}")
                                cyclic_error["file"] = alternative_file
                                cyclic_error["line"] = alternative_line
                                active_error["file"] = alternative_file
                                active_error["line"] = alternative_line
                                new_key = f"{cyclic_error['error_type']}:{alternative_file}"
                                modified_files[new_key] = 0

                        print(
                            f"\n🔄 CYCLIC ERROR DETECTED: {cyclic_error['error_type']} "
                            f"at {cyclic_error['file']}:{cyclic_error['line']}"
                        )
                        print(f"   Error has occurred {cyclic_error['repetitions']} times")

                        is_library_error = any(
                            p in cyclic_error["file"]
                            for p in [
                                "site-packages",
                                "lib/python",
                                "anaconda",
                                "miniconda",
                                "gguf-py/",
                                "gguf/",
                                "_util",
                                "utils/",
                                "helpers/",
                            ]
                        )

                        if is_library_error:
                            full_stack = error_analysis.get("full_stack", [])
                            user_code_caller = None
                            for stack_file, stack_line in full_stack:
                                if not any(p in stack_file for p in ["site-packages", "lib/python", "gguf-py/", "gguf/"]):
                                    user_code_caller = {"file": stack_file, "line": stack_line}
                                    break

                            if user_code_caller:
                                print(
                                    f"   🔍 Library error - redirecting to caller: "
                                    f"{user_code_caller['file']}:{user_code_caller['line']}"
                                )
                                state["current_input"] = (
                                    f"⛔ STOP! The error is in LIBRARY CODE ({cyclic_error['file']}).\n\n"
                                    "You cannot and should not edit library files.\n"
                                    "The library is correctly validating input - YOUR code is passing wrong data.\n\n"
                                    "🎯 FIX YOUR CODE AT:\n"
                                    f"   File: {user_code_caller['file']}\n"
                                    f"   Line: {user_code_caller['line']}\n\n"
                                    "Use: {\"action_type\": \"ReadFileInput\", \"action\": "
                                    f"{{\"file_path\": \"{user_code_caller['file']}\", \"error_line\": {user_code_caller['line']}, "
                                    "\"window_before\": 20, \"window_after\": 30}}}"
                                )
                                chat_history.append({"role": "user", "content": state["current_input"]})
                                continue

                    if (
                        cyclic_error
                        and cyclic_error["error_type"] == "TypeError"
                        and "NoneType" in cyclic_error.get("error_message", "")
                    ):
                        found_related_keyerror = False
                        for prev_err in reversed(state["full_error_history"][:-3]):
                            if prev_err.get("error_type") == "KeyError":
                                print("   💡 This TypeError is likely caused by a bad KeyError fix")
                                print(f"   💡 Previous KeyError: {prev_err.get('message')}")

                                bad_key = prev_err.get("message", "").strip("'\"")
                                valid_keys_hint = ""
                                try:
                                    with open(active_error["file"], "r") as f:
                                        source_lines = f.readlines()

                                    error_line = prev_err.get("line", 10120)
                                    context_start = max(0, error_line - 20)
                                    context_end = min(len(source_lines), error_line + 10)
                                    nearby_content = "".join(source_lines[context_start:context_end])

                                    import re
                                    hparams_keys = re.findall(r'hparams\["([^"]+)"\]', nearby_content)
                                    unique_keys = list(dict.fromkeys(hparams_keys))
                                    if unique_keys:
                                        valid_keys_hint = ", ".join(unique_keys)
                                except Exception:
                                    valid_keys_hint = "(could not extract - READ the file to find valid keys)"

                                keyerror_line = prev_err.get("line", 10120)
                                enriched_input = (
                                    "⚠️ ROOT CAUSE FOUND!\n\n"
                                    f"Your previous fix for KeyError '{bad_key}' used .get(..., None)\n"
                                    "This caused the current TypeError because the value is None.\n\n"
                                    f"The key '{bad_key}' does NOT exist.\n"
                                    "You need to find the CORRECT KEY that actually exists.\n\n"
                                    f"VALID KEYS FOUND IN CODE: {valid_keys_hint}\n\n"
                                    "⚠️ IMPORTANT:\n"
                                    f"1. Fix LINE {keyerror_line}, NOT line {active_error['line']}\n"
                                    "2. Do NOT use .get(..., None) - it will be REJECTED\n"
                                    "3. Use direct key access with the correct key\n"
                                    "4. Look at nearby keys for naming patterns\n"
                                )
                                state["current_input"] = enriched_input
                                chat_history.append({"role": "user", "content": state["current_input"]})
                                found_related_keyerror = True
                                break

                        if found_related_keyerror:
                            continue

                    code_context = ""
                    try:
                        if active_error.get("file"):
                            with open(active_error["file"], "r") as f:
                                source_lines = f.readlines()
                            line_idx = max(active_error.get("line", 1) - 1, 0)
                            context_start = max(0, line_idx - 5)
                            context_end = min(len(source_lines), line_idx + 5)
                            code_context = "".join(source_lines[context_start:context_end])
                    except Exception:
                        pass

                    print("\n🌐 Triggering RAG pipeline to search for solutions...")
                    rag_results = search_online_for_solution(
                        error_type=active_error.get("error_type", "unknown"),
                        error_message=active_error.get("error_message", active_error.get("message", "")),
                        code_context=code_context,
                        file_path=active_error.get("file", ""),
                        error_line=active_error.get("line", 0),
                        env_context=_build_env_context_for_rag(state),
                    )

                    state["rag_searches_performed"] += 1
                    state["last_rag_search_step"] = step

                    analysis = rag_results.get("code_analysis", {})
                    similar_keys = analysis.get("similar_keys", [])
                    class_name = analysis.get("class_name", "")
                    state["last_rag_results"] = rag_results
                    state["last_similar_keys"] = similar_keys

                    enhanced = enhanced_rag_search(
                        active_error.get("error_type", "unknown"),
                        active_error.get("error_message", active_error.get("message", "")),
                        active_error.get("file", ""),
                        active_error.get("line", 0),
                        similar_keys,
                    )

                    suggested_replacement = enhanced.get("suggested_replacement")
                    if suggested_replacement:
                        print(
                            f"   💡 Enhanced RAG suggests: Use '{suggested_replacement}' "
                            f"(confidence: {enhanced['confidence']})"
                        )

                    if active_error.get("error_type") == "KeyError" and similar_keys:
                        enriched_input = build_keyerror_fix_prompt(
                            active_error.get("error_message", active_error.get("message", "")),
                            similar_keys,
                            code_context,
                            active_error.get("file", ""),
                            active_error.get("line", 0),
                            suggested_replacement,
                        )
                    else:
                        package_candidates = rag_results.get("package_candidates", [])
                        llm_suggestion = llm_suggest_fix(
                            error_type=active_error.get("error_type", "unknown"),
                            error_message=active_error.get("error_message", active_error.get("message", "")),
                            code_context=code_context,
                            similar_keys=similar_keys,
                            class_name=class_name,
                            file_path=active_error.get("file", ""),
                            runtime_context=rag_results.get("runtime_context", {}),
                        )

                        rag_hints = "\n".join([f"  - {hint}" for hint in rag_results["solutions"][:5]])

                        code_analysis_section = ""
                        if similar_keys:
                            code_analysis_section = (
                                f"\n📋 VALID KEYS FOUND IN NEARBY CODE:\n"
                                f"  {similar_keys}\n"
                            )
                        if class_name:
                            code_analysis_section += f"\n  Class: {class_name}\n"

                        llm_suggestion_section = ""
                        if llm_suggestion and "FIX:" in llm_suggestion:
                            llm_suggestion_section = f"\n🤖 LLM ANALYSIS & SUGGESTED FIX:\n  {llm_suggestion}\n"
                        elif llm_suggestion:
                            llm_suggestion_section = f"\n🤖 LLM SUGGESTION:\n  {llm_suggestion}\n"

                        package_candidates_section = ""
                        if package_candidates:
                            candidate_lines = []
                            for candidate in package_candidates[:3]:
                                candidate_lines.append(
                                    f"  - {candidate['package']} via {candidate['ecosystem']}: {candidate['recommended_command']}"
                                    + (f" | evidence: {candidate['evidence']}" if candidate.get('evidence') else "")
                                )
                            package_candidates_section = (
                                "\n📦 PACKAGE CANDIDATES FROM RAG:\n"
                                + "\n".join(candidate_lines)
                                + "\n"
                            )

                        headline = (
                            f"⚠️ CYCLIC ERROR DETECTED (occurred {cyclic_error['repetitions']} times)"
                            if cyclic_error
                            else "⚠️ RAG INVESTIGATION TRIGGERED"
                        )

                        enriched_input = (
                            f"{headline}\n\n"
                            f"Error: {active_error.get('error_type', 'unknown')}: {active_error.get('error_message', active_error.get('message', ''))}\n"
                            f"Location: {active_error.get('file', '')}:{active_error.get('line', 0)}\n"
                            f"Reason: {rag_reason or 'general_investigation'}\n"
                            f"{code_analysis_section}"
                            f"{package_candidates_section}"
                            f"{llm_suggestion_section}\n"
                            f"🌐 Additional hints:\n{rag_hints}\n\n"
                            "⚠️ IMPORTANT: Investigate the root cause and apply the LLM's suggested fix only if it matches the code and traceback. Prefer evidence-backed package candidates over generic install commands like pip install <import-name>."
                        )

                        if error_analysis.get("is_raise_statement"):
                            caller = error_analysis.get("caller_info")
                            if caller:
                                enriched_input += (
                                    "\n\n🚨 STOP! You keep modifying a 'raise' statement. This is WRONG.\n"
                                    "The raise is CORRECT - it's catching invalid data.\n"
                                    f"The BUG is at: {caller['file']}:{caller['line']}\n"
                                    "READ that file and fix the code that's passing wrong data."
                                )

                    state["current_input"] = enriched_input
                    chat_history.append({"role": "user", "content": state["current_input"]})
                    continue

                output_text = (state["last_stdout"] or "") + (state["last_stderr"] or "")
                completion_markers = ["Train Loss:", "Epoch", "TestLoss", "Training Done", "completed successfully"]
                train_completed = any(marker in output_text for marker in completion_markers)

                # ===== CASE 1: Expensive function ran - inject mock and rerun =====
                if train_completed and state["strategy"] == "mocking" and not state.get("use_mocks"):
                    print(f"   ✅ Expensive function executed (found output markers)")

                    for func in state["expensive_functions"]:
                        func_key = f"{func['file']}::{func['function']}"

                        if func_key not in state.get("expensive_funcs_executed", set()):
                            state.setdefault("expensive_funcs_executed", set()).add(func_key)
                            state["func_outputs"][func_key] = state["last_stdout"]

                            from system_agent import mock_function
                            mock_res = mock_function(
                                file_path=func["file"],
                                func_name=func["function"],
                                cached_value=state["last_stdout"],
                            )

                            if mock_res.get("success"):
                                state.setdefault("mocked_functions", []).append({
                                    "file": func["file"],
                                    "function": func["function"],
                                })
                                print(f"   ✅ Mocked {func['function']}()")

                    state["use_mocks"] = True
                    state["mocking_phase"] = "complete"

                    print("   🔄 Auto-rerunning with mocked functions...")
                    _release_llm_for_user_run(active_llm, state)
                    active_llm = None
                    out = run_script.invoke({
                        "script_path": state["target_file"], 
                        "args": state["run_args"]
                    })
                    state["last_stdout"] = out.get("stdout", "")
                    state["last_stderr"] = out.get("stderror", "")
                    state["last_exitcode"] = out.get("exitcode", None)
                    review_latest_run_output(state, label="post-mock rerun")

                    if state["last_exitcode"] == 0:
                        print("   ✅ Script passes with mock!")
                        state["current_input"] = "Script completed successfully. Output NoAction if done."
                        chat_history.append({"role": "user", "content": state["current_input"]})
                        continue
                    
                    error_info = extract_error_info(state["last_stderr"])
                    state["error_line"] = error_info["line"]
                    post_mock_env_issue = _classify_environment_compatibility_issue(error_info, state["last_stderr"])
                    if not post_mock_env_issue.get("is_env_compatibility"):
                        state["env_resolution_active"] = False
                    state["env_ready_to_rerun"] = False

                    # Track error in full history
                    state["full_error_history"].append({
                        "error_type": error_info.get("error_type"),
                        "message": error_info.get("message"),
                        "file": error_info.get("file"),
                        "line": error_info.get("line"),
                        "step": step,
                    })

                    error_line_content = ""
                    try:
                        with open(error_info["file"], "r") as f:
                            lines = f.readlines()
                        if 0 < error_info["line"] <= len(lines):
                            error_line_content = lines[error_info["line"] - 1].strip()
                    except:
                        error_line_content = "unknown"

                    # === Check if this is a self-inflicted wrapper error ===
                    mocked_files = [f["file"] for f in state.get("mocked_functions", [])]
                    if any(os.path.abspath(error_info["file"]) == os.path.abspath(mf) for mf in mocked_files):
                        if "IndentationError" in error_info["error_type"] or "SyntaxError" in error_info["error_type"]:
                            print(f"   ⚠️ SELF-INFLICTED ERROR: Wrapper injection failed in {error_info['file']}")
                            print(f"   🔄 Restoring from backup...")

                            backup_path = f"{error_info['file']}.backup"
                            if os.path.exists(backup_path):
                                import shutil
                                shutil.copy(backup_path, error_info['file'])
                                print(f"   ✅ Restored {error_info['file']} from backup")

                                unprotect_file(error_info['file'])

                                state["mocked_functions"] = [
                                    m for m in state["mocked_functions"] 
                                    if os.path.abspath(m["file"]) != os.path.abspath(error_info["file"])
                                ]
                                state["use_mocks"] = False
                                state["mocking_phase"] = "running"

                                state["current_input"] = (
                                    "Mock injection failed. File restored from backup. "
                                    "Mocking is disabled for this run. "
                                    "Now rerun the script to see the original error and fix it normally."
                                )
                                # Force a script run next
                                state["phase"] = "START"
                                chat_history.append({"role": "user", "content": state["current_input"]})
                                continue

                    print(f"   🐛 Post-mock error: {error_info['error_type']} in {error_info['file']}:{error_info['line']}")

                    # Decide how to guide the next step after a post-mock failure.
                    start_line, end_line, stmt_type = get_statement_range(error_info["file"], error_info["line"])

                    if stmt_type == "multiline_call":
                        print(f"   🔍 Detected MULTI-LINE CALL spanning lines {start_line}-{end_line}")

                    if post_mock_env_issue.get("is_env_compatibility"):
                        print("   🧪 Post-mock failure classified as third-party package/version mismatch")
                        state["current_input"] = _build_env_resolution_guidance(post_mock_env_issue, state)
                    elif error_analysis.get("module_error") == "Lib":
                        missing = error_analysis.get("missing_module", "unknown")
                        print(f"   📦 Missing library import: {missing}")
                        runtime_context = _load_runtime_cuda_context()
                        recommended_command, recommendation_reason = _recommended_install_command_for_import(missing, runtime_context)
                        if recommended_command:
                            state["current_input"] = (
                                f"Missing import detected: {missing}.\n"
                                f"This import is runtime-sensitive on this machine.\n"
                                f"Recommended install command based on local machine context: {recommended_command}\n"
                                f"Reason: {recommendation_reason}.\n"
                                f"Do not default to a generic pip install for {missing}. Use the recommended command unless project docs contradict it."
                            )
                        else:
                            state["current_input"] = (
                                f"Missing import detected: {missing}.\n"
                                f"Do NOT assume the install command yet.\n"
                                f"Determine whether '{missing}' is provided by a differently named package, a local module, or an optional dependency.\n"
                                f"Inspect requirements files, import usage across the codebase, and package docs.\n"
                                f"Then choose the next action: read project dependency files, inspect imports, use RAG, or run a targeted install command if justified."
                            )
                    elif stmt_type:
                        state["current_input"] = (
                            f"Mock active. New error: {error_info['error_type']}: {error_info['message']}\n"
                            f"File: {error_info['file']}, Line: {error_info['line']}\n"
                            f"Problematic code: `{error_line_content}`\n"
                            f"This is a '{stmt_type}' block spanning lines {start_line}-{end_line}.\n"
                            f"⚠️ Do NOT just comment it out. Analyze the error and provide a REAL fix."
                        )
                    else:
                        state["current_input"] = (
                            f"Mock active. New error: {error_info['error_type']}: {error_info['message']}\n"
                            f"File: {error_info['file']}, Line: {error_info['line']}\n"
                            f"Problematic code: `{error_line_content}`\n"
                            f"⚠️ IMPORTANT: Do NOT just comment out the line. Analyze the error and provide a REAL fix.\n"
                            f"- For KeyError: The key name is likely WRONG. READ the file to see what keys ARE used successfully nearby - they often share naming conventions. DO NOT use .get() with placeholder defaults.\n"
                            f"- For NameError: Check for typos or missing variable definitions.\n"
                            f"- For TypeError: Check argument types and function signatures.\n"
                            f"Read the surrounding code to understand the intent before fixing."
                        )
                    chat_history.append({"role": "user", "content": state["current_input"]})
                    continue
                
                # ===== CASE 3: Regular error =====
                # ===== CASE 3: Regular error =====
                error_line_content = ""
                try:
                    with open(error_info["file"], "r") as f:
                        lines = f.readlines()
                    if 0 < error_info["line"] <= len(lines):
                        error_line_content = lines[error_info["line"] - 1].strip()
                except:
                    error_line_content = "unknown"

                print(f"   🐛 Error: {error_info['error_type']} in {error_info['file']}:{error_info['line']}")
                
                # Check if we should trace to caller (for TypeError in validation code)
                should_trace, caller_info, trace_reason = should_trace_to_caller(
                    error_info, error_analysis, error_info.get("file", "")
                )
                
                if should_trace:
                    print(f"   🔍 TRACING TO CALLER: {trace_reason}")
                    
                    if caller_info:
                        # Force the agent to look at the CALLER, not the library
                        state["current_input"] = (
                            f"⛔ STOP TRYING TO EDIT LIBRARY CODE!\n\n"
                            f"The error in {error_info.get('file', '')} is a TYPE CHECK that's working correctly.\n"
                            f"A float was passed where a sequence was expected.\n\n"
                            f"🎯 THE BUG IS IN YOUR CODE:\n"
                            f"   File: {caller_info['file']}\n"
                            f"   Line: {caller_info['line']}\n\n"
                            f"You MUST read {caller_info['file']} around line {caller_info['line']} to find where "

                            f"Use: {{\"action_type\": \"ReadFileInput\", \"action\": "
                            f"{{\"file_path\": \"{caller_info['file']}\", \"error_line\": {caller_info['line']}, "
                            f"\"window_before\": 20, \"window_after\": 30}}}}"
                        )
                        state["phase"] = "RAN"  # Allow ReadFileInput
                        chat_history.append({"role": "user", "content": state["current_input"]})
                        continue
                    else:
                        # Library error but couldn't find caller - still warn
                        state["current_input"] = (
                            f"⛔ ERROR IS IN LIBRARY CODE: {error_info.get('file', '')}\n\n"
                            f"DO NOT edit this file. The error indicates your code is passing wrong types.\n"
                            f"Check the traceback to find YOUR code that calls this library function."
                        )
                        chat_history.append({"role": "user", "content": state["current_input"]})
                        continue

                # Check if error is on a 'raise' statement - real bug is in caller
                caller_hint = ""
                user_frame = _find_first_user_stack_frame(error_analysis.get("full_stack", []))
                artifact_target = None
                parser_error_types = {"JSONDecodeError", "json.decoder.JSONDecodeError", "YAMLError", "yaml.YAMLError", "TOMLDecodeError", "tomllib.TOMLDecodeError"}
                if error_info.get("error_type") in parser_error_types:
                    artifact_target = _infer_loaded_artifact_from_user_frame(state, user_frame)
                    if artifact_target:
                        state["artifact_read_target"] = artifact_target
                        state["current_input"] = (
                            f"Structured data/config parsing failed with {error_info.get('error_type')}.\n"
                            f"Do NOT read library parser internals.\n"
                            f"The likely malformed artifact is: {artifact_target['file']}\n"
                            f"Reason: {artifact_target['reason']}.\n"
                            f"Choose ReadFileInput for {artifact_target['file']} to inspect the raw file contents."
                        )
                        state["phase"] = "RAN"
                        chat_history.append({"role": "user", "content": state["current_input"]})
                        continue
                if error_analysis.get("is_raise_statement") and error_analysis.get("caller_info"):
                    caller = error_analysis["caller_info"]
                    caller_hint = (
                        f"\n\n⚠️ CRITICAL: This error is on a 'raise' statement - the code is CORRECTLY catching bad data.\n"
                        f"The REAL BUG is in the CALLER that's passing invalid data.\n"
                        f"Look at: {caller['file']}:{caller['line']}\n"
                        f"Do NOT modify the raise statement. Fix the caller instead."
                    )
                    print(f"   ⚠️ Error on 'raise' statement - real bug at {caller['file']}:{caller['line']}")

                if not env_issue.get("is_env_compatibility") and user_frame and (
                    error_analysis.get("is_raise_statement") or _is_runtime_library_file(error_info.get("file", ""))
                ):
                    state["current_input"] = (
                        f"The traceback currently ends inside runtime/library code: {error_info.get('file', '')}:{error_info.get('line', 0)}.\n"
                        f"Do NOT read or edit that library file.\n"
                        f"The actionable caller in your code is: {user_frame['file']}:{user_frame['line']}.\n"
                        f"Choose ReadFileInput for {user_frame['file']} around line {user_frame['line']} and debug the user-side cause there."
                    )
                    state["phase"] = "RAN"
                    chat_history.append({"role": "user", "content": state["current_input"]})
                    continue
                target_file_to_read = error_info["file"]
                if target_file_to_read == "unknown" or "site-packages" in target_file_to_read:
                    target_file_to_read = state["target_file"]

                # === FIX: Use get_statement_range instead of get_block_range ===
                start_line, end_line, stmt_type = get_statement_range(error_info["file"], error_info["line"])

                if stmt_type:
                    if stmt_type == "multiline_call":
                        print(f"   🔍 Detected MULTI-LINE CALL spanning lines {start_line}-{end_line}")
                        state["current_input"] = (
                            f"Error: {error_info['error_type']}: {error_info['message']}\\n"
                            f"File: {error_info['file']}, Line: {error_info['line']}\\n"
                            f"Problematic code: `{error_line_content}`\\n\\n"
                            f"{caller_hint}\\n"
                            f"⚠️ This is a MULTI-LINE FUNCTION CALL spanning lines {start_line}-{end_line}.\\n"
                            f"Do NOT just comment it out. Analyze WHY this error is happening:\\n"
                            f"- For KeyError: The key name is likely WRONG. READ the file to see what keys ARE used successfully nearby - they often share naming conventions. DO NOT use .get() with placeholder defaults.\\n"
                            f"- For NameError: Check for typos in variable names.\\n"
                            f"- For AttributeError: The method/attribute doesn't exist - find the correct one.\\n"
                            f"Fix the actual problem, don't disable the code."
                        )
                    else:
                        print(f"   🔍 Detected '{stmt_type}' block spanning lines {start_line}-{end_line}")
                        state["current_input"] = (
                            f"Error: {error_info['error_type']}: {error_info['message']}\\n"
                            f"File: {error_info['file']}, Line: {error_info['line']}\\n"
                            f"Problematic code: `{error_line_content}`\\n"
                            f"{caller_hint}\\n"
                            f"This is a '{stmt_type}' block spanning lines {start_line}-{end_line}.\\n"
                            f"⚠️ Do NOT just comment out the block. Analyze the error and provide a REAL fix.\\n"
                            f"Read the surrounding code to understand the intent, then fix the actual bug."
                        )
                else:
                    state["current_input"] = (
                        f"Error: {error_info['error_type']}: {error_info['message']}\\n"
                        f"File: {error_info['file']}, Line: {error_info['line']}\\n"
                        f"Problematic code: `{error_line_content}`\\n\\n"
                        f"{caller_hint}\\n"
                        f"⚠️ IMPORTANT: Do NOT just comment out the line. Analyze the error and provide a REAL fix.\\n"
                        f"- For KeyError: The key name is likely WRONG. READ the file to see what keys ARE used successfully nearby - they often share naming conventions. DO NOT use .get() with placeholder defaults.\\n"
                        f"- For NameError: Variable not defined. Check for typos or look for the correct variable name.\\n"
                        f"- For AttributeError: Method/attribute doesn't exist. Check the class definition.\\n"
                        f"- For TypeError: Wrong argument types. Check function signature.\\n"
                        f"Read the surrounding code context to understand what the code is trying to do, then fix it properly."
                    )
                chat_history.append({"role": "user", "content": state["current_input"]})
                continue

        # =====================================================================
        # RAGSearchInput
        # =====================================================================
        if action.action_type == "RAGSearchInput":
            rag_allowed = bool(
                state.get("rag_eligible", False)
                or state.get("force_rag", False)
                or state.get("user_requested_rag", False)
            )
            if not rag_allowed:
                state["current_input"] = (
                    "RAGSearchInput blocked. By default, RAG is allowed only after cyclic "
                    "error detection. If you explicitly want RAG now, request it in user feedback."
                )
                chat_history.append({"role": "user", "content": state["current_input"]})
                continue

            act = action.action
            last_error = state.get("full_error_history", [])[-1] if state.get("full_error_history") else {}
            rag_signature = _build_rag_signature(
                act.error_type or last_error.get("error_type", ""),
                act.error_message or last_error.get("message", ""),
                act.file_path or last_error.get("file", state.get("target_file", "")),
                act.error_line or last_error.get("line", state.get("error_line", 0) or 0),
            )
            if rag_signature == state.get("last_rag_signature", "") and state.get("last_rag_results"):
                state["force_rag"] = False
                state["user_requested_rag"] = False
                state["current_input"] = (
                    "RAG for this same error signature was already executed. Reuse the existing RAG evidence in state and choose a concrete next action instead of repeating the same lookup."
                )
                chat_history.append({"role": "user", "content": state["current_input"]})
                continue

            error_type = act.error_type or last_error.get("error_type", "")
            error_message = act.error_message or last_error.get("message", "")
            file_path = act.file_path or last_error.get("file", state.get("target_file", ""))
            error_line = act.error_line or last_error.get("line", state.get("error_line", 0) or 0)
            code_context = act.code_context or ""

            if not code_context and file_path and error_line:
                try:
                    with open(file_path, "r") as f:
                        lines = f.readlines()
                    idx = max(0, error_line - 1)
                    start = max(0, idx - 5)
                    end = min(len(lines), idx + 5)
                    code_context = "".join(lines[start:end])
                except Exception:
                    code_context = ""

            print(f"\n🌐 Running RAG tool for {error_type} at {file_path}:{error_line}")
            rag_results = search_online_for_solution(
                error_type=error_type,
                error_message=error_message,
                code_context=code_context,
                file_path=file_path,
                error_line=error_line,
                env_context=_build_env_context_for_rag(state),
            )
            state["rag_searches_performed"] = state.get("rag_searches_performed", 0) + 1
            state["last_rag_search_step"] = state.get("current_step", 0)
            state["last_rag_signature"] = rag_signature
            state["pending_rag_signature"] = ""

            analysis = rag_results.get("code_analysis", {})
            similar_keys = analysis.get("similar_keys", [])
            class_name = analysis.get("class_name", "")
            state["last_rag_results"] = rag_results
            state["last_similar_keys"] = similar_keys
            enhanced = enhanced_rag_search(
                error_type,
                error_message,
                file_path,
                error_line,
                similar_keys,
            )
            suggested_replacement = enhanced.get("suggested_replacement")

            if error_type == "KeyError" and similar_keys:
                enriched_input = build_keyerror_fix_prompt(
                    error_message,
                    similar_keys,
                    code_context,
                    file_path,
                    error_line,
                    suggested_replacement,
                )
            else:
                llm_suggestion = llm_suggest_fix(
                    error_type=error_type,
                    error_message=error_message,
                    code_context=code_context,
                    similar_keys=similar_keys,
                    class_name=class_name,
                    file_path=file_path,
                    runtime_context=rag_results.get("runtime_context", {}),
                )
                rag_hints = "\n".join([f"  - {hint}" for hint in rag_results["solutions"][:5]])
                rag_sources = "\n".join([f"  - {src}" for src in rag_results["sources"][:3]])
                enriched_input = (
                    f"RAG TOOL RESULTS for {error_type} at {file_path}:{error_line}\n\n"
                    f"LLM suggestion: {llm_suggestion}\n\n"
                    f"Hints:\n{rag_hints}\n\n"
                    f"Sources:\n{rag_sources}\n\n"
                    f"Use this context to choose the next action."
                )

            state["current_input"] = enriched_input
            state["phase"] = "RAG"
            state["force_rag"] = False
            state["user_requested_rag"] = False
            state["awaiting_user_feedback"] = True
            chat_history.append({"role": "user", "content": state["current_input"]})
            continue

        # =====================================================================
        # ExecuteTerminalCommand
        # =====================================================================
        if action.action_type == "ExecuteTerminalCommand":
            act = action.action
            normalized_command = _normalize_command_text(act.command)
            preferred_command = _normalize_command_text(state.get("preferred_next_command", ""))
            if preferred_command and normalized_command == preferred_command:
                state["preferred_next_command"] = ""
            if state.get("env_resolution_active") and _is_install_command(act.command):
                signature = state.get("env_issue_signature", "")
                repair_count = sum(1 for item in state.get("env_repair_history", []) if item.get("signature") == signature)
                env_packages = [pkg.lower() for pkg in state.get("env_resolution_packages", []) if pkg]
                command_lower = normalized_command.lower()
                mentions_env_packages = any(pkg in command_lower for pkg in env_packages)
                user_requested_this_command = bool(preferred_command and normalized_command == preferred_command)
                namespace_hint = state.get("env_namespace_shadow_hint", {}) or {}
                root = (namespace_hint.get("root") or "").lower()
                generic_root_upgrade = bool(root and _is_generic_import_install_command(command_lower, root))
                allowed_repair_command = _is_allowed_env_repair_command(normalized_command, state)
                if not user_requested_this_command and not allowed_repair_command:
                    allowed_hint = namespace_hint.get('uninstall_command') or state.get('env_resolution_command') or 'use RAG or a targeted repair tied to the traceback packages'
                    state["current_input"] = (
                        "Rejected environment repair command that is not grounded in the current traceback.\n"
                        f"Command: {normalized_command}\n"
                        f"Active package focus: {', '.join(state.get('env_resolution_packages', [])[:6]) or 'none recorded'}\n"
                        "Do not upgrade or reinstall unrelated/transitive packages just because they appear in metadata.\n"
                        f"Choose one of these instead: rerun if a real repair already happened, use RAG, or run this grounded command: {allowed_hint}"
                    )
                    state["awaiting_user_feedback"] = True
                    chat_history.append({"role": "user", "content": state["current_input"]})
                    continue
                if namespace_hint.get("suspicious") and generic_root_upgrade and not user_requested_this_command:
                    state["current_input"] = (
                        "Rejected generic install/upgrade command during environment resolution.\n"
                        f"Command: {normalized_command}\n"
                        f"Reason: namespace shadowing is suspected for import root '{root}'.\n"
                        "Do not reinforce the exact-root distribution blindly. Use a targeted family diagnosis or remove the shadowing root package instead.\n"
                        f"Recommended next action: Execute terminal command: {namespace_hint.get('uninstall_command') or namespace_hint.get('family_command')}"
                    )
                    state["awaiting_user_feedback"] = True
                    chat_history.append({"role": "user", "content": state["current_input"]})
                    continue
                if signature and repair_count >= 2 and not mentions_env_packages and not user_requested_this_command:
                    packages = ", ".join(state.get("env_resolution_packages", [])[:6]) or "the packages in the traceback"
                    state["current_input"] = (
                        "Repeated compatibility failure is still active after multiple repair attempts.\n"
                        f"Current compatibility packages: {packages}.\n"
                        f"Rejected command: {normalized_command}\n"
                        f"Successful repair attempts already made for this same signature: {repair_count}.\n"
                        "Do not try another broad or unrelated install.\n"
                        "Next action must either inspect the involved package versions directly or use RAG focused on those packages."
                    )
                    state["awaiting_user_feedback"] = True
                    chat_history.append({"role": "user", "content": state["current_input"]})
                    continue
            if state.get("env_resolution_active") and _is_env_diagnostic_command(act.command):
                latest_error = state.get("full_error_history", [])[-1] if state.get("full_error_history") else {}
                blocked, block_message = _check_redundant_env_diagnostic(state, latest_error, act.command)
                if blocked:
                    state["current_input"] = block_message
                    state["awaiting_user_feedback"] = True
                    chat_history.append({"role": "user", "content": state["current_input"]})
                    continue
            result = terminal_command_executor.invoke({"command": act.command})
            state["phase"] = "TERMINAL_COMMAND"
            state["last_terminal_result"] = result
            state["last_stdout"] = result.get("stdout", "") or ""
            state["last_stderr"] = result.get("stderror", "") or ""
            state["last_exitcode"] = result.get("exit_code")
            if _is_install_command(result.get("command", act.command)):
                state["last_env_repair_step"] = state.get("current_step", 0)
                if state.get("env_resolution_active"):
                    state["env_ready_to_rerun"] = True
                if state.get("env_resolution_active") and state.get("env_issue_signature"):
                    state.setdefault("env_repair_history", []).append({
                        "signature": state.get("env_issue_signature", ""),
                        "command": _normalize_command_text(result.get("command", act.command)),
                        "step": state.get("current_step", 0),
                    })
            if state.get("env_resolution_active") and _is_env_diagnostic_command(result.get("command", act.command)) and result.get("exit_code") == 0:
                state["env_ready_to_rerun"] = False
                latest_error = state.get("full_error_history", [])[-1] if state.get("full_error_history") else {}
                _record_env_diagnostic_result(state, latest_error, result.get("command", act.command), result)
            review_latest_run_output(state, label=f"terminal command: {act.command}")
            
            if result.get("exit_code") == 0:
                if state.get("env_resolution_active"):
                    state["current_input"] = _build_env_resolution_followup(result, state)
                    state["phase"] = "TERMINAL_COMMAND"
                    state["awaiting_user_feedback"] = True
                else:
                    state["current_input"] = "Command succeeded. Now rerun the script."
            else:
                preview = (result.get("stderror") or result.get("stdout") or "")[:500]
                state["current_input"] = f"Command failed: {preview}"
            
            chat_history.append({"role": "user", "content": _summarize_terminal_result_for_prompt(act.command, result)})
            continue

        # =====================================================================
        # ReadFileInput
        # =====================================================================
        if action.action_type == "ReadFileInput":
            act = action.action
            artifact_target = state.get("artifact_read_target") or {}
            if artifact_target and os.path.normpath(act.file_path) == os.path.normpath(artifact_target.get("file", "")):
                act.error_line = int(artifact_target.get("line", act.error_line or 1) or 1)
            abs_path = os.path.abspath(act.file_path)
            
            # Get escalated read range if we've read this location before
            error_line_to_use = act.error_line or state.get("error_line", 0) or 0
            before_offset, after_offset, was_escalated = get_escalated_read_range(
                state, act.file_path, error_line_to_use
            )

            if was_escalated:
                print(f"   📖 Escalating read range (attempt #{state['read_attempt_counts'].get(f'{act.file_path}:{error_line_to_use}', 1)})")
    
            is_library = "site-packages" in abs_path or "anaconda3" in abs_path
            if is_library:
                print(f"   ⚠️ Reading library file: {act.file_path}")
                latest_error = state.get("full_error_history", [])[-1] if state.get("full_error_history") else {}
                diag_command = state.get("env_resolution_command") or _build_env_diagnosis_command(
                    state.get("env_resolution_packages", []),
                    error_file=(state.get("last_env_issue", {}) or {}).get("error_file", ""),
                    error_type=(state.get("last_env_issue", {}) or {}).get("error_type", ""),
                )
                if not state.get("env_resolution_active"):
                    inferred_env_issue = _classify_environment_compatibility_issue(latest_error, state.get("last_stderr", "") or "")
                    if inferred_env_issue.get("is_env_compatibility"):
                        state["env_ready_to_rerun"] = False
                        state["current_input"] = _build_env_resolution_guidance(inferred_env_issue, state)
                    else:
                        state["current_input"] = (
                            "Stop reading third-party library source by default.\n"
                            f"Requested library file: {act.file_path}\n"
                            "This failure originates in installed dependency code. Prefer environment repair, targeted RAG, or a one-shot approved library hotfix rather than recursive library inspection."
                        )
                    state["awaiting_user_feedback"] = True
                    chat_history.append({"role": "user", "content": state["current_input"]})
                    continue
                state["current_input"] = (
                    "Stop reading third-party library source for this failure class.\n"
                    "This session is in environment-resolution mode because the traceback indicates a dependency compatibility problem.\n"
                    f"Requested library file: {act.file_path}\n"
                    f"Next action should inspect installed versions in the USER environment: Execute terminal command: {diag_command}"
                )
                state["awaiting_user_feedback"] = True
                chat_history.append({"role": "user", "content": state["current_input"]})
                continue
            
            file_text_res = read_file.invoke({
                "file_path": act.file_path,
                "file_type": act.file_type,
                "error_line": error_line_to_use,
                "window_before": before_offset,   # PASS THE WINDOW
                "window_after": after_offset,     # PASS THE WINDOW
            })
            file_text = file_text_res.get("content") or ""
            state["last_file_text"] = file_text
            state["last_read_file"] = act.file_path
            if state.get("artifact_read_target") and os.path.normpath(act.file_path) == os.path.normpath((state.get("artifact_read_target") or {}).get("file", "")):
                state["artifact_read_target"] = {}
            state["phase"] = "READ"
            
            # Build context message
            escalation_note = ""
            if was_escalated:
                escalation_note = f"\n(Showing wider context: ±{before_offset} lines)\n"
            
            if is_library:
                state["current_input"] = (
                    f"This is a LIBRARY file (read-only). The error is in YOUR code that calls it. "
                    f"Check the traceback for your project files.\\n"
                    f"FILE_SNIPPET: {file_text[:1000]}"
                )
            elif state["strategy"] == "mocking":
                mock_hint = ""
                for func in state["expensive_functions"]:
                    if os.path.abspath(func["file"]) == abs_path:
                        already_mocked = any(
                            m["file"] == func["file"] and m["function"] == func["function"]
                            for m in state["mocked_functions"]
                        )
                        if not already_mocked:
                            mock_hint = f" Note: {func['function']}() can be mocked here."
                            break
                
                state["current_input"] = f"FILE_SNIPPET({act.file_path}):{mock_hint}{escalation_note}\\n{file_text[:1500]}"
            else:
                state["current_input"] = f"FILE_SNIPPET({act.file_path}):{escalation_note}\\n{file_text[:1500]}"
            
            chat_history.append({"role": "user", "content": state["current_input"]})
            continue
        # =====================================================================
        # WriteFileInput
        # =====================================================================
        if action.action_type == "WriteFileInput":
            act = action.action
            
            # ===== BLOCK REPEATED IDENTICAL WRITE ATTEMPTS =====
            write_signature = f"{act.file_path}:{act.start_line}:{act.new_content[:100] if act.new_content else ''}"
            recent_writes = state.get("recent_write_attempts", [])
            
            if write_signature in recent_writes:
                # Count how many times we've blocked writes for this specific location
                block_key = f"blocked:{act.file_path}:{act.start_line}"
                block_count = state.get("block_counts", {}).get(block_key, 0) + 1
                state.setdefault("block_counts", {})[block_key] = block_count
                
                print(f"   ⛔ BLOCKED: Identical write attempt already tried (block #{block_count})")
                
                # After 5+ blocks, force a completely different approach
                # After 5+ blocks, FORCE a read action - don't even give it a choice
                if block_count >= 5:
                    print(f"   🚨 STUCK: Blocked {block_count} times at same location - FORCING READ")
                    
                    # Don't add to chat history - instead, FORCE a read action
                    # by directly invoking read_file and injecting the result
                    
                    # Read a DIFFERENT section - lines BEFORE the error where working keys exist
                    target_line = max(1, act.start_line - 10)
                    file_text_res = read_file.invoke({
                        "file_path": act.file_path,
                        "error_line": target_line,
                        "window_before": 15,
                        "window_after": 5,
                    })
                    file_text = file_text_res.get("content") or ""
                    
                    # Extract keys from this section to show the LLM what ACTUALLY exists
                    import re
                    found_keys = re.findall(r'hparams\["([^"]+)"\]', file_text)
                    # Also try other dict access patterns
                    found_keys += re.findall(r'\["([^"]+)"\]', file_text)
                    unique_keys = list(dict.fromkeys(found_keys))[:10]
                    
                    state["current_input"] = (
                        f"🚨 FORCED READ - You were stuck. Here are lines BEFORE line {act.start_line}:\n\n"
                        f"{file_text}\n\n"
                        f"📋 KEYS ACTUALLY USED IN THIS CODE: {unique_keys}\n\n"
                        f"Use ONE of these exact keys. Do NOT invent new key names."
                    )
                    state["phase"] = "READ"
                    
                    # Reset block count so we don't keep forcing
                    state["block_counts"][block_key] = 0
                    
                    chat_history.append({"role": "user", "content": state["current_input"]})
                    continue

                # Extract what key/value was attempted (for KeyError fixes)
                attempted_key = None
                key_match = re.search(r'\["([^"]+)"\]', act.new_content or "")
                if key_match:
                    attempted_key = key_match.group(1)

                # Get the similar keys that were suggested (if we have them in state)
                similar_keys = state.get("last_similar_keys", [])

                alternatives_msg = ""
                if attempted_key and similar_keys:
                    error_location = f"{act.file_path}:{act.start_line}"
                    remaining, alternatives_msg = track_tried_fix_and_get_alternatives(
                        state, error_location, attempted_key, similar_keys
                    )
                    alternatives_msg = f"\n\n{alternatives_msg}"

                state["current_input"] = (
                    f"⛔ You've already tried this EXACT fix and it didn't work.\n\n"
                    f"STOP repeating the same action. Try something DIFFERENT:\n"
                    f"1. READ the file to understand the context better\n"
                    f"2. Try a completely different approach to fix the error\n"
                    f"3. Check if you're editing the WRONG file"
                    f"{alternatives_msg}"
                )
                chat_history.append({"role": "user", "content": state["current_input"]})
                continue
            is_lib, lib_reason = is_library_or_utility_file(act.file_path)
            allowed_library_patch = _consume_library_patch_allowance(state, act.file_path) if is_lib else False
            if is_lib and not allowed_library_patch:
                print(f"   ⛔ BLOCKED: Cannot edit library file: {lib_reason}")

                # Find the caller file from traceback
                caller_hint = ""
                full_stack = state.get("full_error_history", [{}])[-1].get("full_stack", [])
                for stack_file, stack_line in full_stack:
                    if not is_library_or_utility_file(stack_file)[0]:
                        caller_hint = f"\n\nThe fix should be in YOUR code at: {stack_file}:{stack_line}"
                        break
                    
                state["current_input"] = (
                    f"⛔ CANNOT EDIT LIBRARY FILE: {act.file_path}\n\n"
                    f"Reason: {lib_reason}\n\n"
                    f"Library code should NOT be modified by default. The failure may be a dependency incompatibility rather than a source bug.{caller_hint}\n\n"
                    "If you intentionally want a local hotfix, propose the exact library patch again and approve it with the one-shot library patch option."
                )
                chat_history.append({"role": "user", "content": state["current_input"]})
                continue
            
            # === Check file protection FIRST ===
            if is_file_protected(act.file_path):
                print(f"   ⛔ BLOCKED: File {act.file_path} is protected (contains mocked function)")
                state["current_input"] = (
                    f"CANNOT EDIT {act.file_path} - it contains a mocked expensive function. "
                    f"The wrapper in this file MUST stay intact. "
                    f"Look at the TRACEBACK to find which OTHER file has the actual error."
                )
                chat_history.append({"role": "user", "content": state["current_input"]})
                continue
            
            # Check mocking phase protection
            if state.get("mocking_phase") in ["complete", "running"]:
                for mocked in state.get("mocked_functions", []):
                    if os.path.abspath(mocked["file"]) == os.path.abspath(act.file_path):
                        print(f"   ⚠️ BLOCKED: Cannot edit mocked file {mocked['file']}")
                        state["current_input"] = (
                            f"BLOCKED: {mocked['file']} contains the mocked function {mocked['function']}(). "
                            f"Do NOT edit this file. Fix the error in OTHER files instead."
                        )
                        chat_history.append({
                            "role": "user", 
                            "content": f"BLOCKED: Cannot edit {mocked['file']} - it contains mocked function"
                        })
                        continue
            
            # =================================================================
            # PRE-VALIDATION INDENTATION FIX
            # Apply indentation correction BEFORE syntax validation
            # This ensures the syntax check sees properly indented code
            # =================================================================
            content_for_validation = act.new_content
            if act.start_line and os.path.exists(act.file_path):
                try:
                    with open(act.file_path, 'r') as f:
                        original_file_lines = f.readlines()
                    
                    # Apply indentation fix before validation
                    content_for_validation = preserve_multiline_indentation(
                        act.new_content,
                        original_file_lines,
                        act.start_line,
                        act.end_line or act.start_line
                    )
                except Exception:
                    pass  # If this fails, use original content
            
            # Validate write safety (now with properly indented content)
            is_safe, reason, should_continue = validate_write_safety(
                file_path=act.file_path,
                new_content=content_for_validation,
                start_line=act.start_line,
                end_line=act.end_line,
                state=state,
            )
            import re
            if not is_safe:
                print(f"   {reason}")
                
                # Track cascading syntax errors
                if "SYNTAX ERROR" in reason:
                    syntax_line_match = re.search(r'line (\d+)', reason)
                    if syntax_line_match:
                        current_syntax_line = int(syntax_line_match.group(1))
                        prev_syntax_lines = state.get("syntax_error_lines", [])
                        
                        if current_syntax_line not in prev_syntax_lines:
                            prev_syntax_lines.append(current_syntax_line)
                            state["syntax_error_lines"] = prev_syntax_lines
                        
                        # If syntax errors spread to 3+ different lines, restore and force re-read
                        if len(prev_syntax_lines) >= 3:
                            print(f"   ⚠️ CASCADING SYNTAX ERRORS at lines {prev_syntax_lines}")
                            backup_path = f"{act.file_path}.backup"
                            if os.path.exists(backup_path):
                                import shutil
                                shutil.copy(backup_path, act.file_path)
                                print(f"   🔄 Restored {act.file_path} from backup")
                            
                            # Find the ORIGINAL error (before syntax errors started)
                            # This is typically the first non-SyntaxError/IndentationError in history
                            original_error = None
                            for err in state.get("full_error_history", []):
                                if err.get("error_type") not in ["SyntaxError", "IndentationError"]:
                                    original_error = err
                                    break
                            
                            # Remove all SyntaxError/IndentationError entries from history
                            # These were caused by bad fixes, not real bugs
                            state["full_error_history"] = [
                                err for err in state.get("full_error_history", [])
                                if err.get("error_type") not in ["SyntaxError", "IndentationError"]
                            ]
                            
                            state["syntax_error_lines"] = []
                            
                            # Re-inject original error context if found
                            if original_error:
                                orig_type = original_error.get("error_type", "Unknown")
                                orig_msg = original_error.get("message", "")  # Note: key is "message" not "error_message"
                                orig_file = original_error.get("file", act.file_path)
                                orig_line = original_error.get("line", "unknown")
                                
                                # Reset error_line to the ORIGINAL error location
                                if orig_line and orig_line != "unknown":
                                    state["error_line"] = orig_line
                                
                                state["current_input"] = (
                                    f"⚠️ CASCADING SYNTAX ERRORS detected - your fixes broke the file structure.\n"
                                    f"File restored from backup.\n\n"
                                    f"🎯 ORIGINAL ERROR (focus on this):\n"
                                    f"   {orig_type}: {orig_msg}\n"
                                    f"   File: {orig_file}, Line: {orig_line}\n\n"
                                    f"⚠️ CRITICAL: Your previous fix attempt caused syntax errors.\n"
                                    f"You MUST:\n"
                                    f"1. READ the file first to see the RESTORED original code\n"
                                    f"2. Fix ONLY the original {orig_type}, preserving ALL indentation\n"
                                    f"3. Include the COMPLETE line with proper spacing"
                                )
                            else:
                                state["current_input"] = (
                                    f"⚠️ CASCADING SYNTAX ERRORS detected at lines {prev_syntax_lines}.\n"
                                    f"File restored from backup. You MUST READ the file first.\n"
                                    f"Use ReadFileInput to see the full context before making changes."
                                )
                            chat_history.append({"role": "user", "content": state["current_input"]})
                            continue
                
                # Track repeated .get() blocks for KeyError
                if "BAD PATTERN" in reason and ".get(" in reason:
                    get_block_count = state.get("get_block_count", 0) + 1
                    state["get_block_count"] = get_block_count
                    
                    # After 2 blocked .get() attempts, force a file read with explicit guidance
                    if get_block_count >= 2:
                        state["get_block_count"] = 0  # Reset counter
                        state["force_rag"] = True
                        
                        # Find the error line to read around
                        error_line = state.get("error_line", 1)
                        error_file = state.get("last_error_file", act.file_path)
                        
                        print(f"   🔄 Repeated .get() blocks - forcing file read for context")
                        state["current_input"] = (
                            f"⚠️ You have tried .get() multiple times. THIS PATTERN IS BLOCKED.\n\n"
                            f"You MUST use DIRECT KEY ACCESS with the CORRECT key name.\n"
                            f"Read the file to find valid keys that exist in the codebase.\n\n"
                            f"DO NOT use .get() - use [\"key\"] syntax instead."
                        )
                        chat_history.append({"role": "user", "content": state["current_input"]})
                        
                        # Force a read action
                        state["phase"] = "READ"
                        state["force_read"] = True
                        continue
                
                state["current_input"] = f"Write blocked: {reason}. Please fix and try again."
                chat_history.append({"role": "user", "content": f"BLOCKED: {reason}"})
                continue

            # Process escaped characters
            # Use the pre-indented content from validation
            raw_content = content_for_validation
            raw_content = raw_content.replace("\\n", "\n")
            raw_content = raw_content.replace("\\t", "\t")
            raw_content = raw_content.replace("\\r", "")
            
            # Now check edit size (after processing escapes)
            if act.start_line is not None and act.end_line is not None:
                lines_to_replace = act.end_line - act.start_line + 1
                new_lines_count = raw_content.count('\n') + 1

                if lines_to_replace <= 2 and new_lines_count > 10:
                    print(f"   ⚠️ Edit too large: replacing {lines_to_replace} lines with {new_lines_count} lines")
                    state["current_input"] = (
                        f"Your edit is too large. You're replacing {lines_to_replace} line(s) "
                        f"with {new_lines_count} lines. Make a SMALLER fix."
                    )
                    chat_history.append({"role": "user", "content": f"EDIT TOO LARGE: {state['current_input']}"})
                    continue
            # === BLOCK: Multiline content replacing single line (causes indent errors) ===
            if act.start_line == act.end_line:  # Replacing single line
                new_line_count = raw_content.count('\n')
                if new_line_count > 1:  # More than 1 newline = multiple lines
                    print(f"   ⛔ BLOCKED: Cannot replace 1 line with {new_line_count + 1} lines")
                    state["current_input"] = (
                        f"⛔ Your fix replaces 1 line with {new_line_count + 1} lines.\n"
                        f"This will break indentation. Either:\n"
                        f"1. Keep your fix to a SINGLE line, OR\n"
                        f"2. Specify the correct end_line to replace multiple lines\n\n"
                        f"If you need to add lines, set end_line = start_line + (lines_to_add - 1)"
                    )
                    chat_history.append({"role": "user", "content": state["current_input"]})
                    continue
            # Check for mock stub
            func_to_mock = None
            for func in state.get("expensive_functions", []):
                if os.path.abspath(func["file"]) == os.path.abspath(act.file_path):
                    func_match = re.search(r"def\s+(\w+)\s*\(", raw_content)  # FIXED: removed extra backslashes
                    if func_match and func_match.group(1) == func["function"]:
                        func_to_mock = func
                        break
            
            is_mock_stub = False
            mock_res = None
            
            if func_to_mock and state.get("strategy") == "mocking":
                already_mocked = any(
                    m["file"] == func_to_mock["file"] and m["function"] == func_to_mock["function"]
                    for m in state.get("mocked_functions", [])
                )
                
                if not already_mocked:
                    is_mock_stub = True
                    func_key = f"{func_to_mock['file']}::{func_to_mock['function']}"
                    cached_value = state["func_outputs"].get(func_key, state.get("last_stdout") or "mocked")
                    
                    from system_agent import mock_function
                    mock_res = mock_function(
                        file_path=act.file_path,
                        func_name=func_to_mock["function"],
                        cached_value=cached_value,
                    )
                    
                    if mock_res.get("success"):
                        state.setdefault("mocked_functions", []).append({
                            "file": act.file_path,
                            "function": func_to_mock["function"],
                        })
                        print(f"   ✅ Mocked {func_to_mock['function']}()")
                        state["use_mocks"] = True
                        state["mocking_phase"] = "complete"
            
            if is_mock_stub and mock_res and mock_res.get("success"):
                state["phase"] = "WROTE"
                state["current_input"] = "Function mocked. Rerun to continue."
                chat_history.append({"role": "user", "content": f"MOCKED: {mock_res}"})
                continue
            
            # Read existing file to get indentation
            original_indent = 0
            try:
                with open(act.file_path, "r", encoding="utf-8") as f:
                    existing_lines = f.readlines()
                target_idx = max(0, (act.start_line or 1) - 1)
                if target_idx < len(existing_lines):
                    original_line = existing_lines[target_idx]
                    original_indent = len(original_line) - len(original_line.lstrip())
            except:
                pass
            
            # Clean up content and ensure single trailing newline
            final_new_content = raw_content.rstrip() + "\n"

            # Get original lines being replaced
            original_lines_to_check = []
            if act.start_line and act.end_line and os.path.exists(act.file_path):
                try:
                    with open(act.file_path, 'r') as f:
                        all_lines = f.readlines()
                        start_idx = max(0, act.start_line - 1)
                        end_idx = min(len(all_lines), act.end_line)
                        original_lines_to_check = all_lines[start_idx:end_idx]
                except:
                    pass

            if original_lines_to_check and is_comment_out_fix(original_lines_to_check, final_new_content):
                # User explicitly approved this style of fix in the approval gate.
                if not state.pop("allow_comment_out_once", False):
                    error_key = f"comment_out:{act.file_path}:{act.start_line}"
                    state["blocked_pattern_counts"][error_key] = state["blocked_pattern_counts"].get(error_key, 0) + 1
                    count = state["blocked_pattern_counts"][error_key]

                    print(f"   🚫 REJECTED: Agent tried to comment out code (attempt {count})")
                    # Do not poison repeated-write detection for rejected comment attempts.
                    state["recent_write_attempts"] = [
                        sig for sig in state.get("recent_write_attempts", [])
                        if sig != write_signature
                    ]

                    # ESCALATION STRATEGY
                    if count >= 3:
                        # Level 1: Force trace to caller
                        state["escalation_level"] = 1
                        state["force_rag"] = True

                        # Get the full stack trace
                        full_stack = state.get("full_error_history", [{}])[-1].get("full_stack", [])

                        # Find alternative location to investigate
                        alternative = None
                        for stack_file, stack_line in full_stack:
                            if stack_file != act.file_path and "site-packages" not in stack_file:
                                alternative = {"file": stack_file, "line": stack_line}
                                break

                        if alternative:
                            state["current_input"] = (
                                f"⚠️ ESCALATION: You've tried to comment out this code {count} times.\n"
                                f"THIS APPROACH WILL NEVER WORK.\n\n"
                                f"The error at {act.file_path}:{act.start_line} is a SYMPTOM.\n"
                                f"The ROOT CAUSE is in the code that CALLS this function.\n\n"
                                f"🎯 INVESTIGATE THE CALLER:\n"
                                f"   File: {alternative['file']}\n"
                                f"   Line: {alternative['line']}\n\n"
                                f"Use ReadFileInput to examine {alternative['file']} around line {alternative['line']}.\n"
                                f"The fix is there, not here."
                            )
                        else:
                            state["current_input"] = (
                                f"⚠️ ESCALATION: You've tried to comment out this code {count} times.\n"
                                f"THIS APPROACH WILL NEVER WORK.\n\n"
                                f"MANDATORY: You MUST analyze WHY the error occurs, not suppress it.\n\n"
                                f"For TypeError with len(): A scalar was passed where a sequence was expected.\n"
                                f"For KeyError: The key doesn't exist - find the correct key name.\n"
                                f"For AttributeError: The object type is wrong - trace where it's created.\n\n"
                                f"READ the traceback to find where the BAD DATA originates."
                            )

                        # Reset counter after escalation
                        state["blocked_pattern_counts"][error_key] = 0
                    else:
                        state["current_input"] = (
                            f"❌ COMMENTING OUT CODE IS FORBIDDEN (attempt {count}/3).\n\n"
                            f"You must READ {act.file_path} around line {act.start_line} first.\n"
                            f"Use: {{\"action_type\": \"ReadFileInput\", \"action\": {{\"file_path\": \"{act.file_path}\", \"error_line\": {act.start_line}}}}}\n\n"
                            f"Then provide a REAL fix that handles the root cause.\n"
                            f"After 3 failed attempts, you'll be redirected to investigate the caller."
                        )

                    chat_history.append({"role": "user", "content": state["current_input"]})
                    continue
            # =================================================================

            # =================================================================
            # STRUCTURAL BREAK CHECK: Detect if fix would break code structure
            # =================================================================
            is_struct_safe, struct_warning = detect_structural_break(
                act.file_path,
                raw_content,
                act.start_line or 1,
                act.end_line or 1
            )
            if not is_struct_safe:
                print(f"   ⚠️ STRUCTURAL CHECK: {struct_warning}")
                state["current_input"] = (
                    f"BLOCKED: {struct_warning}\n\n"
                    f"Your fix would introduce undefined variables or break code structure.\n"
                    f"READ the file again and use ACTUAL values from the codebase."
                )
                chat_history.append({"role": "user", "content": f"BLOCKED: {struct_warning}"})
                continue

            # =================================================================
            # IMPROVED INDENTATION: Handle both single and multiline content
            # =================================================================
            try:
                with open(act.file_path, "r", encoding="utf-8") as f:
                    original_file_lines = f.readlines()
                
                # Use the improved multiline indentation handler
                final_new_content = preserve_multiline_indentation(
                    final_new_content,
                    original_file_lines,
                    act.start_line or 1,
                    act.end_line or len(original_file_lines)
                )
            except Exception as e:
                # Fall back to simple indentation handling
                if act.start_line is not None and act.start_line == act.end_line:
                    new_content_stripped = final_new_content.lstrip()
                    new_indent = len(final_new_content) - len(new_content_stripped)
                    
                    if new_indent < original_indent:
                        indent_to_add = " " * (original_indent - new_indent)
                        final_new_content = indent_to_add + final_new_content

            # =================================================================
            # FUNCTION RENAME/DELETION PROTECTION
            # =================================================================
            is_rename, orig_name, new_name, rename_warning = detect_function_rename(
                act.file_path, act.start_line, act.end_line, final_new_content
            )
            if is_rename:
                print(f"   {rename_warning}")
                state["current_input"] = rename_warning
                chat_history.append({"role": "user", "content": state["current_input"]})
                continue


            write_res = write_file.invoke({
                "file_path": act.file_path,
                "new_content": final_new_content,
                "start_line": act.start_line,
                "end_line": act.end_line,
                "create_backup": True,
            })
            
            state["phase"] = "WROTE"
            # PATCH 6: Track modifications for self-inflicted damage detection
            track_code_modification(state, act.file_path, act.start_line, act.end_line, final_new_content)
            
            syntaxcheck = check_syntax(act.file_path)
            
            # Improved syntax error handling
            if syntaxcheck and ("SyntaxError" in syntaxcheck or "IndentationError" in syntaxcheck):
                print(f"   ⚠️ Syntax error after write - reverting")
                
                backup_path = f"{act.file_path}.backup"
                if os.path.exists(backup_path):
                    import shutil
                    shutil.copy(backup_path, act.file_path)
                
                # Extract error line from syntax check
                syntax_line_match = re.search(r'line (\d+)', syntaxcheck)
                syntax_error_line = int(syntax_line_match.group(1)) if syntax_line_match else 0
                
                guidance = ""
                if "unexpected indent" in syntaxcheck.lower():
                    guidance = (
                        "\n\n⚠️ CRITICAL: Your fix caused an 'unexpected indent' error.\n"
                        "This typically means:\n"
                        "1. You replaced a single line with multiline content without proper indentation\n"
                        "2. The lines AFTER your fix now have wrong indentation relative to your changes\n"
                        "3. You may have broken an if/for/while block structure\n\n"
                        "SOLUTION: When replacing a line inside an if/for/while block:\n"
                        "- Keep the SAME indentation as the original line\n"
                        "- If adding multiple lines, indent them consistently\n"
                        "- Do NOT add block headers (if:) without proper indented bodies"
                    )
                
                elif "expected an indented block" in syntaxcheck.lower():
                    guidance = (
                        f"\n\n⚠️ CRITICAL: Your fix ended a block header without a body.\n"
                        f"Line {syntax_error_line} expects an indented body after a colon (:).\n"
                        f"Your edit likely removed or incorrectly modified the block body.\n\n"
                        f"SOLUTION: Either:\n"
                        f"1. Include the full block body in your fix, OR\n"
                        f"2. Don't modify lines that are block headers, OR\n"
                        f"3. Use 'pass' as a placeholder body if needed"
                    )
                
                elif "invalid syntax" in syntaxcheck.lower():
                    guidance = (
                        "\n\nHINT: Check that parentheses, brackets, and quotes are balanced.\n"
                        "Also verify you're not mixing Python 2 and 3 syntax."
                    )
                
                state["current_input"] = f"WRITE REVERTED - Syntax Error: {syntaxcheck}{guidance}"
                chat_history.append({"role": "user", "content": state["current_input"]})
                continue
            else:
                state["current_input"] = "File written. Rerun to verify."
                state["syntax_error_lines"] = []  # Reset on successful write
            
            chat_history.append({"role": "user", "content": f"WRITE: {write_res}, SYNTAX: {syntaxcheck}"})
            continue

        # =====================================================================
        # ConfigInput
        # =====================================================================
        if action.action_type == "ConfigInput":
            act = action.action
            msg = record_system_config.invoke({
                "config_name": act.config_name,
                "config_data": act.config_data,
            })
            state["current_input"] = f"Config recorded: {msg}"
            chat_history.append({"role": "user", "content": state["current_input"]})
            continue

        # =====================================================================
        # NoAction
        # =====================================================================
        if action.action_type == "NoAction":
            if state.get("env_resolution_active"):
                state["current_input"] = (
                    "Environment resolution is still active. Do not stop now. "
                    "Choose the next concrete step: a targeted repair command, a more specific diagnosis, or RAG-backed compatibility lookup."
                )
                chat_history.append({"role": "user", "content": state["current_input"]})
                continue
            return {
                "status": "stopped",
                "reason": "NoAction",
                "state": state,
                "history_len": len(chat_history),
                "strategy": state["strategy"],
                "mocked_functions": state["mocked_functions"],
                "config_changes": state["config_changes"],
                "action_history": state.get("action_history", []),
            }

    return {
        "status": "max_steps_reached",
        "state": state,
        "history_len": len(chat_history),
        "strategy": state["strategy"],
        "mocked_functions": state["mocked_functions"],
        "config_changes": state["config_changes"],
        "action_history": state.get("action_history", []),
    }
