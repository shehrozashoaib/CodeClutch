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

import json
import os
import re
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
import re

RECENT_ACTIONS: list = []
MAX_REPEATED_ACTIONS = 3
# In system_llm_agent.py

# In system_llm_agent.py
RECENT_ACTIONS: list = []
MAX_REPEATED_ACTIONS = 3

# File protection set
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
    Returns: (start_line, end_line, is_multiline_call)
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if error_line < 1 or error_line > len(lines):
            return error_line, error_line, False
        
        line = lines[error_line - 1]
        open_parens = line.count('(') - line.count(')')
        open_brackets = line.count('[') - line.count(']')
        
        if open_parens <= 0 and open_brackets <= 0:
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
        
    except Exception:
        return error_line, error_line, False


def get_statement_range(file_path: str, error_line: int) -> tuple:
    """
    UNIFIED function that detects multiline calls, blocks, or single lines.
    Returns: (start_line, end_line, statement_type)
    """
    start, end, is_call = get_multiline_call_range(file_path, error_line)
    if is_call and end > start:
        return start, end, "multiline_call"
    
    start, end, block_type = get_block_range(file_path, error_line)
    if block_type:
        return start, end, block_type
    
    return error_line, error_line, None



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

LLM_MODEL_PATH = "/home/shehroz/cgcnn_edge/Qwen2.5_Coder/Qwen2.5-Coder-8B-Q4_K_M.gguf"

def create_llm(
    model_path: str = LLM_MODEL_PATH,
    n_ctx: int = 10000,
    n_gpu_layers: int = -1,
    n_threads: int = 4,
    n_batch: int = 512,
    chat_format: str = "chatml",
    verbose: bool = False,
) -> Llama:
    """Factory to create and return a Llama instance. Call this from your entrypoint
    so the GPU is initialised exactly once and then passed around."""
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        n_batch=n_batch,
        chat_format=chat_format,
        verbose=verbose,
    )

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
    
    # Check 2: File exists
    if not os.path.exists(file_path):
        # Allow creating new files in working directory
        return True, "OK - new file", True
    
    # Check 3: Python syntax validation (for .py files)
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
  {"action_type":"ReadFileInput","action":{"file_path":"<path>","error_line":12}}

- ExecuteTerminalCommand:
  {"action_type":"ExecuteTerminalCommand","action":{"command":"pip install numpy"}}

- WriteFileInput:
  {"action_type":"WriteFileInput","action":{
    "file_path":"<path>",
    "start_line": 10,
    "end_line": 12,
    "new_content":"line1\\nline2\\nline3\\n",
    "create_backup": true
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
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def allowed_actions(state: Dict[str, Any]) -> List[str]:
    """Simple phase-based guardrail policy."""

    phase = state.get("phase", "START")
    if phase == "START":
        return ["RunScriptInput"]
    if phase == "RAN":
        return (
            ["NoAction"]
            if state.get("last_exitcode") == 0
            else ["ReadFileInput", "ExecuteTerminalCommand"]
        )
    if phase == "READ":
        return ["WriteFileInput", "ReadFileInput"]
    if phase == "TERMINAL_COMMAND":
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
    return "\n".join([p for p in parts if p])


import re
import json

def extract_json(text: str) -> str:
    """Extract JSON from LLM response with fallback handling."""
    text = text.strip()
    
    # Try to find JSON object
    start = text.find("{")
    if start == -1:
        # No JSON found - try to infer action from text
        return _infer_action_from_text(text)
    
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
    return _infer_action_from_text(text)


def _infer_action_from_text(text: str) -> str:
    """Fallback: infer action type from text when JSON parsing fails."""
    text_lower = text.lower()
    
    # Try to detect intended action
    if "runscript" in text_lower or "run the script" in text_lower or "execute" in text_lower:
        print("   ⚠️ Inferred RunScriptInput from malformed response")
        return '{"action_type": "RunScriptInput", "action": {"script_path": "train_alignn.py"}}'
    
    if "readfile" in text_lower or "read the file" in text_lower or "look at" in text_lower:
        # Try to extract a file path
        file_match = re.search(r'["\']?([a-zA-Z0-9_/]+\.py)["\']?', text)
        file_path = file_match.group(1) if file_match else "train_alignn.py"
        print(f"   ⚠️ Inferred ReadFileInput for {file_path} from malformed response")
        return f'{{"action_type": "ReadFileInput", "action": {{"file_path": "{file_path}"}}}}'
    
    if "writefile" in text_lower or "write" in text_lower or "fix" in text_lower:
        print("   ⚠️ Inferred ReadFileInput (need to read before write) from malformed response")
        return '{"action_type": "ReadFileInput", "action": {"file_path": "train_alignn.py"}}'
    
    if "pip install" in text_lower or "terminal" in text_lower:
        # Try to extract package name
        pip_match = re.search(r'pip install\s+([a-zA-Z0-9_-]+)', text)
        package = pip_match.group(1) if pip_match else "unknown"
        print(f"   ⚠️ Inferred terminal command: pip install {package}")
        return f'{{"action_type": "ExecuteTerminalCommand", "action": {{"command": "pip install {package}"}}}}'
    
    if "noaction" in text_lower or "finish" in text_lower or "done" in text_lower:
        print("   ⚠️ Inferred NoAction from malformed response")
        return '{"action_type": "NoAction", "action": {}}'
    
    # Default fallback: run script again
    print("   ⚠️ Could not parse LLM response, defaulting to RunScriptInput")
    return '{"action_type": "RunScriptInput", "action": {"script_path": "train_alignn.py"}}'





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

def ask_action(
    llm_obj: Llama,
    goal: str, 
    chat_history: List[Dict[str, str]], 
    state: Dict[str, Any]
) -> AgentAction:
    """Ask the LLM which tool action to take next."""
    global RECENT_ACTIONS

    allowed = allowed_actions(state)
    state_summary = build_state_summary(state)
    print(f"   🔍 DEBUG: Sending {len(chat_history)} messages to LLM")
    print(f"   🔍 DEBUG: Last user message: {chat_history[-1]['content'][:200]}...")
    system = f"""You are an automated Python debugging agent.

GOAL:
{goal}

The main file is: {state.get("target_file","<unknown>")}

HARD CONSTRAINTS:
- You MUST choose an action whose action_type is in ALLOWED_ACTIONS.
- Output MUST be a single JSON object and MUST match the exact keys shown below.
- Do NOT use null for writing Python Code.
- Return ONLY JSON. No markdown. No extra text.

ALLOWED_ACTIONS:
{allowed}

{ACTION_SCHEMAS_TEXT}

STATE SUMMARY:
{state_summary}
"""

    base_messages = (
        [{"role": "system", "content": system}]
        + chat_history
        + [{"role": "user", "content": state["current_input"]}]
    )

    def attempt(messages: List[Dict[str, str]]) -> AgentAction:
        raw = call_llama_for_action(llm_obj, messages)
        print(f"   🔍 DEBUG: Raw LLM response: {raw[:500]}...")
        js = extract_json(raw)
        js = sanitize_jsonish(js)
        js = js.replace("\\'", "'")
        obj = json.loads(js)
        return AGENT_ACTION_ADAPTER.validate_python(obj)

    try:
        action = attempt(base_messages)
    except ValidationError:
        correction = (
            "Your JSON did not match the required schema keys. "
            f"ALLOWED_ACTIONS={allowed}. "
            "Return ONLY JSON that exactly matches one of the ACTION SCHEMAS."
        )
        action = attempt(base_messages + [{"role": "user", "content": correction}])

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
    
    RECENT_ACTIONS.append(action.model_dump())
    if len(RECENT_ACTIONS) > 10:
        RECENT_ACTIONS.pop(0)
    # === END LOOP DETECTION ===

    return action





from system_analysis import set_llm_instance


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
    print("INTERACTIVE MOCK SELECTION")
    print("=" * 80)
    print(f"Entry file: {target_file}")
    print(f"Total expensive functions detected: {len(expensive_functions)}")
    
    # Show by category
    print("\n📊 Functions by Category:")
    for cat_name, cat_funcs in categories.items():
        if cat_funcs:
            print(f"\n  {cat_name.upper()}: {len(cat_funcs)} functions")
            for i, func in enumerate(cat_funcs, 1):
                print(f"    {i}. {func['file']}::{func['function']} (score={func['score']})")
                if func.get('expensive_ops'):
                    print(f"       Ops: {', '.join(func['expensive_ops'])}")
                if func.get('side_effects'):
                    print(f"       Effects: {', '.join(func['side_effects'])}")
    
    # Show current selection
    print("\n" + "=" * 80)
    print("CURRENT AUTO-SELECTED FUNCTIONS TO MOCK:")
    print("=" * 80)
    
    if not expensive_functions:
        print("   (none selected)")
        print("\n⚠️  No functions auto-selected for mocking!")
        
        # Ask if user wants to manually add any
        while True:
            response = input("\nWould you like to manually add functions to mock? (yes/no): ").strip().lower()
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
    print("\n" + "=" * 80)
    print("OPTIONS:")
    print("  1. Accept all (proceed with these functions)")
    print("  2. Customize selection (choose which to keep/remove)")
    print("  3. Add more functions")
    print("  4. Skip mocking entirely (use full_run strategy)")
    print("=" * 80)
    
    while True:
        choice = input("\nYour choice (1-4): ").strip()
        
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
            print("   Invalid choice. Please enter 1, 2, 3, or 4.")


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
        selection = input("\nKeep functions: ").strip().lower()
        
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
        selection = input("Add: ").strip().lower()
        
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
        selection = input("Mock: ").strip().lower()
        
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
        error_match = re.match(r'^(\w+Error|\w+Exception|AssertionError): (.+)$', line)
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
        if re.match(r'^\w+Error:', line) or re.match(r'^\w+Exception:', line):
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



def run_llm_debug_agent(
    target_file: str,
    args: List[str] = None,
    max_steps: int = 15,
    llm: Llama = None,
    strategy: str = "mocking"
) -> Dict[str, Any]:
    """
    Unified LLM-driven debugging agent with enhanced state tracking.
    """
    if args is None:
        args = []
    if llm is None:
        llm = globals()["llm"]
    set_llm_instance(llm)
    
    print("\\n" + "=" * 80)
    print("UNIFIED LLM DEBUGGING AGENT")
    print("=" * 80)
    
    # Step 1: Analyze and decide strategy
    print("\\n[Step 1] Analyzing codebase...")
    analysis_results, flows = autonomous_analysis(llm_instance=llm, directory=".")
    strategy, categories = _decide_strategy(analysis_results)


    
    target_flow = flows.get(target_file, {})
    print(f"\\n🔍 DEBUG: Functions in flow from {target_file}:")
    depth_funcs = {}
    for func_name, info in target_flow.items():
        depth = info.get('depth', -1)
        if depth not in depth_funcs:
            depth_funcs[depth] = []
        depth_funcs[depth].append(func_name)
    
    for depth in sorted(depth_funcs.keys()):
        print(f"   Depth {depth}: {depth_funcs[depth][:10]}")
    
    # Get expensive functions
    expensive_functions = []
    if strategy == "mocking":
        for key in ["critical_ml", "compute_heavy", "io_ops"]:
            expensive_functions.extend(categories.get(key, []))
        expensive_functions = _filter_to_entry_flow(target_file, flows, expensive_functions)
    entry_flow = flows.get(target_file, {})
    expensive_functions = _promote_helpers_to_callers(expensive_functions, entry_flow)

    print("\\n🔍 DEBUG: All expensive functions from analysis:")
    for cat_name, cat_funcs in categories.items():
        if cat_funcs:
            print(f"\\n  Category: {cat_name}")
            for func in cat_funcs[:5]:
                print(f"    - {func['file']}::{func['function']} (score={func['score']})")
    
    # Interactive selection
    expensive_functions = interactive_mock_selection(
        expensive_functions,
        categories,
        target_file
    )
    
    if not expensive_functions:
        strategy = "full_run"
        print("\\n   📌 Strategy changed to: FULL_RUN (no mocking)")
    else:
        print(f"\\n   📌 Final selection: {len(expensive_functions)} functions to mock")
        for func in expensive_functions:
            print(f"      - {func['file']}::{func['function']}")
    
    print(f"\\n   Strategy: {strategy.upper()}")
    if expensive_functions:
        print(f"   Expensive functions: {len(expensive_functions)}", expensive_functions)
    
    # Step 2: Apply minimal inputs
    config_changes = {}
    mocked_functions = []
    run_args = args.copy()
    
    if strategy == "mocking":
        print("\\n[Step 2] Applying mocking strategy setup...")
        entry_only_flows = {target_file: flows.get(target_file, {})}
        config_changes = apply_minimal_inputs(
            target_file, 
            expensive_functions, 
            entry_only_flows, 
            llm,
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
    }
    chat_history: List[Dict[str, str]] = [{"role": "user", "content": goal}]
    
    strategy_context = ""
    if strategy == "mocking":
        strategy_context = (
            f"\\nMOCKING STRATEGY: Minimal inputs applied. "
            f"Expensive functions detected: {len(expensive_functions)}. "
            f"If errors occur AFTER expensive functions execute, consider mocking them."
        )
    
    # Step 4: Main loop
    for step in range(max_steps):
        print(f"\\n[Step {step+1}/{max_steps}] Phase: {state['phase']}, Mocking: {state.get('mocking_phase', 'N/A')}")
        
        action = ask_action(llm, goal + strategy_context, chat_history, state)
        chat_history.append({
            "role": "assistant",
            "content": json.dumps(action.model_dump(), ensure_ascii=False),
        })

        # =====================================================================
        # RunScriptInput
        # =====================================================================
        if action.action_type == "RunScriptInput":
            act = action.action
            args_to_use = state["run_args"] if state["run_args"] else act.args

            if state["strategy"] == "mocking" and state["use_mocks"]:
                print("   🔨 Using mocked functions for this run")
                from system_agent import MOCK_CACHE
                print(f"   Mock cache size: {len(MOCK_CACHE)} entries")

            out = run_script.invoke({"script_path": act.script_path, "args": args_to_use})
            state["last_stdout"] = out.get("stdout", "")
            state["last_stderr"] = out.get("stderror", "")
            state["last_exitcode"] = out.get("exitcode", None)
            state["phase"] = "RAN"
            
            if state["strategy"] == "mocking":
                state["mocking_phase"] = "running"

            # === SUCCESS PATH ===
            if state["last_exitcode"] == 0:
                print("   ✅ Script completed successfully!")
                
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
                error_analysis = error_analyzer(state["last_stderr"])
                state["error_line"] = error_analysis.get("error_line", 0) or 0
                error_info = extract_error_info(state["last_stderr"])

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
                    out = run_script.invoke({
                        "script_path": state["target_file"], 
                        "args": state["run_args"]
                    })
                    state["last_stdout"] = out.get("stdout", "")
                    state["last_stderr"] = out.get("stderror", "")
                    state["last_exitcode"] = out.get("exitcode", None)

                    if state["last_exitcode"] == 0:
                        print("   ✅ Script passes with mock!")
                        state["current_input"] = "Script completed successfully. Output NoAction if done."
                        chat_history.append({"role": "user", "content": state["current_input"]})
                        continue
                    
                    error_info = extract_error_info(state["last_stderr"])
                    state["error_line"] = error_info["line"]

                    error_line_content = ""
                    try:
                        with open(error_info["file"], "r") as f:
                            lines = f.readlines()
                        if 0 < error_info["line"] <= len(lines):
                            error_line_content = lines[error_info["line"] - 1].strip()
                    except:
                        error_line_content = "unknown"
                    

                    # === Check if this is a self-inflicted wrapper error ===
                    if error_info["file"] in [f["file"] for f in state.get("mocked_functions", [])]:
                        if "IndentationError" in error_info["error_type"] or "SyntaxError" in error_info["error_type"]:
                            print(f"   ⚠️ SELF-INFLICTED ERROR: Wrapper injection failed in {error_info['file']}")
                            print(f"   🔄 Attempting to restore from backup and retry...")

                            # Restore from backup
                            backup_path = f"{error_info['file']}.backup"
                            if os.path.exists(backup_path):
                                import shutil
                                shutil.copy(backup_path, error_info['file'])
                                print(f"   ✅ Restored {error_info['file']} from backup")

                                # Unprotect the file
                                unprotect_file(error_info['file'])

                                # Remove from mocked_functions
                                state["mocked_functions"] = [
                                    m for m in state["mocked_functions"] 
                                    if m["file"] != error_info["file"]
                                ]
                                state["use_mocks"] = False
                                state["mocking_phase"] = "running"

                                # Rerun without the broken mock
                                state["current_input"] = "Mock injection failed. Restored file. Rerunning..."
                                chat_history.append({"role": "user", "content": state["current_input"]})
                                continue

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

                    # === FIX: Use get_statement_range instead of get_block_range ===
                    start_line, end_line, stmt_type = get_statement_range(error_info["file"], error_info["line"])

                    if stmt_type:
                        if stmt_type == "multiline_call":
                            print(f"   🔍 Detected MULTI-LINE CALL spanning lines {start_line}-{end_line}")
                            state["current_input"] = (
                                f"Mock active. New error: {error_info['error_type']}: {error_info['message']}\\n"
                                f"File: {error_info['file']}, Line: {error_info['line']}\\n"
                                f"Problematic code: `{error_line_content}`\\n\\n"
                                f"⚠️ CRITICAL: This is a MULTI-LINE FUNCTION CALL spanning lines {start_line}-{end_line}.\\n"
                                f"You MUST comment out ALL lines from {start_line} to {end_line}.\\n"
                                f"Use: start_line={start_line}, end_line={end_line}\\n"
                                f"Add '# ' to the start of EVERY line."
                            )
                        else:
                            print(f"   🔍 Detected '{stmt_type}' block spanning lines {start_line}-{end_line}")
                            state["current_input"] = (
                                f"Mock active. New error: {error_info['error_type']}: {error_info['message']}\\n"
                                f"File: {error_info['file']}, Line: {error_info['line']}\\n"
                                f"Problematic code: `{error_line_content}`\\n"
                                f"This is a '{stmt_type}' block that spans lines {start_line}-{end_line}.\\n"
                                f"FIX: Comment out the ENTIRE block (start_line={start_line}, end_line={end_line})."
                            )
                    else:
                        state["current_input"] = (
                            f"Mock active. New error: {error_info['error_type']}: {error_info['message']}\\n"
                            f"File: {error_info['file']}, Line: {error_info['line']}\\n"
                            f"Problematic code: `{error_line_content}`\\n"
                            f"FIX: Comment out or fix this single line."
                        )
                    chat_history.append({"role": "user", "content": state["current_input"]})
                    continue
                
                # ===== CASE 2: Module/import error =====
                if error_analysis.get("module_error") == "Lib":
                    missing = error_analysis.get("missing_module", "unknown")
                    print(f"   📦 Missing library: {missing}")
                    state["current_input"] = f"Missing library: {missing}. Run: pip install {missing}"
                    chat_history.append({"role": "user", "content": state["current_input"]})
                    continue
                
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
                            f"⚠️ CRITICAL: This is a MULTI-LINE FUNCTION CALL spanning lines {start_line}-{end_line}.\\n"
                            f"You MUST comment out ALL lines from {start_line} to {end_line}.\\n"
                            f"Use: start_line={start_line}, end_line={end_line}\\n"
                            f"Add '# ' to the start of EVERY line. Commenting only one line causes IndentationError!"
                        )
                    else:
                        print(f"   🔍 Detected '{stmt_type}' block spanning lines {start_line}-{end_line}")
                        state["current_input"] = (
                            f"Error: {error_info['error_type']}: {error_info['message']}\\n"
                            f"File: {error_info['file']}, Line: {error_info['line']}\\n"
                            f"Problematic code: `{error_line_content}`\\n"
                            f"This is a '{stmt_type}' block spanning lines {start_line}-{end_line}.\\n"
                            f"FIX: Comment out the ENTIRE block (start_line={start_line}, end_line={end_line})."
                        )
                else:
                    state["current_input"] = (
                        f"Error: {error_info['error_type']}: {error_info['message']}\\n"
                        f"File: {error_info['file']}, Line: {error_info['line']}\\n"
                        f"Problematic code: `{error_line_content}`\\n"
                        f"FIX: Comment out or fix this single line."
                    )
                chat_history.append({"role": "user", "content": state["current_input"]})
                continue

        # =====================================================================
        # ExecuteTerminalCommand
        # =====================================================================
        if action.action_type == "ExecuteTerminalCommand":
            act = action.action
            result = terminal_command_executor.invoke({"command": act.command})
            state["phase"] = "TERMINAL_COMMAND"
            
            if result.get("exit_code") == 0:
                state["current_input"] = "Command succeeded. Now rerun the script."
            else:
                state["current_input"] = f"Command failed: {result.get('output', '')[:500]}"
            
            chat_history.append({"role": "user", "content": f"TERMINAL: {result}"})
            continue

        # =====================================================================
        # ReadFileInput
        # =====================================================================
        if action.action_type == "ReadFileInput":
            act = action.action
            abs_path = os.path.abspath(act.file_path)
            
            is_library = "site-packages" in abs_path or "anaconda3" in abs_path
            if is_library:
                print(f"   ⚠️ Reading library file: {act.file_path}")
            
            file_text_res = read_file.invoke({
                "file_path": act.file_path,
                "file_type": act.file_type,
                "error_line": state.get("error_line", 0) or 0,
            })
            file_text = file_text_res.get("content", "")
            state["last_file_text"] = file_text
            state["phase"] = "READ"
            
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
                
                state["current_input"] = f"FILE_SNIPPET({act.file_path}):{mock_hint}\\n{file_text[:1500]}"
            else:
                state["current_input"] = f"FILE_SNIPPET({act.file_path}):\\n{file_text[:1500]}"
            
            chat_history.append({"role": "user", "content": state["current_input"]})
            continue

        # =====================================================================
        # WriteFileInput
        # =====================================================================
        if action.action_type == "WriteFileInput":
            act = action.action
            
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
            
            # Validate write safety
            is_safe, reason, should_continue = validate_write_safety(
                file_path=act.file_path,
                new_content=act.new_content,
                start_line=act.start_line,
                end_line=act.end_line,
                state=state,
            )
            
            if not is_safe:
                print(f"   {reason}")
                state["current_input"] = f"Write blocked: {reason}. Please fix and try again."
                chat_history.append({"role": "user", "content": f"BLOCKED: {reason}"})
                continue

            # Process escaped characters FIRST
            raw_content = act.new_content
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
            
            # Preserve original indentation if replacing a single line
            if act.start_line is not None and act.start_line == act.end_line:
                new_content_stripped = final_new_content.lstrip()
                new_indent = len(final_new_content) - len(new_content_stripped)
                
                if new_indent < original_indent:
                    indent_to_add = " " * (original_indent - new_indent)
                    final_new_content = indent_to_add + final_new_content
                
            write_res = write_file.invoke({
                "file_path": act.file_path,
                "new_content": final_new_content,
                "start_line": act.start_line,
                "end_line": act.end_line,
                "create_backup": True,
            })
            
            state["phase"] = "WROTE"
            syntaxcheck = check_syntax(act.file_path)
            
            # Improved syntax error handling
            if syntaxcheck and ("SyntaxError" in syntaxcheck or "IndentationError" in syntaxcheck):
                print(f"   ⚠️ Syntax error after write - reverting")
                
                backup_path = f"{act.file_path}.backup"
                if os.path.exists(backup_path):
                    import shutil
                    shutil.copy(backup_path, act.file_path)
                
                guidance = ""
                if "unexpected indent" in syntaxcheck.lower():
                    original_error_line = state.get("error_line", 0)
                    if original_error_line > 0:
                        start, end, stmt_type = get_statement_range(act.file_path, original_error_line)
                        
                        if stmt_type == "multiline_call":
                            guidance = (
                                f"\n\n⚠️ CRITICAL: 'unexpected indent' means you only commented PART of a "
                                f"multi-line function call. The call spans lines {start}-{end}. "
                                f"You MUST use start_line={start}, end_line={end} to comment ALL lines."
                            )
                        elif stmt_type:
                            guidance = (
                                f"\n\nHINT: 'unexpected indent' - you only commented part of a "
                                f"'{stmt_type}' block spanning lines {start}-{end}. Comment ALL lines."
                            )
                        else:
                            guidance = (
                                "\n\nHINT: 'unexpected indent' means you broke a multi-line statement. "
                                "Comment out ALL continuation lines."
                            )
                    else:
                        guidance = "\n\nHINT: 'unexpected indent' usually means you commented only part of a multi-line statement."
                
                elif "invalid syntax" in syntaxcheck.lower():
                    guidance = "\n\nHINT: Check that parentheses, brackets, and quotes are balanced."
                
                state["current_input"] = f"WRITE REVERTED - Syntax Error: {syntaxcheck}{guidance}"
                chat_history.append({"role": "user", "content": state["current_input"]})
                continue
            else:
                state["current_input"] = "File written. Rerun to verify."
            
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
            return {
                "status": "stopped",
                "reason": "NoAction",
                "state": state,
                "history_len": len(chat_history),
                "strategy": state["strategy"],
                "mocked_functions": state["mocked_functions"],
                "config_changes": state["config_changes"],
            }

    return {
        "status": "max_steps_reached",
        "state": state,
        "history_len": len(chat_history),
        "strategy": state["strategy"],
        "mocked_functions": state["mocked_functions"],
        "config_changes": state["config_changes"],
    }



