"""
High-level agent that combines:
- runtime debugging tools (from system_tools)
- static analysis + expense detection (from system_analysis)

It decides whether to:
- run full end-to-end debugging, or
- apply a mocking strategy for expensive operations with minimal inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json
import os
from pathlib import Path

from matplotlib import lines
from matplotlib.pylab import rint
from soupsieve import match

from system_tools import (
    run_script,
    read_file,
    write_file,
    error_analyzer,
    list_files,
    check_syntax,
)
from system_analysis import (
    autonomous_analysis,
    analyze_and_rank_expensive_functions,
    categorize_functions_for_mocking,
    discover_entry_points,
)
from system_minimal_inputs import (
    MinimalInputExtractor,
    MinimalInputGenerator,
    generate_complete_minimal_inputs_v2,
)


def _load_minimal_config_candidates() -> List[str]:
    """Return existing minimal-config JSON files in the workspace."""

    candidates = [
        "minimal_inputs_complete_v2.json",
        "minimal_inputs_complete.json",
        "minimal_inputs_summary.json",
        "minimal_input_specs.json",
    ]
    return [c for c in candidates if os.path.exists(c)]

import functools
import pickle
import sys
from typing import Any, Dict, Callable
from pathlib import Path

INSTRUMENTED_FUNCTIONS: Dict[str, Callable] = {}
FUNCTION_OUTPUTS: Dict[str, Any] = {}

def instrument_function(file_path: str, func_name: str) -> bool:
    """
    Inject instrumentation wrapper around a function to record its output.
    
    Returns True if successful, False otherwise.
    """
    try:
        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Create instrumentation wrapper code
        instrumentation_code = f'''
# === INSTRUMENTATION INJECTED BY DEBUGGER ===
import functools
from system_agent import record_function_output

_original_{func_name} = {func_name}

@functools.wraps(_original_{func_name})
def {func_name}(*args, **kwargs):
    """Instrumented wrapper for {func_name}"""
    result = _original_{func_name}(*args, **kwargs)
    record_function_output("{file_path}::{func_name}", result)
    return result
# === END INSTRUMENTATION ===

'''
        
        # Find where to inject (before the function definition)
        import ast
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                # Get the line number
                func_line = node.lineno - 1  # 0-indexed
                
                # Split content into lines
                lines = content.split("\n")
                
                # Insert instrumentation before function
                lines.insert(func_line, instrumentation_code)
                
                # Write back
                new_content = "\n".join(lines)
                
                # Backup original
                backup_path = f"{file_path}.instrumented_backup"
                if not Path(backup_path).exists():
                    with open(backup_path, "w", encoding="utf-8") as f:
                        f.write(content)
                
                # Write instrumented version
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                
                print(f"   🔬 Instrumented {func_name}() in {file_path}")
                return True
        
        return False
    
    except Exception as e:
        print(f"   ⚠️ Instrumentation failed: {e}")
        return False


def record_function_output(func_key: str, output: Any):
    """
    Called by instrumented functions to record their output.
    Stores output in FUNCTION_OUTPUTS for later use in mocking.
    """
    FUNCTION_OUTPUTS[func_key] = output
    
    # Also save to disk for persistence
    output_file = "function_outputs.pkl"
    
    existing_outputs = {}
    if Path(output_file).exists():
        try:
            with open(output_file, "rb") as f:
                existing_outputs = pickle.load(f)
        except:
            pass
    
    existing_outputs[func_key] = output
    
    with open(output_file, "wb") as f:
        pickle.dump(existing_outputs, f)
    
    print(f"   📊 Recorded output for {func_key} (type: {type(output).__name__})")


def remove_instrumentation(file_path: str) -> bool:
    """
    Remove instrumentation and restore original file.
    """
    backup_path = f"{file_path}.instrumented_backup"
    
    if not Path(backup_path).exists():
        return False
    
    # Restore from backup
    with open(backup_path, "r", encoding="utf-8") as f:
        original_content = f.read()
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(original_content)
    
    # Remove backup
    Path(backup_path).unlink()
    
    print(f"   🔄 Removed instrumentation from {file_path}")
    return True


def load_recorded_outputs() -> Dict[str, Any]:
    """
    Load previously recorded function outputs from disk.
    """
    output_file = "function_outputs.pkl"
    
    if not Path(output_file).exists():
        return {}
    
    try:
        with open(output_file, "rb") as f:
            return pickle.load(f)
    except:
        return {}


def _detect_primary_entry() -> str:
    """Choose a primary entry script (prefer certain entry points, else a common training script)."""

    entry_points = discover_entry_points(".")
    if entry_points["certain"]:
        return entry_points["certain"][0]
    if entry_points["likely"]:
        return entry_points["likely"][0]["file"]
    # Fallbacks
    for name in [
        "train_alignn.py",
        "train_dgl_minimal_test.py",
        "train.py",
        "run_alignn_ff.py",
        "my_file.py",
    ]:
        if os.path.exists(name):
            return name
    # Last resort: any non-system python file
    for py in Path(".").glob("*.py"):
        if not py.name.startswith("system_"):
            return str(py)
    raise RuntimeError("No entry script found.")


def update_nested_config(config_data: Dict, path: str, value: Any) -> bool:
    """
    Update nested config value using dot-notation path.
    
    Example: update_nested_config(config, "model.hidden_dim", 8)
    """
    keys = path.split(".")
    current = config_data
    
    # Navigate to parent of target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set final value
    current[keys[-1]] = value
    return True

def _decide_strategy(
    analysis_results: Dict[str, List[Dict[str, Any]]]
) -> Tuple[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Decide between:
    - 'full_run': no significant expensive operations found
    - 'mocking': expensive functions present
    """

    expensive = analyze_and_rank_expensive_functions(analysis_results)
    categories = categorize_functions_for_mocking(expensive)
    if (
        categories["critical_ml"]
        or categories["compute_heavy"]
        or categories["io_ops"]
    ):
        return "mocking", categories
    return "full_run", categories


def _map_minimal_configs_to_targets() -> Dict[str, str]:
    """
    Heuristic mapping from known config files to minimal variants.
    Returned dict: {live_config_path: minimal_source_path}
    """

    minimal_files = _load_minimal_config_candidates()
    mapping: Dict[str, str] = {}
    if not minimal_files:
        return mapping

    # Simple heuristics: prefer v2 as global default
    default_minimal = None
    for c in minimal_files:
        if "v2" in c:
            default_minimal = c
            break
    if not default_minimal:
        default_minimal = minimal_files[0]

    # Common config targets in this repo
    for target in ["config.json", "config_bandgap.json", "config_eform.json"]:
        if os.path.exists(target):
            mapping[target] = default_minimal

    return mapping



def _build_reverse_callmap(flow: dict) -> dict[str, set[str]]:
    """
    flow: {func_name: {"calls": [...], ...}, ...}
    returns: {callee: {caller1, caller2, ...}}
    """
    rev: dict[str, set[str]] = {}
    for caller, info in (flow or {}).items():
        for callee in (info.get("calls") or []):
            rev.setdefault(callee, set()).add(caller)
    return rev


def _promote_helpers_to_callers(funcs: list[dict], entry_flow: dict) -> list[dict]:
    """
    Promote helpers to callers, but only if they're somewhat expensive.
    """
    rev = _build_reverse_callmap(entry_flow)

    promoted: list[dict] = []
    seen = set()

    for f in funcs:
        name = f.get("function", "")
        score = f.get("score", 0)

        # Check if this is a helper (starts with underscore)
        is_underscore_helper = name.startswith("_")
        
        # FIX: Only skip UNDERSCORE helpers with low scores
        # Don't skip regular functions even if they have low scores
        if is_underscore_helper and score < 5:
            # Skip trivial underscore helpers
            continue
        
        if is_underscore_helper and score >= 5:
            # This underscore helper is expensive enough to promote
            callers = list(rev.get(name, []))
            callers.sort(key=lambda x: (x.startswith("_"), x))
            if callers:
                promoted_name = callers[0]
                caller_info = entry_flow.get(promoted_name, {}) or {}
                caller_file = caller_info.get("file") or f.get("file")

                promoted_func = {
                    **f,
                    "function": promoted_name,
                    "file": caller_file,
                    "promoted_from": name,
                    "score": max(score + 5, 5),
                    "reasons": (f.get("reasons") or []) + [f"promoted_from({name})"],
                }

                key = (promoted_func["file"], promoted_func["function"])
                if key not in seen:
                    seen.add(key)
                    promoted.append(promoted_func)
                continue

        # Keep original (non-underscore-helper OR high-score underscore-helper)
        key = (f.get("file"), name)
        if key not in seen:
            seen.add(key)
            promoted.append(f)

    promoted.sort(key=lambda x: x.get("score", 0), reverse=True)
    return promoted




def apply_minimal_inputs(
    entry_file: str,
    expensive_functions: List[Dict[str, Any]] = None,
    flow_graphs: Dict[str, Any] = None,
    llm=None,
    original_args: List[str] = None,
) -> Dict[str, Any]:
    """
    Apply minimal input strategy with proper config injection.
    
    This function:
    1. Generates minimal specs for expensive functions
    2. Extracts minimal values from the analysis
    3. Injects them into BOTH config files AND CLI args
    """
    changes: Dict[str, Any] = {"config_swaps": [], "entry_args": [], "minimal_specs": {}}
    
    if expensive_functions and flow_graphs and llm:
        print("\n🔧 Generating minimal inputs for expensive functions...")
        minimal_specs = generate_complete_minimal_inputs_v2(
            expensive_functions, flow_graphs, llm, "."
        )
        changes["minimal_specs"] = minimal_specs
        
        # DEBUG: Print what we got
        print(f"\n   🔍 DEBUG: minimal_specs keys: {list(minimal_specs.keys())}")
        
        # ===== EXTRACT MINIMAL VALUES FROM all_resolved =====
        # ===== EXTRACT MINIMAL VALUES FROM all_resolved =====
        extracted_minimal = {}
        
        for func_key, spec in minimal_specs.items():
            print(f"\n   🔍 Processing spec for {func_key}")
            
            all_resolved = spec.get("all_resolved", {})
            
            # Extract from critical_minimal (it's a LIST of dicts, not a dict!)
            critical_minimal = all_resolved.get("critical_minimal", [])
            if isinstance(critical_minimal, list):
                for item in critical_minimal:
                    if isinstance(item, dict) and "param" in item:
                        param = item["param"]
                        value = item.get("minimal_value")
                        if value is not None:
                            extracted_minimal[param] = value
                            print(f"      Found critical_minimal: {param} = {value}")
            elif isinstance(critical_minimal, dict):
                # Fallback if it's a dict
                for param, value in critical_minimal.items():
                    extracted_minimal[param] = value
                    print(f"      Found critical_minimal: {param} = {value}")
            
            # Extract from config_attributes (this is a dict)
            config_attrs = all_resolved.get("config_attributes", {})
            if isinstance(config_attrs, dict):
                for attr, info in config_attrs.items():
                    if isinstance(info, dict) and info.get("minimal_value") is not None:
                        # Handle nested attributes like "config.epochs"
                        attr_name = attr.split(".")[-1] if "." in attr else attr
                        extracted_minimal[attr_name] = info["minimal_value"]
                        print(f"      Found config_attr: {attr_name} = {info['minimal_value']} (was {info.get('default')})")
        
        print(f"\n   📋 Extracted minimal values: {extracted_minimal}")
        
        # ===== FIND AND UPDATE CONFIG FILE =====
        config_file = None
        if original_args:
            for i, arg in enumerate(original_args):
                if arg in ["--config_name", "--config", "-c"] and i + 1 < len(original_args):
                    config_file = original_args[i + 1]
                    break
        
        # In apply_minimal_inputs, replace the config update section with this:

        # ===== SAFE PARAMETERS TO MINIMIZE =====
        # Only these parameters should be changed - they control computation time
        # without breaking the model configuration
        SAFE_MINIMAL_OVERRIDES = {
            # These control data size - SAFE to reduce
            "n_train": 100,
            "n_val": 20,
            "n_test": 20,
            "epochs": 1,
            "batch_size": 2,

            # These control computation - SAFE to reduce
            "num_workers": 0,
            "max_neighbors": 8,  # Don't go too low or model breaks

            # These are safe boolean/simple changes
            "pin_memory": False,
            "save_dataloader": False,
            "write_checkpoint": False,
            "progress": True,
        }

        # NEVER change these - they break the model
        NEVER_CHANGE = {
            "model",           # Complex nested dict
            "neighbor_strategy",  # Must be specific string
            "dataset",         # User's choice
            "target",          # User's choice  
            "output_dir",      # User's choice
            "atom_features",   # Model architecture
            "dtype",           # Data type
            "criterion",       # Loss function
            "optimizer",       # Optimizer type
            "scheduler",       # LR scheduler
        }

        config_path = config_file
        if config_file and not os.path.exists(config_path):
            # Try in the entry file's directory
            project_dir = os.path.dirname(os.path.abspath(entry_file))
            config_path = os.path.join(project_dir, config_file)
        
        if config_path and os.path.exists(config_path):
            print(f"   📄 Found config file: {config_path}")
            print(f"\n   📄 Found config file: {config_file}")

            try:
                with open(config_file, "r") as f:
                    config_data = json.load(f)

                # Backup (only once)
                backup_path = f"{config_file}.minimal_backup"
                if not os.path.exists(backup_path):
                    with open(backup_path, "w") as f:
                        json.dump(config_data, f, indent=2)
                    print(f"   💾 Backed up to {backup_path}")

                # ONLY apply safe minimal overrides
                applied_count = 0
                for key, value in SAFE_MINIMAL_OVERRIDES.items():
                    if key in config_data and key not in NEVER_CHANGE:
                        old_val = config_data[key]
                        # Only change if it would actually reduce computation
                        if isinstance(old_val, (int, float)) and isinstance(value, (int, float)):
                            if value < old_val:  # Only reduce, never increase
                                config_data[key] = value
                                print(f"   🔧 Config: {key}: {old_val} → {value}")
                                applied_count += 1
                        elif isinstance(old_val, bool) and isinstance(value, bool):
                            config_data[key] = value
                            print(f"   🔧 Config: {key}: {old_val} → {value}")
                            applied_count += 1

                # Write back
                if applied_count > 0:
                    with open(config_file, "w") as f:
                        json.dump(config_data, f, indent=2)
                    print(f"   ✅ Applied {applied_count} SAFE config changes")

            except Exception as e:
                print(f"   ❌ Failed to update config: {e}")
        else:
            print(f"\n   ⚠️ No config file found in args or file doesn't exist: {config_file}")
        
        # ===== ALSO APPLY HARDCODED FALLBACK MINIMUMS =====
        # These are safe minimal values for common training parameters
        fallback_minimums = {
            "epochs": 1,
            "n_train": 100,
            "n_val": 20,
            "n_test": 20,
            "batch_size": 2,
        }
        
        # Merge extracted with fallbacks (extracted takes priority)
        for key, value in fallback_minimums.items():
            if key not in extracted_minimal:
                extracted_minimal[key] = value
    
    # ===== OVERRIDE CLI ARGS =====
    if original_args:
        # Parse original args into dict
        original_dict = {}
        i = 0
        while i < len(original_args):
            if original_args[i].startswith("--"):
                key = original_args[i][2:]  # Remove --
                if i + 1 < len(original_args) and not original_args[i + 1].startswith("--"):
                    original_dict[key] = original_args[i + 1]
                    i += 2
                else:
                    original_dict[key] = True
                    i += 1
            else:
                i += 1
        
        # Valid CLI args for this script (from earlier error message)
        valid_cli_args = {
            "root_dir", "config_name", "file_format", "classification_threshold",
            "batch_size", "epochs", "subspace_method", "id_dim", "id_ortho",
            "subspace_full_rotation", "target_key", "id_key", "force_key",
            "atomwise_key", "stresswise_key", "additional_output_key",
            "output_dir", "restart_model_path", "device", "id_enable"
        }
        
        # Apply minimal overrides (only for valid CLI args that exist)
        for key, value in extracted_minimal.items():
            if key in original_dict and key in valid_cli_args:
                print(f"   🔧 CLI Override: --{key}: {original_dict[key]} → {value}")
                original_dict[key] = str(value)
        
        # Reconstruct args list
        merged_args = []
        for key, value in original_dict.items():
            merged_args.append(f"--{key}")
            if value is not True:
                merged_args.append(str(value))
        
        changes["entry_args"] = merged_args
    
    return changes

# In system_agent.py

def _build_full_mock_wrapper(func_name: str, cache_dir: str, indent: int = 0) -> str:
    """
    Build a wrapper that:
    1. First run: Records (args, kwargs) -> return_value mapping
    2. Subsequent runs: Looks up return value based on args
    
    This handles functions called multiple times with different arguments.
    """
    ind = " " * indent
    b = " " * (indent + 4)
    b2 = " " * (indent + 8)
    
    return f'''{ind}def {func_name}(*args, **kwargs):
{b}import os
{b}import pickle
{b}import hashlib
{b}
{b}cache_dir = '{cache_dir}'
{b}
{b}# Create a hash of the arguments
{b}def _hash_args(args, kwargs):
{b2}try:
{b2}    # Try to pickle args to get a stable hash
{b2}    arg_bytes = pickle.dumps((args, sorted(kwargs.items())))
{b2}    return hashlib.md5(arg_bytes).hexdigest()[:16]
{b2}except:
{b2}    # Fallback: use repr
{b2}    return hashlib.md5(repr((args, kwargs)).encode()).hexdigest()[:16]
{b}
{b}arg_hash = _hash_args(args, kwargs)
{b}cache_file = os.path.join(cache_dir, f'{func_name}_{{arg_hash}}.pkl')
{b}
{b}# Check for cached result for these specific arguments
{b}if os.path.exists(cache_file):
{b2}print(f'[MOCK] {func_name}(hash={{arg_hash}}) - returning cached result')
{b2}with open(cache_file, 'rb') as f:
{b2}    return pickle.load(f)
{b}
{b}# No cache for these args - execute original
{b}print(f'[CAPTURE] {func_name}(hash={{arg_hash}}) - executing and caching...')
{b}result = _original_{func_name}(*args, **kwargs)
{b}
{b}# Save result
{b}os.makedirs(cache_dir, exist_ok=True)
{b}with open(cache_file, 'wb') as f:
{b2}pickle.dump(result, f)
{b}print(f'[CAPTURE] {func_name}() - cached to {{cache_file}}')
{b}
{b}return result
'''


def _build_mock_stub(func_name: str, cache_file: str, indent: int = 0) -> str:
    """
    Build a mock stub that returns cached value from disk.
    """
    ind = " " * indent
    body_ind = " " * (indent + 4)
    
    return (
        f"{ind}def {func_name}(*args, **kwargs):\n"
        f"{body_ind}import os\n"
        f"{body_ind}import pickle\n"
        f"{body_ind}cache_path = '{cache_file}'\n"
        f"{body_ind}print(f'[MOCK] {func_name}() called - returning cached result')\n"
        f"{body_ind}if os.path.exists(cache_path):\n"
        f"{body_ind}    with open(cache_path, 'rb') as f:\n"
        f"{body_ind}        return pickle.load(f)\n"
        f"{body_ind}print(f'[MOCK WARNING] No cache found at {{cache_path}}, returning None')\n"
        f"{body_ind}return None\n"
    )

# In system_agent.py
def _build_capture_wrapper(func_name: str, cache_file: str, indent: int = 0) -> str:
    """
    Build a wrapper that captures return value on first run,
    returns cached value on subsequent runs.
    
    IMPROVED: Properly handles:
    - Functions that return None (distinguishes from "no cache")
    - Functions that raise exceptions (re-raises them)
    - Pickle errors (falls back to original function)
    """
    ind = " " * indent
    b = " " * (indent + 4)
    b2 = " " * (indent + 8)
    
    return f'''{ind}def {func_name}(*args, **kwargs):
{b}import os
{b}import pickle
{b}
{b}cache_path = '{cache_file}'
{b}
{b}# === CHECK FOR CACHED RESULT ===
{b}if os.path.exists(cache_path):
{b2}try:
{b2}    with open(cache_path, 'rb') as _f:
{b2}        _cached = pickle.load(_f)
{b2}    
{b2}    print(f'[MOCK] {func_name}() - using cached result')
{b2}    
{b2}    if _cached.get("exception") is not None:
{b2}        raise _cached["exception"]
{b2}    
{b2}    return _cached.get("result")
{b2}    
{b2}except Exception as _e:
{b2}    print(f'[MOCK WARNING] Cache load failed: {{_e}}, running original')
{b}
{b}# === FIRST RUN: EXECUTE AND CACHE ===
{b}print(f'[CAPTURE] {func_name}() - first run, executing...')
{b}
{b}_cached = {{"has_result": False, "result": None, "exception": None}}
{b}
{b}try:
{b2}_result = _original_{func_name}(*args, **kwargs)
{b2}_cached["has_result"] = True
{b2}_cached["result"] = _result
{b2}print(f'[CAPTURE] {func_name}() completed (returned {{type(_result).__name__}})')
{b}
{b}except Exception as _e:
{b2}_cached["exception"] = _e
{b2}print(f'[CAPTURE] {func_name}() raised {{type(_e).__name__}}')
{b}
{b}# === SAVE TO CACHE ===
{b}try:
{b2}_cache_dir = os.path.dirname(cache_path)
{b2}if _cache_dir:
{b2}    os.makedirs(_cache_dir, exist_ok=True)
{b2}
{b2}with open(cache_path, 'wb') as _f:
{b2}    pickle.dump(_cached, _f)
{b2}print(f'[CAPTURE] Saved to {{cache_path}}')
{b}
{b}except Exception as _e:
{b2}print(f'[CAPTURE WARNING] Could not save: {{_e}}')
{b}
{b}# === RETURN OR RE-RAISE ===
{b}if _cached.get("exception") is not None:
{b2}raise _cached["exception"]
{b}
{b}return _cached.get("result")
'''

# Global cache used by generated mocks
MOCK_CACHE: Dict[str, Any] = {}

def mock_function(
    file_path: str,
    func_name: str,
    cached_value: Any,  # Not used anymore, kept for compatibility
) -> Dict[str, Any]:
    """
    Inject a capture/replay wrapper around the expensive function.
    - First run: executes original, caches return value to disk
    - Subsequent runs: returns cached value from disk
    """
    import re
    
    cache_dir = ".mock_cache"
    cache_file = f"{cache_dir}/{os.path.basename(file_path).replace('.py', '')}_{func_name}.pkl"
    
    print(f"   🔍 DEBUG mock_function: injecting wrapper for {func_name}")
    print(f"   🔍 DEBUG mock_function: cache file = {cache_file}")
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if wrapper already injected
    if f"_original_{func_name}" in content:
        print(f"   ⚠️ Wrapper already injected for {func_name}")
        return {"success": True, "already_injected": True}
    
    # Find the function definition and its indentation
    pattern = rf"^(\s*)def\s+{func_name}\s*\("
    match = re.search(pattern, content, re.MULTILINE)
    
    if not match:
        print(f"   ❌ Could not find function {func_name}")
        return {"success": False, "error": f"Could not find 'def {func_name}(' in {file_path}"}
    
    func_indent = len(match.group(1))
    print(f"   🔍 DEBUG: Original function indent from regex: {func_indent}")
    print(f"   🔍 DEBUG: Matched whitespace: {repr(match.group(1))}")
    
    # Backup the file
    backup_path = f"{file_path}.backup"
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            f.write(content)
    
    # Step 1: Rename original function to _original_{func_name}
    new_content = re.sub(
        rf"^(\s*)def\s+{func_name}\s*\(",
        rf"\1def _original_{func_name}(",
        content,
        count=1,
        flags=re.MULTILINE
    )
    
    # Step 2: Find where to insert the wrapper
    lines = new_content.split('\n')
    
    func_start = None
    func_end = None
    original_indent = 0
    
    # Find the renamed function
    for idx, line in enumerate(lines):
        if f"def _original_{func_name}(" in line:
            func_start = idx
            original_indent = len(line) - len(line.lstrip())
            break
    
    if func_start is None:
        return {"success": False, "error": f"Could not find renamed function _original_{func_name}"}
    
    print(f"   🔍 DEBUG: func_start={func_start}, original_indent={original_indent}")
    print(f"   🔍 DEBUG: Line at func_start: {repr(lines[func_start][:60])}")
    
    # Find where the function body ends
    for idx in range(func_start + 1, len(lines)):
        line = lines[idx]

        # Skip empty lines
        if not line.strip():
            continue
        
        # Skip comments
        if line.strip().startswith('#'):
            continue
        
        # Skip string continuations (lines that are part of a docstring or multiline string)
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        
        current_indent = len(line) - len(line.lstrip())

        # Function ends when we hit a def/class/if __name__ at same or lesser indentation
        if current_indent <= original_indent:
            # Make sure it's a new statement, not a continuation
            if (stripped.startswith('def ') or 
                stripped.startswith('class ') or 
                stripped.startswith('@') or
                stripped.startswith('if __name__')):
                func_end = idx
                break
    
    if func_end is None:
        func_end = len(lines)
    
    print(f"   🔍 DEBUG: func_end={func_end}")
    
    # Build wrapper with SAME indentation as original function
    wrapper = _build_capture_wrapper(func_name, cache_file, indent=original_indent)
    
    print(f"   🔍 DEBUG: Wrapper first 80 chars: {repr(wrapper[:80])}")
    print(f"   🔍 DEBUG: Line before insert (func_end-1): {repr(lines[func_end-1][:60]) if func_end > 0 else 'N/A'}")
    print(f"   🔍 DEBUG: Line at insert point (func_end): {repr(lines[func_end][:60]) if func_end < len(lines) else 'END OF FILE'}")
    
    # Insert wrapper right after the function ends
    lines.insert(func_end, "")
    lines.insert(func_end + 1, wrapper)
    
    new_content = '\n'.join(lines)
    
    # Write the modified content
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    # Verify syntax
    syntax_error = check_syntax(file_path)
    if syntax_error and ("SyntaxError" in syntax_error or "IndentationError" in syntax_error):
        print(f"   ⚠️ Wrapper created syntax error, reverting: {syntax_error}")
        with open(backup_path, 'r') as f:
            original = f.read()
        with open(file_path, 'w') as f:
            f.write(original)
        return {"success": False, "error": f"Wrapper created syntax error: {syntax_error}"}
    
    # Protect file from LLM edits
    try:
        from system_llm_agent import protect_file
        protect_file(file_path)
        print(f"   🔒 Protected {file_path} from LLM edits")
    except ImportError:
        pass
    
    print(f"   ✅ Injected capture wrapper for {func_name}()")
    print(f"   🔍 DEBUG wrapper preview:\n{wrapper[:300]}")
    
    return {
        "success": True,
        "cache_file": cache_file,
        "mock_info": {
            "function": func_name,
            "file": file_path,
            "wrapper_injected": True,
        }
    }

def full_run_debug(entry_file: str) -> Dict[str, Any]:
    """
    Simple end-to-end run using the basic debugging pattern:
    - run script
    - if failure, analyze error and return info (fixing is left to tools / user)
    """

    out = run_script.invoke({"script_path": entry_file, "args": []})
    if out.get("exitcode") == 0:
        return {"strategy": "full_run", "status": "ok", "run": out}

    err = out.get("stderror") or ""
    analysis = error_analyzer(err)
    return {
        "strategy": "full_run",
        "status": "failed",
        "run": out,
        "error_analysis": analysis,
    }

def _filter_to_entry_flow(entry_file: str, flows: Dict[str, Any], funcs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entry_flow = flows.get(entry_file)
    if not entry_flow:
        print(f"⚠️ No flow found for {entry_file}. No filtering applied.")
        return funcs

    kept = []
    for f in funcs:
        fn = f.get("function")
        fp = os.path.abspath(f.get("file", ""))
        if fn in entry_flow:
            flow_file = (entry_flow.get(fn) or {}).get("file")
            if flow_file and flow_file != "external" and os.path.abspath(flow_file) == fp:
                kept.append(f)
    return kept



def mocking_debug(
    entry_file: str, args: List[str] = None, llm=None
) -> Dict[str, Any]:
    """
    MOCKING STRATEGY: Prepare minimal inputs and expensive function list.
    
    Does NOT run the script - that's the LLM agent's job.
    Just prepares the environment and returns configuration.
    """
    if args is None:
        args = []
    
    print("\n" + "=" * 80)
    print("MOCKING STRATEGY PREPARATION")
    print("=" * 80)
    
    # Step 1: Analyze to get expensive functions
    analysis_results, flows = autonomous_analysis(".", llm_instance=llm)
    strategy, categories = _decide_strategy(analysis_results)
    
    # Step 2: Collect and filter expensive functions
    target_funcs: List[Dict[str, Any]] = []
    for key in ["critical_ml", "compute_heavy", "io_ops"]:
        target_funcs.extend(categories.get(key, []))
    target_funcs = _filter_to_entry_flow(entry_file, flows, target_funcs)

    entry_flow = flows.get(entry_file, {})
    target_funcs = _promote_helpers_to_callers(target_funcs, entry_flow)

    print("\n📌 Expensive functions identified:")
    for func in target_funcs[:10]:
        extra = f" (promoted from {func.get('promoted_from')})" if func.get("promoted_from") else ""
        print(f"   - {func['file']}::{func['function']} score={func['score']}{extra}")
    
    # Step 3: Generate and apply minimal inputs
    print("\n🔧 Generating minimal inputs...")
    entry_only_flows = {entry_file: flows.get(entry_file, {})}
    config_changes = apply_minimal_inputs(entry_file, target_funcs, entry_only_flows, llm)
    
    print(f"   ✓ Generated minimal specs for {len(config_changes['minimal_specs'])} functions")
    print(f"   ✓ Applied {len(config_changes['config_swaps'])} config swaps")
    if config_changes["entry_args"]:
        print(f"   ✓ Minimal CLI args: {' '.join(config_changes['entry_args'])}")
    
    # Return configuration (don't run anything)
    return {
        "strategy": "mocking",
        "target_funcs": target_funcs,
        "config_changes": config_changes,
        "flows": flows,
        "entry_flow": entry_flow,
    }



def run_combined_agent(
    entry_file: Optional[str] = None, args: Optional[List[str]] = None, llm=None
) -> Dict[str, Any]:
    """
    Main entry point with STRATEGY DIFFERENTIATION:
    
    FULL_RUN STRATEGY:
    - No expensive operations detected
    - Direct execution with provided args
    - Simple error analysis on failure
    
    MOCKING STRATEGY:
    - Expensive operations detected (critical_ml, compute_heavy, io_ops)
    - Generate minimal inputs dynamically
    - Apply minimal configs/args
    - Run with minimal inputs
    - On failure: mock expensive functions iteratively
    - Track mocked functions for subsequent runs
    """
    if entry_file is None:
        entry_file = _detect_primary_entry()
    if args is None:
        args = []

    print("\n" + "=" * 80)
    print("COMBINED DEBUGGING AGENT")
    print("=" * 80)
    print(f"Entry file: {entry_file}")
    print(f"Args: {' '.join(args) if args else '(none)'}")

    # Step 1: Analyze codebase
    print("\n[Step 1] Analyzing codebase...")
    analysis_results, flows = autonomous_analysis(".")
    strategy, categories = _decide_strategy(analysis_results)

    print(f"\n📊 Strategy Decision: {strategy.upper()}")
    print(f"   Critical ML functions: {len(categories.get('critical_ml', []))}")
    print(f"   Compute-heavy functions: {len(categories.get('compute_heavy', []))}")
    print(f"   I/O operations: {len(categories.get('io_ops', []))}")

    # Step 2: Route to appropriate strategy
    if strategy == "full_run":
        print("\n[Step 2] Using FULL_RUN strategy (no mocking needed)")
        if args:
            # User provided args, run directly
            run_out = run_script.invoke({"script_path": entry_file, "args": args})
            status = "ok" if run_out.get("exitcode") == 0 else "failed"
            result = {
                "analysis_strategy": strategy,
                "entry_file": entry_file,
                "categories": categories,
                "status": status,
                "run": run_out,
            }
        else:
            # No args, use simple debug
            result = full_run_debug(entry_file)
            result["analysis_strategy"] = strategy
            result["entry_file"] = entry_file
            result["categories"] = categories
    else:
        print("\n[Step 2] Using MOCKING strategy (minimal inputs + function mocking)")
        result = mocking_debug(entry_file, args, llm)
        result["analysis_strategy"] = strategy
        result["entry_file"] = entry_file
        result["categories"] = categories

    return result



