"""
Static analysis + expense/mocking logic derived from the Untitled notebook.

These helpers:
- discover entry points
- trace call graphs
- analyze functions with an LLM for expensive ops / side effects
- rank and categorize functions for mocking
"""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import shared LLM instance
# Module-level LLM instance (set by caller)
llm = None

def set_llm_instance(llm_instance):
    """Set the global LLM instance for this module."""
    global llm
    llm = llm_instance

def call_llm_raw(llm, messages, max_tokens=1024) -> str:
    """LLM call function with output cleaning"""
    resp = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        max_tokens=max_tokens,
        repeat_penalty=1.05,
        stop=[
            "<|im_end|>",
            "<<<CODE",
            "CODE>>>",
            "INPUT CODE",
            "```",
            "\n\nHere",  # NEW: Stop if it starts explaining
            "\n\nThe",   # NEW: Stop if it starts explaining
        ],
    )
    
    output = resp["choices"][0]["message"]["content"]
    
    # NEW: Aggressive cleanup of preamble
    output = output.strip()
    
    # Remove "Here is..." preambles
    import re
    output = re.sub(r'^Here (is|are) the .*?:\s*', '', output, flags=re.IGNORECASE)
    output = re.sub(r'^The JSON.*?:\s*', '', output, flags=re.IGNORECASE)
    
    # Remove markdown fences
    output = re.sub(r'^```json\s*', '', output, flags=re.IGNORECASE)
    output = re.sub(r'^```\s*', '', output)
    output = re.sub(r'```\s*$', '', output)
    
    return output.strip()


def parse_json_stream_safe(text: str) -> List[Any]:
    """
    Safe JSON parser that handles escape sequence issues AND preamble text
    """
    if not text:
        return []

    s = text.strip()
    
    # NEW: Strip common LLM preambles
    preamble_patterns = [
        r'^Here is the JSON.*?:',
        r'^Here are the.*?:',
        r'^The JSON.*?:',
        r'^```json',
        r'^```',
    ]
    
    import re
    for pattern in preamble_patterns:
        s = re.sub(pattern, '', s, flags=re.IGNORECASE | re.MULTILINE)
    
    s = s.strip()
    
    # Strip trailing markdown fences
    s = re.sub(r'```\s*$', '', s)
    s = s.strip()

    def fix_escapes(match: "re.Match[str]") -> str:
        char = match.group(1)
        if char not in ['"', "\\", "/", "b", "f", "n", "r", "t", "u"]:
            return "\\\\" + char
        return match.group(0)

    try:
        s = re.sub(r"\\([^\"\\\/bfnrtu])", fix_escapes, s)
    except Exception:
        pass

    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
        return [obj]
    except Exception:
        pass

    results: List[Any] = []
    i = 0
    n = len(s)

    def skip_ws(idx: int) -> int:
        while idx < n and s[idx].isspace():
            idx += 1
        return idx

    while i < n:
        i = skip_ws(i)
        if i >= n:
            break
        if s[i] not in "{[":
            i += 1
            continue

        start = i
        stack: List[str] = []
        in_str = False
        esc = False

        while i < n:
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    if not stack:
                        break
                    opener = stack.pop()
                    if (opener == "{" and ch != "}") or (
                        opener == "[" and ch != "]"
                    ):
                        stack = []
                        break
                    if not stack:
                        end = i + 1
                        chunk = s[start:end]
                        try:
                            chunk_fixed = re.sub(
                                r"\\([^\"\\\/bfnrtu])", fix_escapes, chunk
                            )
                            parsed = json.loads(chunk_fixed)
                            if isinstance(parsed, list):
                                results.extend(parsed)
                            else:
                                results.append(parsed)
                        except Exception:
                            pass
                        i = end
                        break
            i += 1
        else:
            break

    return results


def discover_entry_points(directory: str = ".") -> Dict[str, Any]:
    """Heuristically discover main execution entry points in a directory."""

    entry_points: Dict[str, Any] = {
        "certain": [],
        "likely": [],
        "possible": [],
    }

    for pyfile in Path(directory).glob("*.py"):
        # Skip system_* helpers
        if pyfile.name.startswith("system_"):
            continue
        try:
            with open(pyfile, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content)

            if 'if __name__ == "__main__"' in content:
                entry_points["certain"].append(str(pyfile))
                continue

            score = 0
            reasons: List[str] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = getattr(node, "module", None)
                    names = [n.name for n in getattr(node, "names", [])]
                    if module in ["argparse", "click", "fire", "typer"]:
                        score += 3
                        reasons.append(f"imports {module}")
                    if "ArgumentParser" in names:
                        score += 3
                        reasons.append("uses ArgumentParser")
                if isinstance(node, ast.FunctionDef):
                    if node.name in ["main", "run", "train", "execute", "cli"]:
                        score += 2
                        reasons.append(f"has {node.name}() function")

            module_level_calls = [
                node
                for node in tree.body
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            ]
            if module_level_calls:
                score += 1
                reasons.append("has module-level calls")

            if score >= 3:
                entry_points["likely"].append(
                    {"file": str(pyfile), "score": score, "reasons": reasons}
                )
            elif score > 0:
                entry_points["possible"].append(
                    {"file": str(pyfile), "score": score, "reasons": reasons}
                )
        except Exception:
            continue

    return entry_points


def extract_calls_from_function(func_node: ast.FunctionDef, source_file: str) -> List[Dict[str, str]]:
    """Extract all function calls within a function."""

    calls: List[Dict[str, str]] = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(
                    {"name": node.func.id, "type": "direct", "from_file": source_file}
                )
            elif isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                calls.append(
                    {
                        "name": f"{node.func.value.id}.{node.func.attr}",
                        "type": "method",
                        "from_file": source_file,
                    }
                )
    return calls


def find_function_definition(
    func_name: str, directory: str = "."
) -> Tuple[Optional[str], Optional[ast.FunctionDef], Optional[str]]:
    """Search all Python files (except system_*) for a function definition."""

    if "." in func_name:
        func_name = func_name.split(".")[-1]

    for pyfile in Path(directory).glob("*.py"):
        if pyfile.name.startswith("system_"):
            continue
        try:
            with open(pyfile, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    return str(pyfile), node, content
        except Exception:
            continue
    return None, None, None


def trace_execution_flow(entry_file: str, max_depth: int = 3) -> Dict[str, Any]:
    """
    Starting from an entry point, recursively trace which functions
    are called and from which files.
    """

    visited: set[str] = set()
    flow_graph: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"calls": [], "called_from": [], "depth": 0, "file": None}
    )

    def trace_recursive(func_name: str, source_file: str, depth: int = 0) -> None:
        if depth > max_depth or func_name in visited:
            return
        visited.add(func_name)
        def_file, func_node, content = find_function_definition(func_name)
        if not func_node:
            flow_graph[func_name]["depth"] = depth
            flow_graph[func_name]["file"] = "external"
            return

        flow_graph[func_name]["depth"] = depth
        flow_graph[func_name]["file"] = def_file
        flow_graph[func_name]["called_from"].append(source_file)

        calls = extract_calls_from_function(func_node, def_file or "")
        flow_graph[func_name]["calls"] = [c["name"] for c in calls]
        for call in calls:
            trace_recursive(call["name"], def_file or "", depth + 1)

    try:
        with open(entry_file, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)
    except Exception:
        return flow_graph

    main_block_calls: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            is_main_guard = False
            if isinstance(test, ast.Compare):
                if (
                    isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value == "__main__"
                ):
                    is_main_guard = True
            if is_main_guard:
                for sub_node in ast.walk(node):
                    if isinstance(sub_node, ast.Call) and isinstance(
                        sub_node.func, ast.Name
                    ):
                        main_block_calls.append(sub_node.func.id)

    entry_functions: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in [
            "main",
            "run",
            "train",
            "execute",
            "cli",
            "train_dgl",
        ]:
            entry_functions.append(node.name)

    module_calls: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                module_calls.append(node.value.func.id)

    all_starting_points = list(set(main_block_calls + entry_functions + module_calls))
    if not all_starting_points:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                calls = extract_calls_from_function(node, entry_file)
                for call in calls:
                    trace_recursive(call["name"], entry_file, depth=1)
    else:
        for func_name in all_starting_points:
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    flow_graph[func_name]["depth"] = 0
                    flow_graph[func_name]["file"] = entry_file
                    flow_graph[func_name]["called_from"].append("__main__")
                    calls = extract_calls_from_function(node, entry_file)
                    flow_graph[func_name]["calls"] = [c["name"] for c in calls]
                    for call in calls:
                        trace_recursive(call["name"], entry_file, depth=1)
                    break
            else:
                trace_recursive(func_name, entry_file, depth=1)

    return flow_graph


def analyze_file_with_context(
    file_path: str, relevant_functions: List[str], flow_graph: Dict[str, Any], llm_instance=None
) -> List[Dict[str, Any]]:
    """Analyze only relevant functions in a file with LLM for expense/side effects."""
    
    if llm_instance is None:
        llm_instance = llm  # Use global instance

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)
    except Exception as e:
        return [{"_error": f"Failed to parse {file_path}: {e}"}]

    relevant_code: List[Dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in relevant_functions:
            try:
                func_source = ast.get_source_segment(content, node)
                if not func_source:
                    lines = content.splitlines()
                    func_source = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                relevant_code.append(
                    {
                        "name": node.name,
                        "code": func_source[:3000],
                        "called_by": flow_graph.get(node.name, {}).get(
                            "called_from", []
                        ),
                        "calls": flow_graph.get(node.name, {}).get("calls", []),
                    }
                )
            except Exception:
                continue

    if not relevant_code:
        return [{"_note": f"No relevant functions found in {file_path}"}]

    functions_list = "\n".join(f"- {f['name']}" for f in relevant_code)
    code_blocks = "\n\n".join(
        f"=== FUNCTION: {f['name']} ===\n{f['code']}" for f in relevant_code
    )

    prompt = f"""Output ONLY valid JSON. No markdown, no text, no commentary.

File: {file_path}

FUNCTIONS TO ANALYZE (use these EXACT names):
{functions_list}

CODE:
{code_blocks}

For EACH function listed above, output ONE JSON object using the EXACT function name shown:
{{"kind":"function","name":"EXACT_NAME_FROM_LIST","purpose":"brief description","side_effects":[],"expensive_ops":[],"calls_internal":[],"external_deps":[],"risk_flags":[],"evidence":["code snippet"]}}

CRITICAL: The "name" field MUST be the exact function name from the list above.

side_effects options: ["io","network","filesystem","gpu","randomness","global_state","db","subprocess"]
expensive_ops options: ["training_loop","model_forward","backprop","dataloader","tokenizer","vector_index","large_matrix_op","pandas_groupby","multiprocessing","disk_io"]

DETECTION RULES (mark as expensive even if not visible in snippet):
- training_loop: 
  * For loops over epochs with optimizer.step() or loss.backward()
  * Functions named train_*, train, training, fit
  * Functions that call model training loops
  * ANY function with "train" in name that iterates
- model_forward: 
  * Calls to model(), forward passes, .forward()
  * Neural network forward propagation
- backprop: 
  * loss.backward(), gradient computation, optimizer.step()
- dataloader: 
  * DataLoader creation, batch iteration
  * Functions that load/prepare training data
- gpu: 
  * .cuda(), .to(device), torch.device
  * GPU memory operations

IMPORTANT DETECTION RULES:
- training_loop: ONLY if function performs the ACTUAL training loop with optimizer.step() and loss.backward()
  * Functions named exactly: train, train_model, train_dgl, fit, training
  * Must show evidence of iterative training (epochs, optimizer, loss)
  * NOT for data preparation functions like get_train_val_loaders, setup_training
- dataloader: For functions that CREATE or SETUP data loaders (DataLoader, get_loaders, setup_data)
  * Functions with "loader" or "dataset" in name are likely dataloader, NOT training_loop

Output valid JSON only.
"""

    messages = [
        {
            "role": "system", 
            "content": "You are a code analyzer. You MUST output ONLY valid JSON. No text before or after the JSON. No markdown. No explanations. Start directly with [ or {."
        },
        {"role": "user", "content": prompt}
    ]

    output = call_llm_raw(llm_instance, messages, max_tokens=3072)
    if 'train_dgl' in str(relevant_functions):
        print(f"\n🔍 DEBUG: LLM analysis for train_dgl:")
        print(f"   Raw output (first 500 chars): {output[:500]}")
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        items = parse_json_stream_safe(output)
    
    # DEBUG: Print parsed items for train_dgl
    if 'train_dgl' in str(relevant_functions):
        print(f"   Parsed {len(items)} items")
        for item in items:
            if isinstance(item, dict) and item.get('name') == 'train_dgl':
                print(f"   train_dgl analysis:")
                print(f"      expensive_ops: {item.get('expensive_ops', [])}")
                print(f"      side_effects: {item.get('side_effects', [])}")
                print(f"      purpose: {item.get('purpose', 'unknown')}")
    
    if not items:
        return [{"_error": "No JSON parsed", "raw": output[:500], "_functions_requested": relevant_functions}]
    
    return [i for i in items if isinstance(i, dict)]


def autonomous_analysis(
    directory: str = ".", llm_instance=None,entry_file: str | None = None
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """End-to-end analysis over likely entry points."""
    
    if llm_instance is None:
        llm_instance = llm  # Use global instance

    if entry_file:
        entry_points = [entry_file]
    else:
        entry_points = discover_entry_points(directory)
        all_entries = entry_points["certain"] + [
            e["file"] for e in entry_points["likely"]
    ]
    if not all_entries:
        all_entries = [
            str(f)
            for f in Path(directory).glob("*.py")
            if not f.name.startswith("system_")
        ]

    all_function_analyses: List[Dict[str, Any]] = []
    analyzed_combinations: set[Tuple[str, Tuple[str, ...]]] = set()
    all_flows: Dict[str, Any] = {}

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        for entry_file in all_entries:
            flow = trace_execution_flow(entry_file)
            all_flows[entry_file] = flow
            files_by_priority: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
            for func_name, info in flow.items():
                if info.get("file") and info["file"] != "external":
                    files_by_priority[info["depth"]].append(
                        (info["file"], func_name)
                    )

            for depth in sorted(files_by_priority.keys()):
                print(f"\n   📍 Processing depth {depth}")  # ← ADD THIS

                file_funcs = defaultdict(list)
                for file_path, func_name in files_by_priority[depth]:
                    file_funcs[file_path].append(func_name)

                for file_path, funcs in file_funcs.items():
                    func_key = (file_path, tuple(sorted(funcs)))

                    if func_key in analyzed_combinations:
                        print(f"      ⏭️  Skipping {file_path} (already analyzed)")  # ← ADD THIS
                        continue
                    
                    analyzed_combinations.add(func_key)

                    print(f"   🔍 Analyzing {file_path} (depth {depth}, {len(funcs)} functions: {funcs[:3]}...)")
                    analysis = analyze_file_with_context(
                        file_path, funcs, flow, llm_instance
                    )
                    for item in analysis:
                        if isinstance(item, dict):
                            item["_source_file"] = file_path
                            item["_entry_point"] = entry_file
                            item["_depth"] = depth
                            all_function_analyses.append(item)

    analysis_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for func_analysis in all_function_analyses:
        file_path = func_analysis.pop("_source_file", "unknown")
        func_analysis.pop("_entry_point", None)
        func_analysis.pop("_depth", None)
        analysis_results[file_path].append(func_analysis)

    return dict(analysis_results), all_flows


"""
Fix for system_analysis.py - Better expense filtering to avoid flagging simple helpers
"""

def analyze_and_rank_expensive_functions(
    analysis_results: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Extract all functions and rank them by expense/risk."""

    EXPENSE_WEIGHTS: Dict[str, int] = {
        "training_loop": 10,
        "backprop": 9,
        "model_forward": 8,
        "dataloader": 7,
        "large_matrix_op": 6,
        "vector_index": 6,
        "pandas_groupby": 5,
        "multiprocessing": 5,
        "disk_io": 5,
        "tokenizer": 4,
        "network": 8,
        "db": 7,
        "gpu": 6,
        "subprocess": 5,
        "filesystem": 3,
        "io": 2,
        "randomness": 2,
        "global_state": 1,
    }
    
    # NEW: Define low-impact operations that shouldn't make a function "expensive"
    LOW_IMPACT_OPS = {"global_state", "randomness"}
    
    # NEW: Helper function name patterns to exclude
    HELPER_PATTERNS = [
        "_atomic_",  # Atomic operations (saves, etc.)
        "_save_",
        "_load_config",
        "_log_",
        "_print_",
        "_format_",
        "_validate_",
    ]

    expensive_functions: List[Dict[str, Any]] = []
    for file_path, analysis in analysis_results.items():
        if not isinstance(analysis, list):
            continue
        for item in analysis:
            if not isinstance(item, dict):
                continue
            if item.get("kind") != "function":
                continue
            func_name = item.get("name", "unknown")
            expensive_ops = item.get("expensive_ops", []) or []
            side_effects = item.get("side_effects", []) or []
            
            # NEW: Skip helper functions with simple patterns
            if any(pattern in func_name.lower() for pattern in HELPER_PATTERNS):
                # Only include if it has truly expensive ops (not just io/filesystem)
                if not any(op in expensive_ops for op in ["training_loop", "backprop", "model_forward", "dataloader"]):
                    continue
            
            score = 0
            reasons: List[str] = []
            
            # Calculate score, but filter out low-impact ops for helpers
            all_ops = set(expensive_ops + side_effects)
            significant_ops = all_ops - LOW_IMPACT_OPS if func_name.startswith("_") else all_ops
            
            for op in expensive_ops:
                if func_name.startswith("_") and op in LOW_IMPACT_OPS:
                    continue  # Don't count low-impact ops for helper functions
                w = EXPENSE_WEIGHTS.get(op, 0)
                score += w
                reasons.append(f"{op}({w})")
                
            for eff in side_effects:
                if func_name.startswith("_") and eff in LOW_IMPACT_OPS:
                    continue
                w = EXPENSE_WEIGHTS.get(eff, 0)
                score += w
                reasons.append(f"{eff}({w})")
            
            # NEW: Increase threshold for helper functions
            min_score = 5 if func_name.startswith("_") else 1
            if score < min_score:
                continue
            
            expensive_functions.append(
                {
                    "file": file_path,
                    "function": func_name,
                    "score": score,
                    "expensive_ops": expensive_ops,
                    "side_effects": side_effects,
                    "reasons": reasons,
                    "purpose": item.get("purpose", "unknown"),
                    "calls_internal": item.get("calls_internal", []),
                    "external_deps": item.get("external_deps", []),
                }
            )

    expensive_functions.sort(key=lambda x: x["score"], reverse=True)
    return expensive_functions

def categorize_functions_for_mocking(
    expensive_functions: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize expensive functions and separate orchestrators."""

    categories: Dict[str, List[Dict[str, Any]]] = {
        "critical_ml": [],
        "compute_heavy": [],
        "io_ops": [],
        "side_effects": [],
        "simple": [],
        "orchestrators": [],
    }

    orchestrator_patterns = ["main", "run", "train_for_", "execute_", "cli", "pipeline"]

    for func in expensive_functions:
        func_name = func["function"]
        ops = set(func["expensive_ops"] + func["side_effects"])
        is_orchestrator = False
        if any(p in func_name.lower() for p in orchestrator_patterns):
            if len(func["expensive_ops"]) > 5:
                is_orchestrator = True
        if len(func.get("calls_internal", [])) > 5:
            is_orchestrator = True

        if is_orchestrator:
            categories["orchestrators"].append(func)
            continue

        if any(op in ops for op in ["training_loop", "backprop"]):
            categories["critical_ml"].append(func)
        elif any(
            op in ops
            for op in ["model_forward", "large_matrix_op", "dataloader", "vector_index"]
        ):
            categories["compute_heavy"].append(func)
        elif any(op in ops for op in ["disk_io", "filesystem", "network", "db", "io"]):
            categories["io_ops"].append(func)
        elif any(
            op in ops for op in ["gpu", "randomness", "global_state", "subprocess"]
        ):
            categories["side_effects"].append(func)
        else:
            categories["simple"].append(func)

    return categories