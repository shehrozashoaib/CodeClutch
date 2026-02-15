"""
Minimal input generation for expensive functions.

Extracted from Untitled.ipynb - provides:
- MinimalInputExtractor: backward slicing to find required inputs
- MinimalInputGenerator: LLM-assisted generation of minimal test configs
- ExistingResourceDiscovery: discovers config files, datasets, checkpoints
- EnhancedMinimalInputGenerator: uses discovered resources
- DataFlowTracer: traces data flow to find parameter origins
- SmartParameterResolver: intelligently resolves parameter values
- ImprovedDataFlowTracer: enhanced tracer for config objects
- EnhancedParameterResolver: enhanced resolver with config attribute resolution
"""

import ast
import json
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_cpp import Llama  # type: ignore[import-not-found]

from system_analysis import parse_json_stream_safe


def call_llm_raw(llm, messages, max_tokens=1024) -> str:
    """LLM call function with output cleaning"""
    resp = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        max_tokens=max_tokens,
        repeat_penalty=1.05,
        stop=["<|im_end|>", "<<<CODE", "CODE>>>", "```"],
    )
    return resp["choices"][0]["message"]["content"]


@dataclass
class InputRequirement:
    """Represents a single input requirement for a function"""
    param_name: str
    param_type: Optional[str] = None
    default_value: Any = None
    source: str = "argument"  # 'argument', 'config', 'cli_argument'
    config_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


class MinimalInputExtractor:
    """
    Extract minimal inputs needed for a function by backward slicing
    """

    def __init__(self, project_dir="."):
        self.project_dir = Path(project_dir)
        self.file_asts = {}
        self.config_sources = []
        self.argparse_info = {}

    def parse_file(self, file_path: str) -> ast.AST:
        """Parse and cache file AST"""
        if file_path not in self.file_asts:
            with open(file_path, "r", encoding="utf-8") as f:
                self.file_asts[file_path] = ast.parse(f.read())
        return self.file_asts[file_path]

    def find_function_node(
        self, file_path: str, func_name: str
    ) -> Optional[ast.FunctionDef]:
        """Find function definition in a file"""
        tree = self.parse_file(file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return node
        return None

    def extract_function_signature(
        self, func_node: ast.FunctionDef
    ) -> List[InputRequirement]:
        """Extract parameter information from function signature"""
        inputs = []

        args = func_node.args

        # Regular arguments
        for i, arg in enumerate(args.args):
            param_name = arg.arg

            # Try to get type annotation
            param_type = None
            if arg.annotation:
                param_type = (
                    ast.unparse(arg.annotation) if hasattr(ast, "unparse") else None
                )

            # Check for default value
            default_value = None
            defaults_offset = len(args.args) - len(args.defaults)
            if i >= defaults_offset:
                default_idx = i - defaults_offset
                default_value = (
                    ast.unparse(args.defaults[default_idx])
                    if hasattr(ast, "unparse")
                    else None
                )

            inputs.append(
                InputRequirement(
                    param_name=param_name,
                    param_type=param_type,
                    default_value=default_value,
                    source="argument",
                )
            )

        # *args
        if args.vararg:
            inputs.append(
                InputRequirement(
                    param_name=f"*{args.vararg.arg}",
                    param_type=None,
                    default_value=None,
                    source="argument",
                )
            )

        # **kwargs
        if args.kwarg:
            inputs.append(
                InputRequirement(
                    param_name=f"**{args.kwarg.arg}",
                    param_type=None,
                    default_value=None,
                    source="argument",
                )
            )

        return inputs

    def trace_parameter_usage(
        self, func_node: ast.FunctionDef, param_name: str
    ) -> Dict[str, Any]:
        """
        Trace how a parameter is used within the function
        Returns info about what the parameter needs to have (methods, attributes, etc.)
        """
        usage_info = {
            "attributes_accessed": set(),
            "methods_called": set(),
            "used_as_dict": False,
            "used_as_list": False,
            "passed_to_functions": [],
            "requires_gpu": False,
            "requires_shape": None,  # For tensors
        }

        for node in ast.walk(func_node):
            # Check attribute access: param.attribute
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == param_name:
                    usage_info["attributes_accessed"].add(node.attr)

            # Check method calls: param.method()
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == param_name
                    ):
                        usage_info["methods_called"].add(node.func.attr)

                        # Special cases
                        if node.func.attr == "cuda":
                            usage_info["requires_gpu"] = True
                        if node.func.attr in ["to"]:
                            # Check if .to(device)
                            for arg in node.args:
                                if isinstance(arg, ast.Name) and "device" in arg.id:
                                    usage_info["requires_gpu"] = True

            # Check subscript access: param[key]
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id == param_name:
                    usage_info["used_as_dict"] = True

            # Check iteration: for x in param
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Name) and node.iter.id == param_name:
                    usage_info["used_as_list"] = True

        return usage_info

    def find_config_files(self) -> List[str]:
        """Find config files in the project"""
        config_files = []

        for pattern in ["config.py", "*/config.py", "config.json", "*.yaml", "*.yml"]:
            config_files.extend(self.project_dir.glob(pattern))

        return [str(f) for f in config_files]

    def extract_config_structure(self, config_file: str) -> Dict[str, Any]:
        """
        Extract configuration structure from a config file
        """
        config_structure = {}

        if config_file.endswith(".py"):
            tree = self.parse_file(config_file)

            # Look for class definitions (common pattern: TrainingConfig, ModelConfig, etc.)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_config = {}

                    for item in node.body:
                        # Look for assignments: attr = value
                        if isinstance(item, ast.AnnAssign) and isinstance(
                            item.target, ast.Name
                        ):
                            attr_name = item.target.id
                            default_val = None
                            attr_type = None

                            if item.annotation:
                                attr_type = (
                                    ast.unparse(item.annotation)
                                    if hasattr(ast, "unparse")
                                    else None
                                )

                            if item.value:
                                try:
                                    default_val = ast.literal_eval(item.value)
                                except:
                                    default_val = (
                                        ast.unparse(item.value)
                                        if hasattr(ast, "unparse")
                                        else None
                                    )

                            class_config[attr_name] = {
                                "type": attr_type,
                                "default": default_val,
                            }

                        # Also check simple assignments
                        elif isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    try:
                                        class_config[target.id] = {
                                            "type": None,
                                            "default": ast.literal_eval(item.value),
                                        }
                                    except:
                                        pass

                    config_structure[node.name] = class_config

            # Also look for top-level assignments
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            try:
                                config_structure[target.id] = ast.literal_eval(node.value)
                            except:
                                pass

        return config_structure

    def extract_argparse_info(self, file_path: str) -> Dict[str, Any]:
        """
        Extract argparse argument definitions from a file
        """
        tree = self.parse_file(file_path)
        argparse_args = {}

        for node in ast.walk(tree):
            # Look for parser.add_argument calls
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                ):
                    # Extract argument info
                    arg_name = None
                    arg_info = {}

                    # First positional arg is the argument name
                    if node.args:
                        if isinstance(node.args[0], ast.Constant):
                            arg_name = node.args[0].value.lstrip("-")

                    # Extract keyword arguments
                    for keyword in node.keywords:
                        key = keyword.arg
                        try:
                            value = ast.literal_eval(keyword.value)
                        except:
                            value = (
                                ast.unparse(keyword.value)
                                if hasattr(ast, "unparse")
                                else None
                            )
                        arg_info[key] = value

                    if arg_name:
                        argparse_args[arg_name] = arg_info

        return argparse_args

    def backward_slice_for_function(
        self, file_path: str, func_name: str, flow_graph: Dict = None
    ) -> Dict[str, Any]:
        """
        Perform backward slicing to find minimal inputs needed for a function
        """
        print(f"\n🔍 Extracting minimal inputs for {file_path}::{func_name}")

        func_node = self.find_function_node(file_path, func_name)
        if not func_node:
            return {"error": f"Function {func_name} not found in {file_path}"}

        # Step 1: Get direct parameters
        direct_inputs = self.extract_function_signature(func_node)

        print(f"   Found {len(direct_inputs)} direct parameters")

        # Step 2: Analyze how each parameter is used
        for inp in direct_inputs:
            if not inp.param_name.startswith("*"):
                usage = self.trace_parameter_usage(func_node, inp.param_name)
                inp.dependencies = list(usage["attributes_accessed"])

                # Add metadata about usage
                setattr(inp, "usage_info", usage)

        # Step 3: Find config sources
        config_files = self.find_config_files()
        config_structures = {}

        for cf in config_files:
            try:
                config_structures[cf] = self.extract_config_structure(cf)
                print(f"   Found config file: {cf}")
            except:
                pass

        # Step 4: Find argparse info (look in entry point files)
        if flow_graph:
            # Check files that call this function
            for func_info in flow_graph.values():
                if func_name in func_info.get("calls", []):
                    caller_file = func_info.get("file")
                    if caller_file and caller_file != "external":
                        try:
                            argparse_info = self.extract_argparse_info(caller_file)
                            if argparse_info:
                                print(f"   Found argparse in: {caller_file}")
                                self.argparse_info.update(argparse_info)
                        except:
                            pass

        # Step 5: Match parameters to config/argparse sources
        for inp in direct_inputs:
            # Try to find config source
            for config_file, structure in config_structures.items():
                for class_name, class_config in structure.items():
                    if isinstance(class_config, dict):
                        for attr, attr_info in class_config.items():
                            if (
                                attr == inp.param_name
                                or attr.lower() == inp.param_name.lower()
                            ):
                                inp.source = "config"
                                inp.config_path = (
                                    f"{config_file}::{class_name}.{attr}"
                                )
                                if isinstance(attr_info, dict):
                                    if inp.default_value is None:
                                        inp.default_value = attr_info.get("default")
                                    if inp.param_type is None:
                                        inp.param_type = attr_info.get("type")

            # Try to find argparse source
            if inp.param_name in self.argparse_info:
                arg_info = self.argparse_info[inp.param_name]
                if inp.source == "argument":  # Don't override config
                    inp.source = "cli_argument"
                if "default" in arg_info and inp.default_value is None:
                    inp.default_value = arg_info["default"]
                if "type" in arg_info and inp.param_type is None:
                    inp.param_type = str(arg_info["type"])

        return {
            "function": func_name,
            "file": file_path,
            "inputs": direct_inputs,
            "config_files": list(config_structures.keys()),
            "has_argparse": bool(self.argparse_info),
        }

    def generate_minimal_input_spec(
        self, expensive_function: Dict[str, Any], flow_graph: Dict = None
    ) -> Dict[str, Any]:
        """
        Generate a complete minimal input specification for an expensive function
        """
        file_path = expensive_function["file"]
        func_name = expensive_function["function"]

        # Backward slice to get input requirements
        input_analysis = self.backward_slice_for_function(
            file_path, func_name, flow_graph
        )

        if "error" in input_analysis:
            return input_analysis

        # Build minimal input specification
        minimal_spec = {
            "function": func_name,
            "file": file_path,
            "purpose": expensive_function.get("purpose", "unknown"),
            "expensive_ops": expensive_function.get("expensive_ops", []),
            "inputs": [],
            "config_requirements": [],
            "cli_requirements": [],
            "synthetic_data_needed": [],
        }

        for inp in input_analysis["inputs"]:
            input_spec = {
                "name": inp.param_name,
                "type": inp.param_type,
                "source": inp.source,
                "default": inp.default_value,
                "config_path": inp.config_path,
            }

            # Add usage information
            if hasattr(inp, "usage_info"):
                usage = inp.usage_info
                input_spec["usage"] = {
                    "attributes": list(usage["attributes_accessed"]),
                    "methods": list(usage["methods_called"]),
                    "is_dict": usage["used_as_dict"],
                    "is_list": usage["used_as_list"],
                    "needs_gpu": usage["requires_gpu"],
                }

                # Determine if we need to generate synthetic data
                if usage["methods_called"]:
                    # This is a complex object, need synthetic data
                    minimal_spec["synthetic_data_needed"].append(
                        {
                            "param": inp.param_name,
                            "type": inp.param_type,
                            "methods": list(usage["methods_called"]),
                            "attributes": list(usage["attributes_accessed"]),
                        }
                    )

            minimal_spec["inputs"].append(input_spec)

            # Categorize by source
            if inp.source == "config":
                minimal_spec["config_requirements"].append(input_spec)
            elif inp.source == "cli_argument":
                minimal_spec["cli_requirements"].append(input_spec)

        return minimal_spec


class MinimalInputGenerator:
    """
    Generate truly minimal inputs by identifying only critical parameters
    and creating small synthetic data
    """

    def __init__(self, llm, project_dir="."):
        self.llm = llm
        self.project_dir = Path(project_dir)
        self.extractor = MinimalInputExtractor(project_dir)

    def identify_critical_parameters(
        self, func_spec: Dict[str, Any], func_code: str
    ) -> Dict[str, Any]:
        """
        Use LLM to identify which parameters are actually critical for minimal execution
        """
        # Extract parameter info
        params_info = []
        for inp in func_spec.get("inputs", []):
            params_info.append(
                {
                    "name": inp["name"],
                    "type": inp["type"],
                    "attributes": inp.get("usage", {}).get("attributes", []),
                    "methods": inp.get("usage", {}).get("methods", []),
                }
            )

        prompt = f"""Analyze this function to identify MINIMAL parameters needed for a quick test run.

FUNCTION: {func_spec['function']}
PURPOSE: {func_spec['purpose']}
EXPENSIVE OPERATIONS: {', '.join(func_spec['expensive_ops'])}

PARAMETERS:
{json.dumps(params_info, indent=2)}

FUNCTION CODE (abbreviated):
{func_code[:2000]}

Your task: Identify the MINIMAL set of parameters needed for a test run that completes quickly.

For ML/training functions, minimal means:
- epochs: 1-2 (not 100)
- batch_size: 2-4 (not 32+)
- dataset_size: 5-10 samples (not thousands)
- num_workers: 0-1 (not 8+)
- hidden_dim/layers: smallest possible

For each parameter, classify as:
1. CRITICAL_MINIMAL: Must set to small value (e.g., epochs=1, batch_size=2)
2. CRITICAL_DEFAULT: Must provide but can use default (e.g., learning_rate=0.001)
3. OPTIONAL: Can skip or use default
4. DERIVED: Computed from other params, don't set directly

Output JSON only:
{{
  "critical_minimal": [
    {{"param": "epochs", "minimal_value": 1, "reason": "controls iteration count"}},
    {{"param": "batch_size", "minimal_value": 2, "reason": "controls data loading"}}
  ],
  "critical_default": [
    {{"param": "learning_rate", "default_value": 0.001, "reason": "needed for optimizer"}}
  ],
  "optional": ["verbose", "log_interval"],
  "derived": ["n_train", "n_val", "n_test"]
}}
"""

        messages = [
            {
                "role": "system",
                "content": "You analyze function parameters to identify minimal test inputs. Output only JSON.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            output = call_llm_raw(self.llm, messages, max_tokens=2048)

            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                items = parse_json_stream_safe(output)

            if items:
                return items[0]
            else:
                return self._fallback_critical_params(func_spec)

        except Exception as e:
            print(f"   ⚠️ LLM analysis failed: {e}")
            return self._fallback_critical_params(func_spec)

    def _fallback_critical_params(self, func_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback heuristic-based critical parameter identification
        """
        critical_minimal = []
        critical_default = []
        optional = []

        # Common patterns for minimal values
        MINIMAL_PARAMS = {
            "epochs": 1,
            "num_epochs": 1,
            "n_epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "max_samples": 10,
            "train_size": 5,
            "val_size": 2,
            "test_size": 2,
            "n_train": 5,
            "n_val": 2,
            "n_test": 2,
            "max_neighbors": 5,
            "hidden_dim": 8,
            "num_layers": 1,
            "n_layers": 1,
        }

        for inp in func_spec.get("inputs", []):
            param_name = inp["name"]
            param_lower = param_name.lower()

            # Check if it matches minimal patterns
            matched = False
            for pattern, value in MINIMAL_PARAMS.items():
                if pattern in param_lower:
                    critical_minimal.append(
                        {
                            "param": param_name,
                            "minimal_value": value,
                            "reason": f"controls {pattern}",
                        }
                    )
                    matched = True
                    break

            if not matched:
                # Check if it's a required config object
                if "config" in param_lower or "model" in param_lower:
                    critical_default.append(
                        {
                            "param": param_name,
                            "default_value": inp.get("default"),
                            "reason": "required object",
                        }
                    )
                # Others are optional
                else:
                    optional.append(param_name)

        return {
            "critical_minimal": critical_minimal,
            "critical_default": critical_default,
            "optional": optional,
            "derived": [],
        }

    def generate_minimal_config_dict(
        self, critical_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a minimal config dictionary"""
        config = {}

        # Add critical minimal params
        for param in critical_params.get("critical_minimal", []):
            config[param["param"]] = param["minimal_value"]

        # Add critical defaults
        for param in critical_params.get("critical_default", []):
            if param.get("default_value") is not None:
                config[param["param"]] = param["default_value"]

        return config

    def generate_synthetic_dataloader(
        self,
        batch_size: int = 2,
        num_samples: int = 5,
        func_spec: Dict[str, Any] = None,
    ) -> str:
        """
        Generate minimal synthetic dataloader code
        """
        # Determine what kind of data based on expensive ops
        ops = func_spec.get("expensive_ops", []) if func_spec else []

        if "model_forward" in ops or "training_loop" in ops:
            # Likely needs tensor data
            code = f"""# Minimal synthetic dataloader
import torch
from torch.utils.data import TensorDataset, DataLoader

# Create tiny synthetic dataset
num_samples = {num_samples}
input_dim = 10  # Minimal feature dimension
output_dim = 1  # Single output

X = torch.randn(num_samples, input_dim)
y = torch.randn(num_samples, output_dim)

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size={batch_size}, shuffle=False)
val_loader = DataLoader(dataset[:2], batch_size={batch_size})  # Even smaller val set
test_loader = DataLoader(dataset[:2], batch_size={batch_size})

train_val_test_loaders = [train_loader, val_loader, test_loader]
"""
        else:
            # Generic iterable
            code = f"""# Minimal synthetic data
train_data = [{{'input': i, 'output': i*2}} for i in range({num_samples})]
val_data = [{{'input': i, 'output': i*2}} for i in range(2)]
test_data = [{{'input': i, 'output': i*2}} for i in range(2)]

train_val_test_loaders = [train_data, val_data, test_data]
"""

        return code


def generate_minimal_inputs_for_expensive_functions(
    expensive_functions: List[Dict],
    flow_graphs: Dict,
    llm,
    project_dir: str = ".",
) -> Dict[str, Any]:
    """
    Main function to generate minimal inputs for all expensive functions
    """
    extractor = MinimalInputExtractor(project_dir)
    generator = MinimalInputGenerator(llm, project_dir)

    print("\n" + "=" * 80)
    print("GENERATING MINIMAL INPUTS FOR EXPENSIVE FUNCTIONS")
    print("=" * 80)

    all_specs = {}

    for func in expensive_functions:
        func_key = f"{func['file']}::{func['function']}"
        print(f"\n🔧 Generating for {func_key}")

        # Find relevant flow graph
        relevant_flow = None
        for flow in flow_graphs.values():
            if func["function"] in flow:
                relevant_flow = flow
                break

        # Get full spec
        spec = extractor.generate_minimal_input_spec(func, relevant_flow)

        # Get function code for LLM analysis
        func_node = extractor.find_function_node(func["file"], func["function"])
        func_code = ""
        if func_node:
            try:
                with open(func["file"], "r") as f:
                    lines = f.readlines()
                    func_code = "".join(
                        lines[func_node.lineno - 1 : func_node.end_lineno]
                    )
            except:
                pass

        critical_params = generator.identify_critical_parameters(spec, func_code)
        spec["critical_params"] = critical_params
        spec["minimal_config"] = generator.generate_minimal_config_dict(
            critical_params
        )

        all_specs[func_key] = spec

    return all_specs


# ============================================================================
# ENHANCED CLASSES FROM UNTITLED.NB - RESOURCE DISCOVERY AND ENHANCED GENERATION
# ============================================================================


class ExistingResourceDiscovery:
    """
    Discover existing configuration files, datasets, and other resources
    that can be used or adapted for minimal inputs
    """

    def __init__(self, project_dir="."):
        self.project_dir = Path(project_dir)
        self.discovered = {
            "config_files": [],
            "data_files": [],
            "checkpoint_files": [],
            "example_scripts": [],
            "dataset_paths": [],
            "json_configs": [],
            "yaml_configs": [],
        }

    def discover_all_resources(self) -> Dict[str, List]:
        """Scan project for all useful resources"""

        print("\n🔍 Discovering existing resources...")

        # Config files
        for pattern in [
            "**/*.json",
            "**/*.yaml",
            "**/*.yml",
            "**/config.py",
            "**/*config*.py",
        ]:
            for file in self.project_dir.glob(pattern):
                if file.is_file() and "test" not in str(file).lower():
                    if file.suffix == ".json":
                        self.discovered["json_configs"].append(str(file))
                    elif file.suffix in [".yaml", ".yml"]:
                        self.discovered["yaml_configs"].append(str(file))
                    elif file.suffix == ".py":
                        self.discovered["config_files"].append(str(file))

        # Data files and directories
        data_patterns = [
            "**/data/**",
            "**/datasets/**",
            "**/*.csv",
            "**/*.jsonl",
            "**/*.pt",
            "**/*.pth",
        ]
        for pattern in data_patterns:
            for path in self.project_dir.glob(pattern):
                if path.is_file():
                    self.discovered["data_files"].append(str(path))
                elif path.is_dir() and "data" in path.name.lower():
                    self.discovered["dataset_paths"].append(str(path))

        # Checkpoint/model files
        for pattern in ["**/*.pth", "**/*.pt", "**/*.ckpt", "**/checkpoints/**"]:
            for file in self.project_dir.glob(pattern):
                if file.is_file():
                    self.discovered["checkpoint_files"].append(str(file))

        # Example/test scripts that might show usage
        for pattern in ["**/example*.py", "**/test*.py", "**/demo*.py"]:
            for file in self.project_dir.glob(pattern):
                if file.is_file():
                    self.discovered["example_scripts"].append(str(file))

        # Print summary
        for key, files in self.discovered.items():
            if files:
                print(f"   Found {len(files)} {key}")
                for f in files[:3]:
                    print(f"     - {f}")
                if len(files) > 3:
                    print(f"     ... and {len(files) - 3} more")

        return self.discovered

    def load_json_config(self, file_path: str) -> Dict:
        """Load and parse JSON config"""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except:
            return {}

    def load_yaml_config(self, file_path: str) -> Dict:
        """Load and parse YAML config"""
        try:
            with open(file_path, "r") as f:
                return yaml.safe_load(f)
        except:
            return {}

    def extract_dataset_info(self, data_file: str) -> Dict[str, Any]:
        """Extract information about a dataset file"""
        path = Path(data_file)
        info = {
            "path": str(path),
            "type": path.suffix,
            "size_mb": path.stat().st_size / (1024 * 1024) if path.exists() else 0,
            "sample_count": None,
        }

        # Try to get sample count for common formats
        if path.suffix == ".jsonl":
            try:
                with open(path, "r") as f:
                    info["sample_count"] = sum(1 for _ in f)
            except:
                pass
        elif path.suffix == ".csv":
            try:
                import csv

                with open(path, "r") as f:
                    info["sample_count"] = (
                        sum(1 for _ in csv.reader(f)) - 1
                    )  # -1 for header
            except:
                pass

        return info

    def analyze_config_for_params(
        self, config_data: Dict, target_params: List[str]
    ) -> Dict[str, Any]:
        """
        Extract values for target parameters from config
        """
        found = {}

        def search_nested(d, prefix=""):
            if not isinstance(d, dict):
                return

            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key

                # Check if this matches any target param
                for target in target_params:
                    if key.lower() == target.lower() or target.lower() in key.lower():
                        found[target] = {
                            "value": value,
                            "path": full_key,
                            "type": type(value).__name__,
                        }

                # Recurse for nested dicts
                if isinstance(value, dict):
                    search_nested(value, full_key)

        search_nested(config_data)
        return found

    def find_example_usage(self, func_name: str) -> List[Dict[str, Any]]:
        """
        Find example usages of a function in example/test scripts
        """
        examples = []

        for script_path in self.discovered["example_scripts"]:
            try:
                with open(script_path, "r") as f:
                    content = f.read()

                # Simple search for function calls
                if func_name in content:
                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            if (
                                isinstance(node.func, ast.Name)
                                and node.func.id == func_name
                            ):
                                # Extract the call
                                call_info = {
                                    "file": script_path,
                                    "args": [],
                                    "kwargs": {},
                                }

                                # Get positional args
                                for arg in node.args:
                                    try:
                                        call_info["args"].append(ast.literal_eval(arg))
                                    except:
                                        call_info["args"].append(
                                            ast.unparse(arg)
                                            if hasattr(ast, "unparse")
                                            else "..."
                                        )

                                # Get keyword args
                                for kw in node.keywords:
                                    try:
                                        call_info["kwargs"][kw.arg] = ast.literal_eval(
                                            kw.value
                                        )
                                    except:
                                        call_info["kwargs"][kw.arg] = (
                                            ast.unparse(kw.value)
                                            if hasattr(ast, "unparse")
                                            else "..."
                                        )

                                examples.append(call_info)
            except:
                pass

        return examples


class EnhancedMinimalInputGenerator(MinimalInputGenerator):
    """
    Enhanced generator that uses existing resources
    """

    def __init__(self, llm, project_dir="."):
        super().__init__(llm, project_dir)
        self.resource_discovery = ExistingResourceDiscovery(project_dir)
        self.resources = None

    def discover_resources(self):
        """Discover all available resources"""
        if self.resources is None:
            self.resources = self.resource_discovery.discover_all_resources()
        return self.resources
    
    def _find_model_import(self, func_spec: Dict[str, Any], file_path: str) -> Optional[Dict[str, str]]:
        """
        Try to find what model class is used in the target function.
        Returns import info if found, None otherwise.
        """
        try:
            with open(file_path, "r") as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Look for model-related imports
            model_imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if "model" in alias.name.lower():
                            model_imports.append({
                                "import_statement": f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""),
                                "class_name": alias.asname or alias.name.split(".")[-1]
                            })
                elif isinstance(node, ast.ImportFrom):
                    if node.module and "model" in node.module.lower():
                        for alias in node.names:
                            model_imports.append({
                                "import_statement": f"from {node.module} import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""),
                                "class_name": alias.asname or alias.name
                            })
            
            # Return the first model import found
            if model_imports:
                return model_imports[0]
                
        except Exception:
            pass
        
        return None

    def _extract_model_config_params(self, critical_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract model-related config parameters that should be minimized.
        """
        model_params = {}
        
        model_keywords = [
            "hidden", "dim", "layer", "head", "embed", "feature",
            "channel", "depth", "width", "size"
        ]
        
        for param in critical_params.get("critical_minimal", []):
            param_name = param["param"].lower()
            if any(kw in param_name for kw in model_keywords):
                model_params[param["param"]] = param["minimal_value"]
        
        # Also check config_attributes if present
        for attr, info in critical_params.get("config_attributes", {}).items():
            attr_lower = attr.lower()
            if any(kw in attr_lower for kw in model_keywords):
                model_params[attr] = info.get("minimal_value", info.get("default"))
        
        return model_params

    def find_config_values(
        self, critical_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Find actual values for critical params from existing configs
        """
        self.discover_resources()

        # Get list of critical param names
        param_names = [
            p["param"] for p in critical_params.get("critical_minimal", [])
        ]
        param_names += [
            p["param"] for p in critical_params.get("critical_default", [])
        ]

        found_values = {}

        # Search JSON configs
        for json_file in self.resources["json_configs"]:
            config_data = self.resource_discovery.load_json_config(json_file)
            matches = self.resource_discovery.analyze_config_for_params(
                config_data, param_names
            )

            for param, info in matches.items():
                if param not in found_values:
                    found_values[param] = {
                        "value": info["value"],
                        "source": json_file,
                        "path": info["path"],
                    }

        # Search YAML configs
        for yaml_file in self.resources["yaml_configs"]:
            config_data = self.resource_discovery.load_yaml_config(yaml_file)
            matches = self.resource_discovery.analyze_config_for_params(
                config_data, param_names
            )

            for param, info in matches.items():
                if param not in found_values:
                    found_values[param] = {
                        "value": info["value"],
                        "source": yaml_file,
                        "path": info["path"],
                    }

        return found_values

    def find_dataset_files(self) -> List[Dict[str, Any]]:
        """
        Find and analyze available dataset files
        """
        self.discover_resources()

        datasets = []
        for data_file in self.resources["data_files"]:
            info = self.resource_discovery.extract_dataset_info(data_file)
            datasets.append(info)

        # Sort by size (prefer smaller files for minimal testing)
        datasets.sort(key=lambda x: x["size_mb"])

        return datasets

    def generate_minimal_input_script_enhanced(
        self,
        func_key: str,
        func_spec: Dict[str, Any],
        critical_params: Dict[str, Any],
    ) -> str:
        """
        Generate script using discovered resources
        """
        file_path = func_spec["file"]
        func_name = func_spec["function"]

        # Find config values from existing files
        found_config_values = self.find_config_values(critical_params)

        # Find example usages
        examples = self.resource_discovery.find_example_usage(func_name)

        # Find datasets
        datasets = self.find_dataset_files()

        script = f'''"""
Minimal input script for testing {func_name}
Auto-generated using discovered project resources

Resources used:
'''

        if found_config_values:
            script += "  Config values from:\n"
            for param, info in list(found_config_values.items())[:5]:
                script += f"    - {param}: {info['source']}\n"

        if datasets:
            script += f"  Datasets found: {len(datasets)} files\n"

        if examples:
            script += f"  Example usages found: {len(examples)} instances\n"

        script += '"""\n\n'

        script += "import sys\nimport torch\nimport torch.nn as nn\nfrom pathlib import Path\n\n"
        script += "# Add project to path\n"
        script += "sys.path.insert(0, str(Path(__file__).parent))\n\n"

        # Import the function
        module_name = Path(file_path).stem
        script += f"from {module_name} import {func_name}\n"

        # Check if there's an existing config file we can import
        config_file_to_import = None
        for config_file in self.resources.get("config_files", []):
            if "config.py" in config_file:
                config_file_to_import = config_file
                break

        needs_config = any(inp["name"] == "config" for inp in func_spec.get("inputs", []))

        if needs_config:
            if config_file_to_import:
                # Import existing config and modify it
                config_module = Path(config_file_to_import).stem
                script += f"from {config_module} import TrainingConfig\n\n"

                script += "# Create minimal config by overriding defaults\n"
                script += "config = TrainingConfig()\n"

                # Override with minimal values
                for param in critical_params.get("critical_minimal", []):
                    param_name = param["param"]

                    # Use found value if available, otherwise use minimal
                    if param_name in found_config_values:
                        # Use the actual value but make it minimal
                        actual_value = found_config_values[param_name]["value"]
                        if isinstance(actual_value, int) and param["minimal_value"] < actual_value:
                            script += f"config.{param_name} = {param['minimal_value']}  # Minimal override (original: {actual_value})\n"
                        else:
                            script += f"config.{param_name} = {repr(actual_value)}  # From {found_config_values[param_name]['source']}\n"
                    else:
                        script += f"config.{param_name} = {param['minimal_value']}  # {param['reason']}\n"

                script += "\n"
            else:
                # Generate minimal config class
                script += "# Minimal configuration\n"
                script += "class MinimalConfig:\n"
                script += "    '''Minimal config for testing'''\n"

                for param in critical_params.get("critical_minimal", []):
                    param_name = param["param"]

                    if param_name in found_config_values:
                        value = found_config_values[param_name]["value"]
                        script += f"    {param_name} = {repr(value)}  # From {found_config_values[param_name]['source']}\n"
                    else:
                        script += f"    {param_name} = {param['minimal_value']}  # {param['reason']}\n"

                # Add defaults
                script += "    learning_rate = 0.001\n"
                script += "    output_dir = './test_output'\n"

                # Check if dataset path found
                if datasets:
                    smallest_dataset = datasets[0]
                    script += f"    dataset = '{smallest_dataset['path']}'  # Smallest dataset found ({smallest_dataset['size_mb']:.2f} MB)\n"
                else:
                    script += "    dataset = 'minimal_test'\n"

                script += "    \n"
                script += "    def dict(self):\n"
                script += "        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}\n"
                script += "\nconfig = MinimalConfig()\n\n"

        # Model generation
        # Model generation - prefer using the actual model with minimal config
        needs_model = any(
            inp["name"] == "model" or "model" in (inp.get("type") or "").lower()
            for inp in func_spec.get("inputs", [])
        )

        if needs_model:
            # Check if we can identify the actual model class from the codebase
            model_import = self._find_model_import(func_spec, file_path)
            
            if model_import:
                script += "# Use actual model with minimal configuration\n"
                script += f"{model_import['import_statement']}\n"
                script += "\n# Create model with minimal dimensions\n"
                
                # If we found model config params, use them with minimal values
                model_config_params = self._extract_model_config_params(critical_params)
                if model_config_params:
                    script += "# Minimal model config overrides:\n"
                    for param, value in model_config_params.items():
                        script += f"# - {param}: {value}\n"
                
                script += f"model = {model_import['class_name']}(config)  # Uses config with minimal values\n\n"
            else:
                # Fallback: just note that model should come from config
                script += "# NOTE: Model should be created by the training function using config\n"
                script += "# The config above contains minimal model parameters\n"
                script += "# If model parameter is required, uncomment below:\n"
                script += "# model = None  # Let the function create it from config\n\n"

        # Dataloader generation
        needs_loaders = any(
            "loader" in inp["name"].lower() or "data" in inp["name"].lower()
            for inp in func_spec.get("inputs", [])
        )

        if needs_loaders or "dataloader" in func_spec.get("expensive_ops", []):
            # Check if we found a real dataset we can use
            if datasets and datasets[0]["size_mb"] < 1.0:  # Less than 1MB
                script += f"# Using smallest dataset found: {datasets[0]['path']}\n"
                script += f"# You can load this real data or use synthetic data below\n\n"

            batch_size = next(
                (
                    p["minimal_value"]
                    for p in critical_params.get("critical_minimal", [])
                    if "batch" in p["param"].lower()
                ),
                2,
            )
            num_samples = next(
                (
                    p["minimal_value"]
                    for p in critical_params.get("critical_minimal", [])
                    if "train" in p["param"].lower() or "sample" in p["param"].lower()
                ),
                5,
            )

            script += self.generate_synthetic_dataloader(batch_size, num_samples, func_spec)
            script += "\n"

        # Show example usage if found
        if examples:
            script += "# Example usage found in codebase:\n"
            for i, example in enumerate(examples[:2], 1):
                script += f"# Example {i} from {example['file']}:\n"
                script += f"#   Args: {example['args']}\n"
                script += f"#   Kwargs: {list(example['kwargs'].keys())}\n"
            script += "\n"

        # Generate function call
        script += f"# Call {func_name} with minimal inputs\n"
        script += "if __name__ == '__main__':\n"
        script += "    print('Running minimal test...')\n"
        script += '    print(f\'Config: epochs={config.epochs if hasattr(config, "epochs") else "N/A"}, batch_size={config.batch_size if hasattr(config, "batch_size") else "N/A"}\')\n'
        script += "    \n"

        # Build argument list
        # Build argument list
        args = []
        model_import = self._find_model_import(func_spec, file_path) if needs_model else None
        
        for inp in func_spec.get("inputs", []):
            param_name = inp["name"]
            if param_name == "config":
                args.append("config=config")
            elif param_name == "model":
                # Only add model if we have one; many training functions create model internally
                if model_import:
                    args.append("model=model")
                else:
                    # Skip - let function create model from config
                    args.append("model=None  # Let function create from config")
            elif "loader" in param_name.lower():
                args.append(f"{param_name}=train_val_test_loaders")
            elif inp.get("default") is not None:
                continue
            else:
                if inp.get("type") == "int":
                    args.append(f"{param_name}=0")
                elif inp.get("type") == "str":
                    args.append(f'{param_name}=""')
                else:
                    args.append(f"{param_name}=None")

        script += f"    result = {func_name}(\n"
        for arg in args:
            script += f"        {arg},\n"
        script += "    )\n"
        script += "    \n"
        script += "    print(f'✓ Test completed! Result type: {type(result)}')\n"

        return script


# ============================================================================
# DATA FLOW TRACING AND PARAMETER RESOLUTION CLASSES
# ============================================================================


class DataFlowTracer:
    """
    Trace data flow to find where parameters like n_train, n_val, dataset come from
    """

    def __init__(self, project_dir="."):
        self.project_dir = Path(project_dir)
        self.file_asts = {}

    def parse_file(self, file_path: str) -> ast.AST:
        """Parse and cache file AST"""
        if file_path not in self.file_asts:
            with open(file_path, "r", encoding="utf-8") as f:
                self.file_asts[file_path] = ast.parse(f.read())
        return self.file_asts[file_path]

    def find_dataloader_creation(
        self, file_path: str, func_name: str
    ) -> Dict[str, Any]:
        """
        Find where dataloaders are created and what parameters they use
        """
        tree = self.parse_file(file_path)

        dataloader_info = {
            "found": False,
            "creation_function": None,
            "parameters": {},
            "dataset_source": None,
        }

        # Look for functions that create dataloaders
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function name suggests dataloader creation
                if any(
                    kw in node.name.lower()
                    for kw in ["dataload", "get_loader", "get_data", "setup_data"]
                ):
                    dataloader_info["found"] = True
                    dataloader_info["creation_function"] = node.name

                    # Extract parameters
                    for arg in node.args.args:
                        dataloader_info["parameters"][arg.arg] = None

                    # Look for dataset loading inside
                    for subnode in ast.walk(node):
                        # Look for calls to DataLoader
                        if isinstance(subnode, ast.Call):
                            if isinstance(subnode.func, ast.Name):
                                if (
                                    "DataLoader" in subnode.func.id
                                    or "Dataset" in subnode.func.id
                                ):
                                    # Extract keyword arguments
                                    for kw in subnode.keywords:
                                        try:
                                            value = ast.literal_eval(kw.value)
                                            dataloader_info["parameters"][
                                                kw.arg
                                            ] = value
                                        except:
                                            pass

                        # Look for file loading
                        if isinstance(subnode, ast.Call):
                            if isinstance(subnode.func, ast.Attribute):
                                if subnode.func.attr in ["read_csv", "load", "open"]:
                                    if subnode.args:
                                        try:
                                            dataloader_info["dataset_source"] = (
                                                ast.literal_eval(subnode.args[0])
                                            )
                                        except:
                                            pass

        return dataloader_info

    def trace_parameter_origin(
        self,
        file_path: str,
        func_name: str,
        param_name: str,
        flow_graph: Dict = None,
    ) -> Dict[str, Any]:
        """
        Trace where a parameter value actually comes from
        """
        origin = {
            "param": param_name,
            "source": "unknown",
            "derivation": None,
            "default_value": None,
            "depends_on": [],
        }

        tree = self.parse_file(file_path)

        # Find the function
        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                func_node = node
                break

        if not func_node:
            return origin

        # Look for assignments to this parameter
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == param_name:
                        # Found assignment!
                        origin["source"] = "computed"

                        # Try to understand the computation
                        if isinstance(node.value, ast.BinOp):
                            # Arithmetic operation
                            origin["derivation"] = (
                                ast.unparse(node.value)
                                if hasattr(ast, "unparse")
                                else "computed"
                            )

                            # Extract dependencies
                            for dep_node in ast.walk(node.value):
                                if isinstance(dep_node, ast.Name):
                                    origin["depends_on"].append(dep_node.id)

                        elif isinstance(node.value, ast.Call):
                            # Function call
                            if isinstance(node.value.func, ast.Name):
                                origin["derivation"] = f"from {node.value.func.id}()"
                            elif isinstance(node.value.func, ast.Attribute):
                                origin["derivation"] = (
                                    f"from .{node.value.func.attr}()"
                                )

                        elif isinstance(node.value, ast.Attribute):
                            # Attribute access (e.g., config.n_train)
                            if isinstance(node.value.value, ast.Name):
                                origin["source"] = "config_attribute"
                                origin["derivation"] = (
                                    f"{node.value.value.id}.{node.value.attr}"
                                )
                                origin["depends_on"].append(node.value.value.id)

        # If not found in function, check if it's a parameter with default
        for arg in func_node.args.args:
            if arg.arg == param_name:
                defaults_offset = len(func_node.args.args) - len(
                    func_node.args.defaults
                )
                arg_idx = func_node.args.args.index(arg)

                if arg_idx >= defaults_offset:
                    default_idx = arg_idx - defaults_offset
                    try:
                        origin["source"] = "function_default"
                        origin["default_value"] = ast.literal_eval(
                            func_node.args.defaults[default_idx]
                        )
                    except:
                        origin["default_value"] = (
                            ast.unparse(func_node.args.defaults[default_idx])
                            if hasattr(ast, "unparse")
                            else None
                        )

        return origin

    def find_dataset_splitting_logic(self, file_path: str) -> Dict[str, Any]:
        """
        Find how datasets are split into train/val/test
        """
        tree = self.parse_file(file_path)

        split_info = {"found": False, "method": None, "ratios": {}, "sizes": {}}

        with open(file_path, "r") as f:
            content = f.read()

        # Look for common patterns
        patterns = {
            "train_ratio": r"train_ratio\s*=\s*([\d.]+)",
            "val_ratio": r"val_ratio\s*=\s*([\d.]+)",
            "test_ratio": r"test_ratio\s*=\s*([\d.]+)",
            "train_size": r"train_size\s*=\s*(\d+)",
            "val_size": r"val_size\s*=\s*(\d+)",
            "test_size": r"test_size\s*=\s*(\d+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                split_info["found"] = True
                try:
                    value = float(match.group(1))
                    if "ratio" in key:
                        split_info["ratios"][key] = value
                    else:
                        split_info["sizes"][key] = int(value)
                except:
                    pass

        # Look for sklearn train_test_split
        if "train_test_split" in content:
            split_info["method"] = "sklearn.train_test_split"

            # Try to find test_size parameter
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Name)
                        and "train_test_split" in node.func.id
                    ):
                        for kw in node.keywords:
                            if kw.arg in ["test_size", "train_size"]:
                                try:
                                    split_info["ratios"][kw.arg] = ast.literal_eval(
                                        kw.value
                                    )
                                except:
                                    pass

        return split_info


class SmartParameterResolver:
    """
    Intelligently resolve parameter values by following data flow and using LLM
    """

    def __init__(self, llm, project_dir="."):
        self.llm = llm
        self.tracer = DataFlowTracer(project_dir)
        self.resource_discovery = ExistingResourceDiscovery(project_dir)

    """
    Fix for system_minimal_inputs.py - SmartParameterResolver.resolve_derived_parameters
    """

    def resolve_derived_parameters(
        self,
        func_spec: Dict[str, Any],
        critical_params: Dict[str, Any],
        flow_graph: Dict = None,
    ) -> Dict[str, Any]:
        """
        Resolve values for derived parameters like n_train, n_val, n_test
        """
        file_path = func_spec["file"]
        func_name = func_spec["function"]

        print(f"\n   🔍 Resolving derived parameters...")

        resolved = {}
        derived_params = critical_params.get("derived", [])

        # FIX: Extract parameter names from dicts
        derived_param_names = []
        for p in derived_params:
            if isinstance(p, dict):
                param_name = p.get("param", "")
                if param_name:
                    derived_param_names.append(param_name)
            elif isinstance(p, str):
                derived_param_names.append(p)
        
        # Trace each derived parameter
        for param in derived_param_names:
            origin = self.tracer.trace_parameter_origin(
                file_path, func_name, param, flow_graph
            )

            print(f"     - {param}: {origin['source']}")

            if origin["source"] == "config_attribute":
                resolved[param] = {
                    "source": "config",
                    "derivation": origin["derivation"],
                    "minimal_value": None,
                }

            elif origin["source"] == "computed":
                resolved[param] = {
                    "source": "computed",
                    "derivation": origin["derivation"],
                    "depends_on": origin["depends_on"],
                }

            elif origin["source"] == "function_default":
                resolved[param] = {
                    "source": "default",
                    "value": origin["default_value"],
                }

        # Find dataset splitting logic
        split_info = self.tracer.find_dataset_splitting_logic(file_path)

        if split_info["found"]:
            print(f"     Found dataset split info: {split_info['ratios']}")

            if split_info["ratios"]:
                total_minimal = 10
                train_ratio = split_info["ratios"].get("train_ratio", 0.8)
                val_ratio = split_info["ratios"].get("val_ratio", 0.1)
                test_ratio = split_info["ratios"].get("test_ratio", 0.1)

                if "n_train" in resolved:
                    resolved["n_train"]["minimal_value"] = max(
                        2, int(total_minimal * train_ratio)
                    )
                if "n_val" in resolved:
                    resolved["n_val"]["minimal_value"] = max(
                        1, int(total_minimal * val_ratio)
                    )
                if "n_test" in resolved:
                    resolved["n_test"]["minimal_value"] = max(
                        1, int(total_minimal * test_ratio)
                    )

        # FIX: Use string parameter names
        unresolved = [
            p
            for p in derived_param_names
            if p not in resolved or resolved[p].get("minimal_value") is None
        ]

        if unresolved:
            resolved_via_llm = self._resolve_with_llm(
                func_spec, unresolved, resolved
            )

            for param, value in resolved_via_llm.items():
                if param in resolved:
                    resolved[param]["minimal_value"] = value
                else:
                    resolved[param] = {
                        "source": "llm_inference",
                        "minimal_value": value,
                    }

        return resolved
    

    def _resolve_with_llm(
        self,
        func_spec: Dict[str, Any],
        unresolved_params: List[str],
        partial_resolved: Dict,
    ) -> Dict[str, int]:
        """
        Use LLM to infer minimal values for unresolved parameters
        """

        prompt = f"""Given this function and its parameters, determine minimal values for testing.

FUNCTION: {func_spec['function']}
PURPOSE: {func_spec['purpose']}

UNRESOLVED PARAMETERS: {unresolved_params}

CONTEXT:
- This is for minimal testing (quick execution)
- For dataset sizes: aim for 5-10 total samples
- For split sizes (n_train, n_val, n_test): should sum to small total
- Common pattern: n_train=5, n_val=2, n_test=2

ALREADY RESOLVED:
{json.dumps(partial_resolved, indent=2)}

Output JSON with minimal integer values:
{{"n_train": 5, "n_val": 2, "n_test": 2}}
"""

        messages = [
            {
                "role": "system",
                "content": "You determine minimal parameter values for testing. Output only JSON.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            output = call_llm_raw(self.llm, messages, max_tokens=512)

            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                items = parse_json_stream_safe(output)

            if items:
                return items[0]
        except:
            pass

        # Fallback defaults
        defaults = {
            "n_train": 5,
            "n_val": 2,
            "n_test": 2,
            "train_size": 5,
            "val_size": 2,
            "test_size": 2,
            "num_samples": 10,
            "dataset_size": 10,
        }

        return {p: defaults.get(p, 5) for p in unresolved_params}

    def find_orchestrator_defaults(
        self, flow_graph: Dict, func_name: str
    ) -> Dict[str, Any]:
        """
        Look in orchestrator functions for how they call this function
        """

        defaults = {}

        # Find who calls this function
        callers = []
        for caller_name, info in flow_graph.items():
            if func_name in info.get("calls", []):
                callers.append((caller_name, info.get("file")))

        if not callers:
            return defaults

        print(f"\n   📞 Found callers of {func_name}:")

        # Analyze each caller
        for caller_name, caller_file in callers[:3]:  # Max 3 callers
            if not caller_file or caller_file == "external":
                continue

            print(f"     - {caller_name} in {caller_file}")

            try:
                tree = self.tracer.parse_file(caller_file)

                # Find the caller function
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == caller_name:
                        # Look for calls to our target function
                        for call_node in ast.walk(node):
                            if isinstance(call_node, ast.Call):
                                # Check if this calls our function
                                is_target_call = False
                                if (
                                    isinstance(call_node.func, ast.Name)
                                    and call_node.func.id == func_name
                                ):
                                    is_target_call = True

                                if is_target_call:
                                    # Extract keyword arguments
                                    for kw in call_node.keywords:
                                        try:
                                            value = ast.literal_eval(kw.value)
                                            defaults[kw.arg] = value
                                            print(f"       Found: {kw.arg}={value}")
                                        except:
                                            # Try to unparse
                                            if hasattr(ast, "unparse"):
                                                defaults[kw.arg] = ast.unparse(kw.value)
            except:
                pass

        return defaults


class ImprovedDataFlowTracer(DataFlowTracer):
    """
    Enhanced tracer that understands config objects and their attributes
    """

    def trace_config_attributes(
        self,
        file_path: str,
        func_name: str,
        config_param_name: str = "config",
    ) -> Dict[str, Any]:
        """
        Trace how config attributes are used in the function
        Returns which config attributes are accessed and how
        """

        tree = self.parse_file(file_path)

        # Find the function
        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                func_node = node
                break

        if not func_node:
            return {}

        config_attrs_used = {}

        # Walk through function body looking for config.attribute accesses
        for node in ast.walk(func_node):
            if isinstance(node, ast.Attribute):
                # Check if this is config.something
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id == config_param_name
                ):
                    attr_name = node.attr

                    if attr_name not in config_attrs_used:
                        config_attrs_used[attr_name] = {
                            "count": 0,
                            "usage_contexts": [],
                        }

                    config_attrs_used[attr_name]["count"] += 1

        return config_attrs_used

    def find_config_class_definition(
        self, config_file: str, class_name: str = None
    ) -> Dict[str, Any]:
        """
        Find config class and extract all its attributes with defaults
        """

        tree = self.parse_file(config_file)

        config_attrs = {}

        # Find config classes (TrainingConfig, ModelConfig, etc.)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                # If no class_name specified, take any config-like class
                if class_name is None and "config" in node.name.lower():
                    class_name = node.name

                if class_name and node.name == class_name:
                    # Extract all attributes
                    for item in node.body:
                        # Type annotated attributes: attr: int = 5
                        if isinstance(item, ast.AnnAssign) and isinstance(
                            item.target, ast.Name
                        ):
                            attr_name = item.target.id
                            attr_type = (
                                ast.unparse(item.annotation)
                                if hasattr(ast, "unparse")
                                else None
                            )
                            default_val = None

                            if item.value:
                                try:
                                    default_val = ast.literal_eval(item.value)
                                except:
                                    default_val = (
                                        ast.unparse(item.value)
                                        if hasattr(ast, "unparse")
                                        else None
                                    )

                            config_attrs[attr_name] = {
                                "type": attr_type,
                                "default": default_val,
                            }

                        # Simple assignments: attr = 5
                        elif isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    try:
                                        config_attrs[target.id] = {
                                            "type": None,
                                            "default": ast.literal_eval(item.value),
                                        }
                                    except:
                                        config_attrs[target.id] = {
                                            "type": None,
                                            "default": (
                                                ast.unparse(item.value)
                                                if hasattr(ast, "unparse")
                                                else None
                                            ),
                                        }

        return config_attrs


class EnhancedParameterResolver(SmartParameterResolver):
    """
    Enhanced resolver that understands config object structure
    """

    def __init__(self, llm, project_dir="."):
        super().__init__(llm, project_dir)
        self.tracer = ImprovedDataFlowTracer(project_dir)

    def resolve_config_based_parameters(
        self, func_spec: Dict[str, Any], critical_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve parameters that come from config object
        """

        file_path = func_spec["file"]
        func_name = func_spec["function"]

        print(f"\n   🔍 Analyzing config-based parameters...")

        # Check if function takes a config parameter
        config_param = None
        for inp in func_spec.get("inputs", []):
            if inp["name"] == "config" or "config" in inp["name"].lower():
                config_param = inp
                break

        if not config_param:
            print(f"     No config parameter found")
            return {}

        # Find what config attributes are used
        config_attrs_used = self.tracer.trace_config_attributes(
            file_path, func_name, config_param["name"]
        )

        print(f"     Found {len(config_attrs_used)} config attributes used:")
        for attr in list(config_attrs_used.keys())[:10]:
            print(f"       - config.{attr}")

        if len(config_attrs_used) > 10:
            print(f"       ... and {len(config_attrs_used) - 10} more")

        # Find the config class definition
        config_file = (
            config_param.get("config_path", "").split("::")[0]
            if config_param.get("config_path")
            else None
        )

        if not config_file or not Path(config_file).exists():
            # Search for config.py
            config_files = list(Path(".").glob("**/config.py"))
            if config_files:
                config_file = str(config_files[0])

        config_class_attrs = {}
        if config_file and Path(config_file).exists():
            print(f"     Analyzing config file: {config_file}")
            config_class_attrs = self.tracer.find_config_class_definition(config_file)
            print(f"     Found {len(config_class_attrs)} attributes in config class")

        # Match used attributes with definitions
        resolved_config_attrs = {}

        for attr in config_attrs_used.keys():
            if attr in config_class_attrs:
                attr_info = config_class_attrs[attr]

                # Determine if this should be minimal
                minimal_value = self._determine_minimal_value(attr, attr_info)

                resolved_config_attrs[attr] = {
                    "source": "config_class",
                    "type": attr_info["type"],
                    "default": attr_info["default"],
                    "minimal_value": minimal_value,
                }

        return resolved_config_attrs

    def _determine_minimal_value(self, attr_name: str, attr_info: Dict) -> Any:
        """
        Determine minimal value for a config attribute
        """

        default = attr_info.get("default")
        attr_type = attr_info.get("type", "")

        # Patterns for minimal values
        if any(kw in attr_name.lower() for kw in ["epoch", "n_epoch", "num_epoch"]):
            return 1

        elif any(kw in attr_name.lower() for kw in ["batch_size", "batch"]):
            return 2

        elif any(
            kw in attr_name.lower()
            for kw in ["n_train", "train_size", "num_train"]
        ):
            return 5

        elif any(
            kw in attr_name.lower()
            for kw in ["n_val", "val_size", "num_val", "n_valid"]
        ):
            return 2

        elif any(
            kw in attr_name.lower() for kw in ["n_test", "test_size", "num_test"]
        ):
            return 2

        elif any(kw in attr_name.lower() for kw in ["num_worker", "n_worker"]):
            return 0

        elif any(kw in attr_name.lower() for kw in ["max_neighbor", "neighbor"]):
            return 5

        elif any(kw in attr_name.lower() for kw in ["hidden", "dim", "embed"]):
            if isinstance(default, int) and default > 32:
                return 8  # Minimal hidden dimension
            return default

        elif any(kw in attr_name.lower() for kw in ["layer", "n_layer", "num_layer"]):
            return 1

        elif "ratio" in attr_name.lower():
            # Keep ratios as-is
            return default

        elif attr_name.lower() in ["learning_rate", "lr"]:
            return default or 0.001

        # For boolean, strings, keep default
        elif attr_type in ["bool", "str"] or isinstance(default, (bool, str)):
            return default

        # For paths, keep default
        elif (
            "path" in attr_name.lower()
            or "dir" in attr_name.lower()
            or "file" in attr_name.lower()
        ):
            return default

        # For None defaults, keep None
        elif default is None:
            return None

        # Otherwise, if it's a number and large, make it small
        elif isinstance(default, int) and default > 10:
            return min(10, default)

        # Default: keep the default value
        return default

    def resolve_all_parameters(
        self,
        func_spec: Dict[str, Any],
        critical_params: Dict[str, Any],
        flow_graph: Dict = None,
    ) -> Dict[str, Any]:
        """
        Complete parameter resolution combining all methods
        """

        all_resolved = {
            "critical_minimal": critical_params.get("critical_minimal", []),
            "critical_default": critical_params.get("critical_default", []),
            "config_attributes": {},
            "derived": {},
            "orchestrator_defaults": {},
        }

        # 1. Resolve config-based parameters
        config_resolved = self.resolve_config_based_parameters(
            func_spec, critical_params
        )
        all_resolved["config_attributes"] = config_resolved

        # Add config attributes to critical_minimal if they need minimal values
        for attr, info in config_resolved.items():
            if info["minimal_value"] != info["default"]:
                # This needs to be set to minimal value
                all_resolved["critical_minimal"].append(
                    {
                        "param": f"config.{attr}",
                        "minimal_value": info["minimal_value"],
                        "default_value": info["default"],
                        "reason": f'config attribute (default: {info["default"]})',
                    }
                )

        # 2. Resolve derived parameters (original method)
        derived_resolved = super().resolve_derived_parameters(
            func_spec, critical_params, flow_graph
        )
        all_resolved["derived"] = derived_resolved

        # 3. Find orchestrator defaults
        orchestrator_defaults = (
            super().find_orchestrator_defaults(flow_graph, func_spec["function"])
            if flow_graph
            else {}
        )
        all_resolved["orchestrator_defaults"] = orchestrator_defaults

        return all_resolved


def generate_complete_minimal_inputs(
    expensive_functions: List[Dict],
    flow_graphs: Dict,
    llm,
    project_dir: str = ".",
) -> Dict[str, Any]:
    """
    Complete pipeline with smart parameter resolution
    """

    generator = EnhancedMinimalInputGenerator(llm, project_dir)
    resolver = SmartParameterResolver(llm, project_dir)

    generator.discover_resources()

    print("\n" + "=" * 80)
    print("COMPLETE MINIMAL INPUT GENERATION")
    print("=" * 80)

    all_specs = {}

    for func in expensive_functions:
        func_key = f"{func['file']}::{func['function']}"
        print(f"\n{'='*80}")
        print(f"Processing: {func_key}")
        print(f"{'='*80}")

        # Get function code
        try:
            with open(func["file"], "r") as f:
                content = f.read()
                tree = ast.parse(content)

            func_code = ""
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func["function"]:
                    func_code = ast.get_source_segment(content, node) or ""
                    break
        except:
            func_code = ""

        # Get flow graph
        relevant_flow = None
        for flow in flow_graphs.values():
            if func["function"] in flow:
                relevant_flow = flow
                break

        # Get full spec
        func_spec = generator.extractor.generate_minimal_input_spec(func, relevant_flow)

        # Identify critical parameters
        critical_params = generator.identify_critical_parameters(func_spec, func_code)

        # NEW: Resolve derived parameters
        derived_resolved = resolver.resolve_derived_parameters(
            func_spec, critical_params, relevant_flow
        )

        # NEW: Find defaults from orchestrator
        orchestrator_defaults = resolver.find_orchestrator_defaults(
            relevant_flow, func["function"]
        )

        if orchestrator_defaults:
            print(
                f"\n   📋 Defaults from orchestrator: {list(orchestrator_defaults.keys())}"
            )

        # Merge resolved derived params into critical params
        for param, info in derived_resolved.items():
            if info.get("minimal_value") is not None:
                critical_params["critical_minimal"].append(
                    {
                        "param": param,
                        "minimal_value": info["minimal_value"],
                        "reason": f"derived: {info.get('derivation', 'computed')}",
                    }
                )

        # Add orchestrator defaults
        for param, value in orchestrator_defaults.items():
            if not any(
                p["param"] == param
                for p in critical_params.get("critical_minimal", [])
            ):
                critical_params["critical_default"].append(
                    {
                        "param": param,
                        "default_value": value,
                        "reason": "from orchestrator call",
                    }
                )

        # Find config values
        found_values = generator.find_config_values(critical_params)

        print(f"\n   📊 Final parameter count:")
        print(
            f"     - Critical minimal: {len(critical_params.get('critical_minimal', []))}"
        )
        print(
            f"     - Critical default: {len(critical_params.get('critical_default', []))}"
        )
        print(f"     - Config values found: {len(found_values)}")

        # Generate script
        script = generator.generate_minimal_input_script_enhanced(
            func_key, func_spec, critical_params
        )

        all_specs[func_key] = {
            "script": script,
            "critical_params": critical_params,
            "derived_resolved": derived_resolved,
            "orchestrator_defaults": orchestrator_defaults,
            "found_config_values": found_values,
        }

        # Save script
        safe_name = func["function"] + "_minimal_test.py"
        with open(safe_name, "w") as f:
            f.write(script)

        print(f"\n   ✅ Saved to {safe_name}")

    return all_specs


def generate_complete_minimal_inputs_v2(
    expensive_functions: List[Dict],
    flow_graphs: Dict,
    llm,
    project_dir: str = ".",
) -> Dict[str, Any]:
    """
    V2: Complete pipeline with config attribute resolution
    """

    generator = EnhancedMinimalInputGenerator(llm, project_dir)
    resolver = EnhancedParameterResolver(llm, project_dir)

    generator.discover_resources()

    print("\n" + "=" * 80)
    print("COMPLETE MINIMAL INPUT GENERATION V2")
    print("=" * 80)

    all_specs = {}

    for func in expensive_functions:
        func_key = f"{func['file']}::{func['function']}"
        print(f"\n{'='*80}")
        print(f"Processing: {func_key}")
        print(f"{'='*80}")

        # Get function code
        try:
            with open(func["file"], "r") as f:
                content = f.read()
                tree = ast.parse(content)

            func_code = ""
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func["function"]:
                    func_code = ast.get_source_segment(content, node) or ""
                    break
        except:
            func_code = ""

        # Get flow graph
        relevant_flow = None
        for flow in flow_graphs.values():
            if func["function"] in flow:
                relevant_flow = flow
                break

        # Get full spec
        func_spec = generator.extractor.generate_minimal_input_spec(func, relevant_flow)

        # Identify critical parameters
        critical_params = generator.identify_critical_parameters(func_spec, func_code)

        # NEW: Complete resolution
        all_resolved = resolver.resolve_all_parameters(
            func_spec, critical_params, relevant_flow
        )

        # Print summary
        print(f"\n   📊 Resolution Summary:")
        print(
            f"     - Direct minimal params: {len(all_resolved['critical_minimal'])}"
        )
        print(f"     - Config attributes: {len(all_resolved['config_attributes'])}")

        if all_resolved["config_attributes"]:
            print(f"\n     Config attributes needing minimal values:")
            for attr, info in list(all_resolved["config_attributes"].items())[:10]:
                if info["minimal_value"] != info["default"]:
                    print(
                        f"       - {attr}: {info['default']} → {info['minimal_value']}"
                    )

        # Update critical_params with all resolved info
        critical_params["critical_minimal"] = all_resolved["critical_minimal"]
        critical_params["critical_default"] = all_resolved["critical_default"]
        critical_params["config_attributes"] = all_resolved["config_attributes"]

        # Generate script
        script = generator.generate_minimal_input_script_enhanced(
            func_key, func_spec, critical_params
        )

        all_specs[func_key] = {
            "script": script,
            "all_resolved": all_resolved,
            "func_spec": func_spec,
        }

        # Save script
        safe_name = func["function"] + "_minimal_test.py"
        with open(safe_name, "w") as f:
            f.write(script)

        print(f"\n   ✅ Saved to {safe_name}")

    return all_specs
