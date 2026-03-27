#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

REQUIRED_MODULES = {
    "pydantic": "pydantic",
    "langchain": "langchain",
    "langchain-core": "langchain_core",
    "PyYAML": "yaml",
    "pyflakes": "pyflakes",
    "torch": "torch",
    "llama-cpp-python": "llama_cpp",
}

OPTIONAL_MODULES = {
    "python-magic": "magic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight checks for the autonomous debugging agent")
    parser.add_argument("--root-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--model-path", default=os.environ.get("DEBUG_AGENT_MODEL_PATH", ""))
    parser.add_argument("--strict-model", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()

    print("=" * 80)
    print("AGENT PREFLIGHT")
    print("=" * 80)
    print(f"Python: {sys.executable}")
    print(f"Root dir: {root_dir}")

    failures: list[str] = []

    for package_name, module_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
            print(f"[OK] {package_name}")
        except Exception as exc:
            failures.append(f"{package_name}: {exc}")
            print(f"[MISSING] {package_name}: {exc}")

    for package_name, module_name in OPTIONAL_MODULES.items():
        try:
            importlib.import_module(module_name)
            print(f"[OK] optional {package_name}")
        except Exception as exc:
            print(f"[WARN] optional {package_name}: {exc}")

    launcher = root_dir / 'system_debug_cli.py'
    if launcher.exists():
        print(f"[OK] launcher: {launcher}")
    else:
        failures.append(f"launcher missing: {launcher}")
        print(f"[MISSING] launcher: {launcher}")

    model_path = args.model_path.strip()
    if model_path:
        if Path(model_path).exists():
            print(f"[OK] model: {model_path}")
        else:
            message = f"model path not found: {model_path}"
            if args.strict_model:
                failures.append(message)
                print(f"[MISSING] {message}")
            else:
                print(f"[WARN] {message}")
    else:
        default_model = root_dir / 'llama.cpp' / 'IQuest_Coder_V1_40B_Q4_K_M.gguf'
        if default_model.exists():
            print(f"[OK] model: {default_model}")
        else:
            print(f"[WARN] model not configured. Set DEBUG_AGENT_MODEL_PATH or place the GGUF at {default_model}")

    if failures:
        print("\nPreflight failed.")
        for item in failures:
            print(f"- {item}")
        return 1

    print("\nPreflight passed.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
