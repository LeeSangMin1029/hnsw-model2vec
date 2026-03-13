#!/usr/bin/env python3
"""Extract function definitions and call edges from Python source files.

Usage:
    python3 python_calls.py <root_dir>

Outputs JSON to stdout:
    {"module::Class::method": ["other_module::func", ...], ...}

Normalisation rules:
  - Module name = relative path without .py, slashes → ::
  - Top-level function:   module::func
  - Class method:          module::Class::method
  - Calls resolved via:
      1. self.method()        → same_module::OwningClass::method
      2. module.func()        → module::func  (if module is imported)
      3. Class.method()       → resolve via imports or same module
      4. bare func()          → same_module::func (if defined there)
"""

import ast
import json
import os
import sys
from pathlib import Path


def module_name(root: Path, filepath: Path) -> str:
    """Convert file path to module-style name (:: separated)."""
    try:
        rel = filepath.relative_to(root)
    except ValueError:
        rel = filepath
    parts = rel.with_suffix("").parts
    # Drop __init__ from package paths
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return "::".join(parts)


def extract_calls(root: Path, filepath: Path):
    """Parse a single Python file and return (definitions, call_map).

    definitions: set of fully qualified names defined in this file
    call_map: dict caller_fqn -> list of callee references (best-effort resolved)
    """
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return set(), {}

    mod = module_name(root, filepath)
    definitions = set()
    call_map = {}

    # Collect top-level and class-level function/method definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Determine owning class (if any)
            fqn = _function_fqn(tree, node, mod)
            definitions.add(fqn)

    # Now walk again to extract calls per function
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fqn = f"{mod}::{node.name}"
            calls = _collect_calls(node, mod, None)
            if calls:
                call_map[fqn] = calls
        elif isinstance(node, ast.ClassDef):
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fqn = f"{mod}::{node.name}::{item.name}"
                    calls = _collect_calls(item, mod, node.name)
                    if calls:
                        call_map[fqn] = calls

    return definitions, call_map


def _function_fqn(tree, func_node, mod_name):
    """Get fully qualified name for a function node."""
    # Check if function is inside a class
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in ast.iter_child_nodes(node):
                if item is func_node:
                    return f"{mod_name}::{node.name}::{func_node.name}"
    return f"{mod_name}::{func_node.name}"


def _collect_calls(func_node, mod_name, class_name):
    """Collect call references inside a function body."""
    calls = []
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        callee = _resolve_call(node, mod_name, class_name)
        if callee:
            calls.append(callee)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in calls:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _resolve_call(call_node, mod_name, class_name):
    """Best-effort resolution of a call target to a qualified name."""
    func = call_node.func

    # self.method() → mod::Class::method
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            if func.value.id == "self" and class_name:
                return f"{mod_name}::{class_name}::{func.attr}"
            if func.value.id == "cls" and class_name:
                return f"{mod_name}::{class_name}::{func.attr}"
            if func.value.id == "super" and class_name:
                # super().method() — can't resolve parent, use placeholder
                return None
            # obj.method() — could be module.func or instance.method
            # Return as module_or_var::method for later resolution
            return f"{func.value.id}::{func.attr}"
        # Chained: a.b.method() — skip for now
        return None

    # bare func()
    if isinstance(func, ast.Name):
        name = func.id
        # Skip builtins
        if name in _BUILTINS:
            return None
        # Assume same module
        return f"{mod_name}::{name}"

    return None


_BUILTINS = frozenset({
    "print", "len", "range", "enumerate", "zip", "map", "filter",
    "sorted", "reversed", "list", "dict", "set", "tuple", "str",
    "int", "float", "bool", "bytes", "bytearray", "type",
    "isinstance", "issubclass", "hasattr", "getattr", "setattr", "delattr",
    "super", "property", "staticmethod", "classmethod",
    "open", "input", "repr", "hash", "id", "dir", "vars", "globals", "locals",
    "abs", "min", "max", "sum", "round", "pow", "divmod",
    "iter", "next", "all", "any",
    "format", "chr", "ord", "hex", "oct", "bin",
    "callable", "compile", "eval", "exec",
    "breakpoint", "exit", "quit",
    "ValueError", "TypeError", "KeyError", "IndexError", "AttributeError",
    "RuntimeError", "StopIteration", "FileNotFoundError", "OSError",
    "ImportError", "NotImplementedError", "Exception", "BaseException",
    "AssertionError", "NameError", "ZeroDivisionError", "OverflowError",
})


def collect_project_calls(root: str):
    """Walk a project directory and collect all Python call edges."""
    root_path = Path(root).resolve()
    all_definitions = set()
    all_calls = {}

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Skip common non-source directories
        dirnames[:] = [
            d for d in dirnames
            if d not in {"__pycache__", ".git", ".venv", "venv", "node_modules",
                         ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
                         ".eggs", "*.egg-info"}
        ]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = Path(dirpath) / fname
            defs, calls = extract_calls(root_path, fpath)
            all_definitions.update(defs)
            all_calls.update(calls)

    # Post-process: resolve cross-module references where possible
    # Build a short-name → fqn index for resolution
    short_to_fqn = {}
    for fqn in all_definitions:
        short = fqn.rsplit("::", 1)[-1]
        # Only index if unambiguous (single definition with that short name)
        if short in short_to_fqn:
            short_to_fqn[short] = None  # ambiguous
        else:
            short_to_fqn[short] = fqn

    # Resolve callee references
    resolved_calls = {}
    for caller, callees in all_calls.items():
        resolved = []
        for callee in callees:
            # Already fully qualified?
            if callee in all_definitions:
                resolved.append(callee)
                continue
            # Try short name resolution
            short = callee.rsplit("::", 1)[-1]
            fqn = short_to_fqn.get(short)
            if fqn and fqn != caller:  # skip self-calls that are just noise
                resolved.append(fqn)
                continue
            # Keep as-is if it looks like a project call (has ::)
            if "::" in callee:
                resolved.append(callee)
        if resolved:
            resolved_calls[caller] = resolved

    json.dump(resolved_calls, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <project_root>", file=sys.stderr)
        sys.exit(1)
    collect_project_calls(sys.argv[1])
