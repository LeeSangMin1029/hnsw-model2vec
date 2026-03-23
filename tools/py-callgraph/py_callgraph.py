#!/usr/bin/env python3
"""Python call graph extractor — outputs edges.jsonl + chunks.jsonl"""

import ast
import json
import os
import sys
from pathlib import Path
from typing import Optional

class CallGraphExtractor(ast.NodeVisitor):
    """AST visitor that extracts function/class definitions and call edges."""

    def __init__(self, file_path: str, module_name: str):
        self.file_path = file_path
        self.module_name = module_name
        self.chunks = []
        self.edges = []
        self.current_function = None  # stack of (name, is_test)
        self.class_stack = []  # current class context

    def _qualified_name(self, name: str) -> str:
        """Build qualified name: module::Class::method or module::function"""
        parts = [self.module_name]
        parts.extend(self.class_stack)
        parts.append(name)
        return "::".join(parts)

    def visit_ClassDef(self, node):
        name = self._qualified_name(node.name)
        self.chunks.append({
            "name": name,
            "file": self.file_path,
            "kind": "class",
            "start_line": node.lineno,
            "end_line": node.end_lineno or node.lineno,
            "signature": f"class {node.name}",
            "visibility": "pub" if not node.name.startswith("_") else "",
            "is_test": False,
        })
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node):
        self._visit_func(node, is_async=True)

    def _visit_func(self, node, is_async=False):
        name = self._qualified_name(node.name)

        # Detect test functions
        is_test = (
            node.name.startswith("test_")
            or any(
                isinstance(d, ast.Name) and d.id in ("pytest.mark", "test")
                or isinstance(d, ast.Attribute) and d.attr in ("mark", "fixture")
                for d in node.decorator_list
            )
            or "test" in self.file_path.lower()
        )

        # Build signature
        args = []
        for arg in node.args.args:
            arg_name = arg.arg
            annotation = ""
            if arg.annotation:
                annotation = f": {ast.unparse(arg.annotation)}"
            args.append(f"{arg_name}{annotation}")

        returns = ""
        if node.returns:
            returns = f" -> {ast.unparse(node.returns)}"

        prefix = "async def" if is_async else "def"
        sig = f"{prefix} {node.name}({', '.join(args)}){returns}"

        # Determine kind
        kind = "method" if self.class_stack else "fn"

        # Visibility
        vis = "" if node.name.startswith("_") else "pub"

        self.chunks.append({
            "name": name,
            "file": self.file_path,
            "kind": kind,
            "start_line": node.lineno,
            "end_line": node.end_lineno or node.lineno,
            "signature": sig,
            "visibility": vis,
            "is_test": is_test,
        })

        # Extract call edges from function body
        prev = self.current_function
        self.current_function = name

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee = self._resolve_call(child)
                if callee:
                    self.edges.append({
                        "caller": name,
                        "caller_file": self.file_path,
                        "caller_kind": kind,
                        "callee": callee,
                        "line": child.lineno,
                        "is_local": True,  # Will be refined later
                    })

        self.current_function = prev

    def _resolve_call(self, node: ast.Call) -> Optional[str]:
        """Resolve a Call node to a callee name."""
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            # obj.method() — try to resolve obj
            parts = []
            current = func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            parts.reverse()
            return "::".join(parts)
        return None


def extract_file(file_path: str, project_root: str) -> tuple:
    """Extract chunks and edges from a single Python file."""
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source, filename=file_path)
    except (SyntaxError, ValueError):
        return [], []

    # Build module name from relative path
    rel = os.path.relpath(file_path, project_root).replace(os.sep, "/")
    module = rel.replace("/", "::").removesuffix(".py").removesuffix("::__init__")

    extractor = CallGraphExtractor(rel, module)
    extractor.visit(tree)
    return extractor.chunks, extractor.edges


def main():
    if len(sys.argv) < 2:
        print("Usage: py_callgraph.py <project_root> [--out-dir <dir>]", file=sys.stderr)
        sys.exit(1)

    project_root = sys.argv[1]
    out_dir = None
    if "--out-dir" in sys.argv:
        idx = sys.argv.index("--out-dir")
        if idx + 1 < len(sys.argv):
            out_dir = sys.argv[idx + 1]

    all_chunks = []
    all_edges = []

    for root, dirs, files in os.walk(project_root):
        # Skip common non-source directories
        dirs[:] = [d for d in dirs if d not in (
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
            ".eggs", "*.egg-info",
        )]
        for fname in files:
            if fname.endswith(".py"):
                fpath = os.path.join(root, fname)
                chunks, edges = extract_file(fpath, project_root)
                all_chunks.extend(chunks)
                all_edges.extend(edges)

    # Mark is_local: callee exists in our chunks
    chunk_names = {c["name"] for c in all_chunks}
    short_names = {}
    for c in all_chunks:
        parts = c["name"].split("::")
        if parts:
            short_names[parts[-1]] = c["name"]

    for edge in all_edges:
        callee = edge["callee"]
        edge["is_local"] = (
            callee in chunk_names
            or callee in short_names
        )

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "python.edges.jsonl"), "w", encoding="utf-8") as f:
            for e in all_edges:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, "python.chunks.jsonl"), "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"[py-callgraph] {len(all_edges)} edges, {len(all_chunks)} chunks", file=sys.stderr)
    else:
        for e in all_edges:
            print(json.dumps(e, ensure_ascii=False))


if __name__ == "__main__":
    main()
