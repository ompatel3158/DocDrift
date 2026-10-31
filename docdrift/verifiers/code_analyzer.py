"""Code analysis, AST syntax validation, symbol checking, and import extraction for DocDrift."""

from __future__ import annotations

import ast
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore


# Standard Python library module set
STDLIB_MODULES: Set[str] = set(getattr(sys, "stdlib_module_names", set()))
STDLIB_MODULES.update(sys.builtin_module_names)
STDLIB_MODULES.update({
    "__future__", "pkg_resources", "site", "distutils", "typing_extensions",
    "os", "sys", "re", "json", "math", "pathlib", "subprocess", "typing",
    "collections", "itertools", "functools", "datetime", "time", "shutil",
    "tempfile", "hashlib", "urllib", "http", "unittest", "logging", "asyncio",
    "ast", "difflib", "traceback", "inspect", "io", "copy", "socket", "ssl",
    "threading", "multiprocessing", "queue", "signal", "platform", "enum",
    "dataclasses", "abc", "contextlib", "glob", "fnmatch", "sqlite3"
})

# Standard Node.js built-in modules
NODE_STDLIB_MODULES: Set[str] = {
    "assert", "async_hooks", "buffer", "child_process", "cluster",
    "console", "constants", "crypto", "dgram", "diagnostics_channel",
    "dns", "domain", "events", "fs", "fs/promises", "http", "http2",
    "https", "inspector", "module", "net", "os", "path", "path/posix",
    "path/win32", "perf_hooks", "process", "punycode", "querystring",
    "readline", "repl", "stream", "stream/promises", "stream/consumers",
    "stream/web", "string_decoder", "sys", "timers", "timers/promises",
    "tls", "trace_events", "tty", "url", "util", "util/types", "v8",
    "vm", "wasi", "worker_threads", "zlib", "test"
}


class CodeAnalyzer:
    """Deterministic AST parser and code validator."""

    # --------------------------------------------------------------------------
    # Syntax & Compilation Validation
    # --------------------------------------------------------------------------

    @classmethod
    def validate_file_syntax(cls, file_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """Checks if a code or configuration file compiles/parses without syntax errors.

        Supports:
        - Python (.py): ast.parse and compile()
        - JSON (.json): json.loads
        - YAML (.yaml, .yml): yaml.safe_load
        - TOML (.toml): tomllib.loads
        - JavaScript (.js, .mjs, .cjs): node --check or balanced bracket heuristic

        Returns:
            (is_valid, error_message_if_any)
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            return False, f"File '{path}' does not exist or is not a regular file."

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return False, f"Unable to read file '{path.name}': {str(e)}"

        suffix = path.suffix.lower()

        # 1. Python Syntax Validation
        if suffix == ".py":
            try:
                compile(content, str(path), "exec")
                return True, None
            except SyntaxError as e:
                line_info = f"line {e.lineno}" if e.lineno else "unknown line"
                col_info = f", col {e.offset}" if e.offset else ""
                msg = e.msg or "syntax error"
                return False, f"Python SyntaxError in '{path.name}': {msg} ({line_info}{col_info})"
            except Exception as e:
                return False, f"Python compilation error in '{path.name}': {str(e)}"

        # 2. JSON Syntax Validation
        elif suffix == ".json":
            try:
                json.loads(content)
                return True, None
            except json.JSONDecodeError as e:
                return False, f"JSON syntax error in '{path.name}': {e.msg} (line {e.lineno}, col {e.colno})"
            except Exception as e:
                return False, f"JSON parsing error in '{path.name}': {str(e)}"

        # 3. YAML Syntax Validation
        elif suffix in (".yaml", ".yml"):
            if yaml is not None:
                try:
                    yaml.safe_load(content)
                    return True, None
                except Exception as e:
                    return False, f"YAML syntax error in '{path.name}': {str(e)}"
            # Fallback when PyYAML is not installed
            valid, err = cls._check_balanced_brackets(content)
            if not valid:
                return False, f"YAML syntax error in '{path.name}': {err}"
            for line_no, line in enumerate(content.splitlines(), 1):
                if line.startswith("\t"):
                    return False, f"YAML syntax error in '{path.name}': tabs are not allowed for indentation at line {line_no}"
                if re.search(r":\s*:\s*\w+", line):
                    return False, f"YAML syntax error in '{path.name}': invalid mapping syntax at line {line_no}"
            return True, None

        # 4. TOML Syntax Validation
        elif suffix == ".toml":
            if tomllib is not None:
                try:
                    tomllib.loads(content)
                    return True, None
                except Exception as e:
                    return False, f"TOML syntax error in '{path.name}': {str(e)}"
            return True, None

        # 5. JavaScript & TypeScript Syntax Validation
        elif suffix in (".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"):
            # If node is available on PATH and file is plain JS, execute node --check
            if suffix in (".js", ".mjs", ".cjs") and shutil.which("node"):
                try:
                    proc = subprocess.run(
                        ["node", "--check", str(path)],
                        capture_output=True,
                        text=True,
                        timeout=3.0,
                    )
                    if proc.returncode != 0:
                        err = proc.stderr.strip() or proc.stdout.strip()
                        first_err_line = err.splitlines()[0] if err else "syntax error"
                        return False, f"JavaScript syntax error in '{path.name}': {first_err_line}"
                    return True, None
                except Exception:
                    pass

            # Heuristic bracket/quote validation fallback for JS, TS, JSX, TSX
            valid, err = cls._check_balanced_brackets(content)
            if not valid:
                lang = "TypeScript" if suffix in (".ts", ".tsx") else "JavaScript"
                return False, f"{lang} syntax error in '{path.name}': {err}"
            return True, None

        return True, None

    @staticmethod
    def _check_balanced_brackets(code: str) -> Tuple[bool, Optional[str]]:
        """Heuristic validator for balanced brackets and quotes in code."""
        stack: List[Tuple[str, int]] = []
        matching = {")": "(", "}": "{", "]": "["}
        lines = code.splitlines()

        in_line_comment = False
        in_block_comment = False
        in_string: Optional[str] = None
        escape = False

        for line_idx, line in enumerate(lines, 1):
            i = 0
            n = len(line)
            while i < n:
                ch = line[i]

                if escape:
                    escape = False
                    i += 1
                    continue

                if ch == "\\" and in_string:
                    escape = True
                    i += 1
                    continue

                # Inside string literal
                if in_string:
                    if ch == in_string:
                        in_string = None
                    i += 1
                    continue

                # Block comments
                if in_block_comment:
                    if ch == "*" and i + 1 < n and line[i + 1] == "/":
                        in_block_comment = False
                        i += 2
                        continue
                    i += 1
                    continue

                # Start of line comment
                if ch == "/" and i + 1 < n and line[i + 1] == "/":
                    break

                # Start of block comment
                if ch == "/" and i + 1 < n and line[i + 1] == "*":
                    in_block_comment = True
                    i += 2
                    continue

                # Start of string
                if ch in ("'", '"', "`"):
                    in_string = ch
                    i += 1
                    continue

                # Bracket tracking
                if ch in ("(", "{", "["):
                    stack.append((ch, line_idx))
                elif ch in (")", "}", "]"):
                    if not stack:
                        return False, f"Unmatched closing bracket '{ch}' at line {line_idx}"
                    top, start_line = stack.pop()
                    if matching[ch] != top:
                        return False, f"Mismatched bracket: expected closing for '{top}' (from line {start_line}), got '{ch}' at line {line_idx}"

                i += 1

            if in_string in ("'", '"'):  # Single/double quotes cannot cross lines
                return False, f"Unclosed string literal ({in_string}) at line {line_idx}"

        if stack:
            unclosed, start_line = stack[-1]
            return False, f"Unclosed bracket '{unclosed}' opened at line {start_line}"

        return True, None

    # --------------------------------------------------------------------------
    # Symbol Extraction & Verification
    # --------------------------------------------------------------------------

    @classmethod
    def extract_declared_symbols(cls, file_path: Union[str, Path]) -> Set[str]:
        """Extracts top-level and exported symbols declared in a Python or JS/TS file."""
        path = Path(file_path).resolve()
        if not path.is_file():
            return set()

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return set()

        suffix = path.suffix.lower()
        symbols: Set[str] = set()

        # 1. Python AST Symbol Extraction
        if suffix == ".py":
            try:
                tree = ast.parse(content, filename=str(path))
            except Exception:
                return symbols

            def _extract_from_nodes(nodes: List[ast.AST]) -> None:
                for node in nodes:
                    # Function definitions
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.add(node.name)
                    # Class definitions
                    elif isinstance(node, ast.ClassDef):
                        symbols.add(node.name)
                    # Variable assignments: app = FastAPI()
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                symbols.add(target.id)
                                # Special case: check __all__
                                if target.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                                    for elt in node.value.elts:
                                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                            symbols.add(elt.value)
                            elif isinstance(target, (ast.Tuple, ast.List)):
                                for elt in target.elts:
                                    if isinstance(elt, ast.Name):
                                        symbols.add(elt.id)
                    # Annotated assignments: app: FastAPI = FastAPI()
                    elif isinstance(node, ast.AnnAssign):
                        if isinstance(node.target, ast.Name):
                            symbols.add(node.target.id)
                    # Imported symbols: from fastapi import FastAPI as App
                    elif isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            symbols.add(alias.asname or alias.name)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            symbols.add(alias.asname or alias.name)
                    # Compound statements at module level (do not traverse inside func/class)
                    elif isinstance(node, ast.If):
                        _extract_from_nodes(node.body)
                        _extract_from_nodes(node.orelse)
                    elif isinstance(node, ast.Try):
                        _extract_from_nodes(node.body)
                        for handler in node.handlers:
                            _extract_from_nodes(handler.body)
                        _extract_from_nodes(node.orelse)
                        _extract_from_nodes(node.finalbody)
                    elif isinstance(node, (ast.With, ast.AsyncWith)):
                        _extract_from_nodes(node.body)

            _extract_from_nodes(tree.body)

        # 2. JS / TS Regex Symbol Extraction
        elif suffix in (".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"):
            # const app = express(), let server, function main(), class App
            decl_patterns = [
                re.compile(r"(?:^|\s)(?:export\s+)?(?:const|let|var|function|class)\s+([a-zA-Z0-9_$]+)"),
                re.compile(r"(?:^|\s)export\s+default\s+([a-zA-Z0-9_$]+)"),
                re.compile(r"exports\.([a-zA-Z0-9_$]+)\s*="),
                re.compile(r"module\.exports\s*=\s*([a-zA-Z0-9_$]+)"),
            ]
            for pat in decl_patterns:
                for match in pat.finditer(content):
                    symbols.add(match.group(1))

            # module.exports = { a, b, c }
            obj_export = re.search(r"module\.exports\s*=\s*\{([^}]+)\}", content)
            if obj_export:
                raw_keys = obj_export.group(1)
                for part in raw_keys.split(","):
                    k = part.split(":")[0].strip()
                    if re.match(r"^[a-zA-Z0-9_$]+$", k):
                        symbols.add(k)

        return symbols

    @classmethod
    def verify_symbol_in_file(
        cls, file_path: Union[str, Path], symbol_name: str
    ) -> Tuple[bool, Set[str], Optional[str]]:
        """Verifies if symbol_name is declared in file_path.

        Returns:
            (is_declared, all_symbols, closest_match_suggestion)
        """
        clean_name = symbol_name.strip().rstrip("()")
        symbols = cls.extract_declared_symbols(file_path)
        if clean_name in symbols or symbol_name in symbols:
            return True, symbols, None

        closest = None
        if symbols:
            # 1. Close match with difflib
            matches = difflib.get_close_matches(symbol_name, list(symbols), n=1, cutoff=0.3)
            if matches:
                closest = matches[0]

            # 2. Semantic web/ASGI synonyms: if symbol is app/server/api/application
            if not closest and symbol_name.lower() in ("app", "server", "api", "application", "handler", "service", "main"):
                for syn in ("server", "app", "api", "application", "handler", "service", "main"):
                    if syn in symbols and syn.lower() != symbol_name.lower():
                        closest = syn
                        break

            # 3. If only one non-private symbol exists, it is the obvious candidate
            if not closest:
                public_syms = [s for s in symbols if not s.startswith("_")]
                if len(public_syms) == 1:
                    closest = public_syms[0]

        return False, symbols, closest

    # --------------------------------------------------------------------------
    # Import Extraction
    # --------------------------------------------------------------------------

    @classmethod
    def extract_imports_from_file(cls, file_path: Union[str, Path]) -> Set[str]:
        """Extracts third-party package imports used in a source file.

        Excludes standard library and built-in modules.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            return set()

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return set()

        suffix = path.suffix.lower()
        imports: Set[str] = set()

        # 1. Python imports
        if suffix == ".py":
            try:
                tree = ast.parse(content, filename=str(path))
            except Exception:
                return imports

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_mod = alias.name.split(".")[0]
                        if root_mod and root_mod not in STDLIB_MODULES:
                            imports.add(root_mod)
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        root_mod = node.module.split(".")[0]
                        if root_mod and root_mod not in STDLIB_MODULES:
                            imports.add(root_mod)

        # 2. JS / TS imports
        elif suffix in (".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"):
            patterns = [
                re.compile(r"(?:^|\n|\;)\s*(?:import|export)\s+(?:.*?from\s+)?['\"]([^'\"]+)['\"]"),
                re.compile(r"(?:^|\n|\;)\s*from\s+['\"]([^'\"]+)['\"]"),
                re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
                re.compile(r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
            ]
            for pat in patterns:
                for match in pat.finditer(content):
                    raw_spec = match.group(1).strip()
                    # Skip local relative paths
                    if raw_spec.startswith((".", "/")):
                        continue
                    if raw_spec.startswith("node:"):
                        continue
                    # Handle @scope/pkg vs pkg
                    parts = raw_spec.split("/")
                    if raw_spec.startswith("@") and len(parts) >= 2:
                        pkg = f"{parts[0]}/{parts[1]}"
                    else:
                        pkg = parts[0]

                    # Validate strict npm package identifier format (letters, digits, @, -, _, .)
                    if not re.match(r"^(@[a-zA-Z0-9_\-\.]+\/)?[a-zA-Z0-9_\-\.]+$", pkg):
                        continue
                    if any(c in pkg for c in ("+", " ", "\\", "*", ":", "'", '"', "=", "(", ")")):
                        continue

                    if pkg and pkg.lower() not in NODE_STDLIB_MODULES:
                        imports.add(pkg)

        return imports
