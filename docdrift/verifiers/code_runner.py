"""Codebase Health & Execution Verifier for DocDrift.

Validates that repository code physically works:
1. Syntax & Compilation: scans source files (.py, .js, .ts, .json, .yaml) for syntax errors.
2. Entrypoints & Exports: validates that declared entrypoints (main.py, server.js, app.py) exist and export valid objects.
3. Dependency Integrity: verifies that code imports match declared manifest dependencies.
4. Test Runner Probing: detects and safely runs test suites (pytest, npm test, cargo test, make test) with bounded timeouts.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from docdrift.verifiers.code_analyzer import CodeAnalyzer
from docdrift.verifiers.dependency_verifier import DependencyVerifier


class CodeHealthReport(BaseModel):
    """Structured report on codebase health and execution viability."""
    repo_path: str
    ecosystems: List[str] = Field(default_factory=list)
    is_monorepo: bool = False
    total_files_checked: int = 0
    syntax_valid_count: int = 0
    syntax_errors: List[Dict[str, str]] = Field(default_factory=list)
    entrypoints_checked: List[Dict[str, Any]] = Field(default_factory=list)
    missing_dependencies: List[str] = Field(default_factory=list)
    test_suite_detected: Optional[str] = None
    test_suite_passed: Optional[bool] = None
    test_output_summary: Optional[str] = None
    is_code_working: bool = True
    summary: str = ""


class CodeHealthVerifier:
    """Verifies that the repository code compiles, runs, and satisfies all dependencies."""

    COMMON_ENTRYPOINTS = [
        "main.py",
        "app.py",
        "server.py",
        "app/main.py",
        "src/main.py",
        "src/app.py",
        "src/server.py",
        "backend/main.py",
        "backend/app.py",
        "backend/server.py",
        "api/index.py",
        "server.js",
        "index.js",
        "app.js",
        "src/index.js",
        "src/server.js",
        "server/index.js",
        "src/index.ts",
        "src/server.ts",
        "src/main.ts",
        "src/main.rs",
        "main.go",
    ]

    @classmethod
    def check_codebase_health(
        cls,
        repo_path: str,
        run_tests: bool = True,
        test_timeout_sec: float = 15.0,
    ) -> CodeHealthReport:
        """Executes a full code health and execution verification."""
        repo = Path(repo_path).resolve()
        report = CodeHealthReport(repo_path=str(repo))

        # 1. Detect Ecosystems
        report.ecosystems, report.is_monorepo = cls._detect_ecosystems(repo)

        # 2. Syntax & Compilation Scan
        syntax_errs: List[Dict[str, str]] = []
        valid_count = 0
        total_files = 0

        target_exts = (".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml")
        for file_path in repo.glob("**/*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in target_exts:
                continue
            if any(d in file_path.parts for d in (".git", "node_modules", ".venv", "venv", "dist", "build", ".next", ".docdrift")):
                continue

            total_files += 1
            is_valid, err = CodeAnalyzer.validate_file_syntax(file_path)
            if is_valid:
                valid_count += 1
            else:
                rel = file_path.relative_to(repo).as_posix()
                syntax_errs.append({"file": rel, "error": err or "Syntax error"})

        report.total_files_checked = total_files
        report.syntax_valid_count = valid_count
        report.syntax_errors = syntax_errs

        # 3. Entrypoint Health & Symbol Exports
        entrypoints_res: List[Dict[str, Any]] = []
        eps_to_check = list(cls.COMMON_ENTRYPOINTS)
        pkg_json = repo / "package.json"
        if pkg_json.is_file():
            try:
                pdata = json.loads(pkg_json.read_text(encoding="utf-8"))
                for k in ("main", "module"):
                    custom_ep = pdata.get(k)
                    if custom_ep and isinstance(custom_ep, str) and custom_ep not in eps_to_check:
                        eps_to_check.append(custom_ep)
            except Exception:
                pass

        for ep in eps_to_check:
            ep_path = repo / ep
            if ep_path.is_file():
                syntax_ok, err = CodeAnalyzer.validate_file_syntax(ep_path)
                symbols = CodeAnalyzer.extract_declared_symbols(ep_path)
                app_symbols = [
                    s for s in symbols
                    if s.lower() in ("app", "server", "main", "router", "handler", "api", "create_app", "application")
                ]
                entrypoints_res.append({
                    "entrypoint": ep,
                    "exists": True,
                    "syntax_valid": syntax_ok,
                    "exported_symbols": sorted(list(symbols))[:8],
                    "app_symbols": app_symbols,
                    "error": err,
                })
        report.entrypoints_checked = entrypoints_res

        # 4. Dependency Consistency Check
        missing_pkgs: List[str] = []
        dep_verifier = DependencyVerifier()
        if (
            (repo / "requirements.txt").is_file()
            or (repo / "backend" / "requirements.txt").is_file()
            or (repo / "pyproject.toml").is_file()
            or any(repo.glob("**/requirements*.txt"))
        ):
            declared = dep_verifier.get_declared_python_dependencies(repo)
            for py_file in repo.glob("**/*.py"):
                if any(d in py_file.parts for d in (".git", "node_modules", ".venv", "venv", "tests", "test", ".docdrift")):
                    continue
                missing = dep_verifier._get_missing_imports_for_file(py_file, declared, repo, include_try_except=False)
                for m in missing:
                    if m not in missing_pkgs:
                        missing_pkgs.append(m)

        if (repo / "package.json").is_file() or (repo / "frontend" / "package.json").is_file():
            declared_node = dep_verifier.get_declared_node_dependencies(repo)
            for js_file in repo.glob("**/*.[jt]s*"):
                if any(d in js_file.parts for d in (".git", "node_modules", ".next", "dist", "build", "tests", "test", "__tests__", ".docdrift")):
                    continue
                imports = CodeAnalyzer.extract_imports_from_file(js_file)
                for imp in imports:
                    if imp.lower() not in declared_node and not dep_verifier._is_local_node_path(imp, js_file.parent, repo):
                        if imp not in missing_pkgs:
                            missing_pkgs.append(imp)

        report.missing_dependencies = missing_pkgs

        # 5. Test Suite Execution Probe (if requested and tests exist)
        if run_tests:
            test_cmd, test_cwd = cls._detect_test_runner(repo)
            if test_cmd:
                report.test_suite_detected = " ".join(test_cmd)
                passed, summary = cls._run_test_suite(test_cmd, test_cwd, timeout=test_timeout_sec)
                report.test_suite_passed = passed
                report.test_output_summary = summary

        # 6. Overall Verdict Calculation
        critical_issues = len(syntax_errs) > 0 or len(missing_pkgs) > 0
        if report.test_suite_passed is False:
            critical_issues = True

        report.is_code_working = not critical_issues

        # Build Summary
        status_word = "HEALTHY & FUNCTIONAL" if report.is_code_working else "ISSUES DETECTED"
        summary_lines = [
            f"Codebase Status: {status_word}",
            f"• Syntax: {report.syntax_valid_count}/{report.total_files_checked} files valid (0 errors)" if not syntax_errs else f"• Syntax: {len(syntax_errs)} syntax error(s) found",
            f"• Entrypoints: {len(entrypoints_res)} detected & validated",
            f"• Dependencies: {'All imports declared' if not missing_pkgs else f'Missing: {', '.join(missing_pkgs)}'}",
        ]
        if report.test_suite_detected:
            test_status = "Passed" if report.test_suite_passed else "Failed"
            summary_lines.append(f"• Test Suite ({report.test_suite_detected}): {test_status}")

        report.summary = "\n".join(summary_lines)
        return report

    @classmethod
    def _detect_ecosystems(cls, repo: Path) -> Tuple[List[str], bool]:
        """Identifies ecosystems and monorepo structure."""
        ecosystems = []
        is_monorepo = False

        if any(repo.glob("*.py")) or any(repo.glob("*/*.py")) or (repo / "requirements.txt").exists() or (repo / "pyproject.toml").exists():
            ecosystems.append("Python")
        if (repo / "package.json").exists() or any(repo.glob("*/package.json")):
            ecosystems.append("Node.js/TypeScript")
        if (repo / "Cargo.toml").exists():
            ecosystems.append("Rust")
        if (repo / "go.mod").exists():
            ecosystems.append("Go")
        if (repo / "Dockerfile").exists() or (repo / "docker-compose.yml").exists():
            ecosystems.append("Docker")

        if (repo / "backend").is_dir() and (repo / "frontend").is_dir():
            is_monorepo = True
        elif any((repo / d).is_dir() for d in ("packages", "apps")):
            is_monorepo = True

        return ecosystems or ["General"], is_monorepo

    @classmethod
    def _detect_test_runner(cls, repo: Path) -> Tuple[Optional[List[str]], Path]:
        """Detects available test runner command and working directory."""
        # Check Node / npm test
        pkg_json = repo / "package.json"
        if pkg_json.is_file():
            try:
                data = json.loads(pkg_json.read_text(encoding="utf-8"))
                if "test" in data.get("scripts", {}):
                    test_script = data["scripts"]["test"].strip()
                    # Skip trivial placeholder 'echo "Error: no test specified" && exit 1'
                    if "no test specified" not in test_script and shutil.which("npm"):
                        return ["npm", "test"], repo
            except Exception:
                pass

        # Check Python pytest or unittest
        test_dir = "tests" if (repo / "tests").is_dir() else ("test" if (repo / "test").is_dir() else "")
        if test_dir or any(repo.glob("test_*.py")):
            target_arg = [test_dir] if test_dir else []
            if shutil.which("pytest"):
                return ["pytest", "-q"] + target_arg, repo
            elif sys.executable:
                return [sys.executable, "-m", "pytest", "-q"] + target_arg, repo
            elif shutil.which("python"):
                return ["python", "-m", "unittest", "discover", "-s", test_dir or "tests"], repo

        # Check monorepo backend tests
        if (repo / "backend").is_dir() and ((repo / "backend" / "tests").is_dir() or any((repo / "backend").glob("test_*.py"))):
            backend_test_dir = "tests" if (repo / "backend" / "tests").is_dir() else ""
            b_target = [backend_test_dir] if backend_test_dir else []
            if shutil.which("pytest"):
                return ["pytest", "-q"] + b_target, repo / "backend"
            elif sys.executable:
                return [sys.executable, "-m", "pytest", "-q"] + b_target, repo / "backend"

        # Check Makefile test
        if (repo / "Makefile").is_file():
            try:
                txt = (repo / "Makefile").read_text(encoding="utf-8")
                if re.search(r"^test\s*:", txt, re.MULTILINE) and shutil.which("make"):
                    return ["make", "test"], repo
            except Exception:
                pass

        return None, repo

    @classmethod
    def _run_test_suite(cls, cmd: List[str], cwd: Path, timeout: float = 15.0) -> Tuple[bool, str]:
        """Safely executes the test runner with bounded timeout."""
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = proc.stdout.strip() or proc.stderr.strip()
            first_lines = "\n".join(out.splitlines()[:5])
            if proc.returncode == 0:
                return True, first_lines or "All tests passed successfully."
            else:
                return False, first_lines or f"Test command exited with code {proc.returncode}."
        except subprocess.TimeoutExpired:
            return False, f"Test execution timed out after {timeout:.1f}s."
        except Exception as e:
            return False, f"Failed to run tests: {str(e)}"


__all__ = ["CodeHealthVerifier", "CodeHealthReport"]
