"""Dependency verifier for Python (requirements.txt, pyproject.toml) and Node (package.json)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore

from docdrift.models import Claim, ClaimType, Verdict, VerificationResult
from docdrift.verifiers.base import BaseVerifier
from docdrift.verifiers.code_analyzer import CodeAnalyzer


# Mapping between common PyPI distribution package names and their top-level import module names
PACKAGE_TO_IMPORT_MAP: Dict[str, Set[str]] = {
    "pyyaml": {"yaml"},
    "python-dotenv": {"dotenv"},
    "scikit-learn": {"sklearn"},
    "pillow": {"pil"},
    "beautifulsoup4": {"bs4"},
    "opencv-python": {"cv2"},
    "opencv-python-headless": {"cv2"},
    "typing-extensions": {"typing_extensions"},
    "pydantic-settings": {"pydantic_settings"},
    "pytest-asyncio": {"pytest_asyncio"},
    "google-genai": {"google", "google_genai"},
    "protobuf": {"google"},
    "pydantic-core": {"pydantic_core"},
    "setuptools": {"setuptools", "pkg_resources"},
    "rich": {"rich"},
    "typer": {"typer"},
    "fastapi": {"fastapi"},
    "uvicorn": {"uvicorn"},
    "gunicorn": {"gunicorn"},
    "pytest": {"pytest"},
    "pydantic": {"pydantic"},
    "sqlalchemy": {"sqlalchemy"},
    "alembic": {"alembic"},
    "psycopg2": {"psycopg2"},
    "psycopg2-binary": {"psycopg2"},
    "asyncpg": {"asyncpg"},
    "pymongo": {"pymongo", "bson"},
    "redis": {"redis"},
    "httpx": {"httpx"},
    "requests": {"requests"},
    "aiohttp": {"aiohttp"},
    "starlette": {"starlette"},
    "flask": {"flask"},
    "django": {"django"},
    "celery": {"celery"},
}

# Reverse mapping from import name to package names
IMPORT_TO_PACKAGE_MAP: Dict[str, Set[str]] = {}
for pkg_name, imp_names in PACKAGE_TO_IMPORT_MAP.items():
    for imp in imp_names:
        IMPORT_TO_PACKAGE_MAP.setdefault(imp.lower(), set()).add(pkg_name.lower())
        IMPORT_TO_PACKAGE_MAP.setdefault(imp.lower().replace("-", "_"), set()).add(pkg_name.lower())

# Packages that bundle or transitively provide well-known top-level modules
KNOWN_TRANSITIVE_IMPORTS: Dict[str, Set[str]] = {
    "fastapi": {"pydantic", "starlette", "typing_extensions"},
    "google-genai": {"pydantic", "pydantic_core"},
    "google_genai": {"pydantic", "pydantic_core"},
    "uvicorn": {"click", "h11"},
    "gunicorn": {"setuptools"},
    "requests": {"urllib3", "certifi", "idna", "charset_normalizer"},
    "pytest": {"pluggy", "iniconfig", "packaging"},
    "flask": {"werkzeug", "jinja2", "itsdangerous", "click", "blinker"},
    "django": {"asgiref", "sqlparse"},
    "pandas": {"numpy", "pytz", "dateutil"},
    "scikit-learn": {"numpy", "scipy", "joblib", "threadpoolctl"},
    "torch": {"typing_extensions", "sympy", "networkx", "jinja2"},
    "typer": {"click", "rich", "shellingham"},
    "next": {"react", "react-dom"},
    "vite": {"esbuild", "rollup"},
}


class DependencyVerifier(BaseVerifier):
    """Verifies that third-party code imports align with declared dependencies in manifests."""

    def can_verify(self, claim: Claim) -> bool:
        """Determines if this verifier can handle the given claim."""
        if claim.claim_type == ClaimType.DEPENDENCY:
            return True

        val = claim.extracted_value.strip().strip("'\"`").lower()
        if claim.claim_type == ClaimType.FILE_PATH:
            base = Path(val).name.lower()
            return base in ("requirements.txt", "package.json", "pyproject.toml")

        if claim.claim_type == ClaimType.COMMAND_SCRIPT:
            return any(val.startswith(p) for p in (
                "pip install", "python -m pip install", "python3 -m pip install",
                "npm install", "npm i ", "yarn add", "pnpm add", "pnpm install",
            ))

        return False

    def verify(self, claim: Claim, repo_path: str) -> VerificationResult:
        """Verifies claimed dependencies or manifest coverage against codebase."""
        repo = Path(repo_path).resolve()
        val = claim.extracted_value.strip().strip("'\"`")
        val_lower = val.lower()

        # 1. pip install -r requirements.txt or direct requirements.txt file
        if "requirements" in val_lower or val_lower.endswith(".txt"):
            return self._verify_python_requirements(claim, repo)

        # 2. npm install or package.json
        if "package.json" in val_lower or val_lower.startswith(("npm ", "yarn ", "pnpm ")):
            return self._verify_node_dependencies(claim, repo)

        # 3. pip install <package>
        if val_lower.startswith(("pip install ", "python -m pip install ")):
            return self._verify_pip_package_install(claim, repo)

        return VerificationResult(
            verdict=Verdict.UNKNOWN,
            evidence=f"Unhandled dependency claim '{val}'.",
            suggested_value=None,
        )

    # --------------------------------------------------------------------------
    # Python Dependencies
    # --------------------------------------------------------------------------

    def _verify_python_requirements(self, claim: Claim, repo: Path) -> VerificationResult:
        """Verifies requirements.txt exists and covers imports across repo."""
        val = claim.extracted_value.strip().strip("'\"`")
        target_name = None
        if "-r" in val:
            parts = val.split()
            if "-r" in parts:
                r_idx = parts.index("-r")
                if len(parts) > r_idx + 1:
                    target_name = parts[r_idx + 1].strip("'\"`")
        elif claim.claim_type == ClaimType.FILE_PATH:
            target_name = val

        doc_file = Path(claim.doc_path) if claim.doc_path else Path("README.md")
        doc_dir = (doc_file if doc_file.is_absolute() else (repo / doc_file)).parent.resolve()

        found_req = None
        if target_name:
            candidates = [
                (doc_dir / target_name).resolve(),
                (repo / target_name).resolve(),
                (repo / "backend" / target_name).resolve(),
                (repo / "requirements" / target_name).resolve(),
            ]
            found_req = next((c for c in candidates if c.is_file()), None)
            if not found_req:
                t_base = Path(target_name).name.lower()
                for r in repo.glob("**/requirements*.txt"):
                    if not any(d in r.parts for d in (".git", "node_modules", ".venv", "venv")):
                        if r.name.lower() == t_base:
                            found_req = r
                            break
            if not found_req:
                return VerificationResult(
                    verdict=Verdict.DRIFTED,
                    evidence=f"Requirements file '{target_name}' not found in repository.",
                    suggested_value=None,
                )
        else:
            req_candidates = [
                repo / "requirements.txt",
                repo / "backend" / "requirements.txt",
                repo / "requirements" / "base.txt",
                repo / "requirements" / "prod.txt",
                repo / "requirements-dev.txt",
            ]
            found_req = next((c for c in req_candidates if c.is_file()), None)
            if not found_req:
                for r in repo.glob("**/requirements*.txt"):
                    if not any(d in r.parts for d in (".git", "node_modules", ".venv", "venv")):
                        found_req = r
                        break

            if not found_req:
                return VerificationResult(
                    verdict=Verdict.DRIFTED,
                    evidence="Requirements file 'requirements.txt' not found in repository.",
                    suggested_value=None,
                )

        rel_req = found_req.relative_to(repo).as_posix()
        declared_pkgs = self.get_declared_python_dependencies(repo)

        # Find third-party imports in Python source files
        missing_by_file: Dict[str, List[str]] = {}
        all_missing: Set[str] = set()

        is_dev_requirements = any(k in found_req.name.lower() for k in ("dev", "test"))

        for py_file in repo.glob("**/*.py"):
            if any(d in py_file.parts for d in (".git", "node_modules", ".venv", "venv", "build", "dist")):
                continue
            # When checking a production requirements file, skip test files
            if not is_dev_requirements and any(p in py_file.parts for p in ("tests", "test")):
                continue
            missing = self._get_missing_imports_for_file(py_file, declared_pkgs, repo)
            if missing:
                rel_py = py_file.relative_to(repo).as_posix()
                missing_by_file[rel_py] = missing
                all_missing.update(missing)

        if all_missing:
            sample_file = next(iter(missing_by_file.keys()))
            missing_list = sorted(all_missing)
            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence=(
                    f"Requirements file '{rel_req}' is missing dependencies imported in source code: "
                    f"{', '.join(missing_list)} (used in '{sample_file}')."
                ),
                suggested_value=None,
                metadata={"missing_dependencies": missing_list, "missing_by_file": missing_by_file},
            )

        return VerificationResult(
            verdict=Verdict.VERIFIED,
            evidence=f"Requirements file '{rel_req}' verified. All source code imports are declared.",
            suggested_value=None,
            metadata={"declared_count": len(declared_pkgs)},
        )

    def _verify_pip_package_install(self, claim: Claim, repo: Path) -> VerificationResult:
        """Verifies 'pip install <package>' matches declared dependencies or imports."""
        tokens = claim.extracted_value.split()
        # Handle editable install 'pip install -e .' or 'pip install .'
        if any(t in tokens for t in (".", "-e")):
            if (repo / "pyproject.toml").is_file() or (repo / "setup.py").is_file():
                return VerificationResult(
                    verdict=Verdict.VERIFIED,
                    evidence="Local package editable installation supported by repository manifest.",
                )

        pkgs = [t for t in tokens[2:] if not t.startswith("-") and t not in (".",)]
        if not pkgs:
            return VerificationResult(verdict=Verdict.VERIFIED, evidence="Standard pip command.")

        target_pkg = pkgs[0].lower().replace("-", "_")
        declared = self.get_declared_python_dependencies(repo)

        if target_pkg in declared or any(target_pkg in imp for imp in PACKAGE_TO_IMPORT_MAP.get(target_pkg, set())):
            return VerificationResult(
                verdict=Verdict.VERIFIED,
                evidence=f"Package '{pkgs[0]}' is declared in repository dependencies.",
            )

        # Check if imported in any source file
        used_imports: Set[str] = set()
        for py_file in repo.glob("**/*.py"):
            if any(d in py_file.parts for d in (".git", "node_modules", ".venv", "venv")):
                continue
            used_imports.update(CodeAnalyzer.extract_imports_from_file(py_file))

        normalized_imports = {i.lower().replace("-", "_") for i in used_imports}
        if target_pkg in normalized_imports or any(target_pkg in IMPORT_TO_PACKAGE_MAP.get(i, set()) for i in normalized_imports):
            return VerificationResult(
                verdict=Verdict.VERIFIED,
                evidence=f"Package '{pkgs[0]}' is imported in repository source code.",
            )

        return VerificationResult(
            verdict=Verdict.DRIFTED,
            evidence=f"Claimed dependency '{pkgs[0]}' is neither declared in repository manifests nor imported in source code.",
            suggested_value=None,
        )

    # --------------------------------------------------------------------------
    # Node / JS Dependencies
    # --------------------------------------------------------------------------

    def _verify_node_dependencies(self, claim: Claim, repo: Path) -> VerificationResult:
        """Verifies package.json exists and covers imports across JS/TS code."""
        pkg_file = repo / "package.json"
        if not pkg_file.is_file():
            # Search monorepo folders
            for sub in ("frontend", "client", "web", "app", "ui"):
                cand = repo / sub / "package.json"
                if cand.is_file():
                    pkg_file = cand
                    break

        if not pkg_file.is_file():
            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence="Package manifest 'package.json' not found in repository.",
                suggested_value=None,
            )

        declared_pkgs = self.get_declared_node_dependencies(repo)
        rel_pkg = pkg_file.relative_to(repo).as_posix()

        # Check if claim is an individual package install like 'npm install express' or 'yarn add lodash'
        tokens = claim.extracted_value.split()
        pkgs = [t for t in tokens[1:] if not t.startswith("-") and t not in ("install", "i", "add", "run")]
        if pkgs:
            target_pkg = pkgs[0].lower()
            if target_pkg in declared_pkgs:
                return VerificationResult(
                    verdict=Verdict.VERIFIED,
                    evidence=f"Package '{pkgs[0]}' is declared in package.json.",
                )
            used_imports: Set[str] = set()
            for code_file in repo.glob("**/*.[jt]s*"):
                if any(d in code_file.parts for d in (".git", "node_modules", ".next", ".nuxt", "dist", "build")):
                    continue
                used_imports.update(CodeAnalyzer.extract_imports_from_file(code_file))

            if target_pkg in {i.lower() for i in used_imports}:
                return VerificationResult(
                    verdict=Verdict.VERIFIED,
                    evidence=f"Package '{pkgs[0]}' is imported in repository source code.",
                )

            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence=f"Claimed package '{pkgs[0]}' is neither declared in package.json nor imported in source code.",
                suggested_value=None,
            )

        all_missing: Set[str] = set()
        missing_by_file: Dict[str, List[str]] = {}

        for code_file in repo.glob("**/*.[jt]s*"):
            # Exclude build, cache, test, and fixture directories for general package install checks
            if any(d in code_file.parts for d in (".git", "node_modules", ".next", ".nuxt", "dist", "build", "test", "tests", "__tests__", "fixtures", "benchmarks", "examples")):
                continue
            imports = CodeAnalyzer.extract_imports_from_file(code_file)
            missing = [
                imp for imp in imports
                if imp.lower() not in declared_pkgs and not self._is_local_node_path(imp, code_file.parent, repo)
            ]
            if missing:
                rel_code = code_file.relative_to(repo).as_posix()
                missing_by_file[rel_code] = missing
                all_missing.update(missing)

        if all_missing:
            missing_list = sorted(all_missing)
            sample_file = next(iter(missing_by_file.keys()))
            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence=(
                    f"Package manifest '{rel_pkg}' is missing dependencies imported in source code: "
                    f"{', '.join(missing_list)} (used in '{sample_file}')."
                ),
                suggested_value=None,  # Do not propose raw string substitution into doc commands
                metadata={"missing_dependencies": missing_list},
            )

        return VerificationResult(
            verdict=Verdict.VERIFIED,
            evidence=f"Package manifest '{rel_pkg}' verified. All source code imports are declared in dependencies.",
            suggested_value=None,
        )

    # --------------------------------------------------------------------------
    # Manifest Parser Utilities
    # --------------------------------------------------------------------------

    @classmethod
    def get_declared_python_dependencies(cls, repo: Path) -> Set[str]:
        """Collects all declared Python packages from requirements.txt, pyproject.toml, and setup.py."""
        packages: Set[str] = set()

        # 1. requirements*.txt
        for req_path in repo.glob("**/requirements*.txt"):
            if any(d in req_path.parts for d in (".git", "node_modules", ".venv", "venv")):
                continue
            try:
                text = req_path.read_text(encoding="utf-8", errors="ignore")
                for line in text.splitlines():
                    cleaned = line.split("#")[0].strip()
                    if not cleaned or cleaned.startswith(("-", "http://", "https://")):
                        continue
                    # Strip version specifiers: fastapi>=0.68.0, requests==2.0
                    pkg_match = re.match(r"^([a-zA-Z0-9_\-\.]+)", cleaned)
                    if pkg_match:
                        raw_pkg = pkg_match.group(1).lower().replace("-", "_")
                        packages.add(raw_pkg)
                        # Add mapped imports
                        for imp in PACKAGE_TO_IMPORT_MAP.get(raw_pkg, set()):
                            packages.add(imp.lower().replace("-", "_"))
            except Exception:
                pass

        # 2. pyproject.toml
        for pyproj in repo.glob("**/pyproject.toml"):
            if any(d in pyproj.parts for d in (".git", "node_modules", ".venv", "venv")):
                continue
            try:
                text = pyproj.read_text(encoding="utf-8", errors="ignore")
                if tomllib is not None:
                    data = tomllib.loads(text)
                    # [project.dependencies]
                    for dep in data.get("project", {}).get("dependencies", []):
                        m = re.match(r"^([a-zA-Z0-9_\-\.]+)", dep.strip())
                        if m:
                            p = m.group(1).lower().replace("-", "_")
                            packages.add(p)
                            for imp in PACKAGE_TO_IMPORT_MAP.get(p, set()):
                                packages.add(imp.lower().replace("-", "_"))

                    # [project.optional-dependencies] (dev, test, docs groups, etc.)
                    for group_deps in data.get("project", {}).get("optional-dependencies", {}).values():
                        if isinstance(group_deps, list):
                            for dep in group_deps:
                                m = re.match(r"^([a-zA-Z0-9_\-\.]+)", dep.strip())
                                if m:
                                    p = m.group(1).lower().replace("-", "_")
                                    packages.add(p)
                                    for imp in PACKAGE_TO_IMPORT_MAP.get(p, set()):
                                        packages.add(imp.lower().replace("-", "_"))

                    # [tool.poetry.dependencies]
                    for k in data.get("tool", {}).get("poetry", {}).get("dependencies", {}).keys():
                        if k.lower() != "python":
                            p = k.lower().replace("-", "_")
                            packages.add(p)
                            for imp in PACKAGE_TO_IMPORT_MAP.get(p, set()):
                                packages.add(imp.lower().replace("-", "_"))

                    # [tool.poetry.dev-dependencies]
                    for k in data.get("tool", {}).get("poetry", {}).get("dev-dependencies", {}).keys():
                        p = k.lower().replace("-", "_")
                        packages.add(p)
                        for imp in PACKAGE_TO_IMPORT_MAP.get(p, set()):
                            packages.add(imp.lower().replace("-", "_"))

                    # [tool.poetry.group.<group>.dependencies]
                    for group_val in data.get("tool", {}).get("poetry", {}).get("group", {}).values():
                        if isinstance(group_val, dict):
                            for k in group_val.get("dependencies", {}).keys():
                                p = k.lower().replace("-", "_")
                                packages.add(p)
                                for imp in PACKAGE_TO_IMPORT_MAP.get(p, set()):
                                    packages.add(imp.lower().replace("-", "_"))
            except Exception:
                pass

        # 3. setup.py install_requires & extras_require
        setup_py = repo / "setup.py"
        if setup_py.is_file():
            try:
                content = setup_py.read_text(encoding="utf-8", errors="ignore")
                req_match = re.search(r"install_requires\s*=\s*\[(.*?)\]", content, re.DOTALL)
                if req_match:
                    for item in re.findall(r"['\"]([a-zA-Z0-9_\-\.]+)[^'\"]*['\"]", req_match.group(1)):
                        p = item.lower().replace("-", "_")
                        packages.add(p)
                        for imp in PACKAGE_TO_IMPORT_MAP.get(p, set()):
                            packages.add(imp.lower().replace("-", "_"))

                extras_match = re.search(r"extras_require\s*=\s*\{(.*?)\}", content, re.DOTALL)
                if extras_match:
                    for item in re.findall(r"['\"]([a-zA-Z0-9_\-\.]+)[^'\"]*['\"]", extras_match.group(1)):
                        p = item.lower().replace("-", "_")
                        packages.add(p)
                        for imp in PACKAGE_TO_IMPORT_MAP.get(p, set()):
                            packages.add(imp.lower().replace("-", "_"))
            except Exception:
                pass

        # Expand declared packages with known transitive dependencies provided by frameworks
        expanded: Set[str] = set(packages)
        for pkg in list(packages):
            for trans in KNOWN_TRANSITIVE_IMPORTS.get(pkg, set()):
                norm_trans = trans.lower().replace("-", "_")
                expanded.add(norm_trans)
                for imp in PACKAGE_TO_IMPORT_MAP.get(norm_trans, set()):
                    expanded.add(imp.lower().replace("-", "_"))
        return expanded

    @classmethod
    def get_declared_node_dependencies(cls, repo: Path) -> Set[str]:
        """Collects all declared dependencies from package.json files."""
        packages: Set[str] = set()
        for pkg_path in repo.glob("**/package.json"):
            if any(d in pkg_path.parts for d in (".git", "node_modules", "dist", "build")):
                continue
            try:
                data = json.loads(pkg_path.read_text(encoding="utf-8"))
                for sec in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
                    for k in data.get(sec, {}).keys():
                        packages.add(k.lower())
            except Exception:
                pass
        expanded: Set[str] = set(packages)
        for pkg in list(packages):
            for trans in KNOWN_TRANSITIVE_IMPORTS.get(pkg, set()):
                expanded.add(trans.lower())
        return expanded

    # --------------------------------------------------------------------------
    # Helper checks
    # --------------------------------------------------------------------------

    def _get_missing_imports_for_file(
        self, py_file: Path, declared_pkgs: Set[str], repo: Path, include_try_except: bool = True
    ) -> List[str]:
        """Identifies any third-party imports in py_file that are not declared."""
        imports = CodeAnalyzer.extract_imports_from_file(py_file, include_try_except=include_try_except)
        missing: List[str] = []

        local_modules = self._get_local_python_modules(repo)

        # Include sibling modules from py_file's parent directory
        if py_file.parent.is_dir():
            for sibling in py_file.parent.iterdir():
                if sibling.is_file() and sibling.suffix == ".py":
                    local_modules.add(sibling.stem.lower().replace("-", "_"))
                elif sibling.is_dir() and ((sibling / "__init__.py").exists() or any(sibling.glob("*.py"))):
                    local_modules.add(sibling.name.lower().replace("-", "_"))

        TEST_FRAMEWORK_MODULES = {
            "pytest", "pytest_asyncio", "pytest_mock", "unittest", "conftest",
            "mock", "hypothesis", "coverage", "responses", "freezegun", "tox"
        }
        is_test_file = (
            any(p in py_file.parts for p in ("tests", "test"))
            or py_file.stem.startswith("test_")
            or py_file.stem.endswith("_test")
            or py_file.stem == "conftest"
        )

        for imp in imports:
            imp_norm = imp.lower().replace("-", "_")
            if is_test_file and imp_norm in TEST_FRAMEWORK_MODULES:
                continue

            if imp_norm in local_modules:
                continue

            # Check direct match
            if imp_norm in declared_pkgs:
                continue

            # Check if import corresponds to a mapped package
            known_pkgs = IMPORT_TO_PACKAGE_MAP.get(imp_norm, set())
            if any(k.lower().replace("-", "_") in declared_pkgs for k in known_pkgs):
                continue

            missing.append(imp)

        return missing

    @staticmethod
    def _get_local_python_modules(repo: Path) -> Set[str]:
        """Identifies top-level local package/module names in the repository."""
        local_mods: Set[str] = set()
        for candidate_root in (
            repo, repo / "src", repo / "backend", repo / "app",
            repo / "lib", repo / "pkg", repo / "packages", repo / "tests"
        ):
            if not candidate_root.is_dir():
                continue
            for item in candidate_root.iterdir():
                if item.name.startswith((".", "_")):
                    continue
                if item.is_dir() and ((item / "__init__.py").exists() or any(item.glob("*.py"))):
                    local_mods.add(item.name.lower().replace("-", "_"))
                elif item.is_file() and item.suffix == ".py":
                    local_mods.add(item.stem.lower().replace("-", "_"))
        return local_mods

    @staticmethod
    def _is_local_node_path(specifier: str, file_dir: Path, repo: Path) -> bool:
        """Checks if a specifier corresponds to a local monorepo package or file."""
        if (file_dir / specifier).exists() or (file_dir / f"{specifier}.ts").exists() or (file_dir / f"{specifier}.js").exists():
            return True
        for pkg_dir in (repo / "packages", repo / "apps"):
            if (pkg_dir / specifier).is_dir():
                return True
        return False
