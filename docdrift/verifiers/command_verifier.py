"""Multi-ecosystem command and manifest verifier for DocDrift."""

import difflib
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

from docdrift.models import Claim, ClaimType, Verdict, VerificationResult
from docdrift.verifiers.base import BaseVerifier


class MultiManifestVerifier(BaseVerifier):
    """Verifies commands, scripts, targets, and docker ports across multiple ecosystems.

    Supported ecosystems:
    - Node / JS / TS: package.json (scripts), pnpm-workspace.yaml, turbo.json
    - Python: pyproject.toml ([project.scripts], [tool.poetry.scripts]), setup.py, requirements.txt
    - Docker: docker-compose.yml (services, ports, env) and Dockerfile (EXPOSE)
    - Rust: Cargo.toml ([[bin]], package.name)
    - Go: go.mod, go files
    - Make: Makefile targets
    """

    def can_verify(self, claim: Claim) -> bool:
        """Checks if this verifier handles COMMAND_SCRIPT or DOCKER_PORT claims."""
        return claim.claim_type in (ClaimType.COMMAND_SCRIPT, ClaimType.DOCKER_PORT)

    def verify(self, claim: Claim, repo_path: str) -> VerificationResult:
        """Verifies the claim against repository manifests."""
        repo = Path(repo_path).resolve()

        if claim.claim_type == ClaimType.DOCKER_PORT:
            return self._verify_docker_port(claim, repo)

        if claim.claim_type == ClaimType.COMMAND_SCRIPT:
            return self._verify_command_script(claim, repo)

        return VerificationResult(
            verdict=Verdict.UNKNOWN,
            evidence=f"Unhandled claim type '{claim.claim_type}'.",
            suggested_value=None,
        )

    # --------------------------------------------------------------------------
    # Docker Port Verification
    # --------------------------------------------------------------------------

    def _verify_docker_port(self, claim: Claim, repo: Path) -> VerificationResult:
        """Verifies a claimed exposed docker port against compose files and Dockerfile."""
        claimed_port_raw = claim.extracted_value.strip().strip("'\"`")
        claimed_ports = self._extract_port_numbers(claimed_port_raw)

        if not claimed_ports:
            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence=f"Could not parse valid port number from '{claim.extracted_value}'.",
                suggested_value=None,
            )

        target_port = claimed_ports[0]

        # Collect exposed ports from compose files and Dockerfiles
        compose_ports = self._get_docker_compose_ports(repo)
        dockerfile_ports = self._get_dockerfile_ports(repo)
        all_declared_ports = list(dict.fromkeys(compose_ports + dockerfile_ports))

        if not all_declared_ports and not self._has_docker_manifests(repo):
            return VerificationResult(
                verdict=Verdict.UNKNOWN,
                evidence=f"No Dockerfile or docker-compose found in repository; port '{target_port}' appears to be a local application port.",
                suggested_value=None,
            )

        # Check for direct match or host:container match
        matched = False
        for p in all_declared_ports:
            declared_nums = self._extract_port_numbers(p)
            if target_port in declared_nums:
                matched = True
                break

        if matched:
            return VerificationResult(
                verdict=Verdict.VERIFIED,
                evidence=f"Docker port '{target_port}' is exposed in repository manifests: {', '.join(all_declared_ports)}.",
                suggested_value=None,
                metadata={"declared_ports": all_declared_ports},
            )

        # Port missing - find closest match
        closest = None
        if all_declared_ports:
            # Match against string representations
            matches = difflib.get_close_matches(target_port, all_declared_ports, n=1, cutoff=0.3)
            if matches:
                closest = matches[0]
            else:
                closest = all_declared_ports[0]

        evidence = (
            f"Port '{claimed_port_raw}' is not exposed in Docker manifests. "
            f"Declared ports: {', '.join(all_declared_ports) if all_declared_ports else 'None'}."
        )
        return VerificationResult(
            verdict=Verdict.DRIFTED,
            evidence=evidence,
            suggested_value=closest,
            metadata={"declared_ports": all_declared_ports},
        )

    # --------------------------------------------------------------------------
    # Command / Script Verification
    # --------------------------------------------------------------------------

    def _verify_command_script(self, claim: Claim, repo: Path) -> VerificationResult:
        """Dispatches verification to the appropriate ecosystem handler."""
        cmd_str = claim.extracted_value.strip().strip("`")
        raw_line = claim.raw_line or ""

        # Normalize command string if full command was in raw_line
        full_cmd = cmd_str
        if not self._is_executable_command(cmd_str) and self._is_executable_command(raw_line):
            full_cmd = self._extract_command_from_line(raw_line) or cmd_str

        tokens = self._tokenize(full_cmd)
        if not tokens:
            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence=f"Empty command claim '{cmd_str}'.",
                suggested_value=None,
            )

        tool = tokens[0].lower()

        # Dispatch based on tool
        if tool in ("npm", "yarn", "pnpm", "bun"):
            return self._verify_node_command(full_cmd, tokens, repo)

        if tool in ("poetry", "uv", "python", "python3"):
            return self._verify_python_command(full_cmd, tokens, repo)

        if tool in ("docker", "docker-compose"):
            return self._verify_docker_command(full_cmd, tokens, repo)

        if tool == "cargo":
            return self._verify_cargo_command(full_cmd, tokens, repo)

        if tool == "go":
            return self._verify_go_command(full_cmd, tokens, repo)

        if tool == "make":
            return self._verify_make_command(full_cmd, tokens, repo)

        # If it's a standalone script or target name (e.g. 'build' or 'test')
        # check if it matches a Makefile or package.json script
        fallback_res = self._verify_standalone_script(cmd_str, repo)
        if fallback_res is not None:
            return fallback_res

        return VerificationResult(
            verdict=Verdict.UNKNOWN,
            evidence=f"Command '{full_cmd}' does not map to a recognized manifest verifier.",
            suggested_value=None,
        )

    # --------------------------------------------------------------------------
    # Node / JS / TS Verifier
    # --------------------------------------------------------------------------

    def _verify_node_command(self, full_cmd: str, tokens: List[str], repo: Path) -> VerificationResult:
        """Validates 'npm run <cmd>', 'yarn <cmd>', 'pnpm <cmd>', 'bun run <cmd>'."""
        tool = tokens[0]
        scripts = self._get_node_scripts(repo)
        pkg_json = repo / "package.json"

        if not pkg_json.exists() and not scripts:
            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence=f"Command '{full_cmd}' requires 'package.json', but none was found in the repository root.",
                suggested_value=None,
            )

        # Extract target script
        target_script = None
        STANDARD_NPM_COMMANDS = {
            "install", "i", "init", "ci", "add", "update", "publish",
            "pack", "link", "unlink", "audit", "outdated", "help", "view",
            "info", "prefix", "root", "bin", "login", "whoami", "version",
            "cache", "config", "dedupe", "prune", "doctor"
        }
        if tool == "npm":
            if len(tokens) >= 2 and tokens[1] in STANDARD_NPM_COMMANDS:
                return VerificationResult(
                    verdict=Verdict.VERIFIED,
                    evidence=f"Standard npm command '{full_cmd}' verified.",
                    suggested_value=None,
                )
            if len(tokens) >= 3 and tokens[1] in ("run", "run-script"):
                target_script = tokens[2]
            elif len(tokens) >= 2 and tokens[1] in ("test", "start", "stop", "restart", "build"):
                target_script = tokens[1]
            elif len(tokens) >= 2:
                target_script = tokens[1]
        elif tool in ("yarn", "pnpm", "bun"):
            # yarn <script> or yarn run <script>
            if len(tokens) >= 3 and tokens[1] == "run":
                target_script = tokens[2]
            elif len(tokens) >= 2:
                # filter out non-script subcommands like 'yarn install', 'pnpm add', 'bun install'
                if tokens[1] not in ("install", "add", "remove", "init", "publish", "pack"):
                    target_script = tokens[1]
                else:
                    return VerificationResult(
                        verdict=Verdict.VERIFIED,
                        evidence=f"Standard package manager command '{full_cmd}' verified.",
                        suggested_value=None,
                    )

        if not target_script:
            # General package manager command like 'npm install'
            return VerificationResult(
                verdict=Verdict.VERIFIED,
                evidence=f"Package manager command '{full_cmd}' verified against package.json.",
                suggested_value=None,
            )

        # Check script existence
        available = list(scripts.keys())
        if target_script in scripts:
            return VerificationResult(
                verdict=Verdict.VERIFIED,
                evidence=f"Script '{target_script}' found in package.json/workspace: '{scripts[target_script]}'.",
                suggested_value=None,
                metadata={"script": target_script, "command": scripts[target_script]},
            )

        # Missing script - find closest match
        closest = self._find_closest_string(target_script, available)
        suggested = None
        if closest:
            suggested = full_cmd.replace(target_script, closest)

        evidence = (
            f"Script '{target_script}' is not defined in package.json. "
            f"Available scripts: {', '.join(available) if available else 'None'}."
        )
        if closest:
            evidence += f" Did you mean '{closest}'?"

        return VerificationResult(
            verdict=Verdict.DRIFTED,
            evidence=evidence,
            suggested_value=suggested,
            metadata={"available_scripts": available, "closest": closest},
        )

    def _get_node_scripts(self, repo: Path) -> Dict[str, str]:
        """Collects scripts from package.json, turbo.json, and pnpm-workspace."""
        scripts: Dict[str, str] = {}

        # 1. Root package.json
        pkg_file = repo / "package.json"
        if pkg_file.exists():
            try:
                data = json.loads(pkg_file.read_text(encoding="utf-8"))
                for k, v in data.get("scripts", {}).items():
                    scripts[k] = str(v)
            except Exception:
                pass

        # 2. turbo.json
        turbo_file = repo / "turbo.json"
        if turbo_file.exists():
            try:
                tdata = json.loads(turbo_file.read_text(encoding="utf-8"))
                tasks = tdata.get("tasks", {}) or tdata.get("pipeline", {})
                for k in tasks.keys():
                    # task can be 'build' or '//#build'
                    clean_task = k.split("#")[-1]
                    if clean_task not in scripts:
                        scripts[clean_task] = f"turbo {clean_task}"
            except Exception:
                pass

        # 3. pnpm-workspace or workspaces subpackages
        packages_dir = repo / "packages"
        if packages_dir.is_dir():
            for subpkg in packages_dir.glob("*/package.json"):
                try:
                    data = json.loads(subpkg.read_text(encoding="utf-8"))
                    for k, v in data.get("scripts", {}).items():
                        if k not in scripts:
                            scripts[k] = str(v)
                except Exception:
                    pass

        return scripts

    # --------------------------------------------------------------------------
    # Python Verifier
    # --------------------------------------------------------------------------

    def _verify_python_command(self, full_cmd: str, tokens: List[str], repo: Path) -> VerificationResult:
        """Validates 'poetry run <cmd>', 'uv run <cmd>', 'python -m <module>'."""
        tool = tokens[0]

        # 1. poetry run <cmd>
        if tool == "poetry":
            if len(tokens) >= 3 and tokens[1] == "run":
                subcmd = tokens[2]
                scripts = self._get_python_scripts(repo)
                if subcmd in scripts:
                    return VerificationResult(
                        verdict=Verdict.VERIFIED,
                        evidence=f"Poetry script '{subcmd}' verified in pyproject.toml.",
                        suggested_value=None,
                        metadata={"script": subcmd, "entrypoint": scripts[subcmd]},
                    )
                available = list(scripts.keys())
                closest = self._find_closest_string(subcmd, available)
                suggested = full_cmd.replace(subcmd, closest) if closest else None
                evidence = (
                    f"Script '{subcmd}' not found under [tool.poetry.scripts] or [project.scripts] in pyproject.toml. "
                    f"Available scripts: {', '.join(available) if available else 'None'}."
                )
                if closest:
                    evidence += f" Did you mean '{closest}'?"
                return VerificationResult(
                    verdict=Verdict.DRIFTED,
                    evidence=evidence,
                    suggested_value=suggested,
                )
            # Other poetry commands (poetry install, poetry add)
            return VerificationResult(
                verdict=Verdict.VERIFIED,
                evidence=f"Poetry command '{full_cmd}' verified.",
                suggested_value=None,
            )

        # 2. uv run <cmd>
        if tool == "uv":
            if len(tokens) >= 3 and tokens[1] == "run":
                subcmd = tokens[2]
                scripts = self._get_python_scripts(repo)
                # Could be a script entrypoint or a local python script
                if subcmd in scripts:
                    return VerificationResult(
                        verdict=Verdict.VERIFIED,
                        evidence=f"uv script '{subcmd}' verified in pyproject.toml: '{scripts[subcmd]}'.",
                        suggested_value=None,
                    )
                # Check if subcmd is a local file, e.g. 'uv run main.py'
                local_file = (repo / subcmd).resolve()
                if local_file.exists():
                    return VerificationResult(
                        verdict=Verdict.VERIFIED,
                        evidence=f"uv target file '{subcmd}' exists in repository.",
                        suggested_value=None,
                    )
                available = list(scripts.keys())
                closest = self._find_closest_string(subcmd, available)
                suggested = full_cmd.replace(subcmd, closest) if closest else None
                evidence = (
                    f"uv command target '{subcmd}' not found in pyproject.toml scripts or as a file in repo. "
                    f"Available scripts: {', '.join(available) if available else 'None'}."
                )
                if closest:
                    evidence += f" Did you mean '{closest}'?"
                return VerificationResult(
                    verdict=Verdict.DRIFTED,
                    evidence=evidence,
                    suggested_value=suggested,
                )
            return VerificationResult(
                verdict=Verdict.VERIFIED,
                evidence=f"uv command '{full_cmd}' verified.",
                suggested_value=None,
            )

        # 3. python -m <module> or python <script.py>
        if tool in ("python", "python3"):
            if len(tokens) >= 3 and tokens[1] == "-m":
                module = tokens[2]
                return self._verify_python_module(module, full_cmd, repo)
            elif len(tokens) >= 2:
                script_path = tokens[1]
                target = (repo / script_path).resolve()
                if target.exists():
                    return VerificationResult(
                        verdict=Verdict.VERIFIED,
                        evidence=f"Python script '{script_path}' exists in repository.",
                        suggested_value=None,
                    )
                all_py = [p.relative_to(repo).as_posix() for p in repo.glob("**/*.py")]
                closest = self._find_closest_string(script_path, all_py)
                suggested = full_cmd.replace(script_path, closest) if closest else None
                return VerificationResult(
                    verdict=Verdict.DRIFTED,
                    evidence=f"Python script '{script_path}' not found in repository.",
                    suggested_value=suggested,
                )

        return VerificationResult(
            verdict=Verdict.UNKNOWN,
            evidence=f"Python command '{full_cmd}' could not be verified.",
            suggested_value=None,
        )

    def _verify_python_module(self, module: str, full_cmd: str, repo: Path) -> VerificationResult:
        """Verifies if python -m <module> exists in repo or src/."""
        # Split module: 'docdrift.cli' -> 'docdrift/cli.py' or 'docdrift/cli/__init__.py'
        mod_parts = module.split(".")
        candidates = [
            repo / (os.path.join(*mod_parts) + ".py"),
            repo / os.path.join(*mod_parts) / "__init__.py",
            repo / "src" / (os.path.join(*mod_parts) + ".py"),
            repo / "src" / os.path.join(*mod_parts) / "__init__.py",
        ]

        for cand in candidates:
            if cand.exists():
                return VerificationResult(
                    verdict=Verdict.VERIFIED,
                    evidence=f"Python module '{module}' verified at '{cand.relative_to(repo).as_posix()}'.",
                    suggested_value=None,
                )

        # Collect available python packages/modules
        available_mods: Set[str] = set()
        for py_path in repo.glob("**/*.py"):
            try:
                rel = py_path.relative_to(repo).as_posix()
            except ValueError:
                continue
            if any(part in rel for part in (".venv", "venv", "__pycache__", "build", "dist", ".git")):
                continue
            # Strip src/ if present
            if rel.startswith("src/"):
                rel = rel[4:]
            if rel.endswith("/__init__.py"):
                available_mods.add(rel[:-12].replace("/", "."))
            elif rel.endswith(".py"):
                available_mods.add(rel[:-3].replace("/", "."))

        closest = self._find_closest_string(module, list(available_mods))
        suggested = full_cmd.replace(module, closest) if closest else None
        evidence = f"Python module '{module}' not found in repository."
        if closest:
            evidence += f" Did you mean '{closest}'?"

        return VerificationResult(
            verdict=Verdict.DRIFTED,
            evidence=evidence,
            suggested_value=suggested,
        )

    def _get_python_scripts(self, repo: Path) -> Dict[str, str]:
        """Reads [project.scripts] and [tool.poetry.scripts] from pyproject.toml."""
        scripts: Dict[str, str] = {}
        pyproj = repo / "pyproject.toml"

        if pyproj.exists():
            content = pyproj.read_text(encoding="utf-8")
            if tomllib is not None:
                try:
                    data = tomllib.loads(content)
                    # 1. [project.scripts]
                    for k, v in data.get("project", {}).get("scripts", {}).items():
                        scripts[k] = str(v)
                    # 2. [tool.poetry.scripts]
                    for k, v in data.get("tool", {}).get("poetry", {}).get("scripts", {}).items():
                        scripts[k] = str(v)
                except Exception:
                    pass
            else:
                # Regex fallback for scripts
                scripts.update(self._parse_toml_scripts_fallback(content))

        # Check setup.py fallback
        setup_py = repo / "setup.py"
        if setup_py.exists():
            content = setup_py.read_text(encoding="utf-8", errors="ignore")
            # entry_points={'console_scripts': ['foo = foo.bar:main']}
            matches = re.findall(r"['\"]([a-zA-Z0-9_\-]+)\s*=\s*[a-zA-Z0-9_\:\.]+['\"]", content)
            for m in matches:
                scripts[m] = "setup.py console_script"

        return scripts

    @staticmethod
    def _parse_toml_scripts_fallback(content: str) -> Dict[str, str]:
        """Regex fallback to extract scripts when tomllib is missing."""
        scripts: Dict[str, str] = {}
        in_section = False
        for line in content.splitlines():
            line_s = line.strip()
            if line_s.startswith("[project.scripts]") or line_s.startswith("[tool.poetry.scripts]"):
                in_section = True
                continue
            if line_s.startswith("[") and in_section:
                in_section = False
            if in_section and "=" in line_s and not line_s.startswith("#"):
                parts = line_s.split("=", 1)
                key = parts[0].strip().strip("'\"")
                val = parts[1].strip().strip("'\"")
                scripts[key] = val
        return scripts

    # --------------------------------------------------------------------------
    # Docker Verifier
    # --------------------------------------------------------------------------

    def _verify_docker_command(self, full_cmd: str, tokens: List[str], repo: Path) -> VerificationResult:
        """Validates docker compose service names and options."""
        # Identify compose subcommand: docker compose up <service> or docker-compose up <service>
        services = self._get_docker_services(repo)

        subcmd_idx = -1
        for i, tok in enumerate(tokens):
            if tok in ("up", "run", "build", "stop", "restart", "logs", "down"):
                subcmd_idx = i
                break

        if subcmd_idx != -1 and len(tokens) > subcmd_idx + 1:
            # Extract target service (skip flags like -d)
            target_service = None
            for tok in tokens[subcmd_idx + 1:]:
                if not tok.startswith("-"):
                    target_service = tok
                    break

            if target_service:
                if target_service in services:
                    return VerificationResult(
                        verdict=Verdict.VERIFIED,
                        evidence=f"Docker service '{target_service}' verified in compose file.",
                        suggested_value=None,
                        metadata={"service": target_service},
                    )

                available = list(services.keys())
                closest = self._find_closest_string(target_service, available)
                suggested = full_cmd.replace(target_service, closest) if closest else None
                evidence = (
                    f"Docker service '{target_service}' not found in compose file. "
                    f"Available services: {', '.join(available) if available else 'None'}."
                )
                if closest:
                    evidence += f" Did you mean '{closest}'?"
                return VerificationResult(
                    verdict=Verdict.DRIFTED,
                    evidence=evidence,
                    suggested_value=suggested,
                )

        # Generic docker command
        return VerificationResult(
            verdict=Verdict.VERIFIED,
            evidence=f"Docker command '{full_cmd}' format verified.",
            suggested_value=None,
        )

    def _get_docker_services(self, repo: Path) -> Dict[str, Any]:
        """Extracts service definitions from docker-compose.yml."""
        compose_files = [
            repo / "docker-compose.yml",
            repo / "docker-compose.yaml",
            repo / "compose.yml",
            repo / "compose.yaml",
        ]

        services: Dict[str, Any] = {}
        for cf in compose_files:
            if not cf.exists():
                continue
            content = cf.read_text(encoding="utf-8", errors="ignore")
            if yaml is not None:
                try:
                    data = yaml.safe_load(content)
                    if isinstance(data, dict) and "services" in data:
                        for s_name, s_cfg in data["services"].items():
                            services[s_name] = s_cfg
                        return services
                except Exception:
                    pass

            # Robust line-by-line fallback parser for services
            in_services = False
            for line in content.splitlines():
                stripped = line.rstrip()
                if re.match(r"^services:\s*$", stripped):
                    in_services = True
                    continue
                if in_services:
                    if re.match(r"^\S", stripped):  # unindented top-level section
                        break
                    match = re.match(r"^[ ]{2,4}([a-zA-Z0-9_\-]+):\s*$", stripped)
                    if match:
                        services[match.group(1)] = {}
            if services:
                return services

        return services

    def _get_docker_compose_ports(self, repo: Path) -> List[str]:
        """Extracts declared port strings from docker-compose files."""
        ports: List[str] = []
        compose_files = [
            repo / "docker-compose.yml",
            repo / "docker-compose.yaml",
            repo / "compose.yml",
            repo / "compose.yaml",
        ]

        for cf in compose_files:
            if not cf.exists():
                continue
            content = cf.read_text(encoding="utf-8", errors="ignore")
            if yaml is not None:
                try:
                    data = yaml.safe_load(content)
                    if isinstance(data, dict) and "services" in data:
                        for s_cfg in data["services"].values():
                            if isinstance(s_cfg, dict):
                                for p in s_cfg.get("ports", []):
                                    ports.append(str(p))
                                for p in s_cfg.get("expose", []):
                                    ports.append(str(p))
                        if ports:
                            return ports
                except Exception:
                    pass

            # Fallback regex extraction for ports
            in_ports = False
            for line in content.splitlines():
                if re.search(r"^\s+(?:ports|expose):\s*$", line):
                    in_ports = True
                    continue
                if in_ports:
                    match = re.search(r"-\s*['\"]?([0-9:\-/]+)['\"]?", line)
                    if match:
                        ports.append(match.group(1))
                    elif re.match(r"^\s+[a-zA-Z0-9_]+:", line):
                        in_ports = False

        return ports

    def _get_dockerfile_ports(self, repo: Path) -> List[str]:
        """Extracts EXPOSE ports from Dockerfiles."""
        ports: List[str] = []
        dockerfiles = list(repo.glob("Dockerfile*"))
        for df in dockerfiles:
            if df.is_file():
                content = df.read_text(encoding="utf-8", errors="ignore")
                for line in content.splitlines():
                    match = re.match(r"^\s*EXPOSE\s+([0-9/a-zA-Z\s]+)", line, re.IGNORECASE)
                    if match:
                        raw = match.group(1).strip()
                        for token in raw.split():
                            ports.append(token.split("/")[0])
        return ports

    def _has_docker_manifests(self, repo: Path) -> bool:
        """Checks if any docker compose or Dockerfile exists."""
        compose_names = ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml")
        if any((repo / name).exists() for name in compose_names):
            return True
        return bool(list(repo.glob("Dockerfile*")))

    # --------------------------------------------------------------------------
    # Rust Cargo Verifier
    # --------------------------------------------------------------------------

    def _verify_cargo_command(self, full_cmd: str, tokens: List[str], repo: Path) -> VerificationResult:
        """Validates 'cargo run', 'cargo test', 'cargo run --bin <name>'."""
        cargo_toml = repo / "Cargo.toml"
        if not cargo_toml.exists():
            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence=f"Cargo command '{full_cmd}' requires 'Cargo.toml', but none was found.",
                suggested_value=None,
            )

        bins = self._get_cargo_bins(repo)

        # Check for --bin <name>
        target_bin = None
        for i, tok in enumerate(tokens):
            if tok == "--bin" and i + 1 < len(tokens):
                target_bin = tokens[i + 1]
                break

        if target_bin:
            if target_bin in bins:
                return VerificationResult(
                    verdict=Verdict.VERIFIED,
                    evidence=f"Rust binary '{target_bin}' verified in Cargo.toml / src/bin.",
                    suggested_value=None,
                )
            closest = self._find_closest_string(target_bin, bins)
            suggested = full_cmd.replace(target_bin, closest) if closest else None
            evidence = (
                f"Binary '{target_bin}' not found in Cargo.toml. "
                f"Available binaries: {', '.join(bins) if bins else 'None'}."
            )
            if closest:
                evidence += f" Did you mean '{closest}'?"
            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence=evidence,
                suggested_value=suggested,
            )

        # Standard cargo run or test
        if len(tokens) >= 2 and tokens[1] in ("run", "test", "build", "check"):
            return VerificationResult(
                verdict=Verdict.VERIFIED,
                evidence=f"Cargo command '{full_cmd}' verified against Cargo.toml.",
                suggested_value=None,
            )

        return VerificationResult(
            verdict=Verdict.VERIFIED,
            evidence=f"Cargo command '{full_cmd}' verified.",
            suggested_value=None,
        )

    def _get_cargo_bins(self, repo: Path) -> List[str]:
        """Extracts declared Rust binaries from Cargo.toml and src/bin/."""
        bins: List[str] = []
        cargo_toml = repo / "Cargo.toml"

        if cargo_toml.exists():
            content = cargo_toml.read_text(encoding="utf-8", errors="ignore")
            # Parse package.name
            pkg_name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
            if pkg_name_match:
                bins.append(pkg_name_match.group(1))

            # Parse [[bin]] tables
            bin_matches = re.findall(r'\[\[bin\]\][\s\S]*?name\s*=\s*["\']([^"\']+)["\']', content)
            bins.extend(bin_matches)

        # Check src/bin/*.rs
        src_bin = repo / "src" / "bin"
        if src_bin.is_dir():
            for f in src_bin.glob("*.rs"):
                bins.append(f.stem)

        return list(dict.fromkeys(bins))

    # --------------------------------------------------------------------------
    # Go Verifier
    # --------------------------------------------------------------------------

    def _verify_go_command(self, full_cmd: str, tokens: List[str], repo: Path) -> VerificationResult:
        """Validates 'go run', 'go build', 'go test'."""
        go_mod = repo / "go.mod"
        go_files = list(repo.glob("*.go")) + list(repo.glob("**/*.go"))

        if not go_mod.exists() and not go_files:
            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence=f"Go command '{full_cmd}' requires 'go.mod' or .go source files, none found.",
                suggested_value=None,
            )

        if len(tokens) >= 2:
            sub = tokens[1]
            if sub in ("test", "build", "run"):
                if len(tokens) >= 3:
                    arg = tokens[2]
                    # If pointing to a specific file like 'go run main.go'
                    if arg.endswith(".go"):
                        target_file = (repo / arg).resolve()
                        if target_file.exists():
                            return VerificationResult(
                                verdict=Verdict.VERIFIED,
                                evidence=f"Go target file '{arg}' exists in repository.",
                                suggested_value=None,
                            )
                        # Find closest .go file
                        go_f_names = [f.name for f in repo.glob("**/*.go")]
                        closest = self._find_closest_string(arg, go_f_names)
                        suggested = full_cmd.replace(arg, closest) if closest else None
                        return VerificationResult(
                            verdict=Verdict.DRIFTED,
                            evidence=f"Go source file '{arg}' does not exist.",
                            suggested_value=suggested,
                        )
                return VerificationResult(
                    verdict=Verdict.VERIFIED,
                    evidence=f"Go command '{full_cmd}' verified against repository.",
                    suggested_value=None,
                )

        return VerificationResult(
            verdict=Verdict.VERIFIED,
            evidence=f"Go command '{full_cmd}' verified.",
            suggested_value=None,
        )

    # --------------------------------------------------------------------------
    # Make Verifier
    # --------------------------------------------------------------------------

    def _verify_make_command(self, full_cmd: str, tokens: List[str], repo: Path) -> VerificationResult:
        """Validates 'make <target>' against Makefile targets."""
        makefile = self._find_makefile(repo)
        if not makefile:
            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence=f"Command '{full_cmd}' requires a Makefile, but none was found in repository root.",
                suggested_value=None,
            )

        targets = self._get_make_targets(makefile)

        # Extract target (skip options like -j4, -s, -f)
        target = None
        for tok in tokens[1:]:
            if not tok.startswith("-"):
                target = tok
                break

        if not target:
            # Bare 'make' invokes default/first target
            first_target = targets[0] if targets else "all"
            return VerificationResult(
                verdict=Verdict.VERIFIED,
                evidence=f"Default make command executes target '{first_target}'.",
                suggested_value=None,
            )

        if target in targets:
            return VerificationResult(
                verdict=Verdict.VERIFIED,
                evidence=f"Target '{target}' verified in '{makefile.name}'.",
                suggested_value=None,
                metadata={"target": target, "makefile": makefile.name},
            )

        closest = self._find_closest_string(target, targets)
        suggested = full_cmd.replace(target, closest) if closest else None
        evidence = (
            f"Make target '{target}' not found in '{makefile.name}'. "
            f"Available targets: {', '.join(targets) if targets else 'None'}."
        )
        if closest:
            evidence += f" Did you mean '{closest}'?"

        return VerificationResult(
            verdict=Verdict.DRIFTED,
            evidence=evidence,
            suggested_value=suggested,
            metadata={"available_targets": targets, "closest": closest},
        )

    def _find_makefile(self, repo: Path) -> Optional[Path]:
        """Locates Makefile in repository."""
        for name in ("Makefile", "makefile", "GNUmakefile"):
            mf = repo / name
            if mf.exists():
                return mf
        return None

    def _get_make_targets(self, makefile: Path) -> List[str]:
        """Parses Makefile target names."""
        targets: List[str] = []
        try:
            content = makefile.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return targets

        for line in content.splitlines():
            # Match target line: target: [prereqs]
            # Exclude special targets starting with . (like .PHONY, .DEFAULT)
            match = re.match(r"^([a-zA-Z0-9_\-\.]+)\s*:", line)
            if match:
                tgt = match.group(1)
                if not tgt.startswith("."):
                    targets.append(tgt)

        return list(dict.fromkeys(targets))

    # --------------------------------------------------------------------------
    # Standalone script fallback
    # --------------------------------------------------------------------------

    def _verify_standalone_script(self, script_name: str, repo: Path) -> Optional[VerificationResult]:
        """Checks if a bare name matches a Makefile target or package.json script."""
        # 1. Check Makefile
        mf = self._find_makefile(repo)
        if mf:
            targets = self._get_make_targets(mf)
            if script_name in targets:
                return VerificationResult(
                    verdict=Verdict.VERIFIED,
                    evidence=f"Target '{script_name}' found in '{mf.name}'.",
                    suggested_value=None,
                )

        # 2. Check package.json scripts
        scripts = self._get_node_scripts(repo)
        if script_name in scripts:
            return VerificationResult(
                verdict=Verdict.VERIFIED,
                evidence=f"Script '{script_name}' found in package.json: '{scripts[script_name]}'.",
                suggested_value=None,
            )

        # 3. Check python scripts
        py_scripts = self._get_python_scripts(repo)
        if script_name in py_scripts:
            return VerificationResult(
                verdict=Verdict.VERIFIED,
                evidence=f"Script '{script_name}' found in pyproject.toml: '{py_scripts[script_name]}'.",
                suggested_value=None,
            )

        return None

    # --------------------------------------------------------------------------
    # Utility helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def _tokenize(cmd_str: str) -> List[str]:
        """Splits a command string safely into tokens."""
        try:
            return shlex.split(cmd_str, posix=True)
        except ValueError:
            return cmd_str.split()

    @staticmethod
    def _is_executable_command(line: str) -> bool:
        """Checks if string looks like an executable CLI command."""
        line_clean = line.strip().strip("`$#")
        if not line_clean:
            return False
        first = line_clean.split()[0].lower()
        return first in (
            "npm", "npx", "yarn", "pnpm", "bun",
            "python", "python3", "poetry", "uv",
            "docker", "docker-compose",
            "cargo", "go", "make",
        )

    @staticmethod
    def _extract_command_from_line(line: str) -> Optional[str]:
        """Extracts command from markdown code span or bash line."""
        # e.g. `npm run dev` or $ npm run dev
        clean = line.strip().lstrip("$# ").strip("`")
        if clean:
            return clean
        return None

    @staticmethod
    def _extract_port_numbers(port_str: str) -> List[str]:
        """Extracts integer port numbers from string like '8080:8080' or '8080'."""
        return re.findall(r"\b\d{2,5}\b", port_str)

    @classmethod
    def _find_closest_string(cls, target: str, choices: List[str]) -> Optional[str]:
        """Finds closest matching string using difflib."""
        if not choices or not target:
            return None
        matches = difflib.get_close_matches(target, choices, n=1, cutoff=0.3)
        if matches:
            return matches[0]
        fallback = difflib.get_close_matches(target, choices, n=1, cutoff=0.0)
        return fallback[0] if fallback else choices[0]
