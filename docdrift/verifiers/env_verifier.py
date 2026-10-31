"""Environment variable verifier for DocDrift."""

import difflib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from docdrift.models import Claim, ClaimType, Verdict, VerificationResult
from docdrift.verifiers.base import BaseVerifier


# File patterns to scan
ENV_TEMPLATE_FILES = (
    ".env.example",
    ".env.template",
    ".env.sample",
    ".env.dist",
    ".env.default",
    ".env.local.example",
    ".env",
)

DOCKER_FILES = (
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    "Dockerfile",
)

SOURCE_EXTENSIONS = (
    ".py",
    ".ts",
    ".js",
    ".go",
    ".rs",
    ".tsx",
    ".jsx",
    ".mjs",
    ".cjs",
)

IGNORE_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    "target",
    "coverage",
    ".pytest_cache",
    "__pycache__",
    ".next",
    ".nuxt",
    ".turbo",
    ".docdrift",
}

COMMON_FALSE_POSITIVES = {
    "GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS",
    "HTTP", "HTTPS", "JSON", "TRUE", "FALSE", "NULL", "NONE",
    "UTF", "ASCII", "ID", "OK", "ERROR", "INFO", "WARN", "DEBUG",
}


class EnvVerifier(BaseVerifier):
    """Verifies ClaimType.ENV_VAR claims against the repository codebase.

    Scans .env.example, .env.template, docker-compose.yml, and source code files
    (.py, .ts, .js, .go, .rs) for occurrences of the environment variable.
    If missing, identifies closest matching environment variable names in the codebase.
    """

    def __init__(self, memory: Optional[Any] = None) -> None:
        self.memory = memory

    def can_verify(self, claim: Claim) -> bool:
        """Determines if this verifier handles the given claim."""
        return claim.claim_type == ClaimType.ENV_VAR

    def verify(self, claim: Claim, repo_path: str) -> VerificationResult:
        """Verifies whether the environment variable exists or has drifted."""
        repo = Path(repo_path).resolve()
        raw_val = claim.extracted_value.strip().strip("'\"`")

        # Strip prefixes like $VAR or ${VAR} or export VAR
        clean_var = re.sub(r"^\$\{?|\}?$", "", raw_val)
        clean_var = re.sub(r"^export\s+", "", clean_var).strip()

        if not clean_var:
            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence=f"Empty or invalid environment variable claim '{claim.extracted_value}'.",
                suggested_value=None,
            )

        occurrences, all_codebase_vars = self._scan_repository(repo, clean_var)

        if occurrences:
            files_list = list(dict.fromkeys(loc["file"] for loc in occurrences))
            evidence = (
                f"Environment variable '{clean_var}' found in {len(occurrences)} location(s) "
                f"across {len(files_list)} file(s): {', '.join(files_list[:5])}"
            )
            if len(files_list) > 5:
                evidence += f" (+{len(files_list) - 5} more)"
            return VerificationResult(
                verdict=Verdict.VERIFIED,
                evidence=evidence,
                suggested_value=None,
                metadata={"occurrences": occurrences, "files": files_list},
            )

        # Env var missing: search for similar env var names
        candidates = [v for v in all_codebase_vars if v != clean_var and v not in COMMON_FALSE_POSITIVES]

        closest_match = None
        if self.memory:
            try:
                # Index discovered candidate symbols into memory
                if candidates:
                    self.memory.index_symbols([
                        {"file_path": str(repo), "symbol_name": c, "symbol_type": "env_var"}
                        for c in candidates
                    ])
                similar = self.memory.find_similar_symbol(clean_var, symbol_type="env_var", limit=1)
                if similar and (similar[0].get("similarity", 0) >= 0.50 or similar[0].get("distance", 1.0) <= 0.50):
                    closest_match = similar[0].get("symbol_name")
            except Exception:
                pass

        if not closest_match:
            closest_match = self._find_closest_env_var(clean_var, candidates)

        if closest_match:
            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence=f"Environment variable '{clean_var}' not found in codebase. Similar variable '{closest_match}' found.",
                suggested_value=closest_match,
                metadata={"closest_match": closest_match, "candidates_count": len(candidates)},
            )

        return VerificationResult(
            verdict=Verdict.DRIFTED,
            evidence=f"Environment variable '{clean_var}' not found in any environment files or source code.",
            suggested_value=None,
            metadata={"candidates_count": len(candidates)},
        )

    def _scan_repository(
        self, repo: Path, target_var: str
    ) -> Tuple[List[Dict[str, Any]], Set[str]]:
        """Scans relevant repo files for target_var occurrences and collects all candidate env vars."""
        occurrences: List[Dict[str, Any]] = []
        codebase_vars: Set[str] = set()
        target_pattern = re.compile(r"\b" + re.escape(target_var) + r"\b")

        if not repo.exists() or not repo.is_dir():
            return occurrences, codebase_vars

        # Identify files to check
        files_to_check: List[Path] = []

        # 1. Root env files and compose files
        for f_name in ENV_TEMPLATE_FILES + DOCKER_FILES:
            target = repo / f_name
            if target.is_file():
                files_to_check.append(target)

        # 2. Walk repo for source files and nested env files
        for root, dirs, files in os.walk(repo):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
            curr_dir = Path(root)

            for f in files:
                f_lower = f.lower()
                is_env = f_lower.startswith(".env") or f_lower.endswith(".env") or f_lower in ("compose.yml", "compose.yaml")
                is_src = any(f_lower.endswith(ext) for ext in SOURCE_EXTENSIONS)
                if is_env or is_src:
                    p = curr_dir / f
                    if p not in files_to_check:
                        files_to_check.append(p)

        # Scan each file
        for file_path in files_to_check:
            try:
                rel_path = file_path.relative_to(repo).as_posix()
            except ValueError:
                rel_path = file_path.as_posix()

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            # Check for target var in file
            if target_pattern.search(content):
                for idx, line in enumerate(content.splitlines(), start=1):
                    if target_pattern.search(line):
                        occurrences.append({
                            "file": rel_path,
                            "line": idx,
                            "snippet": line.strip()[:120],
                        })

            # Extract candidate env vars from file
            file_vars = self._extract_env_vars_from_content(content, file_path.name)
            codebase_vars.update(file_vars)

        return occurrences, codebase_vars

    @staticmethod
    def _extract_env_vars_from_content(content: str, filename: str) -> Set[str]:
        """Extracts candidate environment variable identifiers from file content."""
        vars_found: Set[str] = set()

        # 1. From .env files: lines like KEY=value
        if ".env" in filename.lower():
            for line in content.splitlines():
                stripped = line.strip().lstrip("#").strip()
                match = re.match(r"^([A-Z0-9_]{2,})\s*=", stripped)
                if match:
                    vars_found.add(match.group(1))

        # 2. Code patterns:
        # Python: os.environ[...] / os.getenv(...)
        for m in re.findall(r"os\.(?:environ(?:\[|\.get\()|getenv\()\s*['\"]([A-Z0-9_]{2,})['\"]", content):
            vars_found.add(m)

        # JS/TS: process.env.KEY or process.env['KEY']
        for m in re.findall(r"process\.env\.([A-Z0-9_]{2,})", content):
            vars_found.add(m)
        for m in re.findall(r"process\.env\[['\"]([A-Z0-9_]{2,})['\"]\]", content):
            vars_found.add(m)

        # Go: os.Getenv("KEY") / os.LookupEnv("KEY")
        for m in re.findall(r"os\.(?:Getenv|LookupEnv)\(['\"]([A-Z0-9_]{2,})['\"]", content):
            vars_found.add(m)

        # Rust: env::var("KEY")
        for m in re.findall(r"env::var\(['\"]([A-Z0-9_]{2,})['\"]", content):
            vars_found.add(m)

        # Docker compose / dockerfile: - KEY=val or KEY: val or EXPOSE or ENV KEY=val
        for m in re.findall(r"^\s*-\s*([A-Z0-9_]{2,})(?:=|\s*:)", content, re.MULTILINE):
            vars_found.add(m)
        for m in re.findall(r"^\s*([A-Z0-9_]{2,}):\s*\S", content, re.MULTILINE):
            vars_found.add(m)
        for m in re.findall(r"^\s*ENV\s+([A-Z0-9_]{2,})[=\s]", content, re.MULTILINE):
            vars_found.add(m)

        # General uppercase identifiers in config/env
        if filename.endswith((".yml", ".yaml", ".env")):
            for m in re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", content):
                if m not in COMMON_FALSE_POSITIVES:
                    vars_found.add(m)

        return vars_found

    @classmethod
    def _find_closest_env_var(cls, target: str, candidates: List[str]) -> Optional[str]:
        """Finds closest matching env var name using difflib and semantic token scoring."""
        if not candidates:
            return None

        # 1. Standard difflib match
        matches = difflib.get_close_matches(target, candidates, n=3, cutoff=0.55)
        if matches:
            return matches[0]

        # 2. Token overlap and substring scoring (e.g. MONGO_URL vs MONGODB_URI)
        best_candidate = None
        best_score = 0.0

        target_tokens = set(re.split(r"[_\-]", target.upper()))
        synonym_pairs = [
            ({"URL", "URI", "HOST", "ADDR", "ENDPOINT"}),
            ({"KEY", "TOKEN", "SECRET"}),
            ({"PASS", "PASSWORD", "PWD"}),
            ({"PORT", "LISTEN"}),
            ({"DB", "DATABASE"}),
        ]

        for cand in candidates:
            cand_upper = cand.upper()
            score = difflib.SequenceMatcher(None, target.upper(), cand_upper).ratio()

            cand_tokens = set(re.split(r"[_\-]", cand_upper))

            # Shared base token (e.g. MONGO vs MONGODB)
            has_stem_match = any(
                (t1 in t2 or t2 in t1) and len(min(t1, t2, key=len)) >= 4
                for t1 in target_tokens for t2 in cand_tokens
            )
            if has_stem_match:
                score += 0.20

            # Synonym group overlap (e.g. URL vs URI)
            for group in synonym_pairs:
                if (target_tokens & group) and (cand_tokens & group):
                    score += 0.15
                    break

            if score > best_score:
                best_score = score
                best_candidate = cand

        if best_candidate and best_score >= 0.50:
            return best_candidate

        # 3. Fallback lower threshold
        fallback = difflib.get_close_matches(target, candidates, n=1, cutoff=0.45)
        return fallback[0] if fallback else None
