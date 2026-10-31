"""File path verifier for DocDrift."""

import difflib
import re
import os
from pathlib import Path
from typing import List, Optional, Set

from docdrift.models import Claim, ClaimType, Verdict, VerificationResult
from docdrift.verifiers.base import BaseVerifier
from docdrift.verifiers.code_analyzer import CodeAnalyzer


DEFAULT_IGNORED_DIRS: Set[str] = {
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
    ".docdrift",
    ".turbo",
}


class FileVerifier(BaseVerifier):
    """Verifies ClaimType.FILE_PATH claims against the repository filesystem.

    Resolves paths relative to the repository root and relative to the markdown
    document's parent directory. If not found, scans the repository tree and
    finds moved or renamed candidate paths using difflib.
    """

    def can_verify(self, claim: Claim) -> bool:
        """Checks if this verifier handles the given claim."""
        return claim.claim_type == ClaimType.FILE_PATH

    def verify(self, claim: Claim, repo_path: str) -> VerificationResult:
        """Verifies whether the path exists or has drifted."""
        repo = Path(repo_path).resolve()
        raw_val = claim.extracted_value.strip().strip("'\"`")

        # 0. Check if raw_val is an external HTTP/HTTPS URL
        if re.match(r"^(https?://|ftp://|//)", raw_val, re.IGNORECASE):
            return VerificationResult(
                verdict=Verdict.VERIFIED,
                evidence=f"External documentation URL '{raw_val}' verified.",
                suggested_value=None,
            )

        # 0b. Check if raw_val is the label or target of a markdown link to an external URL in raw_line
        # Example: [GOVERNANCE.md](https://github.com/expressjs/discussions/blob/HEAD/docs/GOVERNANCE.md)
        if claim.raw_line:
            link_pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')
            for m in link_pattern.finditer(claim.raw_line):
                link_text = m.group(1).strip()
                link_url = m.group(2).strip()
                if raw_val == link_text or raw_val in link_text:
                    return VerificationResult(
                        verdict=Verdict.VERIFIED,
                        evidence=f"Label '{raw_val}' links to external documentation: '{link_url}'.",
                        suggested_value=None,
                    )
                if raw_val == link_url or raw_val in link_url:
                    return VerificationResult(
                        verdict=Verdict.VERIFIED,
                        evidence=f"External documentation URL '{link_url}' verified.",
                        suggested_value=None,
                    )

        # 0c. Check if path is absolute OS path or outside repo execution context
        is_abs = (
            raw_val.startswith(("/tmp", "\tmp", "/var", "/etc", "~"))
            or (len(raw_val) >= 2 and raw_val[1] == ":" and raw_val[0].isalpha())
            or getattr(claim, "runs_in", "repo") in ("user_machine", "generated_app")
        )
        if is_abs:
            return VerificationResult(
                verdict=Verdict.UNKNOWN,
                evidence=f"Path '{raw_val}' is an absolute or user machine environment path, not a repository file.",
                suggested_value=None,
            )

        # 0e. Check if target is an HTTP API endpoint route (/api/..., /v1/..., or mentioned with HTTP verbs)
        raw_line_l = (claim.raw_line or "").lower()
        is_api_route = (
            raw_val.startswith(("/api/", "/v1/", "/v2/", "/graphql", "/health", "/auth/"))
            or any(raw_val.lower().startswith(m) for m in ("get ", "post ", "put ", "delete ", "patch "))
            or (raw_val.startswith("/") and any(m in raw_line_l for m in ("post ", "get ", "put ", "delete ", "endpoint", "route", "api")))
        )
        if is_api_route:
            clean_route = re.sub(r"^(get|post|put|delete|patch)\s+", "", raw_val, flags=re.IGNORECASE).strip()
            route_pat = re.compile(re.escape(clean_route))
            found_route = False
            for code_file in repo.glob("**/*"):
                if not code_file.is_file():
                    continue
                if code_file.suffix.lower() not in (".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs"):
                    continue
                if any(d in code_file.parts for d in (".git", "node_modules", "dist", "build", ".venv", "venv")):
                    continue
                try:
                    if route_pat.search(code_file.read_text(encoding="utf-8", errors="ignore")):
                        found_route = True
                        break
                except Exception:
                    pass

            if found_route:
                return VerificationResult(
                    verdict=Verdict.VERIFIED,
                    evidence=f"API endpoint route '{clean_route}' verified in repository codebase.",
                    suggested_value=None,
                )
            else:
                return VerificationResult(
                    verdict=Verdict.UNKNOWN,
                    evidence=f"API endpoint route '{clean_route}' referenced in documentation.",
                    suggested_value=None,
                )

        # 0d. Check if the target is an uncommitted local runtime file (.env, .env.local, .venv)
        LOCAL_RUNTIME_FILES = {
            ".env", ".env.local", ".env.development.local", ".env.test.local",
            ".venv", "venv", ".pytest_cache", "__pycache__"
        }
        raw_val_clean = raw_val.split("#")[0].strip().replace("\\", "/")
        base_name = os.path.basename(raw_val_clean).lower()
        if base_name in LOCAL_RUNTIME_FILES:
            raw_line_lower = (claim.raw_line or "").lower()
            doc_file = Path(claim.doc_path) if claim.doc_path else Path("README.md")
            doc_dir = (doc_file if doc_file.is_absolute() else (repo / doc_file)).parent.resolve()

            templates = [
                doc_dir / ".env.example",
                doc_dir / ".env.sample",
                doc_dir / ".env.template",
                repo / ".env.example",
                repo / ".env.sample",
                repo / ".env.template",
                repo / "backend" / ".env.example",
            ]
            found_template = next((t for t in templates if t.exists()), None)

            is_negative_or_template = any(k in raw_line_lower for k in (
                "never commit", "do not commit", "dont commit", "gitignore",
                "copy", "cp ", "touch ", "create ", "add ", "copy-item", "example"
            ))
            if is_negative_or_template or found_template:
                template_note = f"; template '{found_template.name}' found." if found_template else "."
                return VerificationResult(
                    verdict=Verdict.VERIFIED,
                    evidence=f"Local runtime file '{raw_val}' is correctly omitted from git repository{template_note}",
                    suggested_value=None,
                )

        # Strip URL/markdown anchor fragments (e.g. docs/install.md#prerequisites)
        clean_target = raw_val.split("#")[0].strip()
        if not clean_target:
            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence=f"Invalid or empty file path: '{claim.extracted_value}'.",
                suggested_value=None,
            )

        # Check if target specifies a symbol (e.g. backend/app/main.py:app)
        claimed_symbol = claim.metadata.get("symbol")
        if ":" in clean_target and not (len(clean_target) >= 2 and clean_target[1] == ":" and clean_target[0].isalpha()):
            parts = clean_target.split(":", 1)
            clean_target = parts[0].strip()
            if not claimed_symbol and len(parts) > 1:
                claimed_symbol = parts[1].strip()

        # Normalize relative path: remove leading slashes (e.g. /src/main.py -> src/main.py)
        normalized = clean_target.replace("\\", "/").lstrip("/")
        if normalized.startswith("./"):
            normalized = normalized[2:]

        # 1. Resolve relative to repository root
        root_candidate = (repo / normalized).resolve()
        if self._path_exists(root_candidate, repo):
            rel_display = root_candidate.relative_to(repo).as_posix()
            return self._verify_file_content_and_symbol(root_candidate, raw_val, rel_display, claim, claimed_symbol)

        # 2. Resolve relative to doc_path's directory
        if claim.doc_path:
            doc_file = Path(claim.doc_path)
            doc_dir = (doc_file if doc_file.is_absolute() else (repo / doc_file)).parent.resolve()
            doc_candidate = (doc_dir / clean_target).resolve()
            if self._path_exists(doc_candidate, repo):
                try:
                    rel_display = doc_candidate.relative_to(repo).as_posix()
                except ValueError:
                    rel_display = doc_candidate.as_posix()
                return self._verify_file_content_and_symbol(doc_candidate, raw_val, rel_display, claim, claimed_symbol)

        # 3. Search across all files and directories in the entire repository tree
        all_candidates = self._scan_repo_paths(repo)

        # Check if the filename / relative path exists anywhere in any subdirectory
        target_name = Path(normalized).name.lower()
        exact_basename_matches = [
            p for p in all_candidates
            if p.lower() == normalized.lower() or Path(p).name.lower() == target_name
        ]
        if exact_basename_matches:
            found_rel = exact_basename_matches[0]
            # If the user specified a directory path with slashes that disagrees with reality:
            if "/" in normalized and not found_rel.lower().endswith(normalized.lower()):
                return VerificationResult(
                    verdict=Verdict.DRIFTED,
                    evidence=f"Path '{raw_val}' does not exist at specified directory. Found in repository at '{found_rel}'.",
                    suggested_value=found_rel,
                    metadata={"closest_match": found_rel},
                )
            else:
                found_path = (repo / found_rel).resolve()
                return self._verify_file_content_and_symbol(found_path, raw_val, found_rel, claim, claimed_symbol, found_in_search=True)

        # 4. Fuzzy match for moved or renamed candidates
        closest_match = self._find_closest_match(normalized, all_candidates)
        if closest_match:
            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence=f"Path '{raw_val}' does not exist in repository. Did you mean '{closest_match}'?",
                suggested_value=closest_match,
                metadata={"closest_match": closest_match},
            )

        return VerificationResult(
            verdict=Verdict.DRIFTED,
            evidence=f"Path '{raw_val}' does not exist anywhere in the repository.",
            suggested_value=None,
        )

    def _verify_file_content_and_symbol(
        self,
        candidate: Path,
        raw_val: str,
        rel_display: str,
        claim: Claim,
        claimed_symbol: Optional[str] = None,
        found_in_search: bool = False,
    ) -> VerificationResult:
        """Validates file syntax and declared symbols."""
        verb = "found" if found_in_search else "exists"
        if candidate.is_dir():
            return VerificationResult(
                verdict=Verdict.VERIFIED,
                evidence=f"Directory '{raw_val}' {verb} in repository at '{rel_display}'.",
                suggested_value=None,
                metadata={"resolved_path": str(candidate), "is_dir": True},
            )

        # 1. Validate file syntax (AST / compilation)
        syntax_ok, syntax_err = CodeAnalyzer.validate_file_syntax(candidate)
        if not syntax_ok:
            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence=f"File '{raw_val}' exists in repository at '{rel_display}', but has syntax error: {syntax_err}.",
                suggested_value=None,
                metadata={"resolved_path": str(candidate), "syntax_valid": False, "syntax_error": syntax_err},
            )

        # 2. Check claimed symbol if specified
        target_symbol = claimed_symbol or claim.metadata.get("symbol")
        if target_symbol:
            is_decl, symbols, closest = CodeAnalyzer.verify_symbol_in_file(candidate, target_symbol)
            if not is_decl:
                evidence = f"Symbol '{target_symbol}' is not declared in '{rel_display}'."
                if closest:
                    evidence += f" Did you mean '{closest}'?"
                return VerificationResult(
                    verdict=Verdict.DRIFTED,
                    evidence=evidence,
                    suggested_value=closest,
                    metadata={"resolved_path": str(candidate), "declared_symbols": list(symbols)},
                )

        return VerificationResult(
            verdict=Verdict.VERIFIED,
            evidence=f"File '{raw_val}' {verb} in repository at '{rel_display}'.",
            suggested_value=None,
            metadata={"resolved_path": str(candidate), "is_dir": False, "syntax_valid": True},
        )

    @staticmethod
    def _path_exists(candidate: Path, repo_root: Path) -> bool:
        """Safely checks if candidate exists and is inside or at repo_root."""
        try:
            return candidate.exists()
        except (OSError, ValueError):
            return False

    @staticmethod
    def _scan_repo_paths(repo_root: Path) -> List[str]:
        """Scans the repository and returns relative POSIX paths for files and directories."""
        candidates: List[str] = []
        if not repo_root.exists() or not repo_root.is_dir():
            return candidates

        for root, dirs, files in os.walk(repo_root):
            # Prune ignored directories
            dirs[:] = [
                d for d in dirs
                if d not in DEFAULT_IGNORED_DIRS and not d.startswith(".")
            ]

            curr_dir = Path(root)
            for d in dirs:
                full_dir = curr_dir / d
                try:
                    rel = full_dir.relative_to(repo_root).as_posix()
                    candidates.append(rel)
                except ValueError:
                    pass

            for f in files:
                full_file = curr_dir / f
                try:
                    rel = full_file.relative_to(repo_root).as_posix()
                    candidates.append(rel)
                except ValueError:
                    pass

        return candidates

    @classmethod
    def _find_closest_match(cls, target_path: str, candidates: List[str]) -> Optional[str]:
        """Finds the closest matching file or directory path from candidates."""
        if not candidates or not target_path:
            return None

        target_posix = target_path.replace("\\", "/").strip("/")
        target_name = Path(target_posix).name

        # 1. Exact full-path difflib close matches
        matches = difflib.get_close_matches(target_posix, candidates, n=3, cutoff=0.55)
        if matches:
            return matches[0]

        # 2. Check for moved files where the filename (basename) matches exactly
        basename_matches = [c for c in candidates if Path(c).name == target_name]
        if len(basename_matches) == 1:
            return basename_matches[0]
        if len(basename_matches) > 1:
            # Pick the one with highest path similarity
            scored = difflib.get_close_matches(target_posix, basename_matches, n=1, cutoff=0.0)
            if scored:
                return scored[0]

        # 3. Check for close filename (basename) matches
        candidate_names = {Path(c).name: c for c in candidates}
        close_names = difflib.get_close_matches(target_name, list(candidate_names.keys()), n=1, cutoff=0.6)
        if close_names:
            matching_name = close_names[0]
            for c in candidates:
                if Path(c).name == matching_name:
                    return c

        # 4. Fallback: lower threshold difflib match
        fallback = difflib.get_close_matches(target_posix, candidates, n=1, cutoff=0.4)
        return fallback[0] if fallback else None
